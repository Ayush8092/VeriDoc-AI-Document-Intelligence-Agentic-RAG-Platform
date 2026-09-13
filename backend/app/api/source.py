"""Source Viewer API (Phase 4.1(5), spec group 2): the read-only endpoints
a frontend citation click-through needs to walk

    Citation -> Document -> Page -> Source object -> Bounding box

A `Citation` (see `app/schemas/documents.py`) deliberately carries only
`source_file` (a bare filename), not a `document_id` — adding a
`document_id` to the citation-construction path in `app/rag/llm.py`
would touch the citation architecture this pass is explicitly told not
to rewrite. Instead, the viewer resolves `source_file -> Document` via
`GET /documents/by-filename/{filename}` below, using the SAME
`Document`/`DocumentVersion` metadata `GET /documents` already exposes —
this module adds no new persistence, only new read paths over existing
tables plus on-demand page rendering.

**Coordinate spaces are never touched here.** A citation's `bbox` and
`coordinate_space` (PDF_POINTS or IMAGE_PIXELS — see
`app/documents/models.py::CoordinateSpace`) are returned to the frontend
exactly as stored; this module does not attempt to project a bbox onto
a rendered image itself (that scaling depends on the ACTUAL rendered
image's pixel dimensions vs. the PDF's point dimensions, which only the
frontend, which requested a specific render, knows at the moment it
needs to draw a highlight box — see `frontend/lib/bbox.ts`). What this
module guarantees is that `SourcePageInfo.rendered_width`/
`rendered_height` (the ACTUAL pixel size of the image just rendered) are
always returned alongside the image, so the frontend has everything it
needs to do that projection correctly without guessing a DPI.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask
from PIL import Image

from app import vectorstore
from app.clients import get_index, get_pinecone
from app.core.config import Settings, get_settings
from app.db.models import Document, DocumentVersion, User
from app.db.session import init_db, session_scope
from app.schemas.documents import DocumentOut, DocumentVersionOut, PageObjectOut, PageObjectsResponse
from app.security.auth import get_current_user_optional
from app.services.ingestion_service import _resolve_dir
from app.storage.base import CORPUS_KEY_PREFIX, UPLOADS_KEY_PREFIX, get_backend, local_path

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["source-viewer"])

_PAGE_IMAGE_CACHE_DIR = "data/page_images"
_RENDERABLE_TYPES = {"pdf", "png", "jpg", "jpeg"}


def _get_document_or_404(session, document_id: int, current_user: User | None = None) -> Document:
    """Phase 5 tenant isolation: EVERY endpoint in this module that
    resolves a `document_id` goes through this function — a single
    enforcement point rather than repeating the ownership check at each
    call site, so a future new endpoint can't forget it.

    A document with `owner_id is not None` belonging to someone other
    than `current_user` is treated EXACTLY like a nonexistent document
    (404, not 403) — a 403 would confirm to an unauthorized caller that
    a given `document_id` exists and is just forbidden, which is itself
    information leakage about another user's private data (which ids are
    in use, roughly how many documents they have, ...). See
    `app.security.auth`'s docstring for the same "no side-channel"
    principle applied to login.
    """
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No document with id={document_id}")
    if doc.owner_id is not None and (current_user is None or doc.owner_id != current_user.id):
        raise HTTPException(status_code=404, detail=f"No document with id={document_id}")
    return doc


def _to_document_out(d: Document, version: DocumentVersion | None) -> DocumentOut:
    version_out = DocumentVersionOut(**version.to_dict()) if version else None
    return DocumentOut(
        id=d.id,
        source_root=d.source_root,
        filename=d.filename,
        file_type=d.file_type,
        title=d.title,
        owner_id=d.owner_id,
        created_at=d.created_at.isoformat() if d.created_at else None,
        updated_at=d.updated_at.isoformat() if d.updated_at else None,
        current_version=version_out,
    )


@router.get("/by-filename/{filename}", response_model=DocumentOut)
def get_document_by_filename(
    filename: str, current_user: User | None = Depends(get_current_user_optional)
) -> DocumentOut:
    """Resolve a `Citation.source_file` to the `Document` it came from —
    the first hop of `Citation -> Document`. `source_file` alone (without
    `source_root`) is sufficient in practice: filenames are sanitized and
    deduplicated per-root **and per-owner** at upload time
    (`app/api/documents.py::_safe_filename`, and Phase 5's per-user
    upload subdirectory — see `app.services.ingestion_service`'s
    docstring), and corpus/upload filename collisions are not a case
    this product's ingestion currently guards against either way — if
    one exists, the corpus copy wins (checked first) since it's the
    stable, versioned reference set citations are more likely to point
    at in a demo/eval context.

    Phase 5 tenant isolation: only resolves to a document this
    `current_user` (or the public corpus) actually owns — a citation
    naming a filename that exists ONLY in another user's private uploads
    is treated as not found, same as `_get_document_or_404`.
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        query = session.query(Document).filter(Document.filename == filename)
        if current_user is not None:
            query = query.filter((Document.owner_id.is_(None)) | (Document.owner_id == current_user.id))
        else:
            query = query.filter(Document.owner_id.is_(None))
        doc = query.order_by((Document.source_root != "corpus")).first()
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No document found for filename {filename!r}")
        version = session.get(DocumentVersion, doc.current_version_id) if doc.current_version_id else None
        return _to_document_out(doc, version)


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, current_user: User | None = Depends(get_current_user_optional)) -> DocumentOut:
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        doc = _get_document_or_404(session, document_id, current_user)
        version = session.get(DocumentVersion, doc.current_version_id) if doc.current_version_id else None
        return _to_document_out(doc, version)


@router.get("/{document_id}/capabilities")
def get_source_capabilities(
    document_id: int, current_user: User | None = Depends(get_current_user_optional)
) -> dict:
    """Can this document be shown as page images at all? A `.md`/`.txt`
    source has no page geometry — the viewer must check this before ever
    requesting `/pages/{n}/image`, rather than requesting an image and
    handling a 404 as its "not renderable" signal.
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        doc = _get_document_or_404(session, document_id, current_user)
        version = session.get(DocumentVersion, doc.current_version_id) if doc.current_version_id else None
        page_count = version.page_count if version else 0
        renderable = doc.file_type in _RENDERABLE_TYPES
        body = {"renderable": renderable, "page_count": page_count}
        if not renderable:
            body["reason"] = f"'{doc.file_type}' documents have no page images (text extracted directly)."
        return body


def _source_key(doc: Document) -> str:
    """The storage key (`app.storage.base.StorageBackend`) `doc`'s
    original file lives at — mirrors `app/api/documents.py::_upload_key`
    exactly (same `uploads/<owner_id>/<filename>` / `uploads/<filename>`
    shape) so both modules agree on where an uploaded file is, and adds
    the `corpus/<filename>` shape for the pre-seeded shared corpus.
    """
    if doc.source_root == "uploads" and doc.owner_id is not None:
        return f"{UPLOADS_KEY_PREFIX}/{doc.owner_id}/{doc.filename}"
    if doc.source_root == "uploads":
        return f"{UPLOADS_KEY_PREFIX}/{doc.filename}"
    return f"{CORPUS_KEY_PREFIX}/{doc.filename}"


def _checked_source_key(settings: Settings, doc: Document) -> str:
    """The storage key for `doc`'s original file, after confirming it
    actually exists in storage. Shared by both `_resolve_source_path`
    (used where the caller needs the raw path + backend, to control its
    own cleanup timing — see `get_source_file`) and `get_page_image`
    (which only needs the path for the duration of one function call and
    uses the `local_path()` context manager instead)."""
    backend = get_backend(settings)
    key = _source_key(doc)
    if not backend.exists(key):
        raise HTTPException(
            status_code=404,
            detail=f"{doc.filename} is recorded in the database but not found in storage at key {key!r}.",
        )
    return key


def _resolve_source_path(settings: Settings, doc: Document) -> Path:
    """Resolve `doc`'s original file to a real local path via the
    storage abstraction (`app.storage.base.get_backend`) rather than
    constructing a `pathlib` path directly. For `LocalStorageBackend`
    this is functionally identical to the old direct-pathlib code and
    the returned path is the REAL stored file — releasing it (see
    callers below) is a no-op. For `S3StorageBackend` this downloads to
    a temp file the caller must release; see `get_source_file`/
    `get_page_image` for the two different cleanup strategies a
    "give me a local path" caller needs depending on whether the path
    must outlive this function call (`StorageBackend.get_local_path`'s
    docstring, and `app.storage.base.local_path`'s).
    """
    backend = get_backend(settings)
    key = _checked_source_key(settings, doc)
    local_file_path = backend.get_local_path(key)
    if local_file_path is None:  # pragma: no cover - every current backend returns a path
        raise HTTPException(status_code=500, detail=f"Storage backend could not resolve a local path for {key!r}.")
    return Path(local_file_path)


@router.get("/{document_id}/source")
def get_source_file(
    document_id: int, current_user: User | None = Depends(get_current_user_optional)
) -> FileResponse:
    """Serve the ORIGINAL uploaded/corpus file bytes — `Document -> source
    object` for a person who wants to open/download the actual PDF/DOCX/
    image rather than a rendered page. Read-only; the same file
    `ingest_corpus` already parsed, never a copy or re-derived version.
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        doc = _get_document_or_404(session, document_id, current_user)
        path = _resolve_source_path(settings, doc)
        backend = get_backend(settings)
        # Cleanup must happen AFTER the response body is fully sent, not
        # when this function returns — Starlette's `background` param on
        # `FileResponse` runs exactly then. Using `local_path()`'s `with`
        # block here would delete the file (for a remote backend) before
        # the response ever streams it. `release_local_path` is a no-op
        # for `LocalStorageBackend` (see its docstring), so this is safe
        # for the default/local case too.
        return FileResponse(
            path, filename=doc.filename, background=BackgroundTask(backend.release_local_path, str(path))
        )


def _cache_path(settings: Settings, doc: Document, page_number: int, dpi: int) -> Path:
    cache_dir = _resolve_dir(settings, _PAGE_IMAGE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keyed by content hash (not just filename) so a re-ingested,
    # changed file never serves a stale cached render of its OLD
    # content under the same name.
    key = hashlib.sha256(f"{doc.source_root}/{doc.filename}/{page_number}/{dpi}".encode()).hexdigest()[:24]
    return cache_dir / f"{key}.png"


def _native_page_size(doc: Document, path: Path, page_number: int) -> tuple[float, float] | None:
    """The bbox's OWN native page/image size — required by the frontend
    to scale a PDF_POINTS or IMAGE_PIXELS bbox onto a rendered image at
    an arbitrary DPI (see `frontend/lib/bbox.ts`'s docstring). Returns
    `None` (never a guessed size) if it can't be determined.
    """
    if doc.file_type == "pdf":
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return None
                page = pdf.pages[page_number - 1]
                return float(page.width), float(page.height)
        except Exception:  # noqa: BLE001 - native size is a best-effort header, never fatal
            return None
    if doc.file_type in ("png", "jpg", "jpeg"):
        try:
            with Image.open(path) as im:
                return float(im.width), float(im.height)
        except Exception:  # noqa: BLE001
            return None
    return None


def _render_page_image(settings: Settings, doc: Document, path: Path, page_number: int, dpi: int) -> Image.Image:
    if doc.file_type == "pdf":
        from app.documents.ocr.scanned_pdf import render_pdf_pages

        pages = render_pdf_pages(path, dpi=dpi)
        if page_number < 1 or page_number > len(pages):
            raise HTTPException(
                status_code=404,
                detail=f"{doc.filename} has {len(pages)} page(s); page {page_number} does not exist.",
            )
        return pages[page_number - 1]
    if doc.file_type in ("png", "jpg", "jpeg"):
        if page_number != 1:
            raise HTTPException(status_code=404, detail="Image documents have exactly one page (page 1).")
        return Image.open(path).convert("RGB")
    raise HTTPException(status_code=404, detail=f"'{doc.file_type}' documents have no renderable page images.")


@router.get("/{document_id}/pages/{page_number}/image")
def get_page_image(
    document_id: int,
    page_number: int,
    dpi: int = Query(default=150, ge=72, le=400),
    current_user: User | None = Depends(get_current_user_optional),
) -> Response:
    """Render (or serve a cached render of) one page as a PNG — the
    `Page -> Source object` hop. `dpi` is capped (72-400) since this
    renders on demand; the default (150) is enough for a comfortable
    zoomed reading view without being expensive to render repeatedly.

    Cached under `data/page_images/` (gitignored, same convention as
    OCR's own rendered pages — see `backend/data/README.md`) so
    navigating between citations on the same page/document doesn't
    re-rasterize the PDF every time.

    The response includes `X-Rendered-Width`/`X-Rendered-Height`
    headers — the frontend needs the ACTUAL pixel dimensions of what it
    just received to correctly scale a `PDF_POINTS` bbox onto the image
    (see `frontend/lib/bbox.ts`); reading them from response headers
    avoids a second round trip just to ask "how big is this image".
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        doc = _get_document_or_404(session, document_id, current_user)
        cache_path = _cache_path(settings, doc, page_number, dpi)

        if cache_path.is_file():
            # Cache hit: never touch storage/download the source file at
            # all (matters most for S3 — no point paying for a download
            # just to immediately discard it unused). Trade-off: native
            # PDF page point-size headers (X-Native-Width/Height), which
            # require reading the source file, are only included on a
            # cache MISS below — a cache hit omits them. No current
            # caller depends on them being present on every response
            # (the frontend's bbox scaling only needs X-Rendered-*, see
            # frontend/lib/bbox.ts); revisit if that changes.
            image = Image.open(cache_path)
            width, height = image.size
            return FileResponse(
                cache_path,
                media_type="image/png",
                headers={"X-Rendered-Width": str(width), "X-Rendered-Height": str(height)},
            )

        key = _checked_source_key(settings, doc)
        backend = get_backend(settings)

        with local_path(backend, key) as raw_path:
            path = Path(raw_path)
            native_size = _native_page_size(doc, path, page_number)
            native_headers = (
                {"X-Native-Width": str(native_size[0]), "X-Native-Height": str(native_size[1])}
                if native_size
                else {}
            )

            try:
                image = _render_page_image(settings, doc, path, page_number, dpi)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - rendering failures shouldn't 500 as an opaque trace
                _log.warning("page image render failed for document_id=%s page=%s: %s", document_id, page_number, exc)
                raise HTTPException(status_code=500, detail=f"Could not render page {page_number}: {exc}") from exc

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            try:
                cache_path.write_bytes(buf.getvalue())
            except OSError:
                _log.warning("could not write page image cache at %s (serving uncached)", cache_path)

            return Response(
                content=buf.getvalue(),
                media_type="image/png",
                headers={"X-Rendered-Width": str(image.width), "X-Rendered-Height": str(image.height), **native_headers},
            )

_OBJECT_SNIPPET_CHARS = 240


@router.get("/{document_id}/pages/{page_number}/objects", response_model=PageObjectsResponse)
def get_page_objects(
    document_id: int, page_number: int, current_user: User | None = Depends(get_current_user_optional)
) -> PageObjectsResponse:
    """Every chunk (text/table/figure/chart/visual) whose page range
    covers `page_number` — Phase 4.1(7): the source viewer's "what else
    is on this page" panel, and what powers "citation -> visual object
    navigation" / previous-next-evidence-on-this-page browsing without
    the frontend having to re-run a similarity search just to list
    objects it already knows the page of.

    A metadata-only lookup (`vectorstore.query_by_filter` — see its
    docstring), not a relevance search: everything covering this page is
    returned, unranked. Requires a reachable Pinecone index; if the
    index/namespace is empty or unreachable, this returns an empty list
    rather than a 500 — a source viewer opened before ingestion has run
    should show "nothing indexed for this page yet", not an error page.

    Phase 5 tenant isolation: `_get_document_or_404` already confirms
    `current_user` may see `document_id` itself, but the Pinecone lookup
    below filters ONLY by `source_file` + page range — two different
    users CAN each have their own same-named upload (Phase 5's per-user
    upload subdirectory does not deduplicate filenames ACROSS users), so
    without an explicit `owner_id` filter this would silently return
    chunks from the WRONG user's identically-named file if their page
    ranges happened to overlap. The filter is scoped to `doc.owner_id`
    specifically (this resolved document's actual owner — "" for the
    public corpus), not `current_user`'s whole allowed set, since the
    request is for objects on THIS one document.
    """
    settings = get_settings()
    init_db(settings)
    with session_scope(settings) as session:
        doc = _get_document_or_404(session, document_id, current_user)
        filename = doc.filename
        owner_id_filter_value = str(doc.owner_id) if doc.owner_id is not None else ""

    try:
        pc = get_pinecone(settings)
        index = get_index(pc, settings)
        matches = vectorstore.query_by_filter(
            index,
            settings,
            metadata_filter={
                "source_file": {"$eq": filename},
                "owner_id": {"$eq": owner_id_filter_value},
                "page_start": {"$lte": page_number},
                "page_end": {"$gte": page_number},
            },
        )
    except Exception as exc:  # noqa: BLE001 - an unreachable/empty index shouldn't break the viewer
        _log.warning(
            "page-objects lookup failed for document_id=%s page=%s: %s", document_id, page_number, exc
        )
        matches = []

    objects = [
        PageObjectOut(
            chunk_id=m["chunk_id"],
            object_id=m.get("object_id"),
            block_type=m.get("block_type", "text"),
            section=m.get("section", ""),
            snippet=(m.get("text") or "")[:_OBJECT_SNIPPET_CHARS],
            page_start=m.get("page_start"),
            page_end=m.get("page_end"),
            source=m.get("source", "native"),
            confidence=float(m.get("confidence", 1.0)),
            table_rows=m.get("table_rows"),
            bbox=m.get("bbox"),
            coordinate_space=m.get("coordinate_space"),
            caption=m.get("caption"),
            visual_type=m.get("visual_type"),
            chart_type=m.get("chart_type"),
            related_text=m.get("related_text"),
        )
        for m in matches
    ]
    # Stable, predictable ordering for a browsing panel: block position on
    # the page reads top-to-bottom in a document far more often than not,
    # and `bbox.y0` (top edge) approximates that without needing a real
    # reading-order field on the chunk metadata itself. Objects with no
    # bbox (e.g. plain paragraph text OCR'd with no geometry) sort last,
    # by chunk_id, so they're still present and stably ordered.
    def sort_key(o: PageObjectOut):
        y0 = o.bbox.get("y0") if o.bbox else None
        return (0, y0) if y0 is not None else (1, o.chunk_id)

    objects.sort(key=sort_key)
    return PageObjectsResponse(document_id=document_id, page_number=page_number, objects=objects)