"""Provider client factories: Gemini embeddings, Groq chat, Pinecone.

Kept separate from `graph.py` and `ingestion.py` so both the API server and
the ingestion CLI build the exact same clients from the same settings,
without duplicating setup or leaking keys into more than one place.

Secrets are received from `Settings` as `SecretStr` and only converted to
plain strings right here, at the provider-client boundary — never logged,
never put in the trace, never returned to a caller.

Provider SDKs used:
- `google-genai` (the current Gemini SDK) for embeddings, via `Embedder`.
- `groq` (the official SDK) for chat, via `get_chat`. Both SDKs support
  `response_format={"type": "json_object"}` / structured retries natively,
  which is why this project uses them directly rather than through a
  LangChain wrapper — one fewer layer between us and the actual provider
  behavior (retry semantics, error types, JSON mode) that this project
  depends on.
"""

import time

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

from app.core.config import Settings
from app.observability.trace import current_trace

_INDEX_READY_POLL_SECONDS = 1.0


def _normalize(values: list[float]) -> list[float]:
    """L2-normalize an embedding to unit length.

    `gemini-embedding-001` only guarantees unit-length output at its native
    3072-dimension output. When `output_dimensionality` truncates the vector
    (e.g. to 768, as this project does to keep Pinecone storage small),
    Google's own guidance is to renormalize before using cosine similarity —
    otherwise Pinecone's cosine metric is comparing vectors of inconsistent
    magnitude and retrieval quality quietly degrades.
    """
    vec = np.asarray(values, dtype="float32")
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm else vec.tolist()


class Embedder:
    """Gemini text embeddings via the `google-genai` SDK.

    Exposes the same two-method interface (`embed_documents`, `embed_query`)
    that `app/ingestion.py` and `app/retrieval.py` already call, so swapping
    the underlying SDK required no changes to either module.

    - Uses `RETRIEVAL_DOCUMENT` for corpus chunks and `RETRIEVAL_QUERY` for
      user questions — Gemini optimizes these two embedding types
      differently for asymmetric search, and mixing them up quietly hurts
      retrieval quality without raising any error.
    - Every embedding is L2-normalized (see `_normalize`).
    - Transient provider errors (429 / `RESOURCE_EXHAUSTED` / 5xx) are
      retried with bounded exponential backoff (1s, 2s, 4s, ...), capped at
      `settings.embedding_max_retries` attempts — this is a free-tier
      project and Gemini's free quota returns 429s under normal use.
    """

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self._model = settings.embedding_model
        self._dim = settings.embedding_dimension
        self._retries = settings.embedding_max_retries

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self._call_with_backoff(
            lambda: self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._dim
                ),
            )
        )
        if len(response.embeddings) != len(texts):
            raise RuntimeError(
                f"Gemini returned {len(response.embeddings)} embeddings for "
                f"{len(texts)} input texts."
            )
        self._record_usage(texts)
        return [_normalize(e.values) for e in response.embeddings]

    def _record_usage(self, texts: list[str]) -> None:
        """Best-effort embedding cost tracking (Phase 3B observability).

        `embed_content` responses don't reliably carry a token-usage field
        the way chat completions do, so this falls back to a documented
        ~4-chars-per-token estimate (the same heuristic OpenAI/Anthropic
        publish for English text) rather than pretending to an exact
        count it doesn't have. Skipped entirely outside a traced request
        (e.g. bulk ingestion, which isn't per-question cost-relevant).
        """
        trace = current_trace()
        if trace is None:
            return
        approx_tokens = sum(max(1, len(t) // 4) for t in texts)
        trace.record_llm_usage(model=self._model, prompt_tokens=approx_tokens, completion_tokens=0, purpose="embed")

    def _call_with_backoff(self, call):
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                return call()
            except genai_errors.APIError as exc:
                last_exc = exc
                transient = isinstance(exc, genai_errors.ServerError) or "RESOURCE_EXHAUSTED" in str(exc)
                if not transient or attempt == self._retries - 1:
                    raise
                time.sleep(2**attempt)  # 1, 2, 4, 8, 16s
        raise last_exc  # pragma: no cover — unreachable, satisfies type checkers


def get_embeddings(settings: Settings) -> Embedder:
    return Embedder(settings)


def get_chat(settings: Settings) -> Groq:
    """Groq chat client used for grading, query rewriting, and answer generation.

    `max_retries` is the SDK's own built-in exponential backoff for
    transient errors (429 / 5xx) — Groq's Python client retries these
    automatically, so we don't need to hand-roll that logic here the way we
    do for Gemini.
    """
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(
        api_key=settings.groq_api_key.get_secret_value(),
        max_retries=settings.groq_max_retries,
    )


def get_pinecone(settings: Settings):
    """Returns the vector-backend client object every call site in this
    codebase already expects (`.Index(name)`, `.list_indexes()`,
    `.describe_index(name)`, `.create_index(...)`) — either the real
    Pinecone SDK client, or `app.vectorstore_local.LocalPineconeClient`,
    a free/local, file-persisted equivalent that duck-types the same
    method surface (Phase 5 completion pass, round 4, item 2: "the
    project must be able to run for $0"). The name `get_pinecone` is
    kept (not renamed to something backend-neutral) deliberately: every
    existing call site (`app/services/ingestion_service.py`,
    `app/rag/graph.py`, `app/api/source.py`) and every existing test
    that patches `get_pinecone` (`tests/test_ingestion_service.py`)
    keeps working completely unchanged — the ENTIRE dispatch lives
    inside this one function's body. See
    `app/vectorstore_local.py`'s module docstring for the full design
    rationale.
    """
    if settings.vectorstore_backend == "local":
        from app.vectorstore_local import LocalPineconeClient

        return LocalPineconeClient(settings)
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set and VECTORSTORE_BACKEND is not 'local'. Either set "
            "VECTORSTORE_BACKEND=local (the default free/local mode — no Pinecone account needed) "
            "or provide PINECONE_API_KEY."
        )
    return Pinecone(api_key=settings.pinecone_api_key.get_secret_value())


def _validate_index_dimension(pc: Pinecone, settings: Settings) -> None:
    """Fail fast and clearly if EMBEDDING_DIMENSION doesn't match the index.

    This is the concrete failure mode this project has hit before:
    "Vector dimension 3072 does not match index dimension 768" — a
    confusing Pinecone error that only surfaces deep inside a query or
    upsert call. Checking once, right after connecting, turns that into an
    actionable startup/ingestion error instead.
    """
    description = pc.describe_index(settings.pinecone_index_name)
    index_dim = description.dimension
    if index_dim != settings.embedding_dimension:
        raise RuntimeError(
            f"Pinecone index '{settings.pinecone_index_name}' has dimension "
            f"{index_dim}, but EMBEDDING_DIMENSION is set to "
            f"{settings.embedding_dimension}. Either change EMBEDDING_DIMENSION "
            f"to match the existing index, or delete/recreate the index with "
            f"`python -m app.ingestion --reset` after removing it in the "
            f"Pinecone console."
        )


def ensure_index(pc: Pinecone, settings: Settings):
    """Create the Pinecone serverless index if it does not exist, then return a handle.

    Called from ingestion, which owns index lifecycle: the index should be
    created automatically the first time ingestion runs, without requiring
    a separate manual provisioning step. The API server uses `get_index`
    instead, which assumes ingestion already ran.
    """
    existing = {index["name"] for index in pc.list_indexes()}

    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    deadline = time.monotonic() + settings.pinecone_ready_timeout_seconds
    while True:
        status = pc.describe_index(settings.pinecone_index_name).status
        if status["ready"]:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index '{settings.pinecone_index_name}' did not become "
                f"ready within {settings.pinecone_ready_timeout_seconds}s."
            )
        time.sleep(_INDEX_READY_POLL_SECONDS)

    _validate_index_dimension(pc, settings)
    return pc.Index(settings.pinecone_index_name)


def get_index(pc: Pinecone, settings: Settings):
    """Return a handle to an existing index, failing fast with a clear message.

    The API does not create the index itself — ingestion is the single place
    that owns index lifecycle, so a missing index means "run ingestion first"
    rather than "silently create an empty index and return no results".

    **Local-backend exception**: the free/local vector backend
    (`VECTORSTORE_BACKEND=local`, the default — see `get_pinecone`) has
    none of the cost/quota reasons real Pinecone has for requiring
    ingestion to run first. `LocalPineconeClient.Index()` already
    handles "the on-disk index file doesn't exist yet" gracefully
    (returns an empty, dimension-0 index rather than raising), so for
    local this function returns it directly rather than applying the
    Pinecone-specific "must already exist" / dimension-validation
    guards below — those exist to catch mistakes against a real,
    external, paid service, and were incorrectly firing even for a
    brand-new, never-ingested LOCAL deployment (confirmed: this made
    `/ready`'s vector_store check — `app/api/health.py::_check_vector_store`
    — report NOT ready on first startup, before any ingestion, even
    with zero Pinecone/paid infrastructure involved anywhere — see
    `tests/test_free_mode_e2e.py::test_ready_vector_store_check_succeeds_via_local_backend`,
    which this fixes).

    Dimension validation still applies to the local backend once it
    actually HAS content (`dimension != 0`) — an embedding-model change
    corrupting a previously-populated local index is still worth
    catching; only the "nothing has been ingested yet" case is exempt.
    """
    if settings.vectorstore_backend == "local":
        index = pc.Index(settings.pinecone_index_name)
        description = pc.describe_index(settings.pinecone_index_name)
        if description.dimension and description.dimension != settings.embedding_dimension:
            raise RuntimeError(
                f"Local vector index '{settings.pinecone_index_name}' has dimension "
                f"{description.dimension}, but EMBEDDING_DIMENSION is set to "
                f"{settings.embedding_dimension}. Either change EMBEDDING_DIMENSION "
                f"to match the existing index, or delete "
                f"{settings.local_vector_index_path}/{settings.pinecone_index_name}.json "
                f"and re-ingest."
            )
        return index

    existing = {index["name"] for index in pc.list_indexes()}
    if settings.pinecone_index_name not in existing:
        raise RuntimeError(
            f"Pinecone index '{settings.pinecone_index_name}' does not exist yet. "
            "Run `python -m app.ingestion` first."
        )
    _validate_index_dimension(pc, settings)
    return pc.Index(settings.pinecone_index_name)