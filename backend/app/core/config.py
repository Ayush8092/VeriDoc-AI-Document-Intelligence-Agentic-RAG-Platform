"""Central application configuration.

Loads and validates application settings from environment variables and
`.env` using pydantic-settings. This module is the single source of truth
for configuration — other modules use `get_settings()` rather than reading
environment variables directly. Secrets are never intentionally printed.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Veridoc", validation_alias="APP_NAME")

    # ==========================================================
    # DEPLOYMENT ENVIRONMENT (Phase 5 completion pass, item 2)
    # ==========================================================
    # "development" (default) and "test" are permissive — a weak/default
    # JWT_SECRET_KEY is allowed so local dev and the test suite need no
    # extra setup. "production" is the one value that triggers
    # app.security.auth.assert_secret_is_safe_for_environment's hard
    # startup check (see app/main.py's lifespan). Any other string
    # behaves like "development" (permissive) rather than raising here —
    # the secret check, not this field's validation, is the enforcement
    # point, matching the spec's "add one if it doesn't exist" wording
    # (a free-form flag, not a closed enum).
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")

    # ==========================================================
    # API KEYS (all optional at import time — missing keys fail loudly
    # only when the corresponding provider is actually used, so the app,
    # its docs, and its tests can still run without every credential set)
    # ==========================================================

    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    pinecone_api_key: SecretStr | None = Field(default=None, validation_alias="PINECONE_API_KEY")

    # ==========================================================
    # DATABASE (document + ingestion metadata)
    # ==========================================================

    database_url: str = Field(
        default="sqlite:///./data/veridoc.db",
        validation_alias="DATABASE_URL",
        description=(
            "SQLAlchemy URL. Defaults to a local SQLite file for local dev "
            "and tests. Production deployments should set this to a "
            "PostgreSQL URL, e.g. postgresql+psycopg2://user:pass@host/db."
        ),
    )

    # ==========================================================
    # AUTHENTICATION (Phase 5)
    # ==========================================================
    # Authentication is OPT-IN at the transport layer: every endpoint that
    # accepts an optional current user (see app.security.auth) behaves
    # EXACTLY as it did before Phase 5 when no `Authorization` header is
    # sent — this is what keeps every pre-Phase-5 test passing unchanged
    # (see docs/architecture.md, "Phase 5: authentication is additive").
    # When a valid Bearer token IS presented, requests become scoped to
    # that user's own documents + the public/shared corpus (app.core
    # .config.Settings' "TENANT-SAFE VECTOR RETRIEVAL" contract — see
    # app/retrieval.py, app/lexical_index.py, app/api/*.py).
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("dev-insecure-secret-change-me-before-deploying"),
        validation_alias="JWT_SECRET_KEY",
        description=(
            "HMAC signing key for auth tokens. MUST be overridden via the "
            "JWT_SECRET_KEY environment variable in any non-local "
            "deployment — the default is intentionally obviously-insecure "
            "so it's never mistaken for a real secret (and long enough, "
            ">=32 bytes, that local dev/tests don't trip PyJWT's "
            "InsecureKeyLengthWarning on every run just from the default "
            "itself — that warning should mean something when it fires)."
        ),
    )
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60 * 24, ge=1, validation_alias="JWT_EXPIRE_MINUTES")

    # ==========================================================
    # STORAGE ABSTRACTION (Phase 5) — see app/storage/base.py
    # ==========================================================
    storage_backend: str = Field(
        default="local",
        validation_alias="STORAGE_BACKEND",
        description="'local' or 's3' — see app.storage.base.get_backend.",
    )
    storage_local_root: str = Field(
        default="data",
        validation_alias="STORAGE_LOCAL_ROOT",
        description="Root directory for LocalStorageBackend, relative to the backend project root.",
    )
    s3_bucket: str | None = Field(default=None, validation_alias="S3_BUCKET")
    s3_endpoint_url: str | None = Field(
        default=None,
        validation_alias="S3_ENDPOINT_URL",
        description="Set for S3-compatible providers (MinIO, R2, Spaces, ...); leave unset for real AWS S3.",
    )
    s3_region: str | None = Field(default=None, validation_alias="S3_REGION")
    s3_access_key: SecretStr | None = Field(default=None, validation_alias="S3_ACCESS_KEY")
    s3_secret_key: SecretStr | None = Field(default=None, validation_alias="S3_SECRET_KEY")

    # ==========================================================
    # VECTOR STORE BACKEND (Phase 5 completion pass, round 4, item 2)
    # ==========================================================
    # "local" (default — the free/$0 mode: no Pinecone account needed,
    # see app/vectorstore_local.py) or "pinecone" (opt-in, for a real
    # deployment that wants managed/hosted vector search). Every other
    # Pinecone setting below is only read at all when this is
    # "pinecone" — get_pinecone() in app/clients.py is the single
    # dispatch point.
    vectorstore_backend: str = Field(default="local", validation_alias="VECTORSTORE_BACKEND")
    local_vector_index_path: str = Field(
        default="data/vector_index",
        validation_alias="LOCAL_VECTOR_INDEX_PATH",
        description="Directory for the local vector backend's per-index JSON files. Relative to the backend project root.",
    )

    # ==========================================================
    # PINECONE (opt-in — see VECTORSTORE_BACKEND above)
    # ==========================================================

    pinecone_index_name: str = Field(default="veridoc-documents", validation_alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(default="veridoc-corpus", validation_alias="PINECONE_NAMESPACE")
    pinecone_cloud: str = Field(default="aws", validation_alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", validation_alias="PINECONE_REGION")
    pinecone_ready_timeout_seconds: int = Field(default=120, validation_alias="PINECONE_READY_TIMEOUT_SECONDS")

    # ==========================================================
    # EMBEDDINGS
    # ==========================================================

    embedding_model: str = Field(default="gemini-embedding-001", validation_alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=3072, ge=1, validation_alias="EMBEDDING_DIMENSION")
    embedding_max_retries: int = Field(default=5, ge=1, le=10, validation_alias="EMBEDDING_MAX_RETRIES")

    # ==========================================================
    # ANSWER LLM
    # ==========================================================

    answer_model: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias="ANSWER_MODEL",
        description=(
            "Groq-hosted chat model for answer generation (app/rag/llm.py). "
            "Phase 5 completion pass, round 4, item 8 (\"configuration audit "
            "— do NOT silently keep a stale/decommissioned model config\"): "
            "the previous default, llama-3.1-8b-instant, was deprecated by "
            "Groq (announced 2026-06-17, shut down 2026-08-16 for free/"
            "developer-tier usage — see https://console.groq.com/docs/deprecations); "
            "this project's own .env.example already used the correct "
            "replacement, openai/gpt-oss-20b (Groq's own documented "
            "migration recommendation for llama-3.1-8b-instant), but the "
            "code default here had drifted out of sync with it — any "
            "deployment relying on the code default (not setting "
            "ANSWER_MODEL explicitly) was silently calling a decommissioned "
            "model on every /ask request. Fixed to match .env.example."
        ),
    )
    answer_temperature: float = Field(default=0.0, ge=0.0, le=1.0, validation_alias="ANSWER_TEMPERATURE")
    groq_max_retries: int = Field(default=3, ge=1, le=10, validation_alias="GROQ_MAX_RETRIES")

    # ==========================================================
    # RETRIEVAL
    # ==========================================================

    top_k: int = Field(default=10, ge=1, le=20, validation_alias="TOP_K")
    score_threshold: float = Field(default=0.45, ge=0.0, le=1.0, validation_alias="SCORE_THRESHOLD")
    query_fanout: int = Field(default=3, ge=1, le=5, validation_alias="QUERY_FANOUT")

    # ==========================================================
    # RERANKING
    # ==========================================================

    rerank_enabled: bool = Field(default=True, validation_alias="RERANK_ENABLED")
    rerank_top_k: int = Field(default=8, ge=1, le=20, validation_alias="RERANK_TOP_K")

    cross_encoder_enabled: bool = Field(
        default=True,
        validation_alias="CROSS_ENCODER_ENABLED",
        description=(
            "Prefer a transformer cross-encoder for reranking over the BM25 "
            "lexical reranker. Falls back to BM25 automatically (per-process, "
            "logged once) if the model/package can't be loaded — see "
            "app/rag/cross_encoder.py."
        ),
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias="CROSS_ENCODER_MODEL"
    )

    # ==========================================================
    # HYBRID RETRIEVAL (dense + BM25 fusion) — Phase 3A
    # ==========================================================

    hybrid_retrieval_enabled: bool = Field(
        default=True,
        validation_alias="HYBRID_RETRIEVAL_ENABLED",
        description=(
            "Run BM25 lexical retrieval over the whole corpus (app/lexical_index.py) "
            "alongside dense Pinecone retrieval and fuse both with Reciprocal Rank "
            "Fusion (app/rag/fusion.py) before reranking/grading."
        ),
    )
    bm25_top_k: int = Field(default=10, ge=1, le=50, validation_alias="BM25_TOP_K")
    rrf_k: int = Field(default=60, ge=1, le=1000, validation_alias="RRF_K")
    lexical_index_path: str = Field(default="data/lexical_index.json", validation_alias="LEXICAL_INDEX_PATH")
    lexical_index_cache_enabled: bool = Field(default=True, validation_alias="LEXICAL_INDEX_CACHE_ENABLED")

    # ==========================================================
    # QUERY UNDERSTANDING — Phase 3A
    # ==========================================================

    query_classifier_llm_enabled: bool = Field(
        default=False,
        validation_alias="QUERY_CLASSIFIER_LLM_ENABLED",
        description=(
            "Use an LLM call to refine rule-based query classification for the "
            "LOOKUP-by-default case. Off by default: the deterministic rules "
            "(app/rag/query_classifier.py) already cover the routing-relevant "
            "distinctions without adding latency/cost to every request."
        ),
    )

    # ==========================================================
    # CLAIM-LEVEL GROUNDING — Phase 3A
    # ==========================================================

    claim_grounding_enabled: bool = Field(
        default=True,
        validation_alias="CLAIM_GROUNDING_ENABLED",
        description=(
            "Run claim extraction + per-claim entailment checking on generated "
            "answers (app/rag/grounding.py). Adds LLM calls after generation; "
            "disable for lower latency/cost at the expense of claim-level metrics."
        ),
    )
    claim_grounding_max_claims: int = Field(default=8, ge=1, le=30, validation_alias="CLAIM_GROUNDING_MAX_CLAIMS")

    # ==========================================================
    # CITATION VALIDATION — Phase 7 ablation axis (spec item 2, Pipeline E:
    # "+ Citation Validation"). See app/rag/graph.py::_validate_citations.
    # ==========================================================

    citation_validation_enabled: bool = Field(
        default=True,
        validation_alias="CITATION_VALIDATION_ENABLED",
        description=(
            "Reject any cited chunk_id that does not correspond to a chunk "
            "actually retrieved for this request, and refuse to answer if none "
            "of the model's citations survive that check (app.rag.llm.validate_citations, "
            "called from app.rag.graph._validate_citations). This is the safety "
            "gate that prevents a fabricated/hallucinated citation from ever "
            "reaching the response. ALWAYS true in production — this flag exists "
            "so evaluation/run_ablation.py's config H can measure, causally, how "
            "much this gate actually changes citation trustworthiness/grounded "
            "-answer-rate versus leaving it on (config E). Setting this false "
            "outside of that controlled ablation is not a supported/recommended "
            "configuration."
        ),
    )

    # ==========================================================
    # LANGGRAPH / RAG WORKFLOW
    # ==========================================================

    max_retrieval_loops: int = Field(default=2, ge=0, le=5, validation_alias="MAX_RETRIEVAL_LOOPS")
    recursion_limit: int = Field(default=20, ge=5, le=100, validation_alias="RECURSION_LIMIT")

    # ==========================================================
    # OCR / DOCUMENT PROCESSING
    # ==========================================================

    ocr_language: str = Field(default="eng", validation_alias="OCR_LANGUAGE")
    ocr_dpi: int = Field(default=200, ge=72, le=600, validation_alias="OCR_DPI")
    ocr_min_native_chars_per_page: int = Field(
        default=20,
        ge=0,
        validation_alias="OCR_MIN_NATIVE_CHARS_PER_PAGE",
        description="Below this many extracted native characters, a PDF page is treated as scanned and OCR'd.",
    )
    ocr_confidence_floor: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        validation_alias="OCR_CONFIDENCE_FLOOR",
        description="Tesseract word-confidence (0-100) below which a word is dropped from OCR output.",
    )


    # ==========================================================
    # FIGURES & CHARTS — Phase 4
    # ==========================================================
 
    figure_extraction_enabled: bool = Field(
        default=True,
        validation_alias="FIGURE_EXTRACTION_ENABLED",
        description="Extract embedded images from PDF/DOCX as candidate figures (app/documents/figures/).",
    )
    chart_understanding_enabled: bool = Field(
        default=False,
        validation_alias="CHART_UNDERSTANDING_ENABLED",
        description=(
            "Attempt structured chart data extraction (title/axes/legend/series) via a Gemini vision "
            "call (app/documents/figures/chart_understanding.py). Off by default: this is the one part "
            "of figure/chart handling that costs a live LLM call per chart. Chart TYPE refinement "
            "(bar/line/pie/scatter/area) is geometric-only and always runs regardless of this flag."
        ),
    )
    vision_model: str = Field(default="gemini-2.0-flash", validation_alias="VISION_MODEL")
 



    # ==========================================================
    # INGESTION
    # ==========================================================

    corpus_dir: str = Field(default="data/corpus", validation_alias="CORPUS_DIR")
    upload_dir: str = Field(default="data/uploads", validation_alias="UPLOAD_DIR")
    page_image_dir: str = Field(default="data/page_images", validation_alias="PAGE_IMAGE_DIR")
    max_upload_file_size_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1024, validation_alias="MAX_UPLOAD_FILE_SIZE_BYTES"
    )

    # ==========================================================
    # OBSERVABILITY — Phase 3B
    # ==========================================================

    opik_enabled: bool = Field(
        default=False,
        validation_alias="OPIK_ENABLED",
        description=(
            "Wrap the RAG pipeline's nodes and LLM calls with Opik tracing "
            "(app/observability/opik_integration.py). Off by default: with no "
            "OPIK_API_KEY configured, Opik still doesn't crash the app (it "
            "logs a warning and drops the trace), but there's no reason to "
            "pay the wrapping overhead or emit those warnings unless you're "
            "actually using Opik."
        ),
    )
    opik_api_key: SecretStr | None = Field(default=None, validation_alias="OPIK_API_KEY")
    opik_project_name: str = Field(default="veridoc", validation_alias="OPIK_PROJECT_NAME")
    opik_workspace: str | None = Field(default=None, validation_alias="OPIK_WORKSPACE")

    trulens_enabled: bool = Field(
        default=True,
        validation_alias="TRULENS_ENABLED",
        description=(
            "Record ablation-run feedback scores (app/evaluation/trulens_feedback.py) "
            "into a local TruLens session (data/trulens.sqlite by default — no network "
            "call, no API key needed; TruLens' own DB connector defaults to local "
            "SQLite). Independent of RAGAS and Opik."
        ),
    )
    trulens_database_url: str = Field(
        default="sqlite:///./data/trulens.sqlite", validation_alias="TRULENS_DATABASE_URL"
    )

    pricing_overrides_json: str | None = Field(
        default=None,
        validation_alias="PRICING_OVERRIDES_JSON",
        description=(
            'JSON object overriding app/observability/cost.py\'s default (approximate, '
            'not live) per-model $/1M-token rates, e.g. '
            '\'{"llama-3.3-70b-versatile": [0.59, 0.79]}\'.'
        ),
    )

    # ==========================================================
    # RATE LIMITING (Phase 5 completion pass, item 6) — see app/core/rate_limit.py
    # ==========================================================

    rate_limit_enabled: bool = Field(default=True, validation_alias="RATE_LIMIT_ENABLED")
    rate_limit_auth_per_minute: int = Field(default=20, ge=0, validation_alias="RATE_LIMIT_AUTH_PER_MINUTE")
    rate_limit_upload_per_minute: int = Field(default=10, ge=0, validation_alias="RATE_LIMIT_UPLOAD_PER_MINUTE")
    rate_limit_documents_per_minute: int = Field(
        default=60, ge=0, validation_alias="RATE_LIMIT_DOCUMENTS_PER_MINUTE"
    )
    rate_limit_ask_per_minute: int = Field(default=20, ge=0, validation_alias="RATE_LIMIT_ASK_PER_MINUTE")
    rate_limit_redis_url: str | None = Field(
        default=None,
        validation_alias="RATE_LIMIT_REDIS_URL",
        description=(
            "Optional. When set, RateLimitMiddleware (app/core/rate_limit.py) enforces limits "
            "against a shared Redis sorted-set window instead of this process's own memory — "
            "required for correct enforcement across multiple backend replicas/workers behind a "
            "load balancer (in-memory state is NOT shared between processes). When unset (the "
            "default), the in-memory limiter is used, matching this project's original single "
            "-process deployment target. e.g. redis://localhost:6379/0 . NOT runtime-verified "
            "against a live Redis instance in the environment this was authored in (no Redis "
            "reachable there) — see app/core/rate_limit.py's module docstring."
        ),
    )

    # ==========================================================
    # BACKGROUND INGESTION JOBS (Phase 5 completion pass, item 3.3) —
    # see app/services/ingestion_jobs.py, app/api/ingestion_jobs.py
    # ==========================================================

    ingestion_max_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias="INGESTION_MAX_ATTEMPTS",
        description=(
            "Default max_attempts for a new IngestionJob (app.services.ingestion_jobs.create_job) "
            "when the caller doesn't override it — a transient failure (network/upstream provider "
            "error) is retried up to this many times before the job becomes terminally FAILED. "
            "Was already referenced by create_job() before this field existed on Settings (a real, "
            "confirmed bug — every call to create_job() with max_attempts=None would have raised "
            "AttributeError at runtime); this field closes that gap."
        ),
    )
    ingestion_stale_processing_seconds: int = Field(
        default=15 * 60,
        ge=30,
        validation_alias="INGESTION_STALE_PROCESSING_SECONDS",
        description=(
            "Phase 5 completion pass (round 4), item 2 — crash recovery: a job stuck in PROCESSING "
            "for longer than this (its worker process died mid-run — see app.services.ingestion_jobs."
            "run_job's docstring, 'each state transition is committed in its own short transaction' — "
            "the crash could happen anywhere after that commit) is reclaimed by "
            "poll_and_process_once on its next poll, exactly as if it were freshly QUEUED/RETRYING. "
            "Set well above this deployment's typical ingest_corpus() run time — reclaiming a job "
            "that is still genuinely being processed (not actually stale) would let two workers "
            "process it at once."
        ),
    )

    # ==========================================================
    # API SERVER
    # ==========================================================

    api_host: str = Field(default="127.0.0.1", validation_alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, validation_alias="API_PORT")
    api_reload: bool = Field(default=True, validation_alias="API_RELOAD")
    include_trace: bool = Field(default=False, validation_alias="INCLUDE_TRACE")
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"], validation_alias="CORS_ALLOW_ORIGINS")
    service_init_timeout_seconds: int = Field(default=30, ge=1, le=300, validation_alias="SERVICE_INIT_TIMEOUT_SECONDS")

    def effective_storage_local_root(self) -> "Path | None":
        """The directory `app.storage.base.get_backend`/`validate_storage_config`
        should actually treat as `LocalStorageBackend`'s root — Phase 5
        completion pass, item 3.4.

        `storage_local_root` and `upload_dir`/`corpus_dir` are three
        independently-settable fields that MUST agree for
        `app/api/documents.py`/`app/api/source.py` (which write/read via
        `STORAGE_LOCAL_ROOT` + a fixed `"uploads/"`/`"corpus/"` key
        prefix) and `app.services.ingestion_service` (which still scans
        `UPLOAD_DIR`/`CORPUS_DIR` directly) to be looking at the same
        files. Requiring all three to be configured in lockstep by hand
        is exactly the kind of drift that silently loses uploaded files
        (see `validate_storage_config`'s docstring) — so when
        `storage_local_root` is still at its class default (i.e. NOT
        explicitly customized) and `upload_dir`/`corpus_dir` share a
        common parent directory named the way the fixed key prefixes
        expect (`.../uploads`, `.../corpus`), that shared parent is used
        automatically instead of the literal default. This is what lets
        test fixtures freely override just `UPLOAD_DIR`/`CORPUS_DIR` to
        an isolated `tmp_path` (as most of `tests/` already did before
        this pass, and reasonably keeps doing) without ALSO having to
        remember to override `STORAGE_LOCAL_ROOT` in lockstep.

        Returns `None` when `upload_dir`/`corpus_dir` don't share a
        derivable common root at all (e.g. two unrelated absolute
        paths) — `validate_storage_config` treats that as a hard
        misconfiguration when `storage_local_root` is still at its
        default too, since there's then no single local root that could
        be correct.
        """
        from pathlib import Path

        upload_path = Path(self.upload_dir)
        corpus_path = Path(self.corpus_dir)

        explicit_root = self.storage_local_root != type(self).model_fields["storage_local_root"].default
        if explicit_root:
            return None  # caller compares upload_dir/corpus_dir against the explicit root instead

        if upload_path.name == "uploads" and corpus_path.name == "corpus" and upload_path.parent == corpus_path.parent:
            return upload_path.parent
        return None

    def effective_storage_local_root_for(self, which: str) -> "Path | None":
        """Like `effective_storage_local_root`, but derived from JUST
        `upload_dir` (`which="uploads"`) or JUST `corpus_dir`
        (`which="corpus"`) independently, without requiring the other
        one to also line up.

        Exists because, as of this Phase 5 completion pass, only the
        upload path (`app/api/documents.py`) is actually wired through
        `app.storage.base.get_backend` — `app/api/source.py`'s corpus
        reads still go through `pathlib` directly (see
        `app/storage/base.py`'s "Honest scope note"). Requiring
        `CORPUS_DIR` to line up with a shared root before trusting
        `UPLOAD_DIR`'s OWN, independently-correct root would reject
        configurations (including most of `tests/`, which override only
        `UPLOAD_DIR` for upload-focused tests) that are perfectly valid
        for what's actually wired today. Once `source.py` is migrated
        too, this can go back to requiring one shared root — tracked in
        docs/architecture.md, "Storage abstraction: current integration
        status".
        """
        from pathlib import Path

        target_name = "uploads" if which == "uploads" else "corpus"
        configured = self.upload_dir if which == "uploads" else self.corpus_dir
        path = Path(configured)

        explicit_root = self.storage_local_root != type(self).model_fields["storage_local_root"].default
        if explicit_root:
            return None

        if path.name == target_name:
            return path.parent
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()