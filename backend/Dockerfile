# syntax=docker/dockerfile:1
#
# Phase 5 completion pass, round 2, item 15 ("Docker + Docker Compose").
#
# Multi-stage build:
#   1. `builder` — installs Python deps into a venv (keeps build tools
#      like gcc out of the final image).
#   2. runtime  — copies just the venv + application code, plus the ONE
#      system package the app actually shells out to at runtime:
#      `tesseract-ocr` (OCR itself) and `poppler-utils` (`pdf2image`'s
#      PDF -> image rendering for scanned pages — see
#      app/documents/ocr/scanned_pdf.py and docs/ocr.md).
#
# Deliberately NOT included as a system package: `libmagic`
# (app/security/file_validation.py implements magic-byte checking in
# pure Python — no libmagic dependency).

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- runtime stage -----------------------------------------------------

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 veridoc
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . .

# alembic migrations run as a separate `entrypoint.sh` step so a
# multi-replica deployment doesn't have every replica racing to run
# migrations concurrently at container start — see docker-compose.yml,
# which runs a dedicated one-shot `migrate` service before `backend`
# starts, using this same image. Copied/chmod'd here, as root, BEFORE
# the `USER veridoc` switch below — `/` itself isn't writable by a
# non-root user, so this must happen first.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# data/ (uploads, corpus, sqlite db, page-image cache, lexical index
# snapshot — see data/README.md) must be writable by the non-root user
# this image runs as. In production this directory should be a mounted
# volume (see docker-compose.yml's `backend-data` volume) so it survives
# container recreation; STORAGE_BACKEND=s3 makes most of it disposable
# except the sqlite metadata DB (or DATABASE_URL pointed at a real
# Postgres instead — see .env.example).
RUN mkdir -p data/uploads data/corpus data/page_images \
    && chown -R veridoc:veridoc /app

USER veridoc

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
