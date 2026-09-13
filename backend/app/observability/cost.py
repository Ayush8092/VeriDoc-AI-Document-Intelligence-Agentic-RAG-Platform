"""Approximate per-request cost estimation from token counts.

**These are not live, provider-confirmed prices.** `DEFAULT_PRICING_USD_PER_1M`
is a placeholder table filled in with rates that were plausible at
write-time; provider pricing changes without notice and this module has
no way to verify it's current (no network call is made here on purpose —
cost estimation must stay a pure, offline function of token counts).
Before trusting `total_cost_usd` in `evaluation/reports/` for anything
budget-relevant, confirm the rates below against the providers' current
pricing pages and override via `PRICING_OVERRIDES_JSON` (see
`Settings.pricing_overrides`) rather than editing this file, so the
override is visible in `.env` rather than silently drifting from a code
change.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

_log = logging.getLogger(__name__)

# USD per 1,000,000 tokens, (prompt_rate, completion_rate).
# Groq-hosted models are billed per-token; embedding models are
# prompt-tokens-only (completion_rate unused, kept at 0.0 for a uniform
# lookup shape).
DEFAULT_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    # Groq-hosted chat/completion models (app/rag/llm.py, Settings.answer_model)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),  # deprecated by Groq 2026-08-16 — see Settings.answer_model; kept here
    # only so a report covering an OLDER run (before the config default
    # was updated) can still be priced correctly, not because this is a
    # current recommendation.
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
    # Current default (Settings.answer_model) as of the Phase 5 completion
    # pass, round 4 model-staleness fix — Groq's own list price, verified
    # against multiple independent trackers citing Groq's pricing page
    # (not a reseller/aggregator price, which can differ):
    # $0.075/$0.30 per 1M input/output tokens.
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),  # Groq's recommended replacement for llama-3.3-70b-versatile
    # Embedding model (app/clients.py Embedder, Settings.embedding_model)
    "gemini-embedding-001": (0.15, 0.0),
}

_FALLBACK_RATE = (0.50, 0.75)  # used for an unrecognized model, logged once per model


@lru_cache(maxsize=32)
def _warn_once(model: str) -> None:
    _log.warning(
        "No pricing entry for model '%s' — using a fallback estimate ($%.2f/$%.2f per 1M "
        "prompt/completion tokens). Add it to DEFAULT_PRICING_USD_PER_1M or "
        "PRICING_OVERRIDES_JSON for an accurate figure.",
        model,
        *_FALLBACK_RATE,
    )


def _pricing_table() -> dict[str, tuple[float, float]]:
    table = dict(DEFAULT_PRICING_USD_PER_1M)
    try:
        from app.core.config import get_settings

        overrides_raw = get_settings().pricing_overrides_json
    except Exception:  # noqa: BLE001 - settings unavailable (e.g. a bare script/test); use defaults
        overrides_raw = None
    if overrides_raw:
        try:
            overrides = json.loads(overrides_raw)
            for model, rates in overrides.items():
                table[model] = (float(rates[0]), float(rates[1]))
        except (json.JSONDecodeError, TypeError, KeyError, IndexError, ValueError) as exc:
            _log.warning("Ignoring malformed PRICING_OVERRIDES_JSON: %s", exc)
    return table


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate a single call's cost in USD from its token counts.

    Never raises: an unrecognized model logs a one-time warning and falls
    back to a rough estimate rather than breaking the response that
    happens to be carrying this cost figure.
    """
    table = _pricing_table()
    if model not in table:
        _warn_once(model)
    prompt_rate, completion_rate = table.get(model, _FALLBACK_RATE)
    return (prompt_tokens / 1_000_000) * prompt_rate + (completion_tokens / 1_000_000) * completion_rate