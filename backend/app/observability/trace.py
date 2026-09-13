"""Request-scoped tracing: trace IDs, timed spans, token usage, and cost.

This is Veridoc's own lightweight observability primitive — no external
service, no network calls, nothing that can silently fail in an
environment without live credentials. It's the foundation the optional
Opik integration (`app/observability/opik_integration.py`) and the cost
report both sit on top of: everything here works standalone even with
Opik disabled.

Usage:
    trace = start_trace()                      # call once per request
    with trace.span("retrieve"):
        ...
    trace.record_llm_usage(model="...", prompt_tokens=1, completion_tokens=1)
    trace.finish()
    trace.to_dict()                             # -> API-response-ready dict

`current_trace()` reads the active trace from a contextvar, so deeply
nested code (e.g. `app/rag/llm.py`'s `_chat_json`) can record usage
against "whatever request is currently running" without every function
in the call chain needing a `trace` parameter threaded through it.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.observability.cost import estimate_cost_usd

_current_trace: ContextVar["RequestTrace | None"] = ContextVar("_current_trace", default=None)


@dataclass
class SpanRecord:
    name: str
    started_at: float
    ended_at: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 2)

    def to_dict(self) -> dict:
        return {"name": self.name, "duration_ms": self.duration_ms, **self.metadata}


@dataclass
class LLMUsageRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    purpose: str = ""  # e.g. "grade", "rewrite", "answer", "grounding"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return estimate_cost_usd(self.model, self.prompt_tokens, self.completion_tokens)


@dataclass
class RequestTrace:
    trace_id: str
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    first_token_at: float | None = None
    spans: list[SpanRecord] = field(default_factory=list)
    llm_calls: list[LLMUsageRecord] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **metadata):
        record = SpanRecord(name=name, started_at=time.monotonic(), metadata=metadata)
        self.spans.append(record)
        try:
            yield record
        finally:
            record.ended_at = time.monotonic()

    def record_llm_usage(self, *, model: str, prompt_tokens: int, completion_tokens: int, purpose: str = "") -> None:
        self.llm_calls.append(
            LLMUsageRecord(
                model=model,
                prompt_tokens=max(0, prompt_tokens),
                completion_tokens=max(0, completion_tokens),
                purpose=purpose,
            )
        )

    def mark_first_token(self) -> None:
        """Call exactly once, the moment the first output token/chunk of
        the user-visible answer is available. Used to compute TTFT for
        streaming responses (`app/api/ask_stream.py`); a no-op on repeat
        calls so a defensive extra call from a retry path can't corrupt
        the measurement.
        """
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()

    def finish(self) -> None:
        self.ended_at = time.monotonic()

    @property
    def total_latency_ms(self) -> float | None:
        end = self.ended_at or time.monotonic()
        return round((end - self.started_at) * 1000, 2)

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000, 2)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.llm_calls), 6)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_latency_ms": self.total_latency_ms,
            "ttft_ms": self.ttft_ms,
            "spans": [s.to_dict() for s in self.spans],
            "llm_calls": [
                {
                    "model": c.model,
                    "purpose": c.purpose,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                    "cost_usd": round(c.cost_usd, 6),
                }
                for c in self.llm_calls
            ],
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


def start_trace(trace_id: str | None = None) -> RequestTrace:
    """Create a new trace and make it the "current" one for this request
    (via a contextvar — safe across concurrent async requests, each gets
    its own independent trace).
    """
    trace = RequestTrace(trace_id=trace_id or uuid.uuid4().hex)
    _current_trace.set(trace)
    return trace


def current_trace() -> RequestTrace | None:
    """The active trace for the request currently executing, or None if
    called outside a traced request (e.g. a script, or a test that never
    called `start_trace`). Every caller must tolerate None — tracing is
    always optional instrumentation, never a functional dependency.
    """
    return _current_trace.get()
