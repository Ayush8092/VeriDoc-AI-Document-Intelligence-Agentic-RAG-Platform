"""RAGAS metrics (faithfulness, answer relevancy, context precision/recall).

**Requires a separate virtualenv — see `requirements-ragas.txt`.** `ragas`
is deliberately NOT a dependency of the main app (`requirements.txt`): it
was found, in this project's actual environment, to need a
`langchain-community`/`langchain-openai` chain incompatible with the
`langgraph`/`langchain-core` versions the real RAG pipeline
(`app/rag/graph.py`) needs. That conflict is documented in full, with
the exact errors it produces, in `requirements-ragas.txt` — this isn't a
guess, it's what actually happened when both were installed together.

Every public function here imports `ragas` lazily (inside the function,
not at module scope), so `import app.evaluation.ragas_metrics` never
fails just because `ragas` isn't installed in whatever environment
imported it — only actually *calling* one of these functions does, with
`RagasUnavailableError` naming exactly why and how to fix it, rather than
a bare `ModuleNotFoundError` several frames deep in a third-party
package.

Run these from their own venv:
    python3 -m venv .venv-ragas
    .venv-ragas/bin/pip install -r requirements-ragas.txt -r requirements.txt
    .venv-ragas/bin/python -m evaluation.run_ragas
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


class RagasUnavailableError(RuntimeError):
    """`ragas` (or a package it needs) isn't importable in this venv.

    See this module's docstring / `requirements-ragas.txt` for why RAGAS
    needs its own virtualenv separate from `requirements.txt`.
    """


def _require_ragas():
    """The ONE place every raw `from ragas import ...` / `from ragas.metrics
    import ...` happens in this module. Every public function (score_single,
    and anything added later) must go through this instead of importing
    ragas classes directly, so there is exactly one place that needs to
    change if ragas's API shape changes, and exactly one place that
    produces the typed `RagasUnavailableError` instead of a bare
    `ModuleNotFoundError` surfacing from a random call site.
    """
    try:
        import ragas  # noqa: F401
        from ragas import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            FactualCorrectness,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RagasUnavailableError(
            "ragas is not importable in this environment. Install it in its own "
            "venv per requirements-ragas.txt — do not add it to the main "
            "requirements.txt venv (see that file for the exact conflict this "
            "caused when tried)."
        ) from exc
    return SingleTurnSample, Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall, ResponseRelevancy, FactualCorrectness


def _wrapped_llm(model: str):
    """Wrap this project's own Groq client as a RAGAS-compatible LLM via
    `langchain_groq.ChatGroq` + ragas's LangChain adapter — reusing the
    same `GROQ_API_KEY` / model config the main pipeline uses
    (`app/core/config.py`) rather than introducing a second, disconnected
    provider configuration just for evaluation.
    """
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    from app.core.config import get_settings

    settings = get_settings()
    if not settings.groq_api_key:
        raise RagasUnavailableError("GROQ_API_KEY is not set — RAGAS's LLM-judge metrics need a live LLM to run.")
    chat = ChatGroq(model=model, api_key=settings.groq_api_key.get_secret_value(), temperature=0)
    return LangchainLLMWrapper(chat)


def _wrapped_embeddings():
    """RAGAS's context-precision/recall metrics need an embedding model
    for semantic similarity; this wraps the SAME Gemini embedder
    (`app.clients.Embedder`) the main pipeline retrieves with, rather
    than pulling in a second embeddings provider just for evaluation.

    Uses `LangchainEmbeddingsWrapper`, NOT `embedding_factory` (an
    earlier version of this function called
    `embedding_factory(_EmbedderAdapter())`, which is a real bug fixed
    in the Phase 7 completion pass — verified via `inspect.getsource
    (embedding_factory)` against the actual installed `ragas` package,
    not assumed from its name: `embedding_factory()` ALWAYS constructs
    an `OpenAIEmbeddings` client internally; it does not accept a custom
    adapter argument at all, so the old call would have raised a
    pydantic validation error the first time RAGAS embeddings were
    actually needed. `LangchainEmbeddingsWrapper` is the class that
    accepts any object implementing `embed_query`/`embed_documents` —
    exactly `_EmbedderAdapter` below — confirmed by constructing one
    with a dummy adapter in a real (PyPI-installed) `ragas==0.2.15`.
    """
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    from app.clients import get_embeddings
    from app.core.config import get_settings

    settings = get_settings()
    embedder = get_embeddings(settings)

    class _EmbedderAdapter:
        def embed_query(self, text: str) -> list[float]:
            return embedder.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedder.embed_documents(texts)

    return LangchainEmbeddingsWrapper(_EmbedderAdapter())


def score_single(
    *,
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    model: str | None = None,
    reference_answer: str | None = None,
) -> dict[str, float | None]:
    """Score one Q&A turn with RAGAS's metrics.

    Returns a dict with `faithfulness`, `answer_relevancy`,
    `context_precision`, `context_recall`, and `answer_correctness` — any
    metric RAGAS itself couldn't compute (e.g. an empty
    `retrieved_contexts`) is `None` rather than a fabricated 0.0, so a
    caller can distinguish "genuinely unfaithful" from "couldn't be
    evaluated".

    **`answer_correctness` requires `reference_answer`** (a ground-truth
    answer string) — unlike the other four metrics, which are reference-
    free, RAGAS's `FactualCorrectness` compares the generated answer
    against a known-correct reference. When `reference_answer` is `None`
    (the common case: this project's benchmark datasets —
    `evaluation/datasets/*.jsonl` — record structural ground truth like
    `required_chunks`/`answerable`, not free-text reference answers; see
    `evaluation/README.md`, "Known gaps"), `answer_correctness` is
    `None`, not skipped silently — the key is always present in the
    returned dict so a caller can tell "not computed because no
    reference answer was supplied" apart from "computed and RAGAS itself
    returned no score".

    Raises `RagasUnavailableError` if `ragas`/`langchain_groq` aren't
    installed in the current interpreter — see module docstring.
    """
    (
        SingleTurnSample,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
        FactualCorrectness,
    ) = _require_ragas()

    try:
        from app.core.config import get_settings

        settings = get_settings()
        llm = _wrapped_llm(model or settings.answer_model)
        embeddings = _wrapped_embeddings()
    except ImportError as exc:
        raise RagasUnavailableError(
            "ragas's own dependencies (langchain_groq, or ragas's LangChain "
            "adapter modules) failed to import — see requirements-ragas.txt."
        ) from exc

    sample_kwargs = dict(user_input=question, response=answer, retrieved_contexts=retrieved_contexts or [""])
    if reference_answer:
        sample_kwargs["reference"] = reference_answer
    sample = SingleTurnSample(**sample_kwargs)

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": LLMContextPrecisionWithoutReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
    }
    if reference_answer:
        # Reference-dependent — only attempted when a reference answer
        # was actually supplied (see docstring above).
        metrics["answer_correctness"] = FactualCorrectness(llm=llm)

    scores: dict[str, float | None] = {name: None for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness")}
    for name, metric in metrics.items():
        try:
            scores[name] = metric.single_turn_score(sample)
        except Exception as exc:  # noqa: BLE001 - one metric failing shouldn't blank out the rest
            _log.warning("RAGAS metric '%s' failed for question %r: %s", name, question[:80], exc)
            scores[name] = None
    return scores