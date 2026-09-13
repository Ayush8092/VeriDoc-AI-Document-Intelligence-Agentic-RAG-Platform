"""Deterministic evaluation metrics + benchmark/ablation runner (Phase 3A).

Everything in this package is a measurement, never a claim. Nothing here
fabricates a number — every function either computes a metric from data
you give it, or a script that calls the real pipeline and records what
actually happened. See `evaluation/README.md` at the repo root for how to
run the benchmark and where results land.
"""
