"""The RL training loop's entry points into harness evolution.

Two calls are the whole surface. :func:`select_evidence` turns a window of
training rollouts into an evidence directory; :func:`evolve_once` turns that
directory plus the current harness commit into the commit to train on next.
Diagnosis, gating and evolution memory are arranged behind them, so the
trainer configures none of it.
"""
from __future__ import annotations

from harness_evolution.rl.evolve import EvolveConfig, EvolveResult, evolve_once
from harness_evolution.rl.select import (
    DEFAULT_WINDOW,
    SelectedTrace,
    select_evidence,
    select_sessions,
    task_key,
)

__all__ = [
    "DEFAULT_WINDOW",
    "EvolveConfig",
    "EvolveResult",
    "SelectedTrace",
    "evolve_once",
    "select_evidence",
    "select_sessions",
    "task_key",
]
