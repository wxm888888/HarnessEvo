from __future__ import annotations

from typing import Any

from harness_evolution.core.types import Candidate, MemoryBuildResult


class NoMemory:
    """Disabled memory strategy with no filesystem or prompt side effects."""

    name = "none"
    enabled = False

    def __init__(self, options: dict[str, Any] | None = None, config: Any = None):
        _ = options, config

    def materialize(self, *, workspace: Any) -> MemoryBuildResult:
        _ = workspace
        return MemoryBuildResult(requirement_addendum="", metadata={})

    def record_promotion(
        self,
        *,
        step: int,
        parent_commit: str,
        winner: Candidate,
        baseline_score: dict[str, Any] | None,
        winner_score: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        _ = step, parent_commit, winner, baseline_score, winner_score
        return None
