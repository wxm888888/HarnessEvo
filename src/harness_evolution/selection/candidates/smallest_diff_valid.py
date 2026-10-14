from __future__ import annotations

from harness_evolution.core.types import CandidateSelectionContext, CandidateSelectionDecision


class SmallestDiffValidSelector:
    name = "smallest_diff_valid"

    def select(self, ctx: CandidateSelectionContext) -> CandidateSelectionDecision:
        valid = [candidate for candidate in ctx.candidates if candidate.validation.passed]
        if not valid:
            return CandidateSelectionDecision(
                accepted=False,
                candidate_idx=None,
                fallback_reason="all_candidates_failed_validation",
                metadata={"method": self.name},
            )
        selected = min(valid, key=lambda candidate: (candidate.diff_stats.score, candidate.candidate_idx))
        return CandidateSelectionDecision(
            accepted=True,
            candidate_idx=selected.candidate_idx,
            fallback_reason=None,
            metadata={
                "method": self.name,
                "selected_score": selected.diff_stats.score,
            },
        )
