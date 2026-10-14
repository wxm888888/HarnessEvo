from __future__ import annotations

from harness_evolution.core.types import CandidateSelectionContext, CandidateSelectionDecision


class FirstValidSelector:
    name = "first_valid"

    def select(self, ctx: CandidateSelectionContext) -> CandidateSelectionDecision:
        for candidate in ctx.candidates:
            if candidate.validation.passed:
                return CandidateSelectionDecision(
                    accepted=True,
                    candidate_idx=candidate.candidate_idx,
                    fallback_reason=None,
                    metadata={"method": self.name},
                )
        return CandidateSelectionDecision(
            accepted=False,
            candidate_idx=None,
            fallback_reason="all_candidates_failed_validation",
            metadata={"method": self.name},
        )
