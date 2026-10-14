"""Rank gated candidates for promotion."""
from __future__ import annotations

from typing import Any


def _candidate_score(score: dict[str, Any]) -> tuple[float, float, int, int]:
    """Rank candidates by the fixed held-out benchmark.

    Held-in batches rotate every step and are used to produce the next
    diagnosis/evolution input.  They are deliberately not part of promotion
    or candidate ranking; otherwise a candidate could be rejected merely
    because it was evaluated on a different 32-task TMAX batch.
    """
    in_rate = float(score["held_in"]["rate"])
    out_rate = float(score["held_out"]["rate"])
    return (
        out_rate,
        in_rate,
        int(score["held_out"]["passed_attempts"]),
        -int(score["held_out"]["errors"]),
    )
