from __future__ import annotations

from .first_valid import FirstValidSelector
from .smallest_diff_valid import SmallestDiffValidSelector


CANDIDATE_SELECTION_REGISTRY = {
    FirstValidSelector.name: FirstValidSelector,
    SmallestDiffValidSelector.name: SmallestDiffValidSelector,
}
