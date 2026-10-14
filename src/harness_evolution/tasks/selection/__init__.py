from __future__ import annotations

from .history_scheduler import HistorySchedulerTaskSelection
from .static_list import StaticListTaskSelection


TASK_SELECTION_REGISTRY = {
    StaticListTaskSelection.name: StaticListTaskSelection,
    HistorySchedulerTaskSelection.name: HistorySchedulerTaskSelection,
}
