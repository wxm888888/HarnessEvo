from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_evolution.core.types import RuntimeConfig, TaskSelectionContext, TaskSelectionResult, str_list


class StaticListTaskSelection:
    name = "static_list"

    def __init__(self, options: dict[str, Any], config: RuntimeConfig):
        self.options = options
        self.config = config

    def select(self, ctx: TaskSelectionContext) -> TaskSelectionResult:
        task_names = (
            str_list(self.options, "task_names", allow_empty=True)
            if "task_names" in self.options
            else ctx.static_task_names
        )
        return TaskSelectionResult(
            task_names=task_names,
            metadata={
                "method": self.name,
                "selected_tasks": task_names,
                "source": "task_selection.options.task_names" if "task_names" in self.options else "terminal_bench.task_names",
            },
        )

    def update_after_collection(
        self,
        *,
        iteration_idx: int,
        selected_tasks: list[str],
        job_dir: Path,
    ) -> dict[str, Any] | None:
        return None
