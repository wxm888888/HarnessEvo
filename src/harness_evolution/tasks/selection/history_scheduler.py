from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_evolution.core.types import (
    RuntimeConfig,
    TaskSelectionContext,
    TaskSelectionResult,
    bool_value,
    int_value,
    mapping,
    repo_relative_path,
    str_list,
)
from harness_evolution.tasks.scheduler import (
    discover_cached_terminal_bench_tasks,
    load_task_history,
    save_task_history,
    select_tasks,
    update_task_history_from_job,
)


class HistorySchedulerTaskSelection:
    name = "history_scheduler"

    def __init__(self, options: dict[str, Any], config: RuntimeConfig):
        self.options = options
        self.config = config
        legacy = mapping(config.terminal_bench, "task_scheduler", required=False)
        merged = {**legacy, **options}
        self.enabled = bool_value(merged, "enabled", True)
        self.tasks_per_iteration = int_value(merged, "tasks_per_iteration")
        self.history_path = (
            repo_relative_path(str(merged["history_path"]))
            if merged.get("history_path")
            else config.workspace_root / "task_history.json"
        )
        self.task_cache_root = (
            repo_relative_path(str(merged["task_cache_root"]))
            if merged.get("task_cache_root")
            else None
        )
        self.candidate_tasks = (
            str_list(merged, "candidate_tasks", allow_empty=True)
            if "candidate_tasks" in merged
            else []
        )
        self.success_threshold = float(merged.get("success_threshold", 1.0))
        self.scheduler_config = merged

    def _candidate_tasks(self, static_task_names: list[str]) -> tuple[list[str], str]:
        if self.candidate_tasks:
            return sorted(dict.fromkeys(self.candidate_tasks)), "candidate_tasks"
        if static_task_names:
            return sorted(dict.fromkeys(static_task_names)), "terminal_bench.task_names"
        return discover_cached_terminal_bench_tasks(self.task_cache_root), "harbor_cache"

    def select(self, ctx: TaskSelectionContext) -> TaskSelectionResult:
        if not self.enabled:
            return TaskSelectionResult(
                task_names=ctx.static_task_names,
                metadata={
                    "method": self.name,
                    "enabled": False,
                    "selected_tasks": ctx.static_task_names,
                },
            )
        candidate_tasks, source = self._candidate_tasks(ctx.static_task_names)
        history = load_task_history(self.history_path)
        selection = select_tasks(
            all_tasks=candidate_tasks,
            iteration_idx=ctx.iteration_idx,
            tasks_per_iteration=self.tasks_per_iteration,
            history=history,
            config=self.scheduler_config,
        )
        selection["method"] = self.name
        selection["enabled"] = True
        selection["history_path"] = str(self.history_path)
        selection["candidate_tasks_source"] = source
        save_task_history(self.history_path, history)
        return TaskSelectionResult(task_names=selection["selected_tasks"], metadata=selection)

    def update_after_collection(
        self,
        *,
        iteration_idx: int,
        selected_tasks: list[str],
        job_dir: Path,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        history = load_task_history(self.history_path)
        history = update_task_history_from_job(
            history=history,
            iteration_idx=iteration_idx,
            selected_tasks=selected_tasks,
            job_dir=job_dir,
            success_threshold=self.success_threshold,
        )
        save_task_history(self.history_path, history)
        return {
            "history_path": str(self.history_path),
            "seen_task_count": len(history.get("seen_tasks", [])),
            "iteration_count": len(history.get("iterations", [])),
            "last_iteration": (history.get("iterations") or [{}])[-1],
        }
