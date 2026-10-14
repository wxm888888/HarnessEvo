"""Backend-independent trial records: expansion of tasks into trials, outcomes and per-side aggregation."""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from harness_evolution.tasks.catalog import TaskSpec


@dataclass
class Trial:
    trial_id: str
    side: str
    commit: str
    binary: str
    task: TaskSpec
    attempt: int
    afs_bin: str = ""


@dataclass
class TrialOutcome:
    trial_id: str
    side: str
    commit: str
    task: str
    passed: bool
    error: bool
    detail: str = ""
    # Any terminal error with a durable trial record is scored as zero so it
    # cannot block a two-sided gate. Keep these fields separate from
    # ``error`` so reports retain the raw failure.
    imputed_zero: bool = False
    retry_exhausted: bool = False
    error_type: str = ""


@dataclass
class SideOutcome:
    label: str
    commit: str
    passes: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    job_dir: str = ""
    trials: list[TrialOutcome] = field(default_factory=list)


def expand_trials(
    *,
    sides: list[tuple[str, str] | tuple[str, str, str] | tuple[str, str, str, str]],
    task_names: list[str],
    specs: dict[str, TaskSpec],
    n_attempts: int,
) -> list[Trial]:
    """Duplicate task names become extra attempts (mixed_task_attempts)."""
    trials: list[Trial] = []
    seen: Counter[tuple[str, str]] = Counter()
    for side_info in sides:
        side, commit = side_info[:2]
        binary = side_info[2] if len(side_info) >= 3 else ""
        afs_bin = side_info[3] if len(side_info) >= 4 else ""
        for task_name in task_names:
            for _ in range(max(1, n_attempts)):
                attempt = seen[(side, task_name)]
                seen[(side, task_name)] += 1
                trials.append(
                    Trial(
                        trial_id=f"{side}-{task_name}-a{attempt}",
                        side=side,
                        commit=commit,
                        binary=binary,
                        afs_bin=afs_bin,
                        task=specs[task_name],
                        attempt=attempt,
                    )
                )
    return trials


def _aggregate(trials: list[Trial], outcomes: list[TrialOutcome], jobs_dir: Path) -> dict[str, SideOutcome]:
    by_side: dict[str, SideOutcome] = {}
    commit_by_side = {trial.side: trial.commit for trial in trials}
    for trial in trials:
        by_side.setdefault(
            trial.side,
            SideOutcome(label=trial.side, commit=commit_by_side[trial.side], job_dir=str(jobs_dir / trial.side)),
        )
    for outcome in outcomes:
        side = by_side[outcome.side]
        side.trials.append(outcome)
        side.passes.setdefault(outcome.task, 0)
        if outcome.passed:
            side.passes[outcome.task] += 1
        if outcome.error:
            side.errors += 1
    return by_side


def _trial_outcome_from_json(path: Path, trial: Trial) -> TrialOutcome | None:
    """Load one durable trial result, accepting it only for the same trial."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if (
        str(value.get("trial_id") or "") != trial.trial_id
        or str(value.get("side") or "") != trial.side
        or str(value.get("commit") or "") != trial.commit
        or str(value.get("task") or "") != trial.task.name
    ):
        return None
    return TrialOutcome(
        trial_id=trial.trial_id,
        side=trial.side,
        commit=trial.commit,
        task=trial.task.name,
        passed=bool(value.get("passed")),
        error=bool(value.get("error")),
        detail=str(value.get("detail") or ""),
        imputed_zero=bool(value.get("imputed_zero")),
        retry_exhausted=bool(value.get("retry_exhausted")),
        error_type=str(value.get("error_type") or ""),
    )


def _write_trial_outcome(path: Path, outcome: TrialOutcome) -> None:
    """Atomically persist a trial result so an interrupted gate can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": outcome.trial_id,
        "side": outcome.side,
        "commit": outcome.commit,
        "task": outcome.task,
        "passed": outcome.passed,
        "error": outcome.error,
        "detail": outcome.detail,
        "reward": 1.0 if outcome.passed else 0.0,
        "imputed_zero": outcome.imputed_zero,
        "retry_exhausted": outcome.retry_exhausted,
        "error_type": outcome.error_type,
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
