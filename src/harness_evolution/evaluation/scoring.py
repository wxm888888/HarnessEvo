"""Turn trial outcomes into per-side scores for the gate."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from harness_evolution.evaluation.backends.opensandbox import SideOutcome, TrialOutcome
from harness_evolution.core.types import CollectionResult


def _side_from_collection(collection: CollectionResult, commit: str) -> SideOutcome:
    """Turn a merged sampling result into reusable baseline evidence."""
    side = SideOutcome(label="baseline", commit=commit, job_dir=str(collection.job_dir))
    for raw in collection.metadata.get("outcomes", []):
        imputed_zero = bool(raw.get("imputed_zero"))
        retry_exhausted = bool(raw.get("retry_exhausted"))
        error_type = str(raw.get("error_type") or "")
        outcome = TrialOutcome(
            trial_id=str(raw.get("trial_id") or ""),
            side="baseline",
            commit=str(raw.get("commit") or commit),
            task=str(raw.get("task") or ""),
            passed=bool(raw.get("passed")),
            error=bool(raw.get("error")),
            detail=str(raw.get("detail") or ""),
            imputed_zero=imputed_zero,
            retry_exhausted=retry_exhausted,
            error_type=error_type,
        )
        side.trials.append(outcome)
        side.passes.setdefault(outcome.task, 0)
        if outcome.passed:
            side.passes[outcome.task] += 1
        if outcome.error:
            side.errors += 1
    return side


def _side_for_tasks(
    side: SideOutcome,
    task_names: list[str] | set[str],
    *,
    label: str | None = None,
    commit: str | None = None,
) -> SideOutcome:
    """Return the durable outcomes for exactly one split.

    A parent baseline contains two logically different pieces of evidence:
    the fixed held-out suite and the rotating held-in batch.  Keeping the
    pieces separate lets later steps reuse the fixed 89-task baseline while
    replacing only the new held-in sample.
    """
    wanted = set(task_names)
    selected = SideOutcome(
        label=label or side.label,
        commit=commit or side.commit,
        job_dir=side.job_dir,
    )
    for outcome in side.trials:
        if outcome.task not in wanted:
            continue
        selected.trials.append(outcome)
        selected.passes.setdefault(outcome.task, 0)
        if outcome.passed:
            selected.passes[outcome.task] += 1
        if outcome.error:
            selected.errors += 1
    return selected


def _side_is_complete(side: SideOutcome | None, task_names: list[str], repeats: int) -> bool:
    """Check that a cached split has one usable result per task/repeat."""
    if side is None:
        return False
    score = _score_side(side, task_names, repeats)
    return bool(score["complete"])


def _side_from_trial_jsons(job_dir: Path, commit: str, *, label: str = "baseline") -> SideOutcome:
    """Reconstruct a side outcome from durable per-trial artifacts.

    Resume runs may be interrupted after the sampling collection has been
    written, so the in-memory ``CollectionResult`` is no longer available.
    The gate's ``trial.json`` files are the durable source of truth and let us
    continue without executing the same baseline again.
    """
    side = SideOutcome(label=label, commit=commit, job_dir=str(job_dir))
    trial_paths = sorted(job_dir.rglob("trial.json"))
    for path in trial_paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        # Legacy runs wrote retry_history.json but did not copy the newer
        # imputed_zero/retry_exhausted fields into trial.json. Recover those
        # flags so an exhausted error is scored as a durable zero on resume.
        imputed_zero = bool(raw.get("imputed_zero"))
        retry_exhausted = bool(raw.get("retry_exhausted"))
        error_type = str(raw.get("error_type") or "")
        if bool(raw.get("error")) and not (imputed_zero and retry_exhausted):
            history_path = path.parent / "retry_history.json"
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                history = []
            max_attempts = max(1, int(os.environ.get("SANDBOX_TRIAL_RETRIES", "2")))
            if (
                isinstance(history, list)
                and len(history) >= max_attempts
                and history
                and bool(history[-1].get("error"))
            ):
                imputed_zero = True
                retry_exhausted = True
                error_type = error_type or "legacy_retry_exhausted"

        outcome = TrialOutcome(
            trial_id=str(raw.get("trial_id") or ""),
            side=label,
            commit=str(raw.get("commit") or commit),
            task=str(raw.get("task") or ""),
            passed=bool(raw.get("passed")),
            error=bool(raw.get("error")),
            detail=str(raw.get("detail") or ""),
            imputed_zero=imputed_zero,
            retry_exhausted=retry_exhausted,
            error_type=error_type,
        )
        if not outcome.trial_id or not outcome.task:
            continue
        side.trials.append(outcome)
        side.passes.setdefault(outcome.task, 0)
        if outcome.passed:
            side.passes[outcome.task] += 1
        if outcome.error:
            side.errors += 1
    return side


def _merge_side_outcomes(*sides: SideOutcome, commit: str) -> SideOutcome:
    merged = SideOutcome(label="baseline", commit=commit)
    for side in sides:
        merged.job_dir = merged.job_dir or side.job_dir
        for outcome in side.trials:
            merged.trials.append(outcome)
            merged.passes.setdefault(outcome.task, 0)
            if outcome.passed:
                merged.passes[outcome.task] += 1
            if outcome.error:
                merged.errors += 1
    return merged


def _score_side(side: Any, tasks: list[str], repeats: int) -> dict[str, Any]:
    """Score one side, treating exhausted infrastructure errors as zero.

    An error contributes an ordinary zero reward only when the trial runner
    records that its retry budget was exhausted. Other terminal errors remain
    incomplete. Raw errors and imputed-zero counts are reported separately.
    """
    expected = {task: repeats for task in tasks}
    seen: dict[str, int] = {task: 0 for task in tasks}
    passed: dict[str, int] = {task: 0 for task in tasks}
    errors: dict[str, int] = {task: 0 for task in tasks}
    imputed_zero: dict[str, int] = {task: 0 for task in tasks}
    raw_errors: dict[str, int] = {task: 0 for task in tasks}
    for trial in getattr(side, "trials", []) or []:
        task = str(getattr(trial, "task", ""))
        if task not in expected:
            continue
        seen[task] += 1
        if bool(getattr(trial, "passed", False)):
            passed[task] += 1
        if bool(getattr(trial, "error", False)):
            raw_errors[task] += 1
            if bool(getattr(trial, "imputed_zero", False)) and bool(
                getattr(trial, "retry_exhausted", False)
            ):
                imputed_zero[task] += 1
            else:
                errors[task] += 1

    incomplete = [task for task in tasks if seen[task] != expected[task] or errors[task] > 0]
    passed_attempts = sum(passed.values())
    total_attempts = len(tasks) * repeats
    return {
        "passed_attempts": passed_attempts,
        "total_attempts": total_attempts,
        "rate": passed_attempts / total_attempts if total_attempts else 0.0,
        "task_passes": passed,
        "task_attempts": seen,
        "task_errors": errors,
        "task_raw_errors": raw_errors,
        "task_imputed_zero": imputed_zero,
        "incomplete_tasks": incomplete,
        "complete": not incomplete,
        "errors": sum(errors.values()),
        "raw_errors": sum(raw_errors.values()),
        "imputed_zero_attempts": sum(imputed_zero.values()),
        "job_dir": str(getattr(side, "job_dir", "")),
    }


def _not_run_score(tasks: list[str], *, reason: str) -> dict[str, Any]:
    """Describe a split intentionally excluded from dynamic gating."""
    return {
        "not_run": True,
        "reason": reason,
        "passed_attempts": 0,
        "total_attempts": 0,
        "rate": 0.0,
        "task_passes": {},
        "task_attempts": {},
        "task_errors": {},
        "task_raw_errors": {},
        "task_imputed_zero": {},
        "incomplete_tasks": list(tasks),
        "complete": False,
        "errors": 0,
        "raw_errors": 0,
        "imputed_zero_attempts": 0,
        "job_dir": "",
    }
