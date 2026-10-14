"""Cache the parent's held-out result per commit so later steps reuse it."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from harness_evolution.loop.modular import _write_json
from harness_evolution.evaluation.backends.opensandbox import SideOutcome, TrialOutcome, opensandbox_settings, run_heldout_opensandbox
from harness_evolution.evaluation.scoring import _side_for_tasks, _side_from_trial_jsons, _side_is_complete


def _baseline_cache_signature(
    *,
    held_out: list[str],
    repeats: int,
    gate_config: dict[str, Any],
    task_catalog: dict[str, dict[str, Any]],
) -> str:
    """Fingerprint the inputs that make a held-out baseline reusable."""
    heldout = gate_config.get("heldout_gate") or {}
    opensandbox = dict(heldout.get("opensandbox") or {})
    resolved = opensandbox_settings(heldout)
    # Include only settings which affect the measured baseline.  The run
    # directory and parent commit are separate cache dimensions.
    settings = {
        key: resolved.get(key, opensandbox.get(key))
        for key in (
            "agent_gateway_url",
            "agent_model_name",
            "agent_model_names",
            "agent_timeout_sec",
            "verifier_timeout_sec",
            "trial_retries",
            "opencode_setup",
            "opencode_context_limit",
            "opencode_output_limit",
        )
    }
    payload = {
        "held_out": list(held_out),
        "repeats": int(repeats),
        "backend": heldout.get("backend", "opensandbox"),
        "opensandbox": settings,
        "task_catalog": {task: task_catalog.get(task, {}) for task in held_out},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _baseline_cache_path(run_dir: Path, commit: str) -> Path:
    return run_dir / "baseline-cache" / f"{commit}.json"


def _write_baseline_cache(
    run_dir: Path,
    *,
    side: SideOutcome,
    commit: str,
    held_out: list[str],
    repeats: int,
    signature: str,
) -> None:
    """Persist only the fixed held-out baseline outcomes."""
    path = _baseline_cache_path(run_dir, commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "schema": "opencode_self_harness_loop.baseline_cache.v1",
            "commit": commit,
            "held_out": list(held_out),
            "repeats": int(repeats),
            "signature": signature,
            "side": {
                "label": side.label,
                "commit": side.commit,
                "job_dir": side.job_dir,
                "passes": side.passes,
                "errors": side.errors,
                "trials": [asdict(item) for item in side.trials],
            },
        },
    )


def _read_baseline_cache(
    run_dir: Path,
    *,
    commit: str,
    held_out: list[str],
    repeats: int,
    signature: str,
) -> SideOutcome | None:
    """Load a reusable held-out baseline only after validating its inputs."""
    path = _baseline_cache_path(run_dir, commit)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("commit") != commit
            or payload.get("held_out") != list(held_out)
            or int(payload.get("repeats", 0)) != int(repeats)
            or payload.get("signature") != signature
        ):
            return None
        raw_side = payload.get("side") or {}
        side = SideOutcome(
            label=str(raw_side.get("label") or "baseline"),
            commit=str(raw_side.get("commit") or commit),
            job_dir=str(raw_side.get("job_dir") or ""),
            passes={str(key): int(value) for key, value in (raw_side.get("passes") or {}).items()},
            errors=int(raw_side.get("errors") or 0),
        )
        for raw in raw_side.get("trials") or []:
            if not isinstance(raw, dict):
                continue
            outcome = TrialOutcome(
                trial_id=str(raw.get("trial_id") or ""),
                side="baseline",
                commit=str(raw.get("commit") or commit),
                task=str(raw.get("task") or ""),
                passed=bool(raw.get("passed")),
                error=bool(raw.get("error")),
                detail=str(raw.get("detail") or ""),
                imputed_zero=bool(raw.get("imputed_zero")),
                retry_exhausted=bool(raw.get("retry_exhausted")),
                error_type=str(raw.get("error_type") or ""),
            )
            if outcome.trial_id and outcome.task:
                side.trials.append(outcome)
        if all(item.commit == commit for item in side.trials) and _side_is_complete(side, held_out, repeats):
            return side
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _find_prior_heldout_baseline(
    run_dir: Path,
    *,
    commit: str,
    held_out: list[str],
    repeats: int,
) -> SideOutcome | None:
    """Recover a held-out baseline from an older gate artifact.

    This compatibility path upgrades runs created before the explicit cache
    file existed.  It looks only at completed baseline or promoted-candidate
    directories and never treats a partial validation job as reusable.
    """
    roots = [run_dir / "resume-baseline" / "baseline-heldout-jobs"]
    roots.extend(run_dir.glob("step-*/validation-jobs/*"))
    candidates: list[Path] = []
    for root in roots:
        if root.name == "baseline" and root.is_dir():
            candidates.append(root)
        else:
            candidate = root / "baseline"
            if candidate.is_dir():
                candidates.append(candidate)
            # A promoted candidate has already completed the same fixed
            # held-out suite.  Its trial records are valid baseline evidence
            # for the new parent commit and should be considered here when a
            # run is resumed before the explicit cache file was written.
            candidates.extend(path for path in root.glob("candidate-*") if path.is_dir())
    for candidate in sorted(set(candidates), key=lambda item: item.stat().st_mtime, reverse=True):
        side = _side_from_trial_jsons(candidate, commit, label="baseline")
        heldout_side = _side_for_tasks(side, held_out, label="baseline", commit=commit)
        if all(item.commit == commit for item in heldout_side.trials) and _side_is_complete(
            heldout_side, held_out, repeats
        ):
            return heldout_side
    return None


def _find_completed_resume_baseline(
    run_dir: Path,
    *,
    commit: str,
    task_names: list[str],
    repeats: int,
) -> SideOutcome | None:
    """Find a completed cached resume baseline, newest first.

    A prior resume may have left a partial timestamped directory behind.  We
    only reuse a directory containing every expected task/repeat pair, so a
    partial run cannot be mistaken for valid baseline evidence.
    """
    expected_tasks = set(task_names)
    expected_count = len(expected_tasks) * max(1, repeats)
    root = run_dir / "resume-baseline" / "baseline-heldout-jobs"
    if not root.is_dir() or not expected_tasks:
        return None
    candidates = sorted((path / "baseline" for path in root.iterdir() if path.is_dir()), reverse=True)
    for candidate in candidates:
        side = _side_from_trial_jsons(candidate, commit, label="baseline")
        task_counts: dict[str, int] = {}
        for trial in side.trials:
            if trial.task in expected_tasks:
                task_counts[trial.task] = task_counts.get(trial.task, 0) + 1
        if len(side.trials) >= expected_count and all(
            task_counts.get(task, 0) >= repeats for task in expected_tasks
        ):
            return side
    return None


def _run_heldout_baseline(
    *,
    gate_config: dict[str, Any],
    commit: str,
    binary: str,
    held_out: list[str],
    repeats: int,
    pool_size: int,
    run_dir: Path,
    task_catalog: dict[str, dict[str, Any]],
) -> SideOutcome:
    """Evaluate only the held-out half for a parent whose held-in run exists."""
    config = copy.deepcopy(gate_config)
    paths = config.setdefault("paths", {})
    paths["version_state"] = ""
    paths["baseline_commit"] = commit
    paths["baseline_binary"] = binary
    paths["report_root"] = str(run_dir / "baseline-reports")
    heldout = config.setdefault("heldout_gate", {})
    heldout["n_attempts"] = repeats
    heldout["n_concurrent"] = pool_size
    heldout["jobs_dir"] = str(run_dir / "baseline-heldout-jobs")
    raw = run_heldout_opensandbox(
        config,
        [],
        task_names=held_out,
        task_catalog=task_catalog,
        include_baseline=True,
    )
    baseline = raw.get("baseline")
    if baseline is None:
        raise RuntimeError("held-out baseline evaluation returned no baseline side")
    return baseline
