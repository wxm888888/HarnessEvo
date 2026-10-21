"""Pick evolution evidence out of recent RL rollouts.

Training writes one JSON per sampled session under
``$POLAR_TRAJECTORY_DIR/rollout_XXXXXX/``. An evolution round needs a small,
readable slice of those: a few failures to diagnose, a few successes to
contrast against.

Selection is flat on purpose -- ``n_fail`` failures and ``n_success``
successes, newest step first -- with **one trace per task**. Training samples
the same prompt ``n_samples_per_prompt`` times, so without that rule a single
task's siblings would fill the whole quota and the diagnosis would generalize
from one prompt.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness_evolution.traces.converters.polar_rollout import (
    DEFAULT_SUCCESS_THRESHOLD,
    convert_rollout,
    discover_polar_sessions,
)

DEFAULT_WINDOW = 10

_FAILURE_CATEGORY = "completed_failure"

# Session task ids are position-based ("polar-tb-harbor-{a}-{b}-traj-{i}").
# The "-traj-{i}" suffix is the only part that varies across the independent
# samples of one prompt, so stripping it recovers the shared task identity.
_TRAJ_SUFFIX_RE = re.compile(r"-traj-\d+$")

_DURABLE_ARTIFACT_FILES = (
    "artifact-status.json",
    "capture-status.json",
    "model.diff",
    "openai-messages.jsonl",
    "openai-messages.meta.json",
    "openai-request.meta.json",
    "openai-tools.json",
    "opencode.txt",
    "trace-proxy-status.json",
    "verifier-summary.json",
    "verifier.stdout.log",
    "verifier.stderr.log",
)


def task_key(task_id: object) -> str:
    """The task identity shared by one prompt's independent samples."""
    return _TRAJ_SUFFIX_RE.sub("", str(task_id or ""))


def _session_task(payload: dict[str, Any]) -> str:
    """Prefer the stable dataset id over Polar's step-position task id."""
    return str(payload.get("dataset_task_id") or "").strip() or task_key(payload.get("task_id"))


def _copy_durable_artifacts(
    rollout_root: Path, output_dir: Path, records: list[dict[str, Any]]
) -> None:
    """Join the lightweight rollout dump with its durable wire artifacts."""
    artifact_root = rollout_root / "agent-artifacts"
    for record in records:
        session_id = str(record.get("session_id") or "")
        trial_name = str(record.get("trial_dir") or "")
        if (
            not session_id
            or Path(session_id).name != session_id
            or not trial_name
            or Path(trial_name).name != trial_name
        ):
            raise ValueError(f"unsafe evidence artifact identity: {session_id!r}/{trial_name!r}")
        source = artifact_root / session_id
        target = output_dir / trial_name / "agent"
        copied: list[str] = []
        if source.is_dir() and not source.is_symlink():
            for name in _DURABLE_ARTIFACT_FILES:
                item = source / name
                if item.is_file() and not item.is_symlink():
                    shutil.copy2(item, target / name)
                    copied.append(name)

        # The rollout producer historically called this capture-status.json,
        # while preflight deliberately looks for the generic materialization
        # contract.  Preserve the diagnostic source and synthesize the small
        # canonical status for old as well as new rollout directories.
        status_path = target / "artifact-status.json"
        capture_path = target / "capture-status.json"
        if not status_path.is_file() and capture_path.is_file():
            try:
                capture = json.loads(capture_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                capture = {}
            if isinstance(capture, dict):
                status = {
                    **capture,
                    "complete": bool(capture.get("captured") and capture.get("normalized")),
                    "source": "capture-status.json",
                }
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                copied.append("artifact-status.json")
        record["durable_artifacts"] = sorted(copied)


@dataclass(frozen=True)
class SelectedTrace:
    """One candidate session, without its payload.

    ``discover_polar_sessions`` parses every session JSON in a rollout dir and
    a single dump can run to tens of megabytes. Keeping only these fields per
    step bounds peak memory at one step's worth instead of the whole window's.
    """

    source_path: Path
    category: str
    task: str
    rollout_dir: str

    @property
    def is_failure(self) -> bool:
        return self.category == _FAILURE_CATEGORY


def _allocate(
    pool: list[SelectedTrace], *, n_fail: int, n_success: int
) -> tuple[list[SelectedTrace], list[SelectedTrace]]:
    """Fill the two quotas from ``pool``, one trace per task.

    Failures are allocated first: they are what the diagnosis actually reads,
    so a task that both failed and succeeded inside the window spends its one
    slot on the failure.
    """
    used: set[str] = set()
    failures: list[SelectedTrace] = []
    successes: list[SelectedTrace] = []
    for trace in pool:
        if len(failures) >= n_fail:
            break
        if not trace.is_failure or trace.task in used:
            continue
        used.add(trace.task)
        failures.append(trace)
    for trace in pool:
        if len(successes) >= n_success:
            break
        if trace.is_failure or trace.task in used:
            continue
        used.add(trace.task)
        successes.append(trace)
    return failures, successes


def select_sessions(
    rollout_root: Path | str,
    *,
    end_rollout_id: int,
    n_fail: int,
    n_success: int,
    window: int = DEFAULT_WINDOW,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> tuple[list[SelectedTrace], dict[str, Any]]:
    """Choose the sessions for one evolution round, plus selection stats.

    Scans ``rollout_{end_rollout_id}`` backwards over ``window`` steps, newest
    first, and stops as soon as both quotas are full. A quota that cannot be
    filled within the window is reported in the stats rather than raised: a
    short round is still worth running, an empty one is not (that is the
    caller's call, see :func:`select_evidence`).
    """
    rollout_root = Path(rollout_root)
    if n_fail < 0 or n_success < 0:
        raise ValueError(f"n_fail and n_success must not be negative: {n_fail}, {n_success}")
    if n_fail + n_success <= 0:
        raise ValueError("n_fail + n_success must be positive; nothing to select otherwise")
    if window <= 0:
        raise ValueError(f"window must be positive: {window}")
    if end_rollout_id < 0:
        raise ValueError(f"end_rollout_id must not be negative: {end_rollout_id}")

    pool: list[SelectedTrace] = []
    scanned: list[str] = []
    failures: list[SelectedTrace] = []
    successes: list[SelectedTrace] = []
    first_rollout_id = max(0, end_rollout_id - window + 1)

    for rollout_id in range(end_rollout_id, first_rollout_id - 1, -1):
        rollout_dir = rollout_root / f"rollout_{rollout_id:06d}"
        if not rollout_dir.is_dir():
            continue
        scanned.append(rollout_dir.name)
        # include_multi_trace keeps resumed sessions; sessions that errored,
        # timed out or ran overlong are dropped by discover itself and never
        # become evidence.
        sessions, _skipped = discover_polar_sessions(
            rollout_dir,
            success_threshold=success_threshold,
            include_multi_trace=True,
        )
        pool.extend(
            SelectedTrace(
                source_path=session.source_path,
                category=session.category,
                task=_session_task(session.payload),
                rollout_dir=rollout_dir.name,
            )
            # Deterministic order within a step; across steps the pool stays
            # newest-first, so quotas prefer the most recent behaviour.
            for session in sorted(sessions, key=lambda item: item.source_path.name)
        )
        del sessions
        failures, successes = _allocate(pool, n_fail=n_fail, n_success=n_success)
        if len(failures) >= n_fail and len(successes) >= n_success:
            break

    chosen = failures + successes
    stats: dict[str, Any] = {
        "end_rollout_id": end_rollout_id,
        "window": window,
        "scanned_rollout_dirs": scanned,
        "pooled_sessions": len(pool),
        "requested": {"n_fail": n_fail, "n_success": n_success},
        "picked": {"n_fail": len(failures), "n_success": len(successes)},
        # Distinct tasks on offer, i.e. the ceiling the quotas could have
        # reached -- the useful number when a quota comes up short.
        "available_tasks": {
            "fail": len({trace.task for trace in pool if trace.is_failure}),
            "success": len({trace.task for trace in pool if not trace.is_failure}),
        },
        "short": len(failures) < n_fail or len(successes) < n_success,
        "tasks": [trace.task for trace in chosen],
    }
    return chosen, stats


def select_evidence(
    rollout_root: Path | str,
    *,
    n_fail: int,
    n_success: int,
    output_dir: Path | str,
    end_rollout_id: int,
    window: int = DEFAULT_WINDOW,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> dict[str, Any]:
    """Select this round's traces and convert them into an evidence job dir.

    ``output_dir`` must be empty or absent. Returns the conversion manifest,
    with each record tagged by the selection that produced it and the round's
    stats under ``selection``.
    """
    output_dir = Path(output_dir)
    chosen, stats = select_sessions(
        rollout_root,
        end_rollout_id=end_rollout_id,
        n_fail=n_fail,
        n_success=n_success,
        window=window,
        success_threshold=success_threshold,
    )
    if not chosen:
        raise RuntimeError(
            "no usable session to evolve from; scanned "
            f"{stats['scanned_rollout_dirs'] or 'no rollout dir'} under {rollout_root}"
        )

    category_by_name = {
        trace.source_path.name: ("fail" if trace.is_failure else "success") for trace in chosen
    }
    task_by_name = {trace.source_path.name: trace.task for trace in chosen}

    # Stage the selection into one virtual rollout dir: session file names are
    # session ids, so sessions pulled from different steps cannot collide.
    with tempfile.TemporaryDirectory(prefix="evolution_evidence_") as tmp:
        staged = Path(tmp) / "selected"
        staged.mkdir()
        for trace in chosen:
            shutil.copy2(trace.source_path, staged / trace.source_path.name)
        # No max_* limits: the staged dir holds exactly the chosen sessions,
        # and convert_rollout's own limiting truncates in file-name order,
        # which would drop a different subset than the one selected here.
        manifest = convert_rollout(
            source_dir=staged,
            output_dir=output_dir,
            success_threshold=success_threshold,
            include_multi_trace=True,
        )

    _copy_durable_artifacts(Path(rollout_root), output_dir, manifest.get("records", []))

    if manifest["converted_count"] != len(chosen):
        raise RuntimeError(
            "evidence conversion was incomplete: selected "
            f"{len(chosen)} session(s), converted {manifest['converted_count']} "
            f"(skipped={manifest.get('skipped_counts', {})})"
        )

    for record in manifest.get("records", []):
        name = record.get("source_file")
        record["selection_category"] = category_by_name.get(name)
        record["selection_task"] = task_by_name.get(name)
    manifest["selection"] = stats
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
