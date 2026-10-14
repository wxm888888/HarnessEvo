"""Select evolution evidence from recent Polar training rollouts.

Pools the per-session trajectory dumps of the last WINDOW training steps
(``$POLAR_TRAJECTORY_DIR/rollout_XXXXXX/``), groups them by their underlying
prompt/task (8 independent samples per prompt share one group), and picks a
task-diverse, contrastive evidence set:

  - `pure_fail` tasks: every sample failed -- one failure session each.
  - `mixed` tasks: some samples succeeded and some failed on the *same*
    prompt -- one success + one failure session each, so the evolver can
    directly compare a winning and a losing attempt at the identical task.
  - `pure_success` tasks: every sample succeeded -- one success session each.

Sessions that timed out, hit an execution error, or got truncated/overlong
are excluded up front by discover_polar_sessions() (include_error_trajectories
defaults to False there) and never enter any bucket.

Stages the selection into a virtual rollout directory and converts it with
convert_polar_rollout_to_tb_job into a Terminal-Bench-style job directory the
evolver's agent_analyzed evidence pipeline consumes.

Usage:
  python3 -m harness_evolution.integrations.slime.select_trajectories \
      --trajectory-root $RUN_ROOT/train_trajectories \
      --end-rollout-id 9 --window 10 \
      --pure-fail-tasks 10 --mixed-tasks 6 --pure-success-tasks 3 \
      --output-dir /path/to/new_tb_job_dir
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from harness_evolution.traces.converters.polar_rollout import (
    DEFAULT_SUCCESS_THRESHOLD,
    PolarSession,
    convert_rollout,
    discover_polar_sessions,
)

# Session-level task_id is position-based: "polar-tb-harbor-{a}-{b}-traj-{i}"
# (or "polar-trajectory-group-{g}-...-traj-{i}" for masked/aborted
# placeholders, which discover_polar_sessions() already filters out before
# we ever see them here). The "-traj-{i}" suffix is the only part that
# varies across the n_samples_per_prompt independent samples of one prompt;
# stripping it recovers the shared per-prompt group key.
_TRAJ_SUFFIX_RE = re.compile(r"-traj-\d+$")


def _group_key(task_id: object) -> str:
    return _TRAJ_SUFFIX_RE.sub("", str(task_id or ""))


def select_sessions(
    trajectory_root: Path,
    end_rollout_id: int,
    window: int,
    pure_fail_tasks: int,
    mixed_tasks: int,
    pure_success_tasks: int,
    success_threshold: float,
    *,
    max_extra_scan: int = 40,
) -> tuple[list[Path], dict[str, str], dict]:
    """Return the source paths of the chosen sessions, a
    {source_filename: bucket} map ("pure_fail"/"mixed"/"pure_success" --
    which selection bucket each session came from, so downstream consumers
    can e.g. give noisier "mixed" tasks more held-out gate attempts), plus
    selection stats.

    Scans newest-step-first over `window` steps (extending up to
    `max_extra_scan` additional older steps if the window doesn't contain
    enough distinct groups of each kind -- mixed-outcome groups in
    particular can be rare), pools every clean (non-error, non-timeout,
    non-overlong) session, groups by prompt, and classifies each group as
    pure_fail / mixed / pure_success by the set of outcomes among its
    members.
    """
    groups: dict[str, list[PolarSession]] = defaultdict(list)
    group_first_seen_order: list[str] = []
    scanned_dirs: list[str] = []

    max_steps = window + max(0, max_extra_scan)
    start = max(0, end_rollout_id - max_steps + 1)
    needed = pure_fail_tasks + mixed_tasks + pure_success_tasks

    def _classify() -> tuple[list[list[PolarSession]], list[list[PolarSession]], list[list[PolarSession]]]:
        pure_fail: list[list[PolarSession]] = []
        mixed: list[list[PolarSession]] = []
        pure_success: list[list[PolarSession]] = []
        for key in group_first_seen_order:
            members = groups[key]
            cats = {member.category for member in members}
            if cats == {"completed_failure"}:
                pure_fail.append(members)
            elif cats == {"success"}:
                pure_success.append(members)
            elif "completed_failure" in cats and "success" in cats:
                mixed.append(members)
            # Any other combination shouldn't occur: discover_polar_sessions()
            # only ever yields "success" or "completed_failure" categories.
        return pure_fail, mixed, pure_success

    for step, rollout_id in enumerate(range(end_rollout_id, start - 1, -1)):
        rollout_dir = trajectory_root / f"rollout_{rollout_id:06d}"
        if rollout_dir.is_dir():
            scanned_dirs.append(rollout_dir.name)
            sessions, _skipped = discover_polar_sessions(
                rollout_dir,
                success_threshold=success_threshold,
                include_multi_trace=True,
            )
            for session in sessions:
                key = _group_key(session.payload.get("task_id"))
                if key not in groups:
                    group_first_seen_order.append(key)
                groups[key].append(session)

        # Stop as soon as the *required* window has been scanned AND we
        # already have enough of every bucket -- avoids over-scanning on the
        # common case where quotas fill quickly.
        if step + 1 >= window:
            pure_fail, mixed, pure_success = _classify()
            if len(pure_fail) >= pure_fail_tasks and len(mixed) >= mixed_tasks and len(pure_success) >= pure_success_tasks:
                break

    pure_fail, mixed, pure_success = _classify()

    picked_success: list[Path] = []
    picked_failure: list[Path] = []
    picked_pure_fail_groups = 0
    picked_mixed_groups = 0
    picked_pure_success_groups = 0
    bucket_by_source_name: dict[str, str] = {}

    for members in pure_fail:
        if picked_pure_fail_groups >= pure_fail_tasks:
            break
        picked_failure.append(members[0].source_path)
        bucket_by_source_name[members[0].source_path.name] = "pure_fail"
        picked_pure_fail_groups += 1

    for members in mixed:
        if picked_mixed_groups >= mixed_tasks:
            break
        success_member = next(m for m in members if m.category == "success")
        failure_member = next(m for m in members if m.category == "completed_failure")
        picked_success.append(success_member.source_path)
        picked_failure.append(failure_member.source_path)
        bucket_by_source_name[success_member.source_path.name] = "mixed"
        bucket_by_source_name[failure_member.source_path.name] = "mixed"
        picked_mixed_groups += 1

    for members in pure_success:
        if picked_pure_success_groups >= pure_success_tasks:
            break
        picked_success.append(members[0].source_path)
        bucket_by_source_name[members[0].source_path.name] = "pure_success"
        picked_pure_success_groups += 1

    stats = {
        "scanned_rollout_dirs": scanned_dirs,
        "requested": {
            "pure_fail_tasks": pure_fail_tasks,
            "mixed_tasks": mixed_tasks,
            "pure_success_tasks": pure_success_tasks,
        },
        "picked": {
            "pure_fail_tasks": picked_pure_fail_groups,
            "mixed_tasks": picked_mixed_groups,
            "pure_success_tasks": picked_pure_success_groups,
        },
        "available": {
            "pure_fail_tasks": len(pure_fail),
            "mixed_tasks": len(mixed),
            "pure_success_tasks": len(pure_success),
        },
        "picked_sessions": {
            "success": len(picked_success),
            "completed_failure": len(picked_failure),
        },
    }
    return picked_success + picked_failure, bucket_by_source_name, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--end-rollout-id", type=int, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--pure-fail-tasks", type=int, default=10)
    parser.add_argument("--mixed-tasks", type=int, default=6)
    parser.add_argument("--pure-success-tasks", type=int, default=3)
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument(
        "--max-extra-scan",
        type=int,
        default=40,
        help="extra steps beyond --window to scan if quotas (esp. mixed_tasks) aren't met",
    )
    parser.add_argument("--output-dir", required=True, help="empty/new TB job directory")
    args = parser.parse_args()

    trajectory_root = Path(args.trajectory_root)
    output_dir = Path(args.output_dir)

    chosen, bucket_by_source_name, stats = select_sessions(
        trajectory_root,
        args.end_rollout_id,
        args.window,
        args.pure_fail_tasks,
        args.mixed_tasks,
        args.pure_success_tasks,
        args.success_threshold,
        max_extra_scan=args.max_extra_scan,
    )
    if not chosen:
        print(f"[traj-select] no usable sessions in {stats['scanned_rollout_dirs']}", file=sys.stderr)
        return 1
    print(
        f"[traj-select] picked {stats['picked']['pure_fail_tasks']}/{stats['requested']['pure_fail_tasks']} "
        f"pure-fail + {stats['picked']['mixed_tasks']}/{stats['requested']['mixed_tasks']} mixed + "
        f"{stats['picked']['pure_success_tasks']}/{stats['requested']['pure_success_tasks']} pure-success tasks "
        f"({stats['picked_sessions']['success']} success + {stats['picked_sessions']['completed_failure']} failure "
        f"sessions) from {len(stats['scanned_rollout_dirs'])} rollout dir(s)"
    )
    for bucket in ("pure_fail_tasks", "mixed_tasks", "pure_success_tasks"):
        if stats["picked"][bucket] < stats["requested"][bucket]:
            print(
                f"[traj-select] WARNING: only found {stats['available'][bucket]} available "
                f"{bucket} groups (requested {stats['requested'][bucket]}) within the scanned window",
                file=sys.stderr,
            )

    # Stage into one virtual rollout dir; session filenames are globally
    # unique (session ids), so cross-step collisions cannot happen.
    with tempfile.TemporaryDirectory(prefix="polar_selected_rollout_") as tmp:
        virtual_dir = Path(tmp) / "selected"
        virtual_dir.mkdir()
        for source in chosen:
            shutil.copy2(source, virtual_dir / source.name)
        manifest = convert_rollout(
            source_dir=virtual_dir,
            output_dir=output_dir,
            success_threshold=args.success_threshold,
            max_successes=stats["picked_sessions"]["success"],
            max_completed_failures=stats["picked_sessions"]["completed_failure"],
            max_errors=0,
            include_multi_trace=True,
        )

    if manifest["converted_count"] != len(chosen):
        raise RuntimeError(
            "selected trajectory conversion was incomplete: "
            f"picked {len(chosen)} session(s), converted {manifest['converted_count']} "
            f"(skipped={manifest.get('skipped_counts', {})})"
        )

    # Tag each record with which selection bucket produced it -- lets the
    # held-out gate (see harness_evolution_pipeline.py's
    # _evidence_task_names) give "mixed" (proven-noisy: same prompt won and
    # lost during training) tasks proportionally more gate attempts than
    # pure_fail/pure_success tasks without needing harbor's n_attempts to
    # apply to every task uniformly.
    for record in manifest.get("records", []):
        record["selection_bucket"] = bucket_by_source_name.get(record.get("source_file"))

    manifest["selection"] = stats
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[traj-select] converted {manifest['converted_count']} sessions "
        f"({manifest['converted_counts']}) -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
