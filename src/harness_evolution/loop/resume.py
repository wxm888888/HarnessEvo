"""Resume a partially finished step and prune artifacts of completed steps."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from harness_evolution.loop.modular import _write_json
from harness_evolution.core.types import Candidate, DiffStats, ValidationResult


def _load_partial_step_candidates(
    step_dir: Path,
    *,
    parent_commit: str,
    expected_count: int,
) -> tuple[list[Candidate], dict[str, str]] | None:
    """Recover candidates produced before a process died inside a gate.

    The static gate registers each built binary with the source worktree in
    ``metadata.json``.  Those records are sufficient to reconstruct the
    lightweight ``Candidate`` objects needed by the dynamic gate, so a
    resume can reuse the exact candidate commits instead of invoking the
    evolver again.  A durable candidate checkpoint also covers interruptions
    after source generation but before static validation produced binaries.
    """
    if expected_count < 1:
        return None
    metadata_paths = sorted(
        (step_dir / "static-workspace" / "artifacts" / "binaries").glob("*/metadata.json")
    )
    recovered: list[tuple[int, Candidate, str]] = []
    seen_commits: set[str] = set()
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        commit = str(payload.get("harness_commit") or "").strip()
        workspace_raw = str(payload.get("source_repo_path") or "").strip()
        binary_raw = str(metadata_path.parent / "opencode")
        workspace = Path(workspace_raw).expanduser()
        binary = Path(binary_raw).expanduser()
        if not commit or commit == parent_commit or not workspace.is_dir() or not binary.is_file():
            continue
        if commit in seen_commits:
            continue
        try:
            head = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            ).stdout.strip()
        except OSError:
            continue
        if head != commit:
            continue
        match = re.search(r"rollout_(\d+)$", workspace.name)
        rollout_idx = int(match.group(1)) if match else len(recovered) + 1
        branch = subprocess.run(
            ["git", "-C", str(workspace), "branch", "--show-current"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip() or f"resumed/{commit[:12]}"
        diff = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--name-only", f"{parent_commit}..{commit}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        changed_files = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
        candidate = Candidate(
            candidate_idx=rollout_idx,
            attempt_idx=1,
            rollout_idx=rollout_idx,
            workspace=workspace,
            branch=branch,
            commit=commit,
            commit_created=True,
            changed_files=changed_files,
            summary="recovered candidate from an interrupted dynamic gate",
            trajectory=[],
            evidence={"recovered_from": str(metadata_path)},
            skill_copy={},
            evolver_metadata={"resumed_partial_step": True, "binary_metadata": str(metadata_path)},
            validation=ValidationResult(enabled=False, passed=True, commands=[]),
            diff_stats=DiffStats(files=len(changed_files)),
        )
        recovered.append((rollout_idx, candidate, str(binary)));
        seen_commits.add(commit)

    if len(recovered) == expected_count:
        recovered.sort(key=lambda item: item[0])
        candidates = [item[1] for item in recovered]
        binaries = {item[1].commit: item[2] for item in recovered}
        return candidates, binaries

    checkpoint_path = step_dir / "candidate-checkpoint.json"
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(checkpoint, dict) or str(checkpoint.get("parent_commit") or "") != parent_commit:
        return None
    rows = checkpoint.get("candidates")
    if not isinstance(rows, list) or len(rows) != expected_count:
        return None

    checkpoint_candidates: list[Candidate] = []
    seen_commits = set()
    for position, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            return None
        commit = str(row.get("commit") or "").strip()
        workspace = Path(str(row.get("workspace") or "")).expanduser()
        if not commit or commit == parent_commit or commit in seen_commits or not workspace.is_dir():
            return None
        head = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
        if head != commit:
            return None
        diff = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--name-only", f"{parent_commit}..{commit}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        changed_files = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
        if not changed_files:
            return None
        rollout_idx = int(row.get("rollout_idx") or position)
        branch = str(row.get("branch") or "").strip() or subprocess.run(
            ["git", "-C", str(workspace), "branch", "--show-current"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip() or f"resumed/{commit[:12]}"
        checkpoint_candidates.append(
            Candidate(
                candidate_idx=int(row.get("candidate_idx") or position),
                attempt_idx=int(row.get("attempt_idx") or 1),
                rollout_idx=rollout_idx,
                workspace=workspace,
                branch=branch,
                commit=commit,
                commit_created=True,
                changed_files=changed_files,
                summary=str(row.get("summary") or "recovered candidate checkpoint"),
                trajectory=[],
                evidence={},
                skill_copy={},
                evolver_metadata={
                    "resumed_candidate_checkpoint": True,
                    "candidate_checkpoint": str(checkpoint_path),
                },
                validation=ValidationResult(enabled=False, passed=True, commands=[]),
                diff_stats=DiffStats(files=len(changed_files)),
            )
        )
        seen_commits.add(commit)
    checkpoint_candidates.sort(key=lambda candidate: candidate.rollout_idx)
    return checkpoint_candidates, {}


def _write_candidate_checkpoint(
    step_dir: Path,
    *,
    parent_commit: str,
    candidates: list[Candidate],
) -> Path:
    """Persist source candidates before static validation or binary builds."""
    path = step_dir / "candidate-checkpoint.json"
    _write_json(
        path,
        {
            "schema": "opencode_self_harness_loop.candidate_checkpoint.v1",
            "parent_commit": parent_commit,
            "candidates": [
                {
                    "candidate_idx": candidate.candidate_idx,
                    "attempt_idx": candidate.attempt_idx,
                    "rollout_idx": candidate.rollout_idx,
                    "workspace": str(candidate.workspace),
                    "branch": candidate.branch,
                    "commit": candidate.commit,
                    "changed_files": list(candidate.changed_files),
                    "summary": candidate.summary,
                }
                for candidate in candidates
            ],
        },
    )
    return path


def _find_partial_validation_job(step_dir: Path, candidates: list[Candidate]) -> Path | None:
    """Find the newest incomplete validation job matching resumed candidates."""
    jobs_root = step_dir / "validation-jobs"
    if not jobs_root.is_dir() or not candidates:
        return None
    for job_dir in sorted(
        (path for path in jobs_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        matched_sides = 0
        for index, candidate in enumerate(candidates, 1):
            side_dir = job_dir / f"candidate-{index:02d}"
            if not side_dir.is_dir():
                continue
            for trial_path in side_dir.rglob("trial.json"):
                try:
                    payload = json.loads(trial_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and str(payload.get("commit") or "") == candidate.commit:
                    matched_sides += 1
                    break
        if matched_sides:
            return job_dir
    return None


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(directory.expanduser().resolve(strict=False))
        return True
    except ValueError:
        return False


def _prune_completed_step_artifacts(
    run_dir: Path,
    completed_step: int,
    retention: int,
    *,
    protected_paths: tuple[Path, ...] = (),
) -> None:
    """Remove bulky raw artifacts after a two-step safety window.

    ``retention=2`` means that, once step ``n+2`` has been fully written,
    raw artifacts from step ``n`` are removed.  Small control/summary files
    (``step.json``, ``gate.json``, and the run-level state) remain available
    for auditing and resume.  The function is called only after a step and
    its state have been committed, so an interrupted or failed step cannot
    delete evidence from an earlier step.
    """
    if retention < 0:
        return
    last_old_step = completed_step - retention
    if last_old_step < 0:
        return

    # Keep the compact step metadata and remove only the high-volume raw
    # evidence.  In particular, do not remove the whole step directory: the
    # summary is useful when reviewing a long run and costs almost nothing.
    bulky_names = (
        "sampling",
        "validation-jobs",
        "static-workspace",
        "gate-reports",
        "gate-xdg",
    )
    for old_step in range(last_old_step + 1):
        step_dir = run_dir / f"step-{old_step:02d}"
        if not step_dir.is_dir():
            continue
        removed: list[str] = []
        protected: list[str] = []
        for name in bulky_names:
            target = step_dir / name
            if not target.exists() and not target.is_symlink():
                continue
            if any(_path_is_within(path, target) for path in protected_paths):
                protected.append(name)
                continue
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                removed.append(name)
            except OSError as exc:
                # A failed cleanup should not abort an otherwise completed
                # evolution step.  A later step retries this old directory.
                print(f"[retention] failed to remove {target}: {exc}", file=sys.stderr, flush=True)
        if removed:
            print(
                f"[retention] step-{old_step:02d}: removed raw artifacts ({', '.join(removed)}); "
                f"metadata retained",
                flush=True,
            )
        if protected:
            print(
                f"[retention] step-{old_step:02d}: protected active artifacts "
                f"({', '.join(protected)})",
                flush=True,
            )


def _load_completed_rejected_gate(
    step_dir: Path,
    *,
    parent_commit: str,
    held_in: list[str],
    held_out: list[str],
) -> dict[str, Any] | None:
    """Recover a fully recorded rejected gate after a post-gate crash."""
    path = step_dir / "gate.json"
    if not path.is_file() or (step_dir / "step.json").is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("accepted") or str(payload.get("winner_commit") or ""):
        return None
    if str(payload.get("parent_commit") or "") != parent_commit:
        return None
    if list(payload.get("held_in_tasks") or []) != list(held_in):
        return None
    if list(payload.get("held_out_tasks") or []) != list(held_out):
        return None
    baseline_heldout = ((payload.get("baseline") or {}).get("held_out") or {})
    if not baseline_heldout.get("complete"):
        return None
    candidates = payload.get("candidates") or {}
    if not isinstance(candidates, dict):
        return None
    if any(not ((score or {}).get("held_out") or {}).get("complete") for score in candidates.values()):
        return None
    return payload
