"""Commit what the evolver changed in a candidate worktree and measure the diff."""
from __future__ import annotations

import subprocess
from pathlib import Path
from harness_evolution.core.git import GitHarnessManager
from harness_evolution.core.types import DiffStats


def commit_changes_or_adopt_head(
    manager: GitHarnessManager, workspace: Path, changed_files: list[str], message: str
) -> tuple[str, bool]:
    """Commit the reported changes, or adopt HEAD if the agent already did.

    The evolver reports files the agent committed itself as changed (it diffs
    HEAD before/after the run, not just the dirty worktree). Staging those
    paths yields an empty index, and `git commit` then exits non-zero with
    "nothing to commit" -- which would throw away a perfectly good candidate
    whose changes are already in HEAD. Detect that case and take the agent's
    own commit instead.
    """
    subprocess.run(
        ["git", "add", "--", *changed_files],
        cwd=str(workspace),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    nothing_staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(workspace),
        check=False,
    ).returncode == 0
    if nothing_staged:
        return manager.current_head(workspace), False
    return manager.commit_paths(workspace, changed_files, message), True


def _git_diff_stats(workspace: Path, base_commit: str, commit: str) -> DiffStats:
    result = subprocess.run(
        ["git", "diff", "--numstat", f"{base_commit}..{commit}"],
        cwd=str(workspace),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return DiffStats(score=10**9)
    files = 0
    added = 0
    deleted = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        files += 1
        added += int(parts[0]) if parts[0].isdigit() else 1000
        deleted += int(parts[1]) if parts[1].isdigit() else 1000
    return DiffStats(files=files, added=added, deleted=deleted, score=files * 1000 + added + deleted)
