from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class GitHarnessManager:
    """Public git maintenance interface for harness repositories."""

    def __init__(self, workspace_root: str = ".git_workspaces") -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def prepare_workspace(self, repo_path: str, workspace_name: str) -> Path:
        source = self._validate_source_repo(repo_path)

        target = self.workspace_root / workspace_name / "repo"
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._run_git(["clone", str(source), str(target)], cwd=source.parent)
        return target

    def prepare_shared_workspace(self, repo_path: str, workspace_group: str) -> Path:
        """Clone once for a parallel run group (single git tree root)."""
        source = self._validate_source_repo(repo_path)
        group_root = self.workspace_root / workspace_group
        shared_repo = group_root / "shared_repo"
        if group_root.exists():
            shutil.rmtree(group_root)
        group_root.mkdir(parents=True, exist_ok=True)
        self._run_git(["clone", str(source), str(shared_repo)], cwd=source.parent)
        return shared_repo

    def create_worktree_from_commit(
        self,
        shared_repo: Path,
        worktree_name: str,
        base_commit: str,
        branch_name: str,
    ) -> Path:
        """Create an isolated worktree rooted in the same shared git tree."""
        worktree_root = shared_repo.parent / "worktrees"
        worktree_path = worktree_root / worktree_name
        worktree_root.mkdir(parents=True, exist_ok=True)

        if worktree_path.exists():
            try:
                self._run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=shared_repo)
            except RuntimeError:
                # Fallback when stale files are left by interrupted runs.
                shutil.rmtree(worktree_path, ignore_errors=True)

        self._run_git(["worktree", "add", "-B", branch_name, str(worktree_path), base_commit], cwd=shared_repo)
        return worktree_path

    def checkout_branch_from_commit(self, workspace: Path, base_commit: str, branch_name: str) -> None:
        self._run_git(["checkout", "-B", branch_name, base_commit], cwd=workspace)

    def commit_all(self, workspace: Path, message: str) -> str:
        self._run_git(["add", "."], cwd=workspace)
        return self._commit_staged(workspace, message)

    def commit_paths(self, workspace: Path, paths: list[str], message: str) -> str:
        if not paths:
            raise ValueError("paths must not be empty")
        self._run_git(["add", "--", *paths], cwd=workspace)
        return self._commit_staged(workspace, message)

    def _commit_staged(self, workspace: Path, message: str) -> str:
        self._run_git(
            [
                "-c",
                "user.name=Harness Git Bot",
                "-c",
                "user.email=harness-git-bot@example.com",
                "commit",
                "-m",
                message,
            ],
            cwd=workspace,
        )
        return self._run_git(["rev-parse", "HEAD"], cwd=workspace).strip()

    def merge_commits(self, workspace: Path, commit_shas: list[str], message: str) -> str:
        for commit_sha in commit_shas:
            self._run_git(
                [
                    "-c",
                    "user.name=Harness Git Bot",
                    "-c",
                    "user.email=harness-git-bot@example.com",
                    "merge",
                    "--no-ff",
                    "--no-edit",
                    commit_sha,
                ],
                cwd=workspace,
            )
        self._run_git(
            [
                "-c",
                "user.name=Harness Git Bot",
                "-c",
                "user.email=harness-git-bot@example.com",
                "commit",
                "--amend",
                "-m",
                message,
            ],
            cwd=workspace,
        )
        return self.current_head(workspace)

    def current_head(self, workspace: Path) -> str:
        return self._run_git(["rev-parse", "HEAD"], cwd=workspace).strip()

    def _validate_source_repo(self, repo_path: str) -> Path:
        source = Path(repo_path).resolve()
        if not source.exists():
            raise ValueError(f"repo does not exist: {source}")
        if not (source / ".git").exists():
            raise ValueError(f"repo is not git controlled: {source}")
        return source

    def _run_git(self, args: list[str], cwd: Path) -> str:
        env = os.environ.copy()
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown git error"
            raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
        return result.stdout
