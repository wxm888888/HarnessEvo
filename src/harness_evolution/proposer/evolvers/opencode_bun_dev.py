from __future__ import annotations

import subprocess
from pathlib import Path

from harness_evolution.agents import CLIENT_REGISTRY, BaseClient
from harness_evolution.proposer.evolvers.base import BaseEvolver
from harness_evolution.proposer.evolvers.protocol import EvolverRequest, EvolverResult
from harness_evolution.proposer.evolvers.registry import EVOLVER_REGISTRY


_IGNORED_CHANGE_PREFIXES = (".harness_evolver", ".harness-evolver")


@EVOLVER_REGISTRY.register("opencode_bun_dev")
class OpenCodeBunDevEvolver(BaseEvolver):
    """Evolve a repository by delegating runtime operations to a registered OpenCode client."""

    def __init__(
        self,
        client_name: str = "opencode_bun_dev_client",
        client_init: dict | None = None,
        request_timeout_sec: int = 180,
    ) -> None:
        init_kwargs = dict(client_init or {})

        self.client = self._build_client(client_name=client_name, client_init=init_kwargs)
        self.request_timeout_sec = max(1, int(request_timeout_sec))

    def evolve(self, request: EvolverRequest) -> EvolverResult:
        workspace = request.workspace_resolved()
        if not workspace.exists():
            raise ValueError(f"workspace does not exist: {workspace}")

        prompt = request.build_prompt()
        before = self._list_git_changed_files(workspace)
        before_head = self._current_head(workspace)
        # Evolver stays strategy-only: execution/runtime details live in the client.
        agent_trajectory = self.client.run_instruction(
            workspace=workspace,
            instruction=prompt,
            request_timeout_sec=self.request_timeout_sec,
        )
        after = self._list_git_changed_files(workspace)
        after_head = self._current_head(workspace)
        # The agent may commit its own changes (e.g. `git commit`) instead of
        # leaving them in the working tree. A before/after `git status
        # --porcelain` diff alone would then see a clean tree and misreport
        # the candidate as a no-op, even though real, committed changes
        # exist. Union in the committed diff against the pre-run HEAD to
        # catch that case too.
        committed = self._committed_changed_files(workspace, before_head, after_head)
        changed_files = self._new_or_changed_files(before=before, after=after, committed=committed)
        no_changes_detected = not changed_files

        return EvolverResult(
            changed_files=changed_files,
            summary=(
                "opencode client evolve produced no repository changes"
                if no_changes_detected
                else "opencode client evolve applied"
            ),
            metadata={
                "target_scope": request.target.scope if request.target else "repo",
                "client_name": getattr(self.client, "client_name", "(unknown)"),
                "no_changes_detected": no_changes_detected,
                "agent_trajectory": agent_trajectory,
            },
        )

    def _list_git_changed_files(self, workspace: Path) -> list[str]:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git status failed in workspace {workspace}: {result.stderr.strip()}")
        files: list[str] = []
        for raw in result.stdout.splitlines():
            line = raw.rstrip()
            if len(line) < 4:
                continue
            path_part = line[3:]
            if "->" in path_part:
                path_part = path_part.split("->", 1)[1].strip()
            path = path_part.strip()
            if self._is_ignored_changed_path(path):
                continue
            files.extend(self._expand_changed_path(workspace=workspace, path=path))
        return files

    def _expand_changed_path(self, *, workspace: Path, path: str) -> list[str]:
        normalized = path.rstrip("/")
        if not normalized:
            return []
        if self._is_ignored_changed_path(normalized):
            return []

        absolute = workspace / normalized
        if not absolute.is_dir():
            return [normalized]

        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", normalized],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return [normalized]

        expanded = []
        for raw in result.stdout.splitlines():
            candidate = raw.strip()
            if not candidate:
                continue
            if self._is_ignored_changed_path(candidate):
                continue
            expanded.append(candidate)
        return expanded or [normalized]

    def _is_ignored_changed_path(self, path: str) -> bool:
        return any(path == prefix or path.startswith(f"{prefix}/") for prefix in _IGNORED_CHANGE_PREFIXES)

    def _current_head(self, workspace: Path) -> str | None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _committed_changed_files(
        self, workspace: Path, before_head: str | None, after_head: str | None
    ) -> list[str]:
        if not before_head or not after_head or before_head == after_head:
            return []
        result = subprocess.run(
            ["git", "diff", "--name-only", before_head, after_head],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git diff failed in workspace {workspace}: {result.stderr.strip()}"
            )
        files: list[str] = []
        for raw in result.stdout.splitlines():
            path = raw.strip()
            if not path or self._is_ignored_changed_path(path):
                continue
            files.append(path)
        return files

    def _new_or_changed_files(
        self, *, before: list[str], after: list[str], committed: list[str] | None = None
    ) -> list[str]:
        # Return deterministic, deduplicated list from post-run git status,
        # unioned with any changes the agent committed itself mid-run (see
        # `_committed_changed_files`). We intentionally keep existing dirty
        # files in the output because the agent may further modify them
        # during this evolve call.
        _ = before
        seen: set[str] = set()
        merged: list[str] = []
        for path in [*after, *(committed or [])]:
            if path in seen:
                continue
            seen.add(path)
            merged.append(path)
        return merged

    def _build_client(self, *, client_name: str, client_init: dict) -> BaseClient:
        client = CLIENT_REGISTRY.build(client_name, client_init)
        if client.__class__.run_instruction is BaseClient.run_instruction:
            raise ValueError(f"client '{client_name}' does not implement run_instruction")
        return client
