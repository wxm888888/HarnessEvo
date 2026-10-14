from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_evolution.proposer.evolvers.protocol.request import EvolverRequest
from harness_evolution.core.utils import sanitize_path_part
from harness_evolution.core.utils import json_safe


@dataclass(frozen=True)
class EvolverResult:
    """Standardized output contract for evolver execution."""

    changed_files: list[str]
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def primary_changed_file(self) -> str | None:
        return self.changed_files[0] if self.changed_files else None

    def save(
        self,
        evolver_artifacts_root: Path | str | None = None,
        *,
        request: EvolverRequest | None = None,
        evolver_name: str | None = None,
        run_id: str | None = None,
        output_dir: Path | str | None = None,
    ) -> Path:
        """
        Persist result artifacts for debugging and data reuse.

        The saved directory contains result.json, optional trajectory.jsonl,
        optional changed_files.patch, and optional post-run file snapshots.
        """
        if evolver_artifacts_root is None:
            if output_dir is None:
                raise TypeError("evolver_artifacts_root is required")
            evolver_artifacts_root = output_dir
        elif output_dir is not None:
            raise TypeError("pass only one of evolver_artifacts_root or output_dir")

        created_at = datetime.now(timezone.utc)
        evolver_run_artifact_dir = Path(evolver_artifacts_root).expanduser()
        if run_id is None:
            run_id = self._default_run_id(created_at=created_at, evolver_name=evolver_name)
        evolver_run_artifact_dir = evolver_run_artifact_dir / sanitize_path_part(run_id)
        evolver_run_artifact_dir.mkdir(parents=True, exist_ok=False)

        trajectory = self.metadata.get("agent_trajectory")
        metadata = {
            key: value
            for key, value in self.metadata.items()
            if key != "agent_trajectory"
        }
        record: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "evolver": evolver_name,
            "changed_files": self.changed_files,
            "summary": self.summary,
            "metadata": metadata,
            "artifacts": {},
        }
        if request is not None:
            record["request"] = self._request_record(request)

        if trajectory is not None:
            self._write_trajectory(evolver_run_artifact_dir / "trajectory.jsonl", trajectory)
            record["artifacts"]["trajectory"] = "trajectory.jsonl"

        if request is not None:
            patch = self._build_git_patch(request.workspace_resolved())
            if patch:
                (evolver_run_artifact_dir / "changed_files.patch").write_text(patch, encoding="utf-8")
                record["artifacts"]["patch"] = "changed_files.patch"
            snapshots = self._write_changed_file_snapshots(
                evolver_run_artifact_dir / "files",
                request.workspace_resolved(),
            )
            if snapshots:
                record["artifacts"]["files"] = snapshots

        (evolver_run_artifact_dir / "result.json").write_text(
            json.dumps(json_safe(record), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return evolver_run_artifact_dir

    def _default_run_id(self, *, created_at: datetime, evolver_name: str | None) -> str:
        timestamp = created_at.strftime("%Y%m%d-%H%M%S-%f")
        if evolver_name:
            return f"{timestamp}_{evolver_name}"
        return timestamp

    def _request_record(self, request: EvolverRequest) -> dict[str, Any]:
        target = None
        if request.target is not None:
            target = {
                "scope": request.target.scope,
                "path": request.target.path,
                "options": request.target.options,
            }
        return {
            "workspace": str(request.workspace_resolved()),
            "requirement": request.requirement,
            "target": target,
            "context": request.context,
            "observations": request.observations,
        }

    def _write_trajectory(self, path: Path, trajectory: Any) -> None:
        records = trajectory if isinstance(trajectory, list) else [trajectory]
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")

    def _build_git_patch(self, workspace: Path) -> str:
        if not self.changed_files:
            return ""
        diff_result = subprocess.run(
            ["git", "diff", "--", *self.changed_files],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if diff_result.returncode != 0:
            return ""
        patch_parts = [diff_result.stdout] if diff_result.stdout else []
        patch_parts.extend(self._build_untracked_file_patches(workspace))
        return "\n".join(part for part in patch_parts if part)

    def _build_untracked_file_patches(self, workspace: Path) -> list[str]:
        patches: list[str] = []
        for changed_file in self.changed_files:
            status_result = subprocess.run(
                ["git", "status", "--porcelain", "--", changed_file],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                check=False,
            )
            if status_result.returncode != 0:
                continue
            if not any(line.startswith("?? ") for line in status_result.stdout.splitlines()):
                continue
            diff_result = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", changed_file],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                check=False,
            )
            if diff_result.stdout:
                patches.append(diff_result.stdout)
        return patches

    def _write_changed_file_snapshots(self, output_dir: Path, workspace: Path) -> list[str]:
        saved: list[str] = []
        workspace_resolved = workspace.resolve()
        for changed_file in self.changed_files:
            source = (workspace_resolved / changed_file).resolve()
            try:
                source.relative_to(workspace_resolved)
            except ValueError:
                continue
            if not source.is_file():
                continue
            destination = output_dir / changed_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            saved.append(str(Path("files") / changed_file))
        return saved
