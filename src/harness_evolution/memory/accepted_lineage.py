from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_evolution.core.prompts import render_prompt
from harness_evolution.core.types import Candidate, MemoryBuildResult, RuntimeConfig

_DEFAULT_WORKSPACE_DIR = ".harness_evolver/memory"
_DEFAULT_FILE_NAME = "accepted-lineage.jsonl"


def _relative_dir(raw: Any) -> Path:
    value = str(raw or _DEFAULT_WORKSPACE_DIR).strip().rstrip("/")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"memory.options.workspace_dir must be a safe relative path: {raw}")
    return path


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid memory JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"memory JSONL entry must be an object: {path}:{line_no}")
        records.append(value)
    return records


def _compact_score(score: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(score, dict):
        return None
    fields = ("passed_attempts", "total_attempts", "rate", "complete")
    return {key: score[key] for key in fields if key in score}


def _candidate_summary(candidate: Candidate) -> str:
    metadata = candidate.evolver_metadata if isinstance(candidate.evolver_metadata, dict) else {}
    candidate_report = metadata.get("candidate_report")
    report = candidate_report.get("report") if isinstance(candidate_report, dict) else None
    if not isinstance(report, dict):
        # Resume can reconstruct a candidate from binary metadata without
        # rehydrating evolver_metadata. The uncommitted report remains in the
        # candidate worktree, so memory-enabled runs can recover its summary
        # without changing the existing resume representation.
        report_path = candidate.workspace / ".harness_evolver" / "candidate-report.json"
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        report = loaded if isinstance(loaded, dict) else None
    if isinstance(report, dict):
        behavior_changes = report.get("behavior_changes")
        if isinstance(behavior_changes, list):
            rendered = [str(item).strip() for item in behavior_changes if str(item).strip()]
            if rendered:
                return "; ".join(rendered)
        root_cause = str(report.get("root_cause") or "").strip()
        if root_cause:
            return root_cause
    return str(candidate.summary or "accepted harness change").strip()


class AcceptedLineageMemory:
    """Expose only promoted ancestors as a compact, Git-backed memory."""

    name = "accepted_lineage"
    enabled = True

    def __init__(self, options: dict[str, Any], config: RuntimeConfig):
        self.options = options
        self.config = config
        self.workspace_dir = _relative_dir(options.get("workspace_dir"))
        file_name = str(options.get("file_name") or _DEFAULT_FILE_NAME).strip()
        if not file_name or Path(file_name).name != file_name:
            raise ValueError(f"memory.options.file_name must be a file name: {file_name}")
        self.file_name = file_name
        configured_storage = options.get("storage_path")
        self.storage_path = (
            Path(str(configured_storage)).expanduser().resolve()
            if configured_storage
            else (config.report_root / "memory" / file_name).resolve()
        )
        configured_seed = options.get("seed_path")
        self.seed_path = (
            Path(str(configured_seed)).expanduser().resolve()
            if configured_seed
            else None
        )

    def _records(self) -> list[dict[str, Any]]:
        paths: list[Path] = []
        if self.seed_path is not None:
            paths.append(self.seed_path)
        if self.storage_path not in paths:
            paths.append(self.storage_path)
        records: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        for path in paths:
            for record in _jsonl_records(path):
                key = (record.get("step"), record.get("winner_commit"))
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
        return records

    @staticmethod
    def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        rendered = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)

    def _exclude_workspace_memory_from_git(self, workspace: Path) -> None:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "info/exclude"],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cannot locate git exclude file for memory workspace {workspace}: "
                f"{result.stderr.strip()}"
            )
        exclude_path = Path(result.stdout.strip())
        if not exclude_path.is_absolute():
            exclude_path = (workspace / exclude_path).resolve()
        pattern = f"/{self.workspace_dir.as_posix()}/"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_path.read_text(encoding="utf-8") if exclude_path.is_file() else ""
        if pattern not in existing.splitlines():
            separator = "" if not existing or existing.endswith("\n") else "\n"
            with exclude_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{separator}{pattern}\n")

    def materialize(self, *, workspace: Path) -> MemoryBuildResult:
        target = workspace / self.workspace_dir / self.file_name
        self._exclude_workspace_memory_from_git(workspace)
        records = self._records()
        self._write_records(target, records)
        target.chmod(0o444)
        relative = target.relative_to(workspace).as_posix()
        return MemoryBuildResult(
            requirement_addendum=render_prompt(__package__, "accepted_lineage.md", relative=relative),
            metadata={
                "method": self.name,
                "record_count": len(records),
                "workspace_path": str(target),
                "storage_path": str(self.storage_path),
                "seed_path": str(self.seed_path) if self.seed_path is not None else "",
            },
        )

    def record_promotion(
        self,
        *,
        step: int,
        parent_commit: str,
        winner: Candidate,
        baseline_score: dict[str, Any] | None,
        winner_score: dict[str, Any] | None,
    ) -> dict[str, Any]:
        records = self._records()
        key = (int(step), winner.commit)
        existing = next(
            (
                record
                for record in records
                if (record.get("step"), record.get("winner_commit")) == key
            ),
            None,
        )
        if existing is not None:
            if not self.storage_path.is_file():
                self._write_records(self.storage_path, records)
            return existing
        record = {
            "schema_version": "harness-accepted-lineage-v1",
            "step": int(step),
            "accepted_at": datetime.now(timezone.utc).isoformat(),
            "parent_commit": str(parent_commit),
            "winner_commit": str(winner.commit),
            "candidate_idx": int(winner.candidate_idx),
            "summary": _candidate_summary(winner),
            "changed_files": [str(path) for path in winner.changed_files],
            "baseline_score": _compact_score(baseline_score),
            "winner_score": _compact_score(winner_score),
        }
        records.append(record)
        self._write_records(self.storage_path, records)
        return record
