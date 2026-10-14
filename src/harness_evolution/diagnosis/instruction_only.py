from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from harness_evolution.core.prompts import render_prompt
from harness_evolution.core.types import (
    WORKSPACE_ARTIFACT_ROOT,
    CollectionArtifact,
    EvidenceBuildResult,
    RuntimeConfig,
)
from harness_evolution.traces.sanitize import copy_instruction_if_available, write_sanitized_result


class InstructionOnlyEvidence:
    name = "instruction_only"

    def __init__(self, options: dict[str, Any], config: RuntimeConfig):
        self.options = options
        self.config = config

    def materialize(
        self,
        *,
        artifacts: list[CollectionArtifact],
        workspace: Path,
        task_cache_root: Path,
        current_harness_path: str | Path | None = None,
    ) -> EvidenceBuildResult:
        _ = current_harness_path
        root = workspace / WORKSPACE_ARTIFACT_ROOT
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        records = []
        for idx, artifact in enumerate(artifacts, start=1):
            target_dir = root / f"{idx:03d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            result_path = target_dir / "sanitized_result.json"
            write_sanitized_result(artifact.result_path, result_path)
            instruction_path = copy_instruction_if_available(
                result_path=artifact.result_path,
                task_cache_root=task_cache_root,
                target_dir=target_dir,
            )
            records.append(
                {
                    "source_result_path": str(artifact.result_path),
                    "workspace_result_path": str(result_path),
                    "workspace_instruction_path": str(instruction_path) if instruction_path else "",
                }
            )
        return EvidenceBuildResult(
            requirement_addendum=render_prompt(
                __package__, "evidence_instruction_only.md", name=self.name, root=WORKSPACE_ARTIFACT_ROOT
            ),
            metadata={
                "method": self.name,
                "artifact_root": str(root),
                "artifact_count": len(records),
                "records": records,
            },
        )
