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
from harness_evolution.traces.verifier_feedback import materialize_verifier_feedback


class SanitizedInstructionEvidence:
    name = "sanitized_instruction"

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
            trajectory_path = target_dir / "trajectory.json"
            result_path = target_dir / "sanitized_result.json"
            shutil.copy2(artifact.trajectory_path, trajectory_path)
            context_source = artifact.context_path
            if context_source is None:
                for candidate in (
                    artifact.trajectory_path.parent / "llm-context.jsonl",
                    artifact.result_path.parent / "llm-context.jsonl",
                    artifact.result_path.parent / "agent" / "llm-context.jsonl",
                ):
                    if candidate.is_file():
                        context_source = candidate
                        break
            context_path = target_dir / "llm-context.jsonl"
            if context_source is not None and context_source.is_file():
                shutil.copy2(context_source, context_path)
            openai_source = artifact.openai_messages_path
            if openai_source is None:
                for candidate in (
                    artifact.trajectory_path.parent / "openai-messages.jsonl",
                    artifact.result_path.parent / "openai-messages.jsonl",
                    artifact.result_path.parent / "agent" / "openai-messages.jsonl",
                ):
                    if candidate.is_file():
                        openai_source = candidate
                        break
            openai_path = target_dir / "openai-messages.jsonl"
            if openai_source is not None and openai_source.is_file():
                shutil.copy2(openai_source, openai_path)
                if (openai_source.parent / "openai-sessions.json").is_file():
                    from session_exports import copy_exports
                    copy_exports(openai_source.parent, target_dir)
                openai_meta_source = openai_source.with_name("openai-messages.meta.json")
                if openai_meta_source.is_file():
                    shutil.copy2(openai_meta_source, target_dir / openai_meta_source.name)
            for name in (
                "openai-tools.json",
                "openai-request.meta.json",
                "verifier-summary.json",
            ):
                source_extra = next(
                    (
                        candidate
                        for candidate in (
                            artifact.trajectory_path.parent / name,
                            artifact.result_path.parent / name,
                            artifact.result_path.parent / "agent" / name,
                        )
                        if candidate.is_file()
                    ),
                    None,
                )
                if source_extra is not None:
                    shutil.copy2(source_extra, target_dir / name)
            verifier_feedback_path = materialize_verifier_feedback(
                trajectory_path=artifact.trajectory_path,
                result_path=artifact.result_path,
                target_path=target_dir / "verifier-feedback.json",
            )
            write_sanitized_result(artifact.result_path, result_path)
            instruction_path = copy_instruction_if_available(
                result_path=artifact.result_path,
                task_cache_root=task_cache_root,
                target_dir=target_dir,
            )
            records.append(
                {
                    "source_trajectory_path": str(artifact.trajectory_path),
                    "source_result_path": str(artifact.result_path),
                    "workspace_trajectory_path": str(trajectory_path),
                    "workspace_result_path": str(result_path),
                    "workspace_instruction_path": str(instruction_path) if instruction_path else "",
                    "source_context_path": str(context_source) if context_source else "",
                    "workspace_context_path": str(context_path) if context_path.exists() else "",
                    "source_openai_messages_path": str(openai_source) if openai_source else "",
                    "workspace_openai_messages_path": str(openai_path)
                    if openai_path.exists()
                    else "",
                    "workspace_openai_tools_path": str(target_dir / "openai-tools.json")
                    if (target_dir / "openai-tools.json").exists()
                    else "",
                    "workspace_request_meta_path": str(target_dir / "openai-request.meta.json")
                    if (target_dir / "openai-request.meta.json").exists()
                    else "",
                    "workspace_verifier_summary_path": str(target_dir / "verifier-summary.json")
                    if (target_dir / "verifier-summary.json").exists()
                    else "",
                    "workspace_verifier_feedback_path": str(verifier_feedback_path)
                    if verifier_feedback_path
                    else "",
                }
            )
        return EvidenceBuildResult(
            requirement_addendum=render_prompt(
                __package__, "evidence_sanitized_instruction.md", name=self.name, root=WORKSPACE_ARTIFACT_ROOT
            ),
            metadata={
                "method": self.name,
                "artifact_root": str(root),
                "artifact_count": len(records),
                "records": records,
            },
        )
