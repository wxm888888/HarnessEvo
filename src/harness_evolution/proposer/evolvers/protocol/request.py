from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from harness_evolution.proposer.evolvers.protocol.observation import RunObservation
from harness_evolution.proposer.evolvers.protocol.target import EvolverTarget


@dataclass(frozen=True)
class EvolverRequest:
    """Unified input contract for evolver execution."""

    DEFAULT_AGENT_OUTPUT_INSTRUCTION: ClassVar[str] = (
        "Please apply minimal, correct edits directly to files in this repository. "
        "Do not print full file contents; update files in-place and finish."
    )
    DEFAULT_FILE_EDIT_SYSTEM_PROMPT: ClassVar[str] = (
        "You are editing a repository file. "
        "Keep changes minimal and valid Python when target is .py."
    )
    DEFAULT_FILE_EDIT_OUTPUT_INSTRUCTION: ClassVar[str] = (
        "Return ONLY the full updated target file content. "
        "Do not include markdown fences."
    )
    DEFAULT_OBSERVATION_MAX_CHARS: ClassVar[int] = 12000

    workspace: Path
    requirement: str
    target: EvolverTarget | None = None
    context: dict[str, Any] = field(default_factory=dict)
    observations: list[RunObservation] = field(default_factory=list)

    def workspace_resolved(self) -> Path:
        return self.workspace.resolve()

    def require_file_target(self) -> str:
        """
        Read a file target from request and fail with a clear error if absent.
        File-oriented evolvers can call this helper to keep validation consistent.
        """
        if self.target is None:
            raise ValueError("this evolver requires a file target, but request.target is not set")
        if self.target.scope != "file":
            raise ValueError(f"this evolver requires target scope 'file', got '{self.target.scope}'")
        if not self.target.path:
            raise ValueError("this evolver requires target.path for file scope")
        return self.target.path

    def target_descriptor(self) -> str:
        """Render target metadata in a stable form for model-facing requests."""
        target_scope = self.target.scope if self.target else "repo"
        target_path = self.target.path if self.target else ""
        target_options = self.target.options if self.target else {}
        descriptor = f"scope={target_scope}, path={target_path or '(none)'}"
        if target_options:
            descriptor = f"{descriptor}, options={json.dumps(target_options, ensure_ascii=True)}"
        return descriptor

    def build_prompt(
        self,
        *,
        repo_context: str | None = None,
        target_content: str | None = None,
        output_instruction: str | None = None,
    ) -> str:
        """Build the shared repository-evolution prompt used by evolvers."""
        sections = [
            "You are modifying a local git repository.",
            f"Requirement:\n{self.requirement}",
            f"Target: {self.target_descriptor()}",
        ]
        if repo_context is not None:
            sections.append(f"Repository context (truncated):\n{repo_context}")
        if target_content is not None:
            sections.append(f"Current target file content:\n{target_content}")
        observation_context = self.observations_descriptor()
        if observation_context:
            sections.append(f"Runtime observations from previous harness runs:\n{observation_context}")
        sections.append(output_instruction or self.DEFAULT_AGENT_OUTPUT_INSTRUCTION)
        return "\n\n".join(sections)

    def build_messages(
        self,
        *,
        repo_context: str | None = None,
        target_content: str | None = None,
        output_instruction: str | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build OpenAI-style chat messages from the same request payload."""
        return [
            {
                "role": "system",
                "content": system_prompt or self.DEFAULT_FILE_EDIT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self.build_prompt(
                    repo_context=repo_context,
                    target_content=target_content,
                    output_instruction=output_instruction or self.DEFAULT_FILE_EDIT_OUTPUT_INSTRUCTION,
                ),
            },
        ]

    def observations_descriptor(self) -> str:
        """Render runtime observations in a stable, prompt-friendly form."""
        if not self.observations:
            return ""
        rendered: list[str] = []
        for index, observation in enumerate(self.observations, start=1):
            rendered.append(
                f"[observation {index}]\n"
                f"{observation.to_prompt_text(max_chars=self.DEFAULT_OBSERVATION_MAX_CHARS)}"
            )
        return "\n\n".join(rendered)
