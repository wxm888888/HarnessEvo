from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from harness_evolution.core.utils import json_safe


@dataclass(frozen=True)
class RunObservation:
    """Runtime feedback passed into an evolver.

    The first version intentionally keeps the payload simple: Terminal Bench can
    put its raw trajectory here, while future analyzers can add summaries,
    metrics, or failure modes without changing the loop/evolver boundary.
    """

    source: str
    trajectory: Any
    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self, *, max_chars: int) -> str:
        payload = {
            "source": self.source,
            "artifact_path": self.artifact_path,
            "metadata": self.metadata,
            "trajectory": self.trajectory,
        }
        text = json.dumps(json_safe(payload), ensure_ascii=False, indent=2)
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n... [observation truncated]"
