from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TargetScope = Literal["file", "directory", "repo", "custom"]


@dataclass(frozen=True)
class EvolverTarget:
    """Target descriptor for evolvers with different targeting needs."""

    scope: TargetScope
    path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
