from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseClient(ABC):
    client_name: str

    @abstractmethod
    def generate(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        request_timeout_sec: int,
    ) -> list[dict[str, Any]]:
        """Generate an OpenAI-style message trajectory from a chat-style prompt."""

    def run_instruction(
        self,
        *,
        workspace: Path,
        instruction: str,
        request_timeout_sec: int,
    ) -> list[dict[str, Any]]:
        """
        Execute an instruction against a repository workspace.

        Agent-style clients (for example, local CLI agents) can override this method.
        Chat-only clients can keep the default implementation.
        """
        raise NotImplementedError(f"client '{self.client_name}' does not support run_instruction")
