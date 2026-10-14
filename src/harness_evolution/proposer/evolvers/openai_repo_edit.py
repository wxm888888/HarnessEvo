from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_evolution.agents import CLIENT_REGISTRY, BaseClient
from harness_evolution.proposer.evolvers.base import BaseEvolver
from harness_evolution.proposer.evolvers.protocol import EvolverRequest, EvolverResult
from harness_evolution.proposer.evolvers.registry import EVOLVER_REGISTRY


@EVOLVER_REGISTRY.register("openai_repo_edit")
class OpenAIRepoEditEvolver(BaseEvolver):
    """Use a registered LLM client to evolve target file with repository context."""

    def __init__(
        self,
        model_name: str,
        client_name: str = "openai_chat",
        client_init: dict | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        max_files: int = 20,
        max_chars_each: int = 2500,
        request_timeout_sec: int = 60,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be provided via evolver_init")
        self.model_name = model_name
        self.client = self._build_client(client_name, client_init, api_key, base_url)
        self.max_files = int(max_files)
        self.max_chars_each = int(max_chars_each)
        self.request_timeout_sec = int(request_timeout_sec)

    def evolve(self, request: EvolverRequest) -> EvolverResult:
        workspace = request.workspace_resolved()
        target_file = request.require_file_target()
        file_path = self._resolve_target_file(workspace, target_file)
        target_content = file_path.read_text(encoding="utf-8")
        repo_context = self._build_repo_context(workspace, target_file)
        updated, agent_trajectory = self._call_client(request, target_content, repo_context)
        file_path.write_text(updated, encoding="utf-8")
        return EvolverResult(
            changed_files=[target_file],
            summary="openai repo edit evolve applied",
            metadata={
                "target_scope": "file",
                "client_model": self.model_name,
                "agent_trajectory": agent_trajectory,
            },
        )

    def _resolve_target_file(self, workspace: Path, target_file: str) -> Path:
        file_path = (workspace / target_file).resolve()
        workspace_resolved = workspace.resolve()
        if not str(file_path).startswith(str(workspace_resolved)):
            raise ValueError(f"target path escapes repository root: {target_file}")
        if not file_path.exists():
            raise ValueError(f"target file does not exist: {target_file}")
        return file_path

    def _build_repo_context(self, workspace: Path, target_file: str) -> str:
        snippets: list[str] = []
        root = workspace.resolve()

        files = sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)
        count = 0
        for path in files:
            rel = str(path.relative_to(root))
            if rel == target_file:
                continue
            if path.suffix not in {".py", ".md", ".txt", ".yaml", ".yml", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            snippets.append(f"[file] {rel}\n{text[: self.max_chars_each]}")
            count += 1
            if count >= self.max_files:
                break
        return "\n\n".join(snippets) if snippets else "(no additional text files)"

    def _call_client(
        self,
        request: EvolverRequest,
        target_content: str,
        repo_context: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        messages = request.build_messages(
            repo_context=repo_context,
            target_content=target_content,
        )
        agent_trajectory = self.client.generate(
            model_name=self.model_name,
            messages=messages,
            request_timeout_sec=self.request_timeout_sec,
        )
        content = self._extract_response_content(agent_trajectory).strip()
        if content.startswith("```"):
            content = self._strip_markdown_fence(content)
        if not content:
            raise RuntimeError("client returned empty content")
        return content, agent_trajectory

    def _extract_response_content(self, trajectory: list[dict[str, Any]]) -> str:
        if not isinstance(trajectory, list):
            raise RuntimeError("agent trajectory payload must be a list of messages")
        for message in reversed(trajectory):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if content is None:
                continue
            return content if isinstance(content, str) else str(content)
        raise RuntimeError("agent trajectory is missing assistant content")

    def _build_client(
        self,
        client_name: str,
        client_init: dict | None,
        api_key: str | None,
        base_url: str,
    ) -> BaseClient:
        init_kwargs = dict(client_init or {})
        # Backward compatibility with existing evolver_init configs.
        if api_key and "api_key" not in init_kwargs:
            init_kwargs["api_key"] = api_key
        if base_url and "base_url" not in init_kwargs:
            init_kwargs["base_url"] = base_url
        return CLIENT_REGISTRY.build(client_name, init_kwargs)

    def _strip_markdown_fence(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return text
