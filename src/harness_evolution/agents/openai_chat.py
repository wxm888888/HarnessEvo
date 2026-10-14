from __future__ import annotations

from typing import Any

from openai import OpenAI

from harness_evolution.agents.base import BaseClient
from harness_evolution.agents.registry import CLIENT_REGISTRY


@CLIENT_REGISTRY.register("openai_chat")
class OpenAIChatClient(BaseClient):
    """OpenAI chat completion client wrapper."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        if not api_key:
            raise ValueError("api_key must be provided for openai_chat")
        if not base_url:
            raise ValueError("base_url must not be empty")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        request_timeout_sec: int,
    ) -> list[dict[str, Any]]:
        if not model_name:
            raise ValueError("model_name must not be empty")
        try:
            response = self._client.with_options(timeout=request_timeout_sec).chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"openai request failed: {exc}") from exc
        return self._serialize_trajectory(messages=messages, response=response)

    def _serialize_trajectory(self, *, messages: list[dict[str, str]], response: Any) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = [dict(message) for message in messages]
        trajectory.append(self._response_message(response))
        return trajectory

    def _response_message(self, response: Any) -> dict[str, Any]:
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("openai response is missing choices[0].message") from exc

        try:
            return message.model_dump(mode="json", exclude_none=False)
        except AttributeError:
            return {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", None),
                "tool_calls": getattr(message, "tool_calls", None),
            }
