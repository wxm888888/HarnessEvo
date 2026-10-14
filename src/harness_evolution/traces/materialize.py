"""Materialize HarnessRL evolution artifacts from an OpenCode wire trace.

The OpenCode binary writes one cumulative, provider-native request per line to
``agent/llm-context.jsonl``.  This module turns that trace into the stable
files consumed by the evidence pipeline.  It reuses the TB2.1 converter so
the format stays identical to standalone evaluation artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness_evolution.core.paths import ensure_external_paths
from harness_evolution.traces.verifier_feedback import write_verifier_feedback


ensure_external_paths()

try:
    import agent_artifacts  # type: ignore
except Exception:  # pragma: no cover - only used in a misconfigured worker
    agent_artifacts = None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize(
    session_dir: Path,
    *,
    reward: str,
    status: str,
    error: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write derived evolution files and return counts/status.

    Missing wire capture is reported explicitly.  Callers can keep the trial
    result for infrastructure accounting while excluding it from evidence.
    """
    agent_dir = session_dir / "agent"
    context = agent_dir / "llm-context.jsonl"
    raw_wire_dir = agent_dir / "llm-reqs"
    extra = dict(extra_meta or {})
    if (
        (not context.is_file() or context.stat().st_size == 0)
        and raw_wire_dir.is_dir()
        and agent_artifacts is not None
    ):
        agent_artifacts.write_llm_context(raw_wire_dir, context)
    if not context.is_file() or context.stat().st_size == 0:
        result = {"complete": False, "reason": "missing_llm_context", "context": str(context)}
        _write_json(agent_dir / "artifact-status.json", result)
        return result

    if agent_artifacts is None:
        result = {"complete": False, "reason": "agent_artifacts_import_failed", "context": str(context)}
        _write_json(agent_dir / "artifact-status.json", result)
        return result

    messages = agent_dir / "openai-messages.jsonl"
    tools = agent_dir / "openai-tools.json"
    request_meta = agent_dir / "openai-request.meta.json"
    verifier = agent_dir / "verifier-summary.json"
    verifier_feedback = agent_dir / "verifier-feedback.json"
    n_messages = agent_artifacts.write_openai_messages(context, messages, agent_dir / "openai-messages.meta.json")
    n_tools = agent_artifacts.write_openai_tools(context, tools)
    n_requests = agent_artifacts.write_openai_request_meta(context, request_meta)
    agent_artifacts.write_verifier_summary(
        verifier,
        reward=str(reward),
        status=status,
        error=error,
        extra_meta=extra,
    )
    write_verifier_feedback(
        stdout_path=session_dir / "verifier.stdout.log",
        stderr_path=session_dir / "verifier.stderr.log",
        target_path=verifier_feedback,
        reward=reward,
        status=status,
    )
    result = {
        "complete": n_requests > 0 and messages.is_file() and tools.is_file() and verifier.is_file(),
        "context": str(context),
        "messages": str(messages),
        "tools": str(tools),
        "verifier_summary": str(verifier),
        "verifier_feedback": str(verifier_feedback),
        "n_requests": n_requests,
        "n_messages": n_messages,
        "n_tools": n_tools,
    }
    _write_json(agent_dir / "artifact-status.json", result)
    return result
