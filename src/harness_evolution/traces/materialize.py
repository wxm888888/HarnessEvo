"""Convert temporary wire/context into retained OpenAI-compatible artifacts."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from harness_evolution.core.paths import ensure_external_paths
from harness_evolution.traces.verifier_feedback import write_verifier_feedback

ensure_external_paths()
try:
    import agent_artifacts  # type: ignore
except Exception:  # pragma: no cover
    agent_artifacts = None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _retained_counts(agent_dir: Path) -> tuple[int, int, int] | None:
    """Validate exported evidence before reusing it without the intermediate."""
    try:
        with (agent_dir / "openai-messages.jsonl").open(encoding="utf-8") as stream:
            messages = [json.loads(line) for line in stream if line.strip()]
        tools = json.loads((agent_dir / "openai-tools.json").read_text(encoding="utf-8"))
        metadata = json.loads((agent_dir / "openai-request.meta.json").read_text(encoding="utf-8"))
        message_meta = json.loads((agent_dir / "openai-messages.meta.json").read_text(encoding="utf-8"))
        requests = int(metadata["request_count"])
        if (agent_dir / "openai-sessions.json").exists() or os.environ.get("HARNESSRL_REQUIRE_SESSION_EXPORT") == "1":
            from session_exports import validate_exports
            if not validate_exports(agent_dir):
                return None
        if not messages or not all(isinstance(m, dict) and m.get("role") for m in messages):
            return None
        if not isinstance(tools, list) or requests < 1:
            return None
        if message_meta["message_count"] != len(messages) or message_meta["request_count"] != requests:
            return None
        if not isinstance(metadata.get("requests"), list) or len(metadata["requests"]) != requests:
            return None
        return requests, len(messages), len(tools)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def materialize(
    session_dir: Path,
    *,
    reward: str,
    status: str,
    error: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export OpenAI files; discard context only after successful conversion.

    Sampling materializes a trial again after collecting all outcomes. Reuse
    validated OpenAI exports on that second call rather than requiring a
    deleted context or downloading raw SSE again. An explicit debug override
    HARNESSRL_KEEP_LLM_CONTEXT=1 retains the temporary context.
    """
    agent_dir = session_dir / "agent"
    context = agent_dir / "llm-context.jsonl"
    raw_wire_dir = agent_dir / "llm-reqs"
    messages = agent_dir / "openai-messages.jsonl"
    tools = agent_dir / "openai-tools.json"
    request_meta = agent_dir / "openai-request.meta.json"
    verifier = agent_dir / "verifier-summary.json"
    verifier_feedback = agent_dir / "verifier-feedback.json"
    has_context = context.is_file() and context.stat().st_size > 0
    counts = None if has_context else _retained_counts(agent_dir)

    if not has_context and counts is None and raw_wire_dir.is_dir() and agent_artifacts is not None:
        agent_artifacts.write_llm_context(raw_wire_dir, context)
        has_context = context.is_file() and context.stat().st_size > 0
    if not has_context and counts is None:
        result = {"complete": False, "reason": "missing_llm_context", "context": str(context)}
        _write_json(agent_dir / "artifact-status.json", result)
        return result
    if agent_artifacts is None:
        result = {"complete": False, "reason": "agent_artifacts_import_failed", "context": str(context)}
        _write_json(agent_dir / "artifact-status.json", result)
        return result

    if counts is None:
        expected_messages = agent_artifacts.write_openai_messages(context, messages, agent_dir / "openai-messages.meta.json")
        expected_tools = agent_artifacts.write_openai_tools(context, tools)
        expected_requests = agent_artifacts.write_openai_request_meta(context, request_meta)
        counts = _retained_counts(agent_dir)
        if counts != (expected_requests, expected_messages, expected_tools):
            counts = None
    agent_artifacts.write_verifier_summary(
        verifier, reward=str(reward), status=status, error=error, extra_meta=dict(extra_meta or {})
    )
    write_verifier_feedback(
        stdout_path=session_dir / "verifier.stdout.log",
        stderr_path=session_dir / "verifier.stderr.log",
        target_path=verifier_feedback,
        reward=reward,
        status=status,
    )
    complete = counts is not None and verifier.is_file()
    keep_context = os.environ.get("HARNESSRL_KEEP_LLM_CONTEXT", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if complete and not keep_context:
        context.unlink(missing_ok=True)
    keep_raw = os.environ.get("HARNESSRL_KEEP_RAW_TRACE_WIRE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if complete and not keep_raw and raw_wire_dir.is_dir() and not raw_wire_dir.is_symlink():
        shutil.rmtree(raw_wire_dir)
    n_requests, n_messages, n_tools = counts or (0, 0, 0)
    result = {
        "complete": complete,
        "context": str(context) if context.is_file() else None,
        "context_retained": context.is_file(),
        "messages": str(messages),
        "tools": str(tools),
        "request_meta": str(request_meta),
        "verifier_summary": str(verifier),
        "verifier_feedback": str(verifier_feedback),
        "n_requests": n_requests,
        "n_messages": n_messages,
        "n_tools": n_tools,
        "retention": "openai_exports",
    }
    if not complete:
        result["reason"] = "openai_export_validation_failed"
    _write_json(agent_dir / "artifact-status.json", result)
    return result
