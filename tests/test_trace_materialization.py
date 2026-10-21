"""Regression coverage for retaining usable evidence across materialization."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_evolution.traces import materialize as materializer


@pytest.fixture
def captured_session(tmp_path, monkeypatch):
    # Exercise the external converter used by the deployed sampler, rather
    # than imitating its export schema in this test.
    converter = pytest.importorskip("agent_artifacts")
    monkeypatch.setattr(materializer, "agent_artifacts", converter)
    monkeypatch.delenv("HARNESSRL_KEEP_LLM_CONTEXT", raising=False)
    monkeypatch.delenv("HARNESSRL_KEEP_RAW_TRACE_WIRE", raising=False)
    agent = tmp_path / "agent"
    wire = agent / "llm-reqs"
    wire.mkdir(parents=True)
    tools = [{"type": "function", "function": {
        "name": "shell", "description": "Execute a command",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}}]
    record = {"n": 1, "wire_n": 1, "request": {
        "model": "fixture", "messages": [{"role": "user", "content": "inspect files"}],
        "tools": tools}}
    (agent / "llm-context.jsonl").write_text(json.dumps(record) + "\n")
    (wire / "000001-meta.json").write_text(json.dumps({"n": 1, "stream": False}))
    (wire / "000001-req.json").write_text(json.dumps(record["request"]))
    return tmp_path, agent, wire, tools


def test_successful_exports_survive_context_cleanup_and_second_materialize(captured_session):
    session, agent, wire, tools = captured_session
    first = materializer.materialize(session, reward="0", status="ok")
    assert first["complete"] and first["n_requests"] == 1
    assert json.loads((agent / "openai-tools.json").read_text()) == tools
    assert not (agent / "llm-context.jsonl").exists()
    assert not wire.exists()
    exports = {name: (agent / name).read_bytes() for name in [
        "openai-messages.jsonl", "openai-messages.meta.json",
        "openai-tools.json", "openai-request.meta.json"]}
    second = materializer.materialize(session, reward="1", status="ok")
    assert second["complete"]
    assert (second["n_requests"], second["n_messages"], second["n_tools"]) == (
        first["n_requests"], first["n_messages"], first["n_tools"])
    assert exports == {name: (agent / name).read_bytes() for name in exports}
    assert json.loads((agent / "verifier-summary.json").read_text())["reward"] == "1"


@pytest.mark.parametrize("invalid", ["empty_context", "corrupt_export", "exception"])
def test_failed_conversion_retains_original_context_and_wire(captured_session, monkeypatch, invalid):
    session, agent, wire, _ = captured_session
    if invalid == "empty_context":
        (agent / "llm-context.jsonl").write_text("{}\n")
    elif invalid == "corrupt_export":
        original = materializer.agent_artifacts.write_openai_messages
        def corrupt(*args, **kwargs):
            count = original(*args, **kwargs)
            args[1].write_text("not json\n")
            return count
        monkeypatch.setattr(materializer.agent_artifacts, "write_openai_messages", corrupt)
    else:
        def fail(*args, **kwargs):
            raise OSError("export unavailable")
        monkeypatch.setattr(materializer.agent_artifacts, "write_openai_tools", fail)
    if invalid == "exception":
        with pytest.raises(OSError, match="export unavailable"):
            materializer.materialize(session, reward="0", status="ok")
    else:
        result = materializer.materialize(session, reward="0", status="ok")
        assert not result["complete"]
        assert result["reason"] == "openai_export_validation_failed"
    assert (agent / "llm-context.jsonl").exists()
    assert (wire / "000001-req.json").exists()


def test_debug_flags_keep_context_and_wire(captured_session, monkeypatch):
    session, agent, wire, _ = captured_session
    monkeypatch.setenv("HARNESSRL_KEEP_LLM_CONTEXT", "1")
    monkeypatch.setenv("HARNESSRL_KEEP_RAW_TRACE_WIRE", "1")
    result = materializer.materialize(session, reward="1", status="ok")
    assert result["complete"] and result["context_retained"]
    assert (agent / "llm-context.jsonl").exists() and wire.exists()


def test_corrupt_retained_exports_are_not_reused_without_capture(captured_session):
    session, agent, _, _ = captured_session
    assert materializer.materialize(session, reward="1", status="ok")["complete"]
    (agent / "openai-messages.meta.json").write_text("{}")
    result = materializer.materialize(session, reward="1", status="ok")
    assert not result["complete"] and result["reason"] == "missing_llm_context"
