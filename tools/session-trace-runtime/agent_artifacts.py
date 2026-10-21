"""Agent-run artifacts on gpfs2.

Layout:
  $AGENT_RUNS_ROOT/
    README.md
    index.jsonl                 one line per task attempt (all runs)
    <run_id>/
      run.json                  this batch
      index.jsonl               every attempt in this run
      summary.json              pass@k (batch eval)
      <task_id>/
        summary.json            this task's k attempts
        k01/ k02/ ...
          meta.json             task id, model, sandbox, timings, reward
        instruction.md          original prompt
        reward.txt              0 / 1 / missing
        verifier.txt            pytest output
        run.log                 control-plane log
        trajectory.jsonl        simplified steps (thinking / tool / assistant)
        trajectory.md           human-readable, includes thinking
        llm-turns.jsonl         per-request model turn (reasoning_content + content)
        llm-context.jsonl       complete request messages/tools and reconstructed response
        openai-messages.jsonl   append-only OpenAI-compatible message stream
        openai-tools.json       tool schemas supplied to the model
        openai-request.meta.json request-level settings without message bodies
        verifier-summary.json   sanitized outcome summary for evolution
        raw/                    sandbox /logs dump

Set AGENT_RUN_ID to put many tasks in the same run. Pass@k attempts live
at <task_id>/k01, k02, ....
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(
    os.environ.get(
        "AGENT_RUNS_ROOT",
        "/mnt/shared-storage-gpfs2/safewt-share02/sunyoubang/yuanjiarui/duchenyu/agent-runs",
    )
)

README = """# agent-runs

Grouped by **run**, then by **task**:

```
<run_id>/
  run.json
  index.jsonl
  summary.json
  <task_id>/
    summary.json
    k01/
      meta.json           task id + run info + reward
    instruction.md      prompt given to the agent
    reward.txt          verifier reward (1 pass / 0 fail)
    verifier.txt        pytest -v output
    run.log             EnvGateway control-plane log
        trajectory.jsonl    normalized agent steps, including DeepAgents traces
        trajectory.md       same steps, readable
    llm-turns.jsonl     one row per LLM call (reasoning_content + content)
    llm-context.jsonl   one row per LLM call with complete request context and response
    openai-messages.jsonl append-only OpenAI-compatible messages for evolution
    openai-tools.json  tool schemas supplied to the model
    openai-request.meta.json request-level settings without message bodies
    verifier-summary.json sanitized outcome summary for evolution
    raw/                full /logs dump from the sandbox
```

Put many tasks in one run with `AGENT_RUN_ID=my-eval`.
Pass@k attempts are `<task_id>/k01`, `k02`, ....

Root `index.jsonl` is one JSON object per attempt across all runs.
Each run also has its own `index.jsonl`.
"""

_CURRENT_RUN_ID: str | None = None
_IO_LOCK = threading.RLock()

_TOOL_OUT_LIMIT = 20_000
_TEXT_LIMIT = 100_000


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (value or "").strip())
    return cleaned or "unknown"


def resolve_run_id(explicit: str | None = None) -> str:
    """Reuse one run id per process; override with AGENT_RUN_ID / RUN_ID."""
    global _CURRENT_RUN_ID
    if explicit:
        _CURRENT_RUN_ID = _safe_name(explicit)
        return _CURRENT_RUN_ID
    env = os.environ.get("AGENT_RUN_ID") or os.environ.get("RUN_ID")
    if env:
        _CURRENT_RUN_ID = _safe_name(env)
        return _CURRENT_RUN_ID
    if _CURRENT_RUN_ID:
        return _CURRENT_RUN_ID
    _CURRENT_RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
    return _CURRENT_RUN_ID


def _unique_task_dir(run_root: Path, task_id: str) -> Path:
    base = run_root / _safe_name(task_id)
    if not base.exists():
        return base
    n = 2
    while True:
        candidate = run_root / f"{_safe_name(task_id)}__{n:02d}"
        if not candidate.exists():
            return candidate
        n += 1


def _touch_run_json(run_root: Path, run_id: str) -> None:
    with _IO_LOCK:
        path = run_root / "run.json"
        try:
            data = json.loads(_read_text(path) or "{}") if path.exists() else {}
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("run_id", run_id)
        data.setdefault("started_at", _now())
        data["updated_at"] = _now()
        data["n_tasks"] = sum(1 for p in run_root.iterdir() if p.is_dir())
        _write_json(path, data)


def ensure_readme() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(ROOT / "README.md", README)


def attempt_dir(run_id: str, task_id: str, attempt: int) -> Path:
    return ROOT / _safe_name(run_id) / _safe_name(task_id) / f"k{int(attempt):02d}"


def is_complete(path: Path) -> bool:
    if (path / ".incomplete").exists():
        return False
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        reward = (path / "reward.txt").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return False
    if (meta.get("status"), reward) not in {("ok", "1"), ("fail", "0")}:
        return False
    if meta.get("reward") != reward or meta.get("verifier_rc") != 0 or meta.get("verifier_invalid"):
        return False
    # Existing completed runs predate receipt files. Their terminal status and
    # verifier_rc are the only durable receipt; new generations require a file.
    if not meta.get("attempt_generation"):
        return True
    try:
        receipt = json.loads((path / "verifier_receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (meta.get("verifier_receipt_valid") is True
            and receipt.get("attempt_generation") == meta["attempt_generation"]
            and receipt.get("reward") == reward and receipt.get("verifier_rc") == 0)


def start_run(
    task_id: str,
    instruction: str,
    seed_meta: dict[str, Any],
    run_id: str | None = None,
    attempt: int | None = None,
) -> Path:
    """Create <run_id>/<task_id>/kXX/ and write instruction + in-progress meta."""
    ensure_readme()
    run_id = resolve_run_id(run_id)
    run_root = ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if attempt is None:
        task_dir = _unique_task_dir(run_root, task_id)
    else:
        task_dir = attempt_dir(run_id, task_id, attempt)
    task_dir.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    history: Path | None = None
    with _IO_LOCK:
        old_entries = [p for p in task_dir.iterdir() if p.name != "generations"]
        if old_entries:
            history = task_dir / "generations" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + generation)
            history.mkdir(parents=True)
            for entry in old_entries:
                entry.rename(history / entry.name)
    _touch_run_json(run_root, run_id)
    (task_dir / ".incomplete").write_text("1\n", encoding="utf-8")
    _write(task_dir / "instruction.md", instruction if instruction.endswith("\n") else instruction + "\n")
    meta = {
        "task_id": task_id,
        "run_id": run_id,
        "attempt": attempt,
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "reward": None,
        "attempt_generation": generation,
        "prior_generation_archive": str(history) if history else None,
        "artifacts_dir": str(task_dir),
        **seed_meta,
    }
    _write_json(task_dir / "meta.json", meta)
    return task_dir


def parse_opencode_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type"):
            events.append(obj)
    return events


def parse_deepagents_messages(path: Path) -> list[dict[str, Any]]:
    """Normalize the DeepAgents/LangChain message trace to paper steps."""
    try:
        payload = json.loads(_read_text(path) or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    steps: list[dict[str, Any]] = []
    step_no = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or item.get("class_name") or "").lower()
        text = item.get("text") or item.get("content") or ""
        if isinstance(text, list):
            text = "\n".join(str(x) for x in text)
        if kind in {"human", "humanmessage", "user"}:
            continue
        if kind in {"ai", "assistant", "aimessage"}:
            step_no += 1
            if str(text).strip():
                steps.append({"i": len(steps) + 1, "type": "assistant", "step": step_no, "text": str(text)})
            for call in item.get("tool_calls") or []:
                if isinstance(call, dict):
                    steps.append({
                        "i": len(steps) + 1, "type": "tool", "step": step_no,
                        "name": call.get("name") or call.get("id") or "tool",
                        "status": "requested", "input": call.get("args") or call,
                        "output": "",
                    })
            continue
        if kind in {"tool", "toolmessage"}:
            steps.append({
                "i": len(steps) + 1, "type": "tool", "step": step_no,
                "name": item.get("name") or "tool", "status": "completed",
                "input": None, "output": str(text),
            })
            continue
        if str(text).strip():
            steps.append({"i": len(steps) + 1, "type": "thinking", "step": step_no, "text": str(text)})
    return steps


def parse_deepagents_events(path: Path) -> list[dict[str, Any]]:
    """Fallback parser for streamed DeepAgents event records."""
    if not path.is_file():
        return []
    steps: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        msg = data.get("message") if isinstance(data.get("message"), dict) else None
        if msg is None:
            continue
        kind = str(msg.get("type") or msg.get("class_name") or "").lower()
        text = msg.get("text") or msg.get("content") or ""
        if isinstance(text, list):
            text = "\n".join(str(x) for x in text)
        if kind in {"ai", "assistant", "aimessage"} and str(text).strip():
            steps.append({"i": len(steps) + 1, "type": "assistant", "step": len(steps) + 1, "text": str(text)})
        elif kind in {"tool", "toolmessage"}:
            steps.append({"i": len(steps) + 1, "type": "tool", "step": len(steps) + 1, "name": msg.get("name") or "tool", "status": "completed", "input": None, "output": str(text)})
    return steps


def _clip(value: Any, limit: int) -> tuple[Any, bool]:
    if not isinstance(value, str):
        text = json.dumps(value, ensure_ascii=False)
        if len(text) <= limit:
            return value, False
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]", True
    if len(value) <= limit:
        return value, False
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]", True


def simplify_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    step_no = 0
    for ev in events:
        kind = ev.get("type")
        part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
        ts = ev.get("timestamp")
        if kind == "step_start":
            step_no += 1
            steps.append({"i": len(steps) + 1, "type": "step_start", "step": step_no, "timestamp": ts})
            continue
        if kind == "reasoning":
            text, trunc = _clip(part.get("text") or "", _TEXT_LIMIT)
            if not str(text).strip():
                continue
            rec = {"i": len(steps) + 1, "type": "thinking", "step": step_no, "text": text, "timestamp": ts}
            if trunc:
                rec["truncated"] = True
            steps.append(rec)
            continue
        if kind == "text":
            text, trunc = _clip(part.get("text") or "", _TEXT_LIMIT)
            if not str(text).strip():
                continue
            rec = {"i": len(steps) + 1, "type": "assistant", "step": step_no, "text": text, "timestamp": ts}
            if trunc:
                rec["truncated"] = True
            steps.append(rec)
            continue
        if kind == "tool_use":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            out, out_trunc = _clip(state.get("output") or "", _TOOL_OUT_LIMIT)
            rec = {
                "i": len(steps) + 1,
                "type": "tool",
                "step": step_no,
                "name": part.get("tool") or "tool",
                "status": state.get("status"),
                "input": state.get("input"),
                "output": out,
                "title": part.get("title") or state.get("title"),
                "timestamp": ts,
            }
            if out_trunc:
                rec["truncated"] = True
            steps.append(rec)
            continue
        if kind == "step_finish":
            steps.append(
                {
                    "i": len(steps) + 1,
                    "type": "step_metrics",
                    "step": step_no,
                    "tokens": part.get("tokens"),
                    "cost": part.get("cost"),
                    "reason": part.get("reason"),
                    "timestamp": ts,
                }
            )
            continue
        if kind == "error":
            steps.append({"i": len(steps) + 1, "type": "error", "step": step_no, "payload": part or ev, "timestamp": ts})
    return steps


def render_trajectory_md(
    steps: list[dict[str, Any]],
    *,
    task_id: str,
    reward: str | None,
    instruction: str,
) -> str:
    out: list[str] = [
        f"# Trajectory `{task_id}`",
        "",
        f"- reward: `{reward}`",
        f"- steps: {sum(1 for s in steps if s.get('type') == 'step_start')}",
        "",
        "## Instruction",
        "",
        instruction.rstrip(),
        "",
        "## Agent",
        "",
    ]
    for rec in steps:
        kind = rec.get("type")
        if kind == "step_start":
            out.append(f"### Step {rec.get('step')}")
            out.append("")
        elif kind == "thinking":
            out.append("#### Thinking")
            out.append("")
            out.append(str(rec.get("text") or "").rstrip())
            out.append("")
        elif kind == "assistant":
            out.append("#### Assistant")
            out.append("")
            out.append(str(rec.get("text") or "").rstrip())
            out.append("")
        elif kind == "tool":
            name = rec.get("name") or "tool"
            out.append(f"#### Tool `{name}` ({rec.get('status') or '?'})")
            out.append("")
            inp = rec.get("input")
            out.append("input:")
            out.append("")
            out.append("```json")
            out.append(json.dumps(inp, ensure_ascii=False, indent=2) if not isinstance(inp, str) else inp)
            out.append("```")
            out.append("")
            output = rec.get("output")
            if output:
                out.append("output:")
                out.append("")
                out.append("```")
                out.append(str(output).rstrip())
                out.append("```")
                out.append("")
        elif kind == "step_metrics":
            tok = rec.get("tokens") or {}
            out.append(f"*tokens={tok} reason={rec.get('reason')}*")
            out.append("")
        elif kind == "error":
            out.append("#### Error")
            out.append("")
            out.append("```json")
            out.append(json.dumps(rec.get("payload"), ensure_ascii=False, indent=2)[:8000])
            out.append("```")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def token_totals(events: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "reasoning": 0, "total": 0, "steps": 0}
    for ev in events:
        if ev.get("type") != "step_finish":
            continue
        part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
        tok = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
        totals["steps"] += 1
        for key in ("input", "output", "reasoning", "total"):
            val = tok.get(key)
            if isinstance(val, (int, float)):
                totals[key] += int(val)
    return totals


def _assemble_sse_lines(lines: Iterable[str]) -> dict[str, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    tools: dict[int, dict[str, Any]] = {}
    usage = None
    model = None
    finish = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        model = obj.get("model") or model
        usage = obj.get("usage") or usage
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("content"):
                content.append(str(delta["content"]))
            if delta.get("reasoning_content"):
                reasoning.append(str(delta["reasoning_content"]))
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index") or 0)
                slot = tools.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if fn.get("name"):
                    slot["function"]["name"] += str(fn["name"])
                if fn.get("arguments"):
                    slot["function"]["arguments"] += str(fn["arguments"])
    tool_list = [tools[i] for i in sorted(tools)]
    return {
        "model": model,
        "reasoning_content": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": tool_list or None,
        "finish_reason": finish,
        "usage": usage,
        "stream": True,
    }


def assemble_sse(raw: bytes) -> dict[str, Any]:
    """Keep the bytes API for existing callers and small in-memory payloads."""
    return _assemble_sse_lines(raw.decode("utf-8", errors="replace").splitlines())


def assemble_sse_file(path: Path) -> dict[str, Any]:
    """Read wire incrementally; retain only the assembled response and usage.

    A response may contain gigabytes of repeated per-event token metadata.
    Reading, decoding and splitting the entire file made several full-size
    copies before processing even its first event. Iterate the text stream
    instead. Per-line splitlines preserves the bytes API's handling of CRLF,
    CR and Unicode line separators, including malformed wire records.
    """
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return _assemble_sse_lines(part for line in stream for part in line.splitlines())


def _message_turn(message: dict[str, Any] | None) -> dict[str, Any]:
    msg = message or {}
    return {
        "reasoning_content": msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls"),
    }


def write_llm_turns(llm_dir: Path, dest: Path) -> int:
    if not llm_dir.is_dir():
        return 0
    metas = sorted(llm_dir.glob("*-meta.json"))
    n_written = 0
    with dest.open("w", encoding="utf-8") as fh:
        for meta_path in metas:
            prefix = meta_path.name[: -len("-meta.json")]
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {"n": prefix}
            rec: dict[str, Any] = {
                "n": meta.get("n"),
                "path": meta.get("path"),
                "status": meta.get("status"),
                "model": meta.get("model"),
                "stream": meta.get("stream"),
                "max_tokens": meta.get("max_tokens") or meta.get("max_completion_tokens"),
            }
            sse = llm_dir / f"{prefix}-resp.sse"
            resp_json = llm_dir / f"{prefix}-resp.json"
            if not sse.exists() and not resp_json.exists():
                continue
            if sse.exists():
                assembled = assemble_sse_file(sse)
                model = rec.get("model") or assembled.get("model")
                rec.update(assembled)
                rec["model"] = model
            elif resp_json.exists():
                try:
                    body = json.loads(resp_json.read_text(encoding="utf-8"))
                except Exception:
                    body = {}
                choices = body.get("choices") if isinstance(body, dict) else None
                msg = None
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message")
                    rec["finish_reason"] = choices[0].get("finish_reason")
                rec.update(_message_turn(msg if isinstance(msg, dict) else None))
                rec["usage"] = body.get("usage") if isinstance(body, dict) else None
                rec["stream"] = False
            if meta.get("error_body"):
                rec["error_body"] = meta["error_body"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


def _json_file(path: Path) -> Any:
    """Read one captured JSON wire file without failing finalization."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _response_wire(llm_dir: Path, prefix: str, *, stream: bool) -> Any:
    """Return the reconstructed model response for one captured request."""
    if stream:
        path = llm_dir / f"{prefix}-resp.sse"
        return assemble_sse_file(path) if path.exists() else None
    path = llm_dir / f"{prefix}-resp.json"
    return _json_file(path) if path.exists() else None


def write_llm_context(llm_dir: Path, dest: Path) -> int:
    """Persist complete OpenAI-compatible request/response context.

    The dump proxy captures the request body before forwarding it. Previously
    finalization discarded that body and kept only ``llm-turns.jsonl``. Keeping
    this derived JSONL makes prompt roles, injected messages, tools and
    ``tool_choice`` inspectable by the harness evolver without retaining raw
    HTTP headers or API credentials.
    """
    if not llm_dir.is_dir():
        return 0
    metas = sorted(llm_dir.glob("*-meta.json"))
    n_written = 0
    with dest.open("w", encoding="utf-8") as fh:
        for meta_path in metas:
            prefix = meta_path.name[: -len("-meta.json")]
            meta = _json_file(meta_path)
            if not isinstance(meta, dict):
                meta = {"n": prefix}
            request = _json_file(llm_dir / f"{prefix}-req.json")
            if not isinstance(request, dict):
                continue
            messages = request.get("messages") if isinstance(request, dict) else None
            message_roles = [
                str(message.get("role") or "")
                for message in messages
                if isinstance(message, dict)
            ] if isinstance(messages, list) else []
            request_summary = {
                "model": request.get("model") if isinstance(request, dict) else None,
                "stream": request.get("stream") if isinstance(request, dict) else None,
                "tool_choice": request.get("tool_choice") if isinstance(request, dict) else None,
                "message_roles": message_roles,
                "last_message_role": message_roles[-1] if message_roles else None,
                "n_messages": len(messages) if isinstance(messages, list) else 0,
                "n_tools": len(request.get("tools") or []) if isinstance(request, dict) else 0,
                "max_tokens": request.get("max_tokens") if isinstance(request, dict) else None,
                "max_completion_tokens": request.get("max_completion_tokens") if isinstance(request, dict) else None,
            }
            record = {
                # ``n`` is the compact context sequence. Keep the proxy's
                # global wire sequence separately because discovery requests
                # are intentionally omitted from this file.
                "n": n_written + 1,
                "wire_n": meta.get("n"),
                "request": request,
                "request_summary": request_summary,
                "meta": {
                    key: meta.get(key)
                    for key in (
                        "method", "path", "status", "model", "stream", "max_tokens",
                        "max_completion_tokens", "n_messages", "n_tools", "tool_choice",
                        "error_body", "session_id", "parent_session_id",
                    )
                    if key in meta
                },
                "response": _response_wire(llm_dir, prefix, stream=bool(meta.get("stream"))),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


def _assistant_message(response: Any) -> dict[str, Any] | None:
    """Normalize one captured response to an OpenAI-compatible assistant message."""
    if not isinstance(response, dict):
        return None
    # Non-stream responses retain the standard Chat Completions envelope.
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return dict(message)

    # Streaming responses are reconstructed by assemble_sse().
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.get("content") or None,
    }
    reasoning = response.get("reasoning_content")
    if reasoning:
        # Provider extension used by Qwen/OpenCode; retain it for evolution.
        message["reasoning_content"] = reasoning
    tool_calls = response.get("tool_calls")
    if tool_calls:
        message["tool_calls"] = tool_calls
    if message.get("content") is None and not message.get("tool_calls") and not reasoning:
        return None
    return message


def _message_key(message: Any) -> str:
    """Stable comparison key for cumulative OpenAI messages."""
    if not isinstance(message, dict):
        return json.dumps(message, ensure_ascii=False, sort_keys=True)
    # Reasoning fields may be omitted by a provider when the same assistant
    # message is replayed in the next request. They do not change ordering.
    comparable = {
        key: value
        for key, value in message.items()
        if key != "reasoning_content"
    }
    # The proxy/provider can reserialize tool-call arguments between rounds
    # (for example, adding spaces after JSON separators).  Compare parsed
    # arguments structurally so this harmless wire-level change does not make
    # the same assistant turn appear twice in the incremental stream.
    tool_calls = comparable.get("tool_calls")
    if isinstance(tool_calls, list):
        normalized_calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                normalized_calls.append(call)
                continue
            normalized = dict(call)
            function = normalized.get("function")
            if isinstance(function, dict):
                function = dict(function)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except (TypeError, ValueError):
                        pass
                normalized["function"] = function
            normalized_calls.append(normalized)
        comparable["tool_calls"] = normalized_calls
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_openai_messages_legacy(context_path: Path, dest: Path, meta_dest: Path | None = None) -> int:
    """Convert cumulative request snapshots into an append-only message stream.

    This is deliberately prompt-agnostic. Every message that appears after the
    common prefix of two consecutive request histories is appended as-is, so
    injected ``user`` messages (including max-step prompts) are preserved
    without matching their text or type specially.
    """
    if not context_path.is_file():
        return 0
    rows: list[dict[str, Any]] = []
    for raw in context_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)

    messages: list[dict[str, Any]] = []
    previous_history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        request = row.get("request")
        history = request.get("messages") if isinstance(request, dict) else None
        history = [message for message in history if isinstance(message, dict)] if isinstance(history, list) else []

        common = 0
        while (
            common < len(previous_history)
            and common < len(history)
            and _message_key(previous_history[common]) == _message_key(history[common])
        ):
            common += 1

        start = len(messages)
        for message in history[common:]:
            messages.append(dict(message))

        assistant = _assistant_message(row.get("response"))
        if assistant is not None:
            messages.append(assistant)

        # The next request normally contains the assistant response verbatim;
        # include it in the comparison history so only the following tool/user
        # messages are emitted on the next iteration.
        previous_history = list(history)
        if assistant is not None:
            previous_history.append(assistant)

        records.append(
            {
                "request_n": row.get("n"),
                "wire_n": row.get("wire_n"),
                "message_start": start,
                "message_end": len(messages),
                "new_messages": len(messages) - start,
                "finish_reason": (row.get("response") or {}).get("finish_reason")
                if isinstance(row.get("response"), dict)
                else None,
            }
        )

    with dest.open("w", encoding="utf-8") as fh:
        for message in messages:
            fh.write(json.dumps(message, ensure_ascii=False) + "\n")
    if meta_dest is not None:
        _write_json(
            meta_dest,
            {
                "schema_version": "openai-messages-v1",
                "source_context": str(context_path),
                "message_count": len(messages),
                "request_count": len(rows),
                "requests": records,
            },
        )
    return len(messages)


def write_openai_messages(context_path: Path, dest: Path, meta_dest: Path | None = None) -> int:
    from session_exports import export_messages
    return export_messages(context_path, dest, meta_dest, _assistant_message, _write_openai_messages_legacy)


def _context_rows(context_path: Path) -> list[dict[str, Any]]:
    """Read valid context records without exposing malformed wire rows."""
    if not context_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in context_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_openai_tools(context_path: Path, dest: Path) -> int:
    """Save the first stable OpenAI tool schema once, without prompt history."""
    manifest = Path(dest).parent / "openai-sessions.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        main = next(item for item in metadata["sessions"] if item["session_id"] == metadata["main_session_id"])
        tool_list = json.loads((manifest.parent / main["tools_path"]).read_text())
        _write_json(Path(dest), tool_list)
        return len(tool_list)
    tools: list[Any] = []
    for row in _context_rows(context_path):
        request = row.get("request")
        candidate = request.get("tools") if isinstance(request, dict) else None
        if isinstance(candidate, list) and candidate:
            tools = candidate
            break
    _write_json(dest, tools)
    return len(tools)


def write_openai_request_meta(context_path: Path, dest: Path) -> int:
    """Save request settings while deliberately omitting message bodies."""
    from session_exports import _rows
    rows = _rows(context_path)
    requests: list[dict[str, Any]] = []
    for row in rows:
        request = row.get("request")
        if not isinstance(request, dict):
            continue
        requests.append(
            {
                "request_n": row.get("n"),
                "wire_n": row.get("wire_n"),
                "session_id": (row.get("meta") or {}).get("session_id"),
                "parent_session_id": (row.get("meta") or {}).get("parent_session_id"),
                "model": request.get("model"),
                "stream": request.get("stream"),
                "tool_choice": request.get("tool_choice"),
                "max_tokens": request.get("max_tokens"),
                "max_completion_tokens": request.get("max_completion_tokens"),
                "n_messages": len(request.get("messages") or []),
                "n_tools": len(request.get("tools") or []),
                "finish_reason": (row.get("response") or {}).get("finish_reason")
                if isinstance(row.get("response"), dict)
                else None,
            }
        )
    first = requests[0] if requests else {}
    _write_json(
        dest,
        {
            "schema_version": "openai-request-meta-v1",
            "model": first.get("model"),
            "stream": first.get("stream"),
            "tool_choice": first.get("tool_choice"),
            "max_tokens": first.get("max_tokens"),
            "max_completion_tokens": first.get("max_completion_tokens"),
            "request_count": len(requests),
            "requests": requests,
        },
    )
    return len(requests)


def _verifier_failure_class(*, reward: str, status: str, error: str | None,
                            extra_meta: dict[str, Any]) -> str:
    if str(reward) == "1" and extra_meta.get("verifier_rc") == 0:
        return "passed"
    text = " ".join(
        str(value or "")
        for value in (
            error,
            extra_meta.get("failure_phase"),
            extra_meta.get("verifier_terminal_reason"),
        )
    ).lower()
    if "verifier" in text and "timeout" in text:
        return "verifier_timeout"
    if "timeout" in text:
        return "agent_timeout"
    if extra_meta.get("verifier_invalid"):
        return "verifier_invalid"
    if extra_meta.get("verifier_rc") not in (None, 0):
        return "verifier_failure"
    if status == "error" or error:
        return "runtime_error"
    return "wrong_or_missing_output"


def write_verifier_summary(
    dest: Path,
    *,
    reward: str,
    status: str,
    error: str | None,
    extra_meta: dict[str, Any],
) -> None:
    """Write only structured verifier outcome; never copy verifier text/code."""
    _write_json(
        dest,
        {
            "schema_version": "verifier-summary-v1",
            "reward": str(reward),
            "status": status,
            "failure_class": _verifier_failure_class(
                reward=str(reward), status=status, error=error, extra_meta=extra_meta
            ),
            "verifier_rc": extra_meta.get("verifier_rc"),
            "verifier_receipt_valid": bool(extra_meta.get("verifier_receipt_valid")),
            "agent_exit": extra_meta.get("opencode_exit"),
            "failure_phase": extra_meta.get("failure_phase"),
            "error_present": bool(error),
            "agent_timeout": "timeout" in " ".join(
                str(value or "") for value in (error, extra_meta.get("failure_phase"))
            ).lower(),
        },
    )


def extract_logs_tar(tar_bytes: bytes, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as archive:
        try:
            archive.extractall(raw_dir, filter="data")
        except TypeError:
            archive.extractall(raw_dir)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def finalize_run(
    run_dir: Path,
    *,
    reward: str | None,
    status: str,
    error: str | None,
    extra_meta: dict[str, Any],
    tar_bytes: bytes | None = None,
    verifier_stdout: str | None = None,
    started_monotonic: float | None = None,
) -> dict[str, Any]:
    """Write trajectory / reward / meta / index. Safe to call more than once."""
    raw_dir = run_dir / "raw"
    if tar_bytes:
        try:
            extract_logs_tar(tar_bytes, raw_dir)
        except Exception as exc:
            extra_meta = {**extra_meta, "tar_extract_error": f"{type(exc).__name__}: {exc}"}

    opencode_txt = _read_text(raw_dir / "agent" / "opencode.txt")
    events = parse_opencode_events(opencode_txt) if opencode_txt else []
    steps = simplify_events(events)
    trajectory_source = "opencode"
    if not steps:
        # DeepAgents traces are collected from /logs/agent by the same tar
        # path as OpenCode.  Keep a host-side fallback for resumed attempts.
        for base in (run_dir / "agent", run_dir / "deepagents", raw_dir / "agent", raw_dir / "deepagents"):
            msg_path = base / "messages.json"
            event_path = base / "events.jsonl"
            if msg_path.is_file():
                steps = parse_deepagents_messages(msg_path)
                trajectory_source = "deepagents.messages"
            if not steps and event_path.is_file():
                steps = parse_deepagents_events(event_path)
                trajectory_source = "deepagents.events"
            if steps:
                break
    instruction = _read_text(run_dir / "instruction.md")
    generation = extra_meta.get("attempt_generation")
    if not generation:
        try:
            generation = json.loads(_read_text(run_dir / "meta.json") or "{}").get("attempt_generation")
        except ValueError:
            generation = None
    receipt_valid = (status in {"ok", "fail"} and extra_meta.get("verifier_receipt_valid") is True
                     and extra_meta.get("verifier_rc") == 0 and str(reward) in {"0", "1"}
                     and bool(generation) and not extra_meta.get("verifier_invalid"))
    reward = str(reward) if receipt_valid else "missing"
    receipt_path = run_dir / "verifier_receipt.json"
    if receipt_valid:
        _write_json(receipt_path, {"attempt_generation": generation, "reward": reward,
                                   "verifier_rc": 0, "verified_at": _now()})
    else:
        receipt_path.unlink(missing_ok=True)
    _write(run_dir / "reward.txt", str(reward).strip() + "\n")
    _write(run_dir / "trajectory.jsonl", "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in steps))
    _write(
        run_dir / "trajectory.md",
        render_trajectory_md(
            steps,
            task_id=str(extra_meta.get("task_id") or run_dir.name),
            reward=reward,
            instruction=instruction,
        ),
    )

    verifier_txt = _read_text(raw_dir / "verifier" / "test-stdout.txt") or (verifier_stdout or "")
    if verifier_txt:
        _write(run_dir / "verifier.txt", verifier_txt if verifier_txt.endswith("\n") else verifier_txt + "\n")

    llm_wire_dir = raw_dir / "agent" / "llm-reqs"
    n_turns = write_llm_turns(llm_wire_dir, run_dir / "llm-turns.jsonl")
    n_contexts = write_llm_context(llm_wire_dir, run_dir / "llm-context.jsonl")
    n_messages = write_openai_messages(
        run_dir / "llm-context.jsonl",
        run_dir / "openai-messages.jsonl",
        run_dir / "openai-messages.meta.json",
    )
    n_tools = write_openai_tools(
        run_dir / "llm-context.jsonl",
        run_dir / "openai-tools.json",
    )
    n_request_meta = write_openai_request_meta(
        run_dir / "llm-context.jsonl",
        run_dir / "openai-request.meta.json",
    )
    write_verifier_summary(
        run_dir / "verifier-summary.json",
        reward=str(reward),
        status=status,
        error=error,
        extra_meta=extra_meta,
    )
    # Keep the derived complete context and model turns but discard raw
    # request/SSE wire captures unless explicitly requested for debugging.
    if os.environ.get("KEEP_RAW_LLM_WIRE", "0") != "1":
        shutil.rmtree(llm_wire_dir, ignore_errors=True)

    try:
        existing = json.loads(_read_text(run_dir / "meta.json") or "{}")
    except json.JSONDecodeError:
        existing = {}
    duration = None
    if started_monotonic is not None:
        duration = round(time.monotonic() - started_monotonic, 3)
    meta = {
        **existing,
        **extra_meta,
        "finished_at": _now(),
        "duration_sec": duration,
        "status": status,
        "reward": str(reward).strip(),
        "reward_numeric": int(reward) if receipt_valid else None,
        "verifier_receipt_valid": receipt_valid,
        "error": error,
        "n_opencode_events": len(events),
        "harness": "deepagents" if trajectory_source.startswith("deepagents") else "opencode",
        "trajectory_source": trajectory_source,
        "n_trajectory_steps": len(steps),
        "n_thinking_spans": sum(1 for s in steps if s.get("type") == "thinking"),
        "n_tool_calls": sum(1 for s in steps if s.get("type") == "tool"),
        "n_llm_turns": n_turns,
        "n_llm_contexts": n_contexts,
        "n_openai_messages": n_messages,
        "n_openai_tools": n_tools,
        "n_openai_request_meta": n_request_meta,
        "tokens": token_totals(events),
        "artifacts_dir": str(run_dir),
    }
    _write_json(run_dir / "meta.json", meta)

    index_row = {
        "task_id": meta.get("task_id"),
        "run_id": meta.get("run_id"),
        "attempt": meta.get("attempt"),
        "reward": meta.get("reward"),
        "status": status,
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "duration_sec": duration,
        "model": meta.get("model"),
        "image_tag": meta.get("image_tag"),
        "error": error,
        "path": str(run_dir),
    }
    line = json.dumps(index_row, ensure_ascii=False) + "\n"
    run_id = str(meta.get("run_id") or extra_meta.get("run_id") or "")
    run_root = ROOT / _safe_name(run_id) if run_id else run_dir.parent
    with _IO_LOCK:
        with (ROOT / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line)
        run_root.mkdir(parents=True, exist_ok=True)
        with (run_root / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line)
        _touch_run_json(run_root, run_id or run_root.name)

    marker = run_dir / ".incomplete"
    try:
        marker.unlink()
    except OSError:
        pass
    return meta
