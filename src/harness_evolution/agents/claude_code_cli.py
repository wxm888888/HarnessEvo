from __future__ import annotations

import json
import os
import queue
import signal
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from harness_evolution.agents.base import BaseClient
from harness_evolution.agents.registry import CLIENT_REGISTRY


@CLIENT_REGISTRY.register("claude_code_cli_client")
class ClaudeCodeCliClient(BaseClient):
    """Client wrapper for Claude Code CLI in non-interactive print mode."""

    def __init__(
        self,
        command_prefix: str = "claude",
        env: dict[str, str] | None = None,
        output_format: str = "stream-json",
        permission_mode: str = "bypassPermissions",
        extra_args: list[str] | None = None,
        stream_output: bool = False,
        output_log_path: str | None = None,
        output_prefix: str = "[claude-evolver] ",
    ) -> None:
        if not command_prefix.strip():
            raise ValueError("command_prefix must not be empty")
        self.command_prefix = command_prefix
        self.env = {str(key): str(value) for key, value in (env or {}).items() if value is not None}
        self.output_format = output_format
        self.permission_mode = permission_mode
        self.extra_args = [str(arg) for arg in (extra_args or [])]
        self.stream_output = bool(stream_output)
        self.output_log_path = str(output_log_path) if output_log_path else None
        self.output_prefix = str(output_prefix)

    def generate(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        request_timeout_sec: int,
    ) -> list[dict[str, Any]]:
        _ = model_name
        _ = messages
        _ = request_timeout_sec
        raise NotImplementedError("claude_code_cli_client is agent-style; use run_instruction instead")

    def run_instruction(
        self,
        *,
        workspace: Path,
        instruction: str,
        request_timeout_sec: int,
    ) -> list[dict[str, Any]]:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if not workspace.exists():
            raise ValueError(f"workspace does not exist: {workspace}")

        command = shlex.split(self.command_prefix)
        command.extend(
            [
                "--verbose",
                f"--output-format={self.output_format}",
                f"--permission-mode={self.permission_mode}",
                *self.extra_args,
                "--print",
                "--",
                instruction,
            ]
        )
        run_env = {
            **os.environ,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
            **self.env,
        }

        log_path = self._resolve_log_path(workspace)
        log_file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            log_file.write(
                f"\n===== claude code evolver run {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
            )
            log_file.write(f"cwd: {workspace}\n")
            log_file.write(f"command: {' '.join(shlex.quote(part) for part in command)}\n\n")
            log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=run_env,
            start_new_session=True,
        )
        output_parts: list[str] = []
        output_queue: queue.Queue[str] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + max(1, int(request_timeout_sec))
            while True:
                self._drain_output_queue(output_queue, output_parts, log_file)
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(cmd=command, timeout=request_timeout_sec, output="".join(output_parts))
                if process.poll() is not None:
                    reader.join(timeout=1)
                    self._drain_output_queue(output_queue, output_parts, log_file)
                    break
                time.sleep(0.1)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_group(process)
            output = self._coerce_output(exc.output) or "".join(output_parts)
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=request_timeout_sec,
                output=(
                    f"claude code cli timed out after {request_timeout_sec}s; "
                    "terminated its process group.\n"
                    f"{output}"
                ),
            ) from exc
        finally:
            if log_file is not None:
                log_file.close()

        stdout = "".join(output_parts)
        trajectory = self._build_agent_trajectory(prompt=instruction, output=stdout)
        if process.returncode != 0:
            raise RuntimeError(
                f"claude code cli command failed: {' '.join(shlex.quote(part) for part in command)}\n"
                f"trajectory:\n{json.dumps(trajectory, ensure_ascii=False, indent=2)}\n\nraw output:\n{stdout}"
            )
        return trajectory

    def _resolve_log_path(self, workspace: Path) -> Path | None:
        if not self.output_log_path:
            return None
        path = Path(self.output_log_path).expanduser()
        if not path.is_absolute():
            path = workspace / path
        return path

    def _drain_output_queue(
        self,
        output_queue: queue.Queue[str],
        output_parts: list[str],
        log_file: Any,
    ) -> None:
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                return
            output_parts.append(line)
            if self.stream_output:
                sys.stderr.write(f"{self.output_prefix}{line}")
                sys.stderr.flush()
            if log_file is not None:
                log_file.write(line)
                log_file.flush()

    def _terminate_process_group(self, process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
            process.wait()
            return

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def _coerce_output(self, output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output

    def _build_agent_trajectory(self, *, prompt: str, output: str) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        assistant_parts: list[str] = []
        tool_events: list[dict[str, Any]] = []

        for raw in output.splitlines():
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "assistant":
                self._collect_assistant_text(event, assistant_parts)
            elif event_type in {"tool_use", "tool_result"}:
                tool_events.append(event)
            elif event_type == "result" and isinstance(event.get("result"), str):
                assistant_parts.append(event["result"])

        content = "\n".join(part for part in assistant_parts if part).strip()
        assistant: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_events:
            assistant["tool_events"] = tool_events
        trajectory.append(assistant)
        return trajectory

    def _collect_assistant_text(self, event: dict[str, Any], parts: list[str]) -> None:
        message = event.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            return
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
