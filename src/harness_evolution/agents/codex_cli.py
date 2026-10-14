from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class CodexCliRunResult:
    command: list[str]
    cwd: Path
    target_dir: Path
    prompt: str
    returncode: int
    stdout: str
    trajectory: list[dict[str, Any]]


@CLIENT_REGISTRY.register("codex_cli_client")
class CodexCliClient(BaseClient):
    """Run the authenticated Codex CLI as a repository or evidence agent."""

    def __init__(
        self,
        command_prefix: str = "codex",
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "high",
        sandbox: str = "workspace-write",
        approve_for_me: bool = False,
        ephemeral: bool = True,
        skip_git_repo_check: bool = True,
        env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
        stream_output: bool = False,
        output_log_path: str = ".harness_evolver/codex-run.jsonl",
        last_message_path: str = ".harness_evolver/codex-last-message.md",
        output_prefix: str = "[codex-evolver] ",
    ) -> None:
        if not command_prefix.strip():
            raise ValueError("command_prefix must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        self.command_prefix = command_prefix
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.sandbox = sandbox
        self.approve_for_me = bool(approve_for_me)
        self.ephemeral = bool(ephemeral)
        self.skip_git_repo_check = bool(skip_git_repo_check)
        self.env = {str(key): str(value) for key, value in (env or {}).items() if value is not None}
        self.extra_args = [str(arg) for arg in (extra_args or [])]
        self.stream_output = bool(stream_output)
        self.output_log_path = output_log_path
        self.last_message_path = last_message_path
        self.output_prefix = output_prefix

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
        raise NotImplementedError("codex_cli_client is agent-style; use run_instruction instead")

    def run_instruction(
        self,
        *,
        workspace: Path,
        instruction: str,
        request_timeout_sec: int,
    ) -> list[dict[str, Any]]:
        return self.run_instruction_detailed(
            workspace=workspace,
            instruction=instruction,
            request_timeout_sec=request_timeout_sec,
            raise_on_error=True,
        ).trajectory

    def run_instruction_detailed(
        self,
        *,
        workspace: Path,
        instruction: str,
        request_timeout_sec: int,
        raise_on_error: bool = True,
    ) -> CodexCliRunResult:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        workspace = workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace does not exist: {workspace}")

        log_path = self._workspace_path(workspace, self.output_log_path)
        last_message_path = self._workspace_path(workspace, self.last_message_path)
        for path in (log_path, last_message_path):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)

        command = shlex.split(self.command_prefix)
        command.extend(["exec", "-m", self.model, "-C", str(workspace)])
        if self.approve_for_me:
            command.append("--approve-for-me")
        elif self.sandbox:
            command.extend(["--sandbox", self.sandbox])
        if self.ephemeral:
            command.append("--ephemeral")
        if self.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        command.extend(["--json", "-c", f"model_reasoning_effort={self.reasoning_effort}"])
        if last_message_path is not None:
            command.extend(["--output-last-message", str(last_message_path)])
        command.extend(self.extra_args)
        command.append("-")

        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, **self.env},
            start_new_session=True,
        )
        assert process.stdin is not None
        process.stdin.write(instruction)
        process.stdin.close()

        output_parts: list[str] = []
        output_queue: queue.Queue[str] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        log_file = log_path.open("w", encoding="utf-8") if log_path is not None else None
        timeout_sec = max(1, int(request_timeout_sec))
        try:
            deadline = time.monotonic() + timeout_sec
            while True:
                self._drain_output(output_queue, output_parts, log_file)
                if process.poll() is not None:
                    reader.join(timeout=1)
                    self._drain_output(output_queue, output_parts, log_file)
                    break
                if time.monotonic() > deadline:
                    self._terminate_process_group(process)
                    raise subprocess.TimeoutExpired(
                        cmd=command,
                        timeout=timeout_sec,
                        output="".join(output_parts),
                    )
                time.sleep(0.1)
        finally:
            if log_file is not None:
                log_file.close()

        stdout = "".join(output_parts)
        trajectory = self._build_agent_trajectory(instruction, stdout, last_message_path)
        result = CodexCliRunResult(
            command=command,
            cwd=workspace,
            target_dir=workspace,
            prompt=instruction,
            returncode=int(process.returncode or 0),
            stdout=stdout,
            trajectory=trajectory,
        )
        if result.returncode != 0 and raise_on_error:
            raise RuntimeError(
                f"codex cli command failed ({result.returncode}): "
                f"{' '.join(shlex.quote(part) for part in command)}\n{stdout[-20000:]}"
            )
        return result

    def _workspace_path(self, workspace: Path, raw: str | None) -> Path | None:
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path if path.is_absolute() else workspace / path

    def _drain_output(
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
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
            if self.stream_output:
                sys.stderr.write(f"{self.output_prefix}{line}")
                sys.stderr.flush()

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

    def _build_agent_trajectory(
        self,
        instruction: str,
        stdout: str,
        last_message_path: Path | None,
    ) -> list[dict[str, Any]]:
        final_text = ""
        if last_message_path is not None and last_message_path.is_file():
            final_text = last_message_path.read_text(encoding="utf-8", errors="replace").strip()
        events: list[dict[str, Any]] = []
        for raw in stdout.splitlines():
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": final_text or None, "events": events},
        ]
