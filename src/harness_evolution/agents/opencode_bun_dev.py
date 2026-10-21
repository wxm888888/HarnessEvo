from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harness_evolution.agents.base import BaseClient
from harness_evolution.agents.registry import CLIENT_REGISTRY
from harness_evolution.harnesses.opencode.build import bun_command_env


@dataclass
class OpenCodeDevRunResult:
    command: list[str]
    cwd: Path
    target_dir: Path
    prompt: str
    returncode: int
    stdout: str
    trajectory: list[dict[str, Any]]


@CLIENT_REGISTRY.register("opencode_bun_dev_client")
class OpenCodeBunDevClient(BaseClient):
    """Client wrapper for OpenCode dev-cli mode."""

    def __init__(
        self,
        opencode_repo_path: str,
        dev_run_command_prefix: str = "bun dev -- run",
        dev_run_add_dir_arg: bool = True,
        dev_run_env: dict[str, str] | None = None,
    ) -> None:
        repo_path = Path(opencode_repo_path).expanduser().resolve()
        if not repo_path.exists():
            raise ValueError(f"opencode_repo_path does not exist: {repo_path}")
        if not repo_path.is_dir():
            raise ValueError(f"opencode_repo_path must be a directory: {repo_path}")
        if not dev_run_command_prefix.strip():
            raise ValueError("dev_run_command_prefix must not be empty")

        self.opencode_repo_path = repo_path
        self.dev_run_command_prefix = dev_run_command_prefix
        self.dev_run_add_dir_arg = bool(dev_run_add_dir_arg)
        self.dev_run_env = {str(key): str(value) for key, value in (dev_run_env or {}).items()}

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
        raise NotImplementedError("opencode_bun_dev_client is agent-style; use run_instruction instead")

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
        timeout_sec = max(1, int(request_timeout_sec))
        result = self.run_instruction_detailed(
            workspace=workspace,
            instruction=instruction,
            request_timeout_sec=timeout_sec,
            raise_on_error=True,
        )
        return result.trajectory

    def run_instruction_detailed(
        self,
        *,
        workspace: Path,
        instruction: str,
        request_timeout_sec: int,
        raise_on_error: bool = True,
    ) -> OpenCodeDevRunResult:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if not workspace.exists():
            raise ValueError(f"workspace does not exist: {workspace}")
        timeout_sec = max(1, int(request_timeout_sec))
        return self._run_opencode_dev_cli_command(
            workspace=workspace,
            prompt=instruction,
            timeout_sec=timeout_sec,
            raise_on_error=raise_on_error,
        )

    def _run_opencode_dev_cli_command(
        self,
        *,
        workspace: Path,
        prompt: str,
        timeout_sec: int,
        raise_on_error: bool,
    ) -> OpenCodeDevRunResult:
        command = shlex.split(self.dev_run_command_prefix)
        # Agent shell tools need the controller's Node/Bun compatibility too.
        run_env = bun_command_env({**os.environ, **self.dev_run_env})
        if command and Path(command[0]).name == "bun":
            self._ensure_bun_install(self.opencode_repo_path)
        if not self._has_cli_option(command, "--format"):
            command.extend(["--format", "json"])
        if not self._has_cli_option(command, "--thinking"):
            command.append("--thinking")
        if self.dev_run_add_dir_arg:
            command.extend(["--dir", str(workspace)])
        command.append(prompt)
        process = subprocess.Popen(
            command,
            cwd=str(self.opencode_repo_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=run_env,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_group(process)
            output = self._coerce_output(exc.output)
            trajectory = self._build_agent_trajectory(prompt=prompt, output=output)
            raise subprocess.TimeoutExpired(
                cmd=command,
                timeout=timeout_sec,
                output=(
                    f"opencode dev-cli run timed out after {timeout_sec}s; "
                    "terminated its process group.\n"
                    f"{output}"
                ),
            ) from exc

        result_returncode = process.returncode
        result_stdout = output or ""
        trajectory = self._build_agent_trajectory(prompt=prompt, output=result_stdout)
        result = OpenCodeDevRunResult(
            command=command,
            cwd=self.opencode_repo_path,
            target_dir=workspace,
            prompt=prompt,
            returncode=result_returncode,
            stdout=result_stdout,
            trajectory=trajectory,
        )

        if result_returncode != 0 and raise_on_error:
            raise RuntimeError(
                f"opencode dev-cli run command failed: {' '.join(command)}\n"
                f"trajectory:\n{json.dumps(trajectory, ensure_ascii=False, indent=2)}\n\nraw output:\n{result_stdout}"
            )
        return result

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

    def _ensure_bun_install(self, repo_path: Path | None = None) -> None:
        repo_path = (repo_path or self.opencode_repo_path).expanduser().resolve()
        marker_path = repo_path / "node_modules" / ".opencode_bun_install.ok"
        if marker_path.exists():
            return

        attempt = 0
        while True:
            attempt += 1
            print(f"running bun install for {repo_path} (attempt {attempt})", file=sys.stderr)
            try:
                run_env = {**os.environ, **self.dev_run_env, "HUSKY": "0"}
                bun_bin = str(run_env.get("BUN_BIN") or "bun")
                if Path(bun_bin).is_absolute():
                    bun_dir = str(Path(bun_bin).expanduser().resolve().parent)
                    run_env["PATH"] = os.pathsep.join(
                        item for item in (bun_dir, run_env.get("PATH", "")) if item
                    )
                process = subprocess.Popen(
                    [bun_bin, "install"],
                    cwd=str(repo_path),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    env=run_env,
                )
            except OSError as exc:
                print(f"bun install could not start: {exc}; retrying in 10s", file=sys.stderr)
                time.sleep(10)
                continue

            output_lines: list[str] = []
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="", file=sys.stderr)
                    output_lines.append(line)

            returncode = process.wait()
            output = "".join(output_lines)

            if returncode == 0:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text(
                    json.dumps(
                        {
                            "opencode_repo_path": str(repo_path),
                            "installed_at": datetime.now(timezone.utc).isoformat(),
                            "attempts": attempt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return

            print(
                f"bun install failed with exit code {returncode}; retrying in 10s\n{output}",
                file=sys.stderr,
            )
            time.sleep(10)

    def _has_cli_option(self, command: list[str], option: str) -> bool:
        prefix = f"{option}="
        return any(token == option or token.startswith(prefix) for token in command)

    def _build_agent_trajectory(self, *, prompt: str, output: str) -> list[dict[str, Any]]:
        events = self._parse_json_events(output)
        trajectory: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        pending_assistant: dict[str, Any] = self._empty_assistant_message()

        for event in events:
            event_type = event.get("type")
            part = event.get("part") if isinstance(event.get("part"), dict) else {}

            if event_type == "step_start":
                pending_assistant = self._flush_assistant_message(trajectory, pending_assistant)
                continue

            if event_type == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    pending_assistant["content"] = f"{pending_assistant['content'] or ''}\n{text}".strip()
                continue

            if event_type == "tool_use":
                tool_call, tool_message = self._convert_tool_event(part)
                if tool_call:
                    pending_assistant.setdefault("tool_calls", []).append(tool_call)
                if tool_message:
                    pending_assistant = self._flush_assistant_message(trajectory, pending_assistant)
                    trajectory.append(tool_message)
                continue

            if event_type == "step_finish":
                pending_assistant = self._flush_assistant_message(trajectory, pending_assistant)
                continue

        self._flush_assistant_message(trajectory, pending_assistant)
        return trajectory

    def _parse_json_events(self, output: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in output.splitlines():
            raw = line.strip()
            if not raw.startswith("{"):
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("type"), str):
                events.append(value)
        return events

    def _empty_assistant_message(self) -> dict[str, Any]:
        return {"role": "assistant", "content": None, "tool_calls": []}

    def _flush_assistant_message(
        self,
        trajectory: list[dict[str, Any]],
        message: dict[str, Any],
    ) -> dict[str, Any]:
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []
        if content is not None or tool_calls:
            normalized = {"role": "assistant", "content": content}
            if tool_calls:
                normalized["tool_calls"] = tool_calls
            trajectory.append(normalized)
        return self._empty_assistant_message()

    def _convert_tool_event(self, part: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        call_id = part.get("callID")
        tool_name = part.get("tool")
        tool_call = None
        if isinstance(call_id, str) and isinstance(tool_name, str):
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(state.get("input", {}), ensure_ascii=False),
                },
            }

        tool_message = None
        if state.get("status") == "completed" or "output" in state:
            tool_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": state.get("output", ""),
            }
        elif state.get("status") == "error" or "error" in state:
            tool_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": state.get("error", ""),
                "error": True,
            }

        return tool_call, tool_message
