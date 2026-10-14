"""MiMo task setup and verification inside an OpenSandbox trial.

A MiMo task is one JSON definition (``TaskSpec.mimo_task_file``) with its own
image, working directory and a hidden verifier.  The trial runs OpenCode the
same way as for Terminal-Bench tasks; this module supplies the steps around it:

- ``prepare`` (before the agent): check the working directory and, for
  ``opensource-code`` tasks, record a git baseline and remove history after
  it so the agent cannot read the reference fix.
- ``capture_model_diff`` (after the agent): save the agent's change for the
  evidence pipeline.
- ``verify``: ``opensource-code`` restores the files touched by the hidden
  test patch, applies it and runs ``test_command`` (exit 0 = pass);
  ``terminal_bench`` uploads the embedded ``tests/`` and runs ``test.sh``.

The procedure follows the MiMo screening runner
(``duchenyu/reports/mimo-smoke-20260926/run_smoke.py``) so held-in rewards are
comparable with the screening pass rates.  Anything that prevents a trustworthy
verifier result raises ``MimoInfrastructureError`` and the trial is recorded as
an infrastructure error, never as an agent failure.
"""
from __future__ import annotations

import base64
import json
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harness_evolution.core.paths import ensure_external_paths

ensure_external_paths()

from terminal_bench_opensandbox.verification import infrastructure_error, verifier_command

# OpenCode keeps its config under $HOME; keep it out of the task repository.
MIMO_HOME = "/home/user"
# The agent runs with HOME=MIMO_HOME, but setup and verification run as the
# image's root user, as in the screening runner: /root/.bashrc is what puts
# pyenv and similar toolchains on PATH for the tasks' test scripts.
ROOT_ENV = {"HOME": "/root"}
_CODE_RECEIPT = re.compile(r"__MIMO_VERIFIER_RC__=([0-9]+)\s*\Z")
_TERMINAL_RECEIPT = re.compile(r"__POLAR_VERIFIER_REWARD__=([01])\s*__MIMO_VERIFIER_RC__=[0-9]+\s*\Z")


class MimoInfrastructureError(RuntimeError):
    """The sandbox or verifier did not produce a trustworthy result."""


@dataclass
class MimoVerification:
    passed: bool
    return_code: int
    stdout: str
    stderr: str


def load_task(path: str | Path) -> dict[str, Any]:
    task = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise ValueError(f"MiMo task file is not a JSON object: {path}")
    return task


def _safe_relative(raw: str) -> str:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] == ".git":
        raise ValueError(f"unsafe path in MiMo task: {raw!r}")
    return raw


def patch_paths(task: dict[str, Any]) -> list[str]:
    """Files touched by the hidden test patch.

    The screening inputs store the list git itself parsed
    (``verified_patch_paths``); fall back to the ``diff --git`` headers.
    """
    recorded = task.get("verified_patch_paths")
    if isinstance(recorded, list) and recorded:
        return [_safe_relative(str(item)) for item in recorded]
    paths: list[str] = []
    for line in str(task.get("test_patch") or "").splitlines():
        if not line.startswith("diff --git "):
            continue
        fields = shlex.split(line)
        if len(fields) != 4:
            raise ValueError("unsupported diff header in MiMo test patch")
        for raw in fields[2:]:
            if not raw.startswith(("a/", "b/")):
                raise ValueError("invalid path in MiMo test patch")
            rel = _safe_relative(raw[2:])
            if rel not in paths:
                paths.append(rel)
    if not paths:
        raise ValueError("MiMo test patch contains no file headers")
    return paths


def parse_code_receipt(return_code: int | None, stdout: str) -> int:
    match = _CODE_RECEIPT.search(stdout or "")
    if return_code != 0 or match is None:
        raise MimoInfrastructureError(f"verifier has no valid completion receipt (rc={return_code})")
    rc = int(match.group(1))
    if rc in {124, 125, 126, 127} or rc >= 128:
        raise MimoInfrastructureError(f"verifier infrastructure exit {rc}")
    return rc


def parse_terminal_receipt(return_code: int | None, stdout: str, stderr: str) -> tuple[int, bool]:
    rc = parse_code_receipt(return_code, stdout)
    normalized = (stdout or "").replace("__POLAR_VERIFIER_REWARD__=", "\n__POLAR_VERIFIER_REWARD__=")
    reason = infrastructure_error(normalized + "\n" + (stderr or ""))
    if reason:
        raise MimoInfrastructureError(reason)
    match = _TERMINAL_RECEIPT.search(stdout or "")
    if match is None or rc not in {0, 1}:
        raise MimoInfrastructureError(f"invalid terminal reward receipt (verifier rc={rc})")
    passed = match.group(1) == "1"
    if passed and (rc != 0 or re.search(r"\b\d+\s+(?:failed|errors?)\b", normalized, re.I)):
        raise MimoInfrastructureError("terminal reward=1 contradicts verifier failure")
    return rc, passed


async def _run(
    runtime: Any,
    command: str,
    *,
    name: str,
    log_dir: Path,
    cwd: str = "/",
    timeout: int = 60,
    check: bool = True,
) -> Any:
    result = await runtime.exec(command, cwd=cwd, env=dict(ROOT_ENV), timeout_sec=timeout)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}.stdout").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / f"{name}.stderr").write_text(result.stderr or "", encoding="utf-8")
    if check and result.return_code != 0:
        tail = (result.stderr or result.stdout or "")[-1200:]
        raise MimoInfrastructureError(f"MiMo {name} failed: rc={result.return_code}: {tail}")
    return result


async def _upload_bytes(runtime: Any, data: bytes, remote_path: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(data)
        local = Path(handle.name)
    try:
        await runtime.upload_file(str(local), remote_path)
    finally:
        local.unlink(missing_ok=True)


def _log_dir(session_dir: Path) -> Path:
    return session_dir / "mimo"


async def prepare(runtime: Any, task: dict[str, Any], *, session_dir: Path) -> dict[str, Any]:
    """Run before the agent; returns the state ``verify`` needs."""
    logs = _log_dir(session_dir)
    cwd = shlex.quote(str(task["cwd"]))
    state: dict[str, Any] = {"dataset_type": task["dataset_type"]}
    await _run(
        runtime,
        f"test -d {cwd} && mkdir -p {MIMO_HOME} /logs/agent /logs/verifier && command -v bash",
        name="preflight",
        log_dir=logs,
    )
    hidden = await _run(
        runtime,
        f"cd {cwd} && test ! -e mimo_test_command.sh && test ! -e /tests/test.sh && echo HIDDEN_VERIFIER_ABSENT",
        name="hidden_before",
        log_dir=logs,
        check=False,
    )
    state["hidden_verifier_absent"] = "HIDDEN_VERIFIER_ABSENT" in (hidden.stdout or "")
    if task["dataset_type"] != "opensource-code":
        return state

    await _run(
        runtime,
        f"git config --global --add safe.directory {cwd} && cd {cwd} && "
        "(git rev-parse --git-dir >/dev/null 2>&1 || (git init -q && git add -A && "
        "git -c user.name=harness -c user.email=harness@localhost commit -q -m baseline --allow-empty))",
        name="git_setup",
        log_dir=logs,
    )
    head = await _run(runtime, f"cd {cwd} && git rev-parse HEAD", name="base_ref", log_dir=logs)
    base = (head.stdout or "").strip().split()[-1] if (head.stdout or "").strip() else ""
    if not re.fullmatch("[a-f0-9]{40}", base):
        raise MimoInfrastructureError("invalid MiMo base ref")
    state["base_ref"] = base
    extra = await _run(runtime, f"cd {cwd} && git rev-list --all --not {base}", name="history", log_dir=logs)
    later = (extra.stdout or "").split()
    if later:
        # Commits after the baseline can contain the reference fix.
        state["future_commits_removed"] = len(later)
        strip = "\n".join(
            [
                "set -euo pipefail",
                f"cd {cwd}",
                f"git checkout --detach {base}",
                'git for-each-ref --format="%(refname)" refs/heads refs/replace refs/notes refs/stash '
                'refs/remotes refs/pull | while read -r ref; do git update-ref -d "$ref"; done',
                'git remote | while read -r remote; do git remote remove "$remote"; done',
                'git for-each-ref --format="%(refname)" refs/tags | while read -r tag; do '
                f'if ! git merge-base --is-ancestor "$tag" {base}; then git update-ref -d "$tag"; fi; done',
                "git reflog expire --expire=now --all",
                "git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --prune=now",
                f'test -z "$(git rev-list --all --not {base})"',
            ]
        )
        await _run(runtime, "bash -c " + shlex.quote(strip), name="strip_history", log_dir=logs, timeout=600)
        probe = " && ".join(f"! git cat-file -e {shlex.quote(sha)} 2>/dev/null" for sha in later[:10])
        await _run(runtime, f"cd {cwd} && {probe}", name="removed_object_check", log_dir=logs)
    return state


async def capture_model_diff(runtime: Any, task: dict[str, Any], state: dict[str, Any], *, session_dir: Path) -> None:
    """Save the agent's change as ``agent/model.diff``; never fails the trial."""
    base = state.get("base_ref")
    if not base:
        return
    cwd = shlex.quote(str(task["cwd"]))
    try:
        result = await runtime.exec(
            f"cd {cwd} && git diff --binary {base}; echo '# untracked files:'; "
            "git ls-files --others --exclude-standard | sed 's/^/#   /'",
            cwd="/",
            env=dict(ROOT_ENV),
            timeout_sec=120,
        )
        agent_dir = session_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "model.diff").write_text(result.stdout or "", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        (_log_dir(session_dir)).mkdir(parents=True, exist_ok=True)
        (_log_dir(session_dir) / "model_diff.error").write_text(str(exc), encoding="utf-8")


async def verify(runtime: Any, task: dict[str, Any], state: dict[str, Any], *, session_dir: Path) -> MimoVerification:
    logs = _log_dir(session_dir)
    cwd_raw = str(task["cwd"])
    cwd = shlex.quote(cwd_raw)
    timeout = int(float(task.get("verifier_timeout_sec") or 1800))
    if task["dataset_type"] == "opensource-code":
        base = state.get("base_ref")
        if not base:
            raise MimoInfrastructureError("MiMo code verification needs a base ref")
        commands = [f"cd {cwd}"]
        for path in patch_paths(task):
            quoted = shlex.quote(path)
            obj = shlex.quote(f"{base}:{path}")
            commands.append(
                f"if git cat-file -e {obj} 2>/dev/null; then "
                f"git restore --source={base} --staged --worktree -- {quoted}; else rm -f -- {quoted}; fi"
            )
        await _run(runtime, " && ".join(commands), name="restore_test_paths", log_dir=logs)
        await _upload_bytes(runtime, str(task["test_patch"]).encode("utf-8"), "/tmp/mimo-hidden.patch")
        await _run(
            runtime,
            f"cd {cwd} && git apply --check /tmp/mimo-hidden.patch && git apply --verbose /tmp/mimo-hidden.patch "
            "&& rm -f /tmp/mimo-hidden.patch",
            name="apply_test_patch",
            log_dir=logs,
        )
        script = "\n".join(
            ["#!/bin/bash", "set +e", f"cd {cwd} || exit 125", str(task["test_command"]),
             "rc=$?", "echo __MIMO_VERIFIER_RC__=$rc", "exit 0", ""]
        )
        await _upload_bytes(runtime, script.encode("utf-8"), "/tmp/mimo-verify.sh")
        result = await _run(runtime, "bash /tmp/mimo-verify.sh", name="verifier", log_dir=logs,
                            timeout=timeout + 10, check=False)
        rc = parse_code_receipt(result.return_code, result.stdout or "")
        return MimoVerification(passed=rc == 0, return_code=rc, stdout=result.stdout or "", stderr=result.stderr or "")

    if task["dataset_type"] == "terminal_bench":
        files = json.loads(task["tests_files"]) if isinstance(task["tests_files"], str) else task["tests_files"]
        parents = sorted({str(PurePosixPath("/tests", _safe_relative(rel)).parent) for rel in files})
        await _run(runtime, "mkdir -p " + " ".join(shlex.quote(p) for p in parents), name="tests_dirs", log_dir=logs)
        for rel, payload in files.items():
            await _upload_bytes(runtime, base64.b64decode(payload, validate=True), f"/tests/{_safe_relative(rel)}")
        script = "\n".join(
            ["#!/bin/bash", "set +e", f"cd {cwd} || exit 125",
             verifier_command("TEST_DIR=/tests bash /tests/test.sh"),
             "rc=$?", "echo __MIMO_VERIFIER_RC__=$rc", "exit 0", ""]
        )
        await _upload_bytes(runtime, script.encode("utf-8"), "/tmp/mimo-verify.sh")
        result = await _run(runtime, "bash /tmp/mimo-verify.sh", name="verifier", log_dir=logs,
                            timeout=timeout + 30, check=False)
        rc, passed = parse_terminal_receipt(result.return_code, result.stdout or "", result.stderr or "")
        return MimoVerification(passed=passed, return_code=rc, stdout=result.stdout or "", stderr=result.stderr or "")

    raise ValueError(f"unsupported MiMo dataset_type {task['dataset_type']!r}")
