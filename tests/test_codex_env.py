import os
import socket
import subprocess

import pytest

from harness_evolution.proposer import batch


@pytest.mark.parametrize("stdin_kind", ["pipe", "devnull", "socket"])
@pytest.mark.parametrize("shlvl", [None, "0", "1", "2"])
def test_codex_bash_env_keeps_inherited_path_entries(tmp_path, monkeypatch, stdin_kind, shlvl):
    """BASH_ENV must merge into PATH: Bun's node_modules/.bin has to survive it."""
    bun_dir = tmp_path / "bun" / "bin"
    bun_dir.mkdir(parents=True)
    tool_dir = tmp_path / "node_modules" / ".bin"
    tool_dir.mkdir(parents=True)
    tool = tool_dir / "faketool"
    tool.write_text("#!/bin/sh\necho FAKE-TOOL-FOUND\n")
    tool.chmod(0o755)

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(
        batch,
        "_command_env",
        lambda config: {"BUN_BIN": str(bun_dir / "bun"), "PATH": f"{bun_dir}:/usr/bin:/bin"},
    )
    env = batch._codex_client_env(None, {})
    assert env["BASH_ENV"]

    run_env = {"BASH_ENV": env["BASH_ENV"], "PATH": f"{tool_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    if shlvl is not None:
        run_env["SHLVL"] = shlvl
    if stdin_kind == "socket":
        left, stdin = socket.socketpair()
    elif stdin_kind == "pipe":
        left, stdin = os.pipe()[0], None
        stdin = left
    else:
        stdin = subprocess.DEVNULL
    result = subprocess.run(
        ["/bin/bash", "-c", "faketool; echo PATH=$PATH"],
        stdin=stdin,
        env=run_env,
        capture_output=True,
        text=True,
    )
    assert "FAKE-TOOL-FOUND" in result.stdout, result.stderr
    path = result.stdout.split("PATH=", 1)[1].strip().split(":")
    assert str(tool_dir) in path
    # Bash skips BASH_ENV for remote-looking shells (socket stdin, SHLVL < 1); otherwise the
    # prepared directories keep priority and inherited entries follow without duplicates.
    if stdin_kind != "socket" or shlvl not in (None, "0"):
        assert path[0] == str(bun_dir)
    assert len(path) == len(set(path))
