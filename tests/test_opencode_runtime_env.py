"""Agent tools must use the same tsgo-compatible runtime as static checks."""
import os
import subprocess
import sys
from pathlib import Path

from harness_evolution.agents.opencode_bun_dev import OpenCodeBunDevClient
from harness_evolution.harnesses.opencode.build import bun_command_env


def _executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)
    return path


def _env(tmp_path, node_works):
    bun = _executable(tmp_path / "bun-bin" / "bun", "#!/bin/sh\necho BUN-RUNTIME\n")
    node = _executable(tmp_path / "node-bin" / "node", "#!/bin/sh\nexit " + ("0" if node_works else "1") + "\n")
    return {"BUN_BIN": str(bun), "PATH": str(node.parent) + ":/usr/bin:/bin", "TMPDIR": str(tmp_path)}


def test_old_node_uses_bun_without_losing_inherited_tool_paths(tmp_path):
    env = _env(tmp_path, False)
    prepared = bun_command_env(env)
    result = subprocess.run(["/bin/sh", "-c", "node --version"], env=prepared, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "BUN-RUNTIME"
    assert env["PATH"] in prepared["PATH"]
    assert "harnessrl-bun-node-shim" not in env["PATH"]


def test_compatible_node_is_preserved(tmp_path):
    env = _env(tmp_path, True)
    prepared = bun_command_env(env)
    assert "harnessrl-bun-node-shim" not in prepared["PATH"]
    assert prepared["PATH"].split(":")[0] == str(Path(env["BUN_BIN"]).parent)


def test_opencode_child_shell_inherits_tsgo_runtime_fix(tmp_path, monkeypatch):
    env = _env(tmp_path, False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    probe = tmp_path / "probe.py"
    probe.write_text("import subprocess,sys\nr=subprocess.run(['/bin/sh','-c','node --version'],capture_output=True,text=True)\nprint(r.stdout,end='')\nsys.exit(r.returncode)\n")
    client = OpenCodeBunDevClient(str(tmp_path), dev_run_command_prefix=f"{sys.executable} {probe}", dev_run_env={"OPENCODE_DISABLE_MODELS_FETCH": "1"})
    result = client.run_instruction_detailed(workspace=tmp_path, instruction="runtime smoke", request_timeout_sec=10)
    assert result.returncode == 0
    assert "BUN-RUNTIME" in result.stdout
