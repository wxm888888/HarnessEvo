"""Gate configuration, logging and git helpers shared by the static and held-out gates."""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
from harness_evolution.core.paths import ensure_external_paths

ensure_external_paths()

from harness_dist import state as version_state


def load_gate_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"gate config root must be a mapping: {path}")
    for section in ("paths", "static_gate", "build", "heldout_gate"):
        if section not in value:
            raise ValueError(f"gate config missing required section '{section}'")
    return value


def _paths(config: dict[str, Any]) -> dict[str, Any]:
    return config["paths"]


def _report_root(config: dict[str, Any]) -> Path:
    root = Path(_paths(config)["report_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_baseline(config: dict[str, Any]) -> dict[str, str]:
    """The gate baseline is whatever harness the training currently runs:
    the active version from the state file when configured (this is what
    makes chained evolution v0 -> v1 -> v2 compare against the right
    predecessor), falling back to the static config baseline otherwise."""
    state_path = _paths(config).get("version_state")
    if state_path and Path(state_path).exists():
        return version_state.read_active(state_path)
    paths = _paths(config)
    commit = str(paths["baseline_commit"])
    binary = str(paths.get("baseline_binary") or "")
    return {"commit": commit, "commit_short": commit[:7], "binary_path": binary}


def _log(msg: str) -> None:
    print(f"[gate {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
