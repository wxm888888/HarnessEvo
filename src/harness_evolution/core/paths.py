"""Filesystem roots and external source trees used by the package.

``REPO_ROOT`` is this checkout; relative paths in loop configs resolve
against it.  ``harnessrl_root()`` is the workspace that holds the sibling
projects this package still imports from source (Polar, harness_dist,
reward-design, the TB2.1 runtime).  Every location can be overridden with an
environment variable, so nothing below is tied to one cluster layout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def harnessrl_root() -> Path:
    """Workspace holding polar/, harness_dist/ and reward-design/ ($HARNESSRL_ROOT)."""
    raw = os.environ.get("HARNESSRL_ROOT")
    return Path(raw).expanduser().resolve() if raw else REPO_ROOT.parent


def default_output_root() -> Path:
    """Root for relative run directories ($HARNESSRL_OUTPUT_ROOT, else <repo>/runs)."""
    raw = os.environ.get("HARNESSRL_OUTPUT_ROOT")
    return Path(raw).expanduser() if raw else REPO_ROOT / "runs"


def reward_design_root() -> Path:
    raw = os.environ.get("HARNESSRL_REWARD_DESIGN_ROOT")
    return Path(raw).expanduser() if raw else harnessrl_root() / "reward-design"


def tb21_runtime_root() -> Path:
    """Directory providing the ``agent_artifacts`` TB2.1 trace converter."""
    raw = os.environ.get("HARNESSRL_TB21_RUNTIME")
    return Path(raw).expanduser() if raw else harnessrl_root().parent / "tb21-eval-qwen36-fixed" / "runtime"


def external_source_paths() -> list[Path]:
    root = harnessrl_root()
    return [
        root / "polar" / "src",
        root / "polar" / "examples",
        root / "harness_dist" / "src",
        tb21_runtime_root(),
    ]


def ensure_external_paths() -> None:
    """Put the sibling source trees on ``sys.path`` when they exist.

    These projects are not pip-installable yet.  Modules that import them call
    this once before the import instead of each carrying its own path logic.
    """
    for path in reversed(external_source_paths()):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
