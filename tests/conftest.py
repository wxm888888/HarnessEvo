"""Make the package importable from a plain checkout and expose sibling sources."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harness_evolution.core.paths import ensure_external_paths  # noqa: E402

ensure_external_paths()
