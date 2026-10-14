"""Every module in the package must import, apart from missing third-party dependencies.

A missing ``harness_evolution.*`` module or any other import-time error means a
broken internal reference, which is exactly what a file move can introduce.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import conftest  # noqa: F401

import harness_evolution


def _check_all() -> tuple[list[str], list[str]]:
    broken: list[str] = []
    skipped: list[str] = []
    for info in pkgutil.walk_packages(harness_evolution.__path__, "harness_evolution."):
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "harness_evolution" or missing.startswith("harness_evolution."):
                broken.append(f"{info.name}: {exc}")
            else:
                skipped.append(f"{info.name}: missing third-party module {missing!r}")
        except Exception as exc:  # noqa: BLE001 - report every import-time failure
            broken.append(f"{info.name}: {type(exc).__name__}: {exc}")
    return broken, skipped


def test_all_modules_import() -> None:
    broken, _ = _check_all()
    assert not broken, "\n".join(broken)


if __name__ == "__main__":
    broken, skipped = _check_all()
    for line in skipped:
        print("skip  ", line)
    for line in broken:
        print("BROKEN", line)
    print(f"broken={len(broken)} skipped={len(skipped)}")
    raise SystemExit(1 if broken else 0)
