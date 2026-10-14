"""Resolve task names to task directories and images from JSONL catalogs.

Two task formats are supported:

- Terminal-Bench: ``metadata.task_dir`` points at a directory with
  ``instruction.md`` and ``tests/test.sh``.
- MiMo: ``metadata.task_file`` points at one JSON task definition (problem
  statement, image, working directory and a hidden verifier).  See
  ``harness_evolution.evaluation.backends.mimo``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskSpec:
    name: str
    task_dir: Path
    image: str
    instruction: str
    tests_dir: Path
    agent_timeout_sec: int | None = None
    verifier_timeout_sec: int | None = None
    # Per-task sandbox working directory; None uses the backend default.
    workdir: str | None = None
    # Set for MiMo tasks: the JSON definition that carries the hidden verifier.
    mimo_task_file: Path | None = None
    # Per-task sandbox resources; None uses the backend default.
    cpu: str | None = None
    memory: str | None = None

    @property
    def is_mimo(self) -> bool:
        return self.mimo_task_file is not None


def _first_nonempty(*values: Any, default: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


def load_jsonl_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    """Polar prompt JSONL -> {task_id: {task_dir, image_tag}}."""
    catalog: dict[str, dict[str, str]] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        meta = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(meta, dict):
            continue
        task_id = str(meta.get("task_id") or "").strip()
        if not task_id:
            continue
        entry = catalog.setdefault(task_id, {})
        task_dir = str(meta.get("task_dir") or "").strip()
        image = str(meta.get("image_tag") or meta.get("docker_image") or "").strip()
        if task_dir and not entry.get("task_dir"):
            entry["task_dir"] = task_dir
        if image and not entry.get("image_tag"):
            entry["image_tag"] = image
        for key in ("agent_timeout_sec", "verifier_timeout_sec", *MIMO_CATALOG_KEYS):
            value = meta.get(key)
            if value not in (None, "") and key not in entry:
                entry[key] = value
    return catalog


# Catalog metadata that identifies a MiMo task (see ``_resolve_mimo_task_spec``).
MIMO_CATALOG_KEYS = ("task_file", "task_sha256", "mimo_dataset_type")
MIMO_DATASET_TYPES = ("opensource-code", "terminal_bench")
_MIMO_MIN_MEMORY_MB = 8192
# The screening run gave every MiMo task a 900 s agent budget; code tasks do
# not carry their own, so keep that budget to stay comparable with screening.
MIMO_DEFAULT_AGENT_TIMEOUT_SEC = 900


def _resolve_mimo_task_spec(name: str, info: dict[str, Any]) -> TaskSpec:
    task_file = Path(str(info["task_file"]))
    if not task_file.is_file():
        raise ValueError(f"MiMo task {name!r} is missing {task_file}")
    raw = task_file.read_bytes()
    expected = str(info.get("task_sha256") or "").strip()
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"MiMo task {name!r} changed on disk: {task_file} does not match task_sha256")
    task = json.loads(raw)
    family = str(task.get("dataset_type") or "")
    if family not in MIMO_DATASET_TYPES:
        raise ValueError(f"MiMo task {name!r} has unsupported dataset_type={family!r}")
    image = str(info.get("image_tag") or task.get("public_image") or "").strip()
    if not image:
        raise ValueError(f"MiMo task {name!r} has no image (catalog image_tag or task public_image)")
    workdir = str(task.get("cwd") or "").strip()
    if not workdir.startswith("/"):
        raise ValueError(f"MiMo task {name!r} has no absolute cwd")
    instruction = str(task.get("problem_statement") or "")
    if not instruction.strip():
        raise ValueError(f"MiMo task {name!r} has an empty problem_statement")

    def _seconds(key: str) -> int | None:
        value = task.get(key) or info.get(key)
        return int(float(value)) if value not in (None, "") and float(value) > 0 else None

    # Match the screening runner: at least 8 GiB and the task's own CPU count.
    memory_mb = max(_MIMO_MIN_MEMORY_MB, int(task.get("memory_mb") or 0))
    return TaskSpec(
        name=name,
        task_dir=task_file,
        image=image,
        instruction=instruction,
        tests_dir=task_file,
        agent_timeout_sec=_seconds("agent_timeout_sec") or MIMO_DEFAULT_AGENT_TIMEOUT_SEC,
        verifier_timeout_sec=_seconds("verifier_timeout_sec"),
        workdir=workdir,
        mimo_task_file=task_file,
        cpu=str(max(1, int(task.get("cpus") or 1))),
        memory=f"{memory_mb}Mi",
    )


def _image_from_task_dir(task_dir: Path) -> str:
    toml_path = task_dir / "task.toml"
    if toml_path.is_file():
        try:
            import tomllib
        except ImportError:  # python 3.10
            tomllib = None  # type: ignore[assignment]
        if tomllib is not None:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            for section in (data, data.get("environment") or {}, (data.get("environments") or {}).get("docker") or {}):
                if isinstance(section, dict) and section.get("docker_image"):
                    return str(section["docker_image"])
    for name in ("image_tag", "image.txt"):
        marker = task_dir / name
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
    meta_path = task_dir / "metadata.json"
    if meta_path.is_file():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return str(payload.get("image_tag") or payload.get("docker_image") or "").strip()
    return ""


def resolve_task_spec(
    name: str,
    *,
    tasks_dir: str | Path,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> TaskSpec:
    info = dict((catalog or {}).get(name) or {})
    if info.get("task_file") and not info.get("task_dir"):
        return _resolve_mimo_task_spec(name, info)
    tasks_root = Path(tasks_dir) if tasks_dir else None
    raw_dir = info.get("task_dir") or (str(tasks_root / name) if tasks_root else "")
    if not raw_dir:
        raise ValueError(f"held-out task {name!r} has no task_dir (pass task_catalog or tasks_dir)")
    task_dir = Path(raw_dir)
    image = str(info.get("image_tag") or info.get("image") or "").strip() or _image_from_task_dir(task_dir)
    if not image:
        raise ValueError(
            f"held-out task {name!r} has no image_tag; set dataset_image_tag on the evidence "
            "manifest, heldout_gate.tasks_index / PROMPT_DATA, or a task.toml docker_image"
        )
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.is_file():
        raise ValueError(f"held-out task {name!r} is missing {instruction_path}")
    tests_dir = task_dir / "tests"
    if not (tests_dir / "test.sh").is_file():
        raise ValueError(f"held-out task {name!r} is missing {tests_dir / 'test.sh'}")
    def _timeout(key: str) -> int | None:
        value = info.get(key)
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"task {name!r} has invalid {key}={value!r}") from exc
        if parsed < 1:
            raise ValueError(f"task {name!r} has invalid {key}={parsed}")
        return parsed

    return TaskSpec(
        name=name,
        task_dir=task_dir,
        image=image,
        instruction=instruction_path.read_text(encoding="utf-8"),
        tests_dir=tests_dir,
        agent_timeout_sec=_timeout("agent_timeout_sec"),
        verifier_timeout_sec=_timeout("verifier_timeout_sec"),
    )


def _concrete(value: Any) -> str:
    text = str(value or "").strip()
    if not text or (text.startswith("__") and text.endswith("__")):
        return ""
    return text


def _tasks_index_path(heldout: dict[str, Any]) -> str:
    return _first_nonempty(
        _concrete(heldout.get("tasks_index")),
        os.environ.get("HARNESS_GATE_TASKS_INDEX"),
        os.environ.get("PROMPT_DATA"),
    )


def resolve_task_specs(
    heldout: dict[str, Any],
    task_names: list[str],
    task_catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, TaskSpec]:
    catalog = dict(task_catalog or {})
    index_path = _tasks_index_path(heldout)
    if index_path and Path(index_path).is_file():
        for key, value in load_jsonl_catalog(index_path).items():
            catalog.setdefault(key, {}).update({k: v for k, v in value.items() if v})
    tasks_dir = _first_nonempty(
        _concrete(heldout.get("tasks_dir")),
        os.environ.get("HARNESS_GATE_TASKS_DIR"),
    )
    specs: dict[str, TaskSpec] = {}
    for name in dict.fromkeys(task_names):
        specs[name] = resolve_task_spec(name, tasks_dir=tasks_dir, catalog=catalog)
    return specs
