"""Held-in / held-out task splits and the per-step held-in schedule."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from harness_evolution.tasks.catalog import MIMO_CATALOG_KEYS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_split(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"split must be a JSON array of non-empty strings: {path}")
    result = [item.strip() for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"split contains duplicate task names: {path}")
    return result


def _read_catalog_task_ids(path: Path) -> list[str]:
    """Read task ids in catalog order, preserving the catalog's order."""
    ids: list[str] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in task catalog {path}:{line_no}: {exc}") from exc
        metadata = row.get("metadata") if isinstance(row, dict) else None
        task_id = str(metadata.get("task_id") or "").strip() if isinstance(metadata, dict) else ""
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        ids.append(task_id)
    if not ids:
        raise ValueError(f"task catalog has no metadata.task_id entries: {path}")
    return ids


def _build_held_in_schedule(
    *,
    held_in_path: Path | None,
    held_in_catalog: Path | None,
    held_in_size: int,
    held_in_seed: int,
    steps: int,
) -> list[list[str]]:
    """Build the per-step held-in task batches.

    With ``held_in_catalog`` each step receives a fresh, non-overlapping batch
    from the catalog.  The fixed ``held_in`` split remains supported for old
    runs; it is repeated per step because there is no larger source pool.
    """
    if held_in_size < 1:
        raise ValueError("held-in size must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if held_in_catalog is not None:
        ids = _read_catalog_task_ids(held_in_catalog)
        if steps * held_in_size > len(ids):
            raise ValueError(
                f"held-in catalog has {len(ids)} tasks, but {steps} steps x "
                f"{held_in_size} tasks requires {steps * held_in_size} unique tasks"
            )
        rng = random.Random(held_in_seed)
        rng.shuffle(ids)
        return [ids[offset : offset + held_in_size] for offset in range(0, steps * held_in_size, held_in_size)]
    if held_in_path is None:
        raise ValueError("provide either --held-in or --held-in-catalog")
    fixed = _read_split(held_in_path)
    if len(fixed) != held_in_size:
        raise ValueError(
            f"fixed held-in split has {len(fixed)} tasks; expected --held-in-size={held_in_size}"
        )
    return [list(fixed) for _ in range(steps)]


def _load_task_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Load TB2.1 task_id -> task directory/image metadata from JSONL."""
    catalog: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        metadata = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(metadata, dict):
            continue
        task_id = str(metadata.get("task_id") or "").strip()
        if not task_id:
            continue
        task_dir = str(metadata.get("task_dir") or "").strip()
        image = str(metadata.get("image_tag") or metadata.get("docker_image") or "").strip()
        entry: dict[str, Any] = {"task_dir": task_dir, "image_tag": image}
        for key in ("agent_timeout_sec", "verifier_timeout_sec", *MIMO_CATALOG_KEYS):
            value = metadata.get(key)
            if value not in (None, ""):
                entry[key] = value
        catalog[task_id] = entry
    return catalog
