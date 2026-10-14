"""Recover dataset_task_id/dataset_task_dir/dataset_image_tag for evolution
evidence sessions purely from on-disk instruction text -- no polar/slime
source changes required.

Background: the trajectory dump's dataset_task_id/dataset_task_dir/
dataset_image_tag fields (see harness_evolution_pipeline.py's
_evidence_task_names) are always null in practice -- the original dataset-row
metadata never survives the Sample-metadata reconstruction that happens
inside polar/src/slime_bridge/{adapter,rollout}.py on every rollout, so it
never reaches disk in the first place. Fixing that upstream would touch live
training code.

Each harbor task's instruction is fixed, unique text reused verbatim in the
rollout prompt (verified: 0 hash collisions across the full ~14.6k task
pool), so we can recover the mapping entirely offline by matching each
evidence session's already-converted instruction.md (written by
convert_polar_rollout_to_tb_job.py) against every task's instruction.md
under `tasks_dir`, by exact content hash. The trajectory conversion wraps
the captured prompt text in one outer pair of quotes and backslash-escapes
any quotes/backslashes that were already inside it, so normalization undoes
that before hashing.

Usage:
  python3 resolve_evidence_task_ids.py \
      --tasks-dir /path/to/harbor/train \
      --job-dir /path/to/tb_job \
      [--cache-file /path/to/task_instruction_index_cache.json]

Mutates conversion_manifest.json (and each matched session's result.json)
in place under --job-dir, filling in the three fields wherever a match is
found. Exits 0 even when some/all sessions can't be matched -- callers
(harness_evolution_pipeline.py) treat unresolved sessions the same as before
this script existed: _evidence_task_names() skips null dataset_task_id and
the held-out gate falls back to its fixed canary set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path


def _normalize_instruction(text: str) -> str:
    """Undo the outer-quote wrapping + escaping that trace-message capture
    leaves on the instruction text, so it matches the task's raw instruction.md
    byte-for-byte."""
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = text.replace('\\"', '"').replace("\\\\", "\\")
    return text.strip()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_AMBIGUOUS = object()  # sentinel: hash collided across >1 distinct task, never match it


def build_task_index(tasks_dir: Path) -> dict[str, str]:
    """hash(normalized instruction) -> task_id.

    A collision (two different tasks with byte-identical instructions) is
    logged and that hash is made permanently unmatchable rather than raising
    -- one ambiguous pair should not take down the whole evolution step.
    """
    index: dict[str, object] = {}
    n_tasks = 0
    for task_path in sorted(tasks_dir.iterdir()):
        if not task_path.is_dir():
            continue
        instr_path = task_path / "instruction.md"
        if not instr_path.exists():
            continue
        n_tasks += 1
        key = _hash(_normalize_instruction(instr_path.read_text(encoding="utf-8", errors="replace")))
        existing = index.get(key)
        if existing is _AMBIGUOUS:
            continue
        if existing is not None and existing != task_path.name:
            print(
                f"[resolve-task-ids] WARNING: instruction hash collision between "
                f"{existing!r} and {task_path.name!r}; this instruction will never match",
                file=sys.stderr,
            )
            index[key] = _AMBIGUOUS
            continue
        index[key] = task_path.name
    if n_tasks == 0:
        raise ValueError(f"no task instruction.md files found under {tasks_dir}")
    return {key: value for key, value in index.items() if value is not _AMBIGUOUS}


def load_or_build_task_index(tasks_dir: Path, cache_file: Path | None) -> dict[str, str]:
    if cache_file is not None and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get("tasks_dir") == str(tasks_dir) and cached.get("index"):
            return cached["index"]
    index = build_task_index(tasks_dir)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"tasks_dir": str(tasks_dir), "index": index}, indent=2),
            encoding="utf-8",
        )
    return index


def _task_image(tasks_dir: Path, task_id: str) -> str | None:
    task_toml = tasks_dir / task_id / "task.toml"
    if not task_toml.exists():
        return None
    data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    for section in (data, data.get("environment", {}), data.get("environments", {}).get("docker", {})):
        if isinstance(section, dict) and section.get("docker_image"):
            return str(section["docker_image"])
    return None


def resolve_manifest(job_dir: Path, tasks_dir: Path, index: dict[str, str]) -> dict[str, int]:
    manifest_path = job_dir / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = {"matched": 0, "unmatched": 0, "already_set": 0}
    for record in manifest.get("records", []):
        if record.get("dataset_task_id"):
            stats["already_set"] += 1
            continue
        session_id = record["session_id"]
        instr_path = job_dir / session_id / "instruction.md"
        if not instr_path.exists():
            stats["unmatched"] += 1
            continue
        key = _hash(_normalize_instruction(instr_path.read_text(encoding="utf-8", errors="replace")))
        task_id = index.get(key)
        if task_id is None:
            stats["unmatched"] += 1
            continue

        task_dir = str(tasks_dir / task_id)
        image_tag = _task_image(tasks_dir, task_id)
        record["dataset_task_id"] = task_id
        record["dataset_task_dir"] = task_dir
        record["dataset_image_tag"] = image_tag
        stats["matched"] += 1

        # Keep the per-session result.json's polar_metadata consistent too --
        # some readers (evolver evidence, future gate tooling) may look there
        # instead of the manifest.
        result_path = job_dir / session_id / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            polar_meta = result.setdefault("polar_metadata", {})
            polar_meta["dataset_task_id"] = task_id
            polar_meta["dataset_task_dir"] = task_dir
            polar_meta["dataset_image_tag"] = image_tag
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--cache-file", default=None, help="cache the tasks_dir instruction index across invocations")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    job_dir = Path(args.job_dir)
    cache_file = Path(args.cache_file) if args.cache_file else None

    if not (job_dir / "conversion_manifest.json").exists():
        print(f"[resolve-task-ids] no conversion_manifest.json under {job_dir}; nothing to do", file=sys.stderr)
        return 0

    index = load_or_build_task_index(tasks_dir, cache_file)
    stats = resolve_manifest(job_dir, tasks_dir, index)
    print(
        f"[resolve-task-ids] matched {stats['matched']}, unmatched {stats['unmatched']}, "
        f"already_set {stats['already_set']} (task index size: {len(index)})"
    )
    if stats["unmatched"] > 0:
        print(
            f"[resolve-task-ids] WARNING: {stats['unmatched']} evidence session(s) could not be "
            "matched to a dataset task; they will be excluded from the held-out gate's task list",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
