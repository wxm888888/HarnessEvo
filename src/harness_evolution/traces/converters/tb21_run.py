#!/usr/bin/env python3
"""Adapt the TB2.1 JSONL artifacts to the existing-TB evolver input layout.

The modular evolver consumes ``<task>/<trial>/agent/trajectory.json`` and a
nearby ``result.json``.  The TB2.1 runner writes ``trajectory.jsonl`` plus
``meta.json``/``reward.txt`` instead, so this script preserves every JSONL
event in one valid JSON document and creates the small result summary expected
by the evolver.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="TB2.1 agent-runs/<run-id> directory")
    parser.add_argument("--output-job-dir", required=True, type=Path, help="Directory consumed by the modular evolver")
    parser.add_argument("--task", action="append", dest="tasks", help="Task name to include; repeatable (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output directory before writing")
    return parser.parse_args()


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected an object in {path}:{line_no}")
        events.append(value)
    return events


def _convert_trial(source: Path, target: Path) -> dict[str, Any]:
    trajectory_jsonl = source / "trajectory.jsonl"
    meta = _json(source / "meta.json", {})
    if not trajectory_jsonl.exists():
        raise ValueError(f"missing trajectory.jsonl: {source}")
    if not isinstance(meta, dict):
        meta = {}
    reward = (source / "reward.txt").read_text(encoding="utf-8", errors="replace").strip() if (source / "reward.txt").exists() else "missing"
    events = _events(trajectory_jsonl)

    (target / "agent").mkdir(parents=True, exist_ok=True)
    trajectory = {
        "schema_version": "TB21-JSONL-adapter-v1",
        "task_name": meta.get("task_id", source.parent.name),
        "trial_name": source.name,
        "run_id": meta.get("run_id"),
        "model": meta.get("model"),
        "opencode_version": meta.get("opencode_version"),
        "source_trajectory": str(trajectory_jsonl),
        "llm_context": "agent/llm-context.jsonl" if (source / "llm-context.jsonl").exists() else None,
        "openai_messages": "agent/openai-messages.jsonl"
        if (source / "openai-messages.jsonl").exists()
        else None,
        "openai_tools": "agent/openai-tools.json"
        if (source / "openai-tools.json").exists()
        else None,
        "events": events,
    }
    (target / "agent" / "trajectory.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(trajectory_jsonl, target / "trajectory.jsonl")
    instruction = source / "instruction.md"
    if instruction.exists():
        shutil.copy2(instruction, target / "instruction.md")
    context = source / "llm-context.jsonl"
    if context.exists():
        shutil.copy2(context, target / "agent" / "llm-context.jsonl")
    openai_messages = source / "openai-messages.jsonl"
    if openai_messages.exists():
        shutil.copy2(openai_messages, target / "agent" / "openai-messages.jsonl")
    openai_messages_meta = source / "openai-messages.meta.json"
    if openai_messages_meta.exists():
        shutil.copy2(openai_messages_meta, target / "agent" / "openai-messages.meta.json")
    for name in (
        "openai-tools.json",
        "openai-request.meta.json",
        "verifier-summary.json",
        "verifier-feedback.json",
    ):
        source_meta = source / name
        if source_meta.exists():
            shutil.copy2(source_meta, target / "agent" / name)

    result = {
        "id": f"{meta.get('run_id', '')}::{meta.get('task_id', source.parent.name)}::{source.name}",
        "task_name": meta.get("task_id", source.parent.name),
        "trial_name": source.name,
        "task_id": {"name": meta.get("task_id", source.parent.name)},
        "source": "tb21-adapter",
        "reward": reward,
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": None, "exception_message": None},
        "agent_result": {},
        "config": {},
    }
    (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "task_name": result["task_name"],
        "trial_name": source.name,
        "source": str(source),
        "trajectory": str(target / "agent" / "trajectory.json"),
        "result": str(target / "result.json"),
        "context": str(target / "agent" / "llm-context.jsonl") if context.exists() else None,
        "openai_messages": str(target / "agent" / "openai-messages.jsonl")
        if openai_messages.exists()
        else None,
        "openai_tools": str(target / "agent" / "openai-tools.json")
        if (source / "openai-tools.json").exists()
        else None,
        "verifier_summary": str(target / "agent" / "verifier-summary.json")
        if (source / "verifier-summary.json").exists()
        else None,
        "reward": reward,
        "events": len(events),
    }


def main() -> None:
    args = _args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output_job_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory does not exist: {run_dir}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.tasks or [])
    records = []
    for task_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and (not selected or path.name in selected)):
        for trial_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            if not (trial_dir / "trajectory.jsonl").exists():
                continue
            records.append(_convert_trial(trial_dir, output / task_dir.name / trial_dir.name))
    if not records:
        raise SystemExit(f"no trajectory.jsonl files found under {run_dir}")
    manifest = {"source_run_dir": str(run_dir), "output_job_dir": str(output), "records": records}
    (output / "tb21_adapter_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_job_dir": str(output), "trials": len(records), "records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
