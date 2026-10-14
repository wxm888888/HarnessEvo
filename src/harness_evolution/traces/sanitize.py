from __future__ import annotations

import json
import shutil
from pathlib import Path

from harness_evolution.core.types import load_json_object, safe_mapping


def _sanitize_task_id(task_id: object) -> object:
    if not isinstance(task_id, dict):
        return task_id
    sanitized = dict(task_id)
    path = sanitized.get("path")
    if isinstance(path, str):
        sanitized["path"] = Path(path).name
    return sanitized


def sanitize_terminal_bench_result(result_path: Path) -> dict:
    result = load_json_object(result_path)
    verifier_result = safe_mapping(result.get("verifier_result"))
    rewards = safe_mapping(verifier_result.get("rewards"))
    exception_info = safe_mapping(result.get("exception_info"))
    agent_result = safe_mapping(result.get("agent_result"))
    config = safe_mapping(result.get("config"))
    return {
        "sanitized": True,
        "source_schema": "terminal_bench_result_summary_v1",
        "id": result.get("id"),
        "task_name": result.get("task_name"),
        "trial_name": result.get("trial_name"),
        "task_id": _sanitize_task_id(result.get("task_id")),
        "source": result.get("source"),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "reward": rewards.get("reward"),
        "exception": {
            "type": exception_info.get("exception_type"),
            "message": exception_info.get("exception_message"),
        },
        "agent_result": {
            "n_input_tokens": agent_result.get("n_input_tokens"),
            "n_output_tokens": agent_result.get("n_output_tokens"),
            "n_total_tokens": agent_result.get("n_total_tokens"),
        },
        "timeouts": {
            "timeout_multiplier": config.get("timeout_multiplier"),
            "agent_timeout_multiplier": config.get("agent_timeout_multiplier"),
            "verifier_timeout_multiplier": config.get("verifier_timeout_multiplier"),
            "agent_setup_timeout_multiplier": config.get("agent_setup_timeout_multiplier"),
            "environment_build_timeout_multiplier": config.get("environment_build_timeout_multiplier"),
        },
    }


def resolve_task_package_dir(*, result_path: Path, task_cache_root: Path) -> Path | None:
    result = load_json_object(result_path)
    task_id = result.get("task_id") if isinstance(result.get("task_id"), dict) else {}
    task_name = task_id.get("name")
    task_ref = task_id.get("ref")
    if not isinstance(task_name, str) or not isinstance(task_ref, str):
        return None
    task_slug = task_name.split("/")[-1]
    task_hash = task_ref.removeprefix("sha256:")
    candidate = task_cache_root / task_slug / task_hash
    if candidate.exists():
        return candidate.resolve()
    return None


def copy_instruction_if_available(*, result_path: Path, task_cache_root: Path, target_dir: Path) -> Path | None:
    local_instruction = result_path.parent / "instruction.md"
    if local_instruction.exists():
        target = target_dir / "instruction.md"
        shutil.copy2(local_instruction, target)
        return target

    task_package = resolve_task_package_dir(result_path=result_path, task_cache_root=task_cache_root)
    if task_package is None:
        return None
    instruction = task_package / "instruction.md"
    if not instruction.exists():
        return None
    target = target_dir / "instruction.md"
    shutil.copy2(instruction, target)
    return target


def write_sanitized_result(source_result_path: Path, target_path: Path) -> None:
    text = json.dumps(sanitize_terminal_bench_result(source_result_path), ensure_ascii=False, indent=2) + "\n"
    if "/home/" in text or "file:///home/" in text:
        raise ValueError(f"sanitized result still contains a host path: {source_result_path}")
    target_path.write_text(text, encoding="utf-8")
