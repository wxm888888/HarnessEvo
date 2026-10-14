from __future__ import annotations

import json
import random
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def discover_cached_terminal_bench_tasks(cache_root: str | Path | None = None) -> list[str]:
    """从 Harbor 本地 task cache 中发现 Terminal-Bench task 名称。"""
    root = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache/harbor/tasks/packages/terminal-bench"
    if not root.exists():
        return []

    task_names: set[str] = set()
    for task_toml in root.glob("*/**/task.toml"):
        task_dir_name = task_toml.parent.parent.name
        try:
            data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        task_config = data.get("task")
        configured_name = task_config.get("name") if isinstance(task_config, dict) else None
        if isinstance(configured_name, str) and configured_name:
            task_names.add(configured_name.rsplit("/", 1)[-1])
        elif task_dir_name:
            task_names.add(task_dir_name)
    return sorted(task_names)


def load_task_history(path: str | Path) -> dict[str, Any]:
    """读取 scheduler 历史；没有历史文件时返回空历史结构。"""
    history_path = Path(path)
    if not history_path.exists():
        return {"seen_tasks": [], "task_stats": {}, "iterations": []}
    data = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"task history must be a JSON object: {history_path}")
    data.setdefault("seen_tasks", [])
    data.setdefault("task_stats", {})
    data.setdefault("iterations", [])
    return data


def save_task_history(path: str | Path, history: dict[str, Any]) -> None:
    """把 scheduler 历史写回 JSON，供下一轮 task 选择使用。"""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_tasks(
    *,
    all_tasks: list[str],
    iteration_idx: int,
    tasks_per_iteration: int,
    history: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """根据策略和历史状态，为当前 iteration 选择 Terminal-Bench tasks。"""
    if tasks_per_iteration <= 0:
        raise ValueError("tasks_per_iteration must be positive")
    ordered_tasks = sorted(dict.fromkeys(all_tasks))
    if not ordered_tasks:
        raise ValueError("task scheduler cannot select from an empty task list")

    strategy = str(config.get("strategy", "mix_failed_and_new"))
    seed = int(config.get("seed", 0))
    rng = random.Random(seed + iteration_idx)
    limit = min(tasks_per_iteration, len(ordered_tasks))

    seen_tasks = set(str(task) for task in history.get("seen_tasks", []))
    task_stats = history.get("task_stats", {})
    if not isinstance(task_stats, dict):
        task_stats = {}

    unseen_tasks = [task for task in ordered_tasks if task not in seen_tasks]
    failed_tasks = [
        task
        for task in ordered_tasks
        if isinstance(task_stats.get(task), dict) and task_stats[task].get("last_success") is False
    ]

    if strategy == "sequential_shards":
        selected = _select_sequential_shard(ordered_tasks, iteration_idx, limit)
    elif strategy == "random_unseen":
        selected = _sample_random_unseen(
            rng=rng,
            ordered_tasks=ordered_tasks,
            unseen_tasks=unseen_tasks,
            limit=limit,
        )
    elif strategy == "retry_failed":
        selected = _select_retry_failed(
            rng=rng,
            ordered_tasks=ordered_tasks,
            failed_tasks=failed_tasks,
            unseen_tasks=unseen_tasks,
            limit=limit,
        )
    elif strategy == "mix_failed_and_new":
        failed_ratio = float(config.get("failed_ratio", 0.4))
        selected = _select_mix_failed_and_new(
            rng=rng,
            ordered_tasks=ordered_tasks,
            failed_tasks=failed_tasks,
            unseen_tasks=unseen_tasks,
            limit=limit,
            failed_ratio=failed_ratio,
        )
    else:
        raise ValueError(f"unknown task_scheduler strategy: {strategy}")

    selected = selected[:limit]
    return {
        "strategy": strategy,
        "iteration": iteration_idx + 1,
        "tasks_per_iteration": tasks_per_iteration,
        "selected_tasks": selected,
        "candidate_task_count": len(ordered_tasks),
        "seen_task_count_before": len(seen_tasks),
        "unseen_task_count_before": len(unseen_tasks),
        "failed_task_count_before": len(failed_tasks),
        "seed": seed,
    }


def update_task_history_from_job(
    *,
    history: dict[str, Any],
    iteration_idx: int,
    selected_tasks: list[str],
    job_dir: str | Path,
    success_threshold: float = 1.0,
) -> dict[str, Any]:
    """根据 Harbor job 结果更新每个 task 的成功/失败统计。"""
    job_path = Path(job_dir)
    task_outcomes = _read_task_outcomes(job_path, selected_tasks, success_threshold)
    seen_tasks = set(str(task) for task in history.get("seen_tasks", []))
    task_stats = history.setdefault("task_stats", {})
    iterations = history.setdefault("iterations", [])

    completed_tasks: list[str] = []
    failed_tasks: list[str] = []
    missing_result_tasks: list[str] = []

    for task_name in selected_tasks:
        seen_tasks.add(task_name)
        outcome = task_outcomes.get(task_name) or {
            "task_name": task_name,
            "success": False,
            "reward": None,
            "trial_name": None,
            "result_path": None,
            "exception_type": "MissingResult",
            "exception_message": f"no result.json found for selected task {task_name}",
        }
        stats = task_stats.setdefault(
            task_name,
            {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
            },
        )
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        stats["last_iteration"] = iteration_idx + 1
        stats["last_reward"] = outcome.get("reward")
        stats["last_success"] = bool(outcome.get("success"))
        stats["last_trial_name"] = outcome.get("trial_name")
        stats["last_result_path"] = outcome.get("result_path")
        stats["last_exception_type"] = outcome.get("exception_type")
        stats["last_exception_message"] = outcome.get("exception_message")

        if outcome.get("success"):
            stats["successes"] = int(stats.get("successes", 0)) + 1
            completed_tasks.append(task_name)
        else:
            stats["failures"] = int(stats.get("failures", 0)) + 1
            failed_tasks.append(task_name)
            if outcome.get("exception_type") == "MissingResult":
                missing_result_tasks.append(task_name)

    history["seen_tasks"] = sorted(seen_tasks)
    iterations.append(
        {
            "iteration": iteration_idx + 1,
            "selected_tasks": selected_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "missing_result_tasks": missing_result_tasks,
            "job_dir": str(job_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return history


def _select_sequential_shard(ordered_tasks: list[str], iteration_idx: int, limit: int) -> list[str]:
    """按 task 排序结果切连续分片；超过末尾时回绕补齐。"""
    start = iteration_idx * limit
    selected = ordered_tasks[start : start + limit]
    if len(selected) < limit:
        selected.extend(ordered_tasks[: limit - len(selected)])
    return selected


def _sample_random_unseen(
    *,
    rng: random.Random,
    ordered_tasks: list[str],
    unseen_tasks: list[str],
    limit: int,
) -> list[str]:
    """优先随机选择未见任务，不足时从全量任务中随机补齐。"""
    selected = _sample(rng, unseen_tasks, min(limit, len(unseen_tasks)))
    return _fill_unique(rng, selected, ordered_tasks, limit)


def _select_retry_failed(
    *,
    rng: random.Random,
    ordered_tasks: list[str],
    failed_tasks: list[str],
    unseen_tasks: list[str],
    limit: int,
) -> list[str]:
    """优先重试历史失败任务，不足时用未见任务和全量任务补齐。"""
    selected = _sample(rng, failed_tasks, min(limit, len(failed_tasks)))
    selected = _fill_unique(rng, selected, unseen_tasks, limit)
    return _fill_unique(rng, selected, ordered_tasks, limit)


def _select_mix_failed_and_new(
    *,
    rng: random.Random,
    ordered_tasks: list[str],
    failed_tasks: list[str],
    unseen_tasks: list[str],
    limit: int,
    failed_ratio: float,
) -> list[str]:
    """按比例混合历史失败任务和未见任务，兼顾修复回测与新任务探索。"""
    failed_ratio = max(0.0, min(1.0, failed_ratio))
    failed_target = min(limit, round(limit * failed_ratio))
    selected = _sample(rng, failed_tasks, min(failed_target, len(failed_tasks)))
    selected = _fill_unique(rng, selected, unseen_tasks, limit)
    selected = _fill_unique(rng, selected, failed_tasks, limit)
    return _fill_unique(rng, selected, ordered_tasks, limit)


def _sample(rng: random.Random, items: list[str], count: int) -> list[str]:
    """随机采样并保持返回顺序稳定可复现。"""
    if count <= 0:
        return []
    unique_items = sorted(dict.fromkeys(items))
    if count >= len(unique_items):
        shuffled = list(unique_items)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(unique_items, count)


def _fill_unique(
    rng: random.Random,
    selected: list[str],
    candidates: list[str],
    limit: int,
) -> list[str]:
    """从候选集中随机补齐 selected，避免同一轮重复 task。"""
    result = list(selected)
    selected_set = set(result)
    remaining = [item for item in sorted(dict.fromkeys(candidates)) if item not in selected_set]
    rng.shuffle(remaining)
    for item in remaining:
        if len(result) >= limit:
            break
        result.append(item)
        selected_set.add(item)
    return result


def _read_task_outcomes(
    job_path: Path,
    selected_tasks: list[str],
    success_threshold: float,
) -> dict[str, dict[str, Any]]:
    """读取 job 目录下每个 trial 的 result.json，并按 task 汇总最后结果。"""
    selected_set = set(selected_tasks)
    outcomes: dict[str, dict[str, Any]] = {}
    for result_path in sorted(job_path.glob("*/result.json")):
        trial_dir = result_path.parent
        task_name = trial_dir.name.split("__", 1)[0]
        if task_name not in selected_set:
            continue
        result = _load_json(result_path)
        reward = _extract_reward(result)
        exception_info = result.get("exception_info") if isinstance(result, dict) else None
        exception_type = exception_info.get("exception_type") if isinstance(exception_info, dict) else None
        exception_message = exception_info.get("exception_message") if isinstance(exception_info, dict) else None
        success = reward is not None and float(reward) >= success_threshold and exception_type is None
        outcomes[task_name] = {
            "task_name": task_name,
            "success": success,
            "reward": reward,
            "trial_name": result.get("trial_name") if isinstance(result, dict) else trial_dir.name,
            "result_path": str(result_path),
            "exception_type": exception_type,
            "exception_message": exception_message,
        }
    return outcomes


def _load_json(path: Path) -> dict[str, Any]:
    """读取 JSON object；解析失败时返回空 dict。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _extract_reward(result: dict[str, Any]) -> float | None:
    """从 Terminal-Bench result.json 中提取 reward 字段。"""
    verifier_result = result.get("verifier_result")
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None
