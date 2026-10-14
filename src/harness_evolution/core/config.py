"""Loop configuration: YAML loading, typed accessors and RuntimeConfig construction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import yaml
from harness_evolution.core.paths import REPO_ROOT
from harness_evolution.core.types import RuntimeConfig


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    value = yaml.safe_load(config_path.resolve().read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"yaml root must be a mapping: {config_path}")
    return value


def _mapping(config: dict[str, Any], key: str, *, required: bool = True) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        if required:
            raise ValueError(f"config section '{key}' is required")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section '{key}' must be a mapping")
    return value


def _str_value(config: dict[str, Any], key: str, default: str | None = None) -> str:
    value = config.get(key, default)
    if value is None or str(value).strip() == "":
        raise ValueError(f"config key '{key}' is required")
    return str(value)


def _int_value(config: dict[str, Any], key: str, default: int | None = None) -> int:
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"config key '{key}' is required")
    return int(value)


def _optional_int_value(config: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = config.get(key, default)
    if value is None:
        return None
    return int(value)


def _bool_value(config: dict[str, Any], key: str, default: bool | None = None) -> bool:
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"config key '{key}' is required")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _memory_section(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the optional memory setting without changing old configs."""
    value = raw.get("memory")
    if value is None:
        evolution = raw.get("evolution")
        if isinstance(evolution, dict):
            value = evolution.get("memory")
    if value is None:
        return {}
    if isinstance(value, str):
        return {"method": value}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("config section 'memory' must be a method name or mapping")


def _str_list(config: dict[str, Any], key: str, *, allow_empty: bool = False) -> list[str]:
    value = config.get(key)
    if value is None:
        raise ValueError(f"config key '{key}' is required")
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        raise ValueError(f"config key '{key}' must be a list of strings")
    result = [item for item in items if item]
    if not result and not allow_empty:
        raise ValueError(f"config key '{key}' must not be empty")
    return result


def _repo_relative_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _relative_workspace_dir(raw: Any, *, key: str) -> str:
    value = str(raw).strip() or "skills"
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts) or not path.parts:
        raise ValueError(f"{key} must be a relative workspace path: {raw}")
    return path.as_posix()


def _tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    raw = _load_yaml_config(path)
    initial = _mapping(raw, "initial")
    loop = _mapping(raw, "loop")
    terminal_bench = _mapping(raw, "terminal_bench")
    build = _mapping(raw, "build", required=False)
    opencode_evolver = _mapping(raw, "opencode_evolver", required=False)
    task_selection = _mapping(raw, "task_selection", required=False)
    evidence = _mapping(raw, "evidence", required=False)
    candidate_selection = _mapping(raw, "candidate_selection", required=False)
    validation = _mapping(raw, "validation", required=False)
    memory = _memory_section(raw)

    rollouts_per_attempt = _int_value(
        loop,
        "rollouts_per_attempt",
        loop.get("rollouts_per_iteration"),
    )
    if rollouts_per_attempt <= 0:
        raise ValueError("loop.rollouts_per_attempt must be positive")

    return RuntimeConfig(
        raw=raw,
        initial_harness_path=str(_repo_relative_path(_str_value(initial, "harness_path"))),
        initial_base_commit=_str_value(initial, "base_commit", "HEAD"),
        iterations=_int_value(loop, "iterations"),
        rollouts_per_attempt=rollouts_per_attempt,
        workspace_root=_repo_relative_path(_str_value(loop, "workspace_root")),
        report_root=_repo_relative_path(_str_value(loop, "report_root")),
        evolver_timeout_sec=_int_value(loop, "evolver_timeout_sec", 1800),
        terminal_bench=terminal_bench,
        build=build,
        opencode_evolver=opencode_evolver,
        task_selection=task_selection,
        evidence=evidence,
        candidate_selection=candidate_selection,
        validation=validation,
        memory=memory,
    )
