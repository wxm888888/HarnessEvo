"""Instantiate the pluggable strategies (task selection, evidence, memory, ...) named in the config."""
from __future__ import annotations

from typing import Any
from harness_evolution.core.types import MemoryStrategy, RuntimeConfig
from harness_evolution.memory import MEMORY_REGISTRY
from harness_evolution.core.config import _mapping, _str_value


def _build_strategy(registry: dict[str, Any], section: dict[str, Any], default_method: str, config: RuntimeConfig) -> Any:
    method = _str_value(section, "method", default_method)
    options = _mapping(section, "options", required=False)
    if method not in registry:
        raise ValueError(f"unknown strategy method '{method}', available: {', '.join(sorted(registry))}")
    cls = registry[method]
    try:
        return cls(options, config)
    except TypeError:
        return cls()


def _build_memory_strategy(config: RuntimeConfig) -> MemoryStrategy:
    return _build_strategy(MEMORY_REGISTRY, config.memory, "none", config)
