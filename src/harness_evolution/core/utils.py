"""Small helpers shared across the package."""
from __future__ import annotations

def sanitize_path_part(value: str, *, fallback: str = "run") -> str:
    sanitized = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return sanitized.strip("._") or fallback


from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
