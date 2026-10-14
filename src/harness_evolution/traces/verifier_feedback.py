from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


_MAX_STREAM_CHARS = 12000
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PYTEST_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<label>failed|passed|error|errors|skipped|xfailed|xpassed|deselected)\b",
    re.IGNORECASE,
)
_PYTEST_TEST_NAME_RE = re.compile(r"_{5,}\s*(test_[A-Za-z0-9_]+)\s*_{5,}")
_PYTEST_NODE_RE = re.compile(r"FAILED\s+([^\s]+::test_[A-Za-z0-9_]+)")
_PYTEST_ASSERTION_RE = re.compile(
    r"AssertionError:\s*(.*?)(?=(?:E\s+(?:assert|\+)|={8,}|!{8,}|FAILED\s+|$))",
    re.DOTALL,
)


def _read_stream(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_stream(value: str) -> str:
    value = _ANSI_ESCAPE_RE.sub("", value)
    return _CONTROL_RE.sub("", value).strip()


def _excerpt(value: str) -> tuple[str, bool]:
    if len(value) <= _MAX_STREAM_CHARS:
        return value, False
    return value[-_MAX_STREAM_CHARS:], True


def _pytest_details(stdout: str) -> dict[str, Any]:
    collected_match = re.search(r"collected\s+(\d+)\s+items?", stdout, re.IGNORECASE)
    counts: dict[str, int] = {}
    for match in _PYTEST_COUNT_RE.finditer(stdout):
        label = match.group("label").lower()
        if label == "errors":
            label = "error"
        counts[label] = int(match.group("count"))

    names = _PYTEST_TEST_NAME_RE.findall(stdout)
    node_ids = _PYTEST_NODE_RE.findall(stdout)
    assertions = [
        " ".join(match.group(1).split())[:4000]
        for match in _PYTEST_ASSERTION_RE.finditer(stdout)
        if match.group(1).strip()
    ]
    failed_count = max(len(names), len(node_ids), len(assertions))
    failed_checks = []
    for index in range(failed_count):
        node_id = node_ids[index] if index < len(node_ids) else ""
        name = names[index] if index < len(names) else ""
        if not name and node_id:
            name = node_id.rsplit("::", 1)[-1]
        failed_checks.append(
            {
                "name": name,
                "node_id": node_id,
                "assertion": assertions[index] if index < len(assertions) else "",
            }
        )
    return {
        "collected": int(collected_match.group(1)) if collected_match else None,
        "counts": counts,
        "failed_checks": failed_checks,
    }


def build_verifier_feedback(
    *,
    stdout: str,
    stderr: str,
    reward: str | int | float | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    clean_stdout = _clean_stream(stdout)
    clean_stderr = _clean_stream(stderr)
    stdout_excerpt, stdout_truncated = _excerpt(clean_stdout)
    stderr_excerpt, stderr_truncated = _excerpt(clean_stderr)
    is_pytest = "pytest" in clean_stdout.lower() or "test session starts" in clean_stdout.lower()
    reward_match = re.search(r"__POLAR_VERIFIER_REWARD__=([^\s]+)", clean_stdout)
    return {
        "schema_version": "verifier-feedback-v1",
        "available": bool(clean_stdout or clean_stderr),
        "framework": "pytest" if is_pytest else "unknown",
        "reward": reward if reward is not None else (reward_match.group(1) if reward_match else None),
        "status": status,
        "pytest": _pytest_details(clean_stdout) if is_pytest else None,
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
        "truncated": {
            "stdout": stdout_truncated,
            "stderr": stderr_truncated,
            "max_chars_per_stream": _MAX_STREAM_CHARS,
        },
    }


def write_verifier_feedback(
    *,
    stdout_path: Path | None,
    stderr_path: Path | None,
    target_path: Path,
    reward: str | int | float | None = None,
    status: str | None = None,
) -> Path:
    payload = build_verifier_feedback(
        stdout=_read_stream(stdout_path),
        stderr=_read_stream(stderr_path),
        reward=reward,
        status=status,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target_path


def materialize_verifier_feedback(
    *,
    trajectory_path: Path,
    result_path: Path,
    target_path: Path,
) -> Path | None:
    source_dirs = (
        trajectory_path.parent,
        result_path.parent / "agent",
        result_path.parent,
        trajectory_path.parent.parent,
    )
    for directory in source_dirs:
        existing = directory / "verifier-feedback.json"
        if existing.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, target_path)
            return target_path

    stdout_path = next(
        (
            directory / "verifier.stdout.log"
            for directory in source_dirs
            if (directory / "verifier.stdout.log").is_file()
        ),
        None,
    )
    stderr_path = next(
        (
            directory / "verifier.stderr.log"
            for directory in source_dirs
            if (directory / "verifier.stderr.log").is_file()
        ),
        None,
    )
    if stdout_path is None and stderr_path is None:
        return None
    return write_verifier_feedback(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        target_path=target_path,
    )
