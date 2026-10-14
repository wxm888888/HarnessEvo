"""Deterministic eligibility checks for evolution evidence.

The sampling gate can produce a durable trial record even when the sandbox,
verifier, or trace collector failed before the task was actually evaluated.
Those records are useful for operations, but they are not evidence about the
harness and must not be handed to an evolver.  This module keeps that decision
outside the model prompt and returns only complete, verifier-backed trials.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from harness_evolution.core.types import CollectionArtifact, safe_mapping


_ENVIRONMENT_ERROR_PATTERNS = (
    re.compile(r"\bpython(?:3)?\b[^\n]{0,120}(?:command not found|not found|no such file)", re.I),
    re.compile(r"\bpython(?:3)?\b[^\n]{0,120}(?:not installed|missing)", re.I),
    re.compile(r"no such file or directory[^\n]{0,120}\b(?:python|pip|apt|conda)\b", re.I),
    re.compile(r"(?:missing|unable to locate|could not find)[^\n]{0,120}(?:package|dependency|module|command|python|pip)", re.I),
    re.compile(r"(?:apt(?:-get)?|pip(?:3)?|conda)[^\n]{0,160}(?:error|failed|failure|unable|not found)", re.I),
    re.compile(r"(?:network|dns|internet|connection)[^\n]{0,120}(?:error|fail(?:ed|ure)?|unavailable|timeout|refused)", re.I),
    re.compile(r"(?:permission denied|operation not permitted)", re.I),
    re.compile(r"(?:gpu|cuda|nvidia)[^\n]{0,120}(?:error|unavailable|not found)", re.I),
    re.compile(r"(?:disk|filesystem|quota|no space left)[^\n]{0,120}(?:full|space|error|exhaust)", re.I),
    re.compile(r"(?:port|address)[^\n]{0,120}(?:already in use|occupied|bind[^\n]{0,40}fail)", re.I),
    re.compile(r"(?:route not active|route not found|command receipt[^\n]{0,80}(?:failed|error)|trace proxy[^\n]{0,80}(?:unavailable|failed|error))", re.I),
    re.compile(r"(?:sandbox|opensandbox|control plane)[^\n]{0,120}(?:error|fail(?:ed|ure)?|unavailable|closed|not found)", re.I),
    re.compile(r"(?:verifier|test(?:s)?)[^\n]{0,120}(?:dependency missing|did not execute|not executed)", re.I),
)

_INFRA_FAILURE_CLASSES = {
    "runtime_error",
    "verifier_timeout",
    "agent_timeout",
    "verifier_invalid",
    "verifier_failure",
}


@dataclass(frozen=True)
class PreflightDecision:
    eligible: bool
    reason_code: str
    reason: str
    task_name: str = ""
    trial_name: str = ""
    result_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _nonempty(path: Path | None) -> bool:
    return bool(path and path.is_file() and path.stat().st_size > 0)


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if _nonempty(candidate):
            return candidate
    return None


def _structured_text_matches(values: Iterable[Any]) -> str | None:
    """Return a matching marker from structured error fields only.

    We intentionally do not scan the whole trajectory.  Task instructions or
    legitimate command output can mention Python, ports, or packages without
    being an environment failure.
    """

    text = "\n".join(str(value) for value in values if value not in (None, "")).strip()
    if not text:
        return None
    for pattern in _ENVIRONMENT_ERROR_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)[:240]
    return None


def _decision(artifact: CollectionArtifact, code: str, reason: str, *, eligible: bool = False) -> PreflightDecision:
    result = _load_json(artifact.result_path) or {}
    return PreflightDecision(
        eligible=eligible,
        reason_code=code,
        reason=reason,
        task_name=str(result.get("task_name") or ""),
        trial_name=str(result.get("trial_name") or ""),
        result_path=str(artifact.result_path),
    )


def classify_collection_artifact(artifact: CollectionArtifact) -> PreflightDecision:
    """Classify one collected trial as usable evolution evidence or not."""

    if not artifact.result_path.is_file():
        return _decision(artifact, "missing_result", "result.json is missing")
    result = _load_json(artifact.result_path)
    if result is None:
        return _decision(artifact, "invalid_result", "result.json is not a JSON object")

    trial = _load_json(artifact.result_path.parent / "trial.json") or {}
    if bool(trial.get("error")):
        return _decision(
            artifact,
            "trial_error",
            f"trial.json error: {trial.get('error_type') or trial.get('detail') or 'unknown'}",
        )
    if str(trial.get("error_type") or "").strip():
        return _decision(artifact, "trial_error_type", f"trial.json error_type={trial['error_type']}")

    exception_info = safe_mapping(result.get("exception_info"))
    exception_type = exception_info.get("exception_type")
    exception_message = exception_info.get("exception_message")
    if exception_type or exception_message:
        return _decision(
            artifact,
            "exception",
            f"structured trial exception: {exception_type or exception_message}",
        )

    verifier_result = safe_mapping(result.get("verifier_result"))
    rewards = safe_mapping(verifier_result.get("rewards"))
    reward = rewards.get("reward", result.get("reward"))
    if reward in (None, "", "missing"):
        return _decision(artifact, "missing_reward", "no verifier reward was recorded")
    try:
        reward_value = float(reward)
    except (TypeError, ValueError):
        return _decision(artifact, "invalid_reward", f"invalid verifier reward={reward!r}")
    if reward_value not in (0.0, 1.0):
        return _decision(artifact, "invalid_reward", f"non-binary verifier reward={reward!r}")

    status_paths = (
        artifact.result_path.parent / "agent" / "artifact-status.json",
        artifact.result_path.parent / "artifact-status.json",
        artifact.trajectory_path.parent / "artifact-status.json",
    )
    status_path = next((path for path in status_paths if path.is_file()), None)
    if status_path is None:
        return _decision(artifact, "missing_artifact_status", "trace materialization status is missing")
    artifact_status = _load_json(status_path)
    if artifact_status is None or artifact_status.get("complete") is not True:
        return _decision(artifact, "incomplete_artifacts", f"trace materialization incomplete: {status_path}")

    summary_path = _first_existing(
        (
            artifact.result_path.parent / "agent" / "verifier-summary.json",
            artifact.result_path.parent / "verifier-summary.json",
            artifact.trajectory_path.parent / "verifier-summary.json",
        )
    )
    if summary_path is None:
        return _decision(artifact, "missing_verifier_summary", "verifier-summary.json is missing")
    summary = _load_json(summary_path)
    if summary is None:
        return _decision(artifact, "invalid_verifier_summary", "verifier-summary.json is invalid")
    summary_status = str(summary.get("status") or "").lower()
    summary_reward = str(summary.get("reward") or "").lower()
    if summary_status not in {"ok", "fail"} or summary_reward not in {"0", "1"}:
        return _decision(artifact, "verifier_not_executed", "verifier summary does not show a valid execution")
    failure_class = str(summary.get("failure_class") or "").lower()
    if failure_class in _INFRA_FAILURE_CLASSES:
        return _decision(artifact, "infrastructure_failure", f"verifier failure_class={failure_class}")

    marker = _structured_text_matches(
        (
            trial.get("detail"),
            trial.get("error_type"),
            exception_type,
            exception_message,
            summary.get("failure_phase"),
        )
    )
    if marker:
        return _decision(artifact, "environment_failure", f"environment/setup marker: {marker}")

    messages_path = _first_existing(
        (
            *((artifact.openai_messages_path,) if artifact.openai_messages_path else ()),
            artifact.trajectory_path.parent / "openai-messages.jsonl",
            artifact.result_path.parent / "openai-messages.jsonl",
            artifact.result_path.parent / "agent" / "openai-messages.jsonl",
        )
    )
    tools_path = _first_existing(
        (
            artifact.trajectory_path.parent / "openai-tools.json",
            artifact.result_path.parent / "openai-tools.json",
            artifact.result_path.parent / "agent" / "openai-tools.json",
        )
    )
    if not _nonempty(artifact.trajectory_path) or messages_path is None or tools_path is None:
        missing = [
            name
            for name, present in (
                ("trajectory.json", _nonempty(artifact.trajectory_path)),
                ("openai-messages.jsonl", messages_path is not None),
                ("openai-tools.json", tools_path is not None),
            )
            if not present
        ]
        return _decision(artifact, "missing_trace_files", "missing normalized trace files: " + ", ".join(missing))

    return _decision(artifact, "eligible", "complete trace with a valid verifier outcome", eligible=True)


def filter_collection_artifacts(
    artifacts: Iterable[CollectionArtifact],
) -> tuple[list[CollectionArtifact], list[PreflightDecision]]:
    """Return eligible artifacts and a decision for every input artifact."""

    eligible: list[CollectionArtifact] = []
    decisions: list[PreflightDecision] = []
    for artifact in artifacts:
        decision = classify_collection_artifact(artifact)
        decisions.append(decision)
        if decision.eligible:
            eligible.append(artifact)
    return eligible, decisions
