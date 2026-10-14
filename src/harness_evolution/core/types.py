from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness_evolution.core.paths import REPO_ROOT


WORKSPACE_ARTIFACT_ROOT = ".harness_evolver/terminal_bench_artifacts"
DEFAULT_TASK_CACHE_ROOT = Path.home() / ".cache" / "harbor" / "tasks" / "packages" / "terminal-bench"


def mapping(config: dict[str, Any], key: str, *, required: bool = True) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        if required:
            raise ValueError(f"config section '{key}' is required")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section '{key}' must be a mapping")
    return value


def str_value(config: dict[str, Any], key: str, default: str | None = None) -> str:
    value = config.get(key, default)
    if value is None or str(value).strip() == "":
        raise ValueError(f"config key '{key}' is required")
    return str(value)


def int_value(config: dict[str, Any], key: str, default: int | None = None) -> int:
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"config key '{key}' is required")
    return int(value)


def optional_int_value(config: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = config.get(key, default)
    if value is None:
        return None
    return int(value)


def bool_value(config: dict[str, Any], key: str, default: bool | None = None) -> bool:
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"config key '{key}' is required")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def str_list(config: dict[str, Any], key: str, *, allow_empty: bool = False) -> list[str]:
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


def repo_relative_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]
    initial_harness_path: str
    initial_base_commit: str
    iterations: int
    rollouts_per_attempt: int
    workspace_root: Path
    report_root: Path
    evolver_timeout_sec: int
    terminal_bench: dict[str, Any]
    build: dict[str, Any]
    opencode_evolver: dict[str, Any]
    task_selection: dict[str, Any]
    evidence: dict[str, Any]
    candidate_selection: dict[str, Any]
    validation: dict[str, Any]
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSelectionContext:
    iteration_idx: int
    static_task_names: list[str]
    jobs_root: Path


@dataclass
class TaskSelectionResult:
    task_names: list[str]
    metadata: dict[str, Any]


@dataclass
class CollectionArtifact:
    trajectory_path: Path
    result_path: Path
    # Optional complete LLM request/response context captured by the TB2.1
    # runtime. Older jobs do not have this file.
    context_path: Path | None = None
    # OpenAI-compatible incremental message stream generated from the raw
    # cumulative request context. This is the preferred evolution input.
    openai_messages_path: Path | None = None


@dataclass
class CollectionResult:
    job_name: str
    job_dir: Path
    log_path: Path
    returncode: int
    artifacts: list[CollectionArtifact]
    metadata: dict[str, Any]


@dataclass
class EvidenceBuildResult:
    requirement_addendum: str
    metadata: dict[str, Any]


@dataclass
class MemoryBuildResult:
    requirement_addendum: str
    metadata: dict[str, Any]


@dataclass
class CommandValidationResult:
    command: str
    passed: bool
    returncode: int
    duration_sec: float
    log_path: str
    output_tail: str


@dataclass
class ValidationResult:
    enabled: bool
    passed: bool
    commands: list[CommandValidationResult] = field(default_factory=list)


@dataclass
class DiffStats:
    files: int = 0
    added: int = 0
    deleted: int = 0
    score: int = 0


@dataclass
class Candidate:
    candidate_idx: int
    attempt_idx: int
    rollout_idx: int
    workspace: Path
    branch: str
    commit: str
    commit_created: bool
    changed_files: list[str]
    summary: str
    trajectory: list[dict[str, Any]]
    evidence: dict[str, Any]
    skill_copy: dict[str, Any]
    evolver_metadata: dict[str, Any]
    validation: ValidationResult
    diff_stats: DiffStats


@dataclass
class CandidateSelectionContext:
    candidates: list[Candidate]
    options: dict[str, Any]


@dataclass
class CandidateSelectionDecision:
    accepted: bool
    candidate_idx: int | None
    fallback_reason: str | None
    metadata: dict[str, Any]


class TaskSelectionStrategy(Protocol):
    name: str

    def select(self, ctx: TaskSelectionContext) -> TaskSelectionResult:
        ...

    def update_after_collection(
        self,
        *,
        iteration_idx: int,
        selected_tasks: list[str],
        job_dir: Path,
    ) -> dict[str, Any] | None:
        ...


class EvidenceStrategy(Protocol):
    name: str

    def materialize(
        self,
        *,
        artifacts: list[CollectionArtifact],
        workspace: Path,
        task_cache_root: Path,
        current_harness_path: str | Path | None = None,
    ) -> EvidenceBuildResult:
        ...


class CandidateSelectionStrategy(Protocol):
    name: str

    def select(self, ctx: CandidateSelectionContext) -> CandidateSelectionDecision:
        ...


class MemoryStrategy(Protocol):
    name: str
    enabled: bool

    def materialize(self, *, workspace: Path) -> MemoryBuildResult:
        ...

    def record_promotion(
        self,
        *,
        step: int,
        parent_commit: str,
        winner: Candidate,
        baseline_score: dict[str, Any] | None,
        winner_score: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        ...
