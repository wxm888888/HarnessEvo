"""The instruction handed to the evolver agent and the report it must write back."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from harness_evolution.core.types import EvidenceBuildResult, MemoryBuildResult, RuntimeConfig
from harness_evolution.core.config import _bool_value, _mapping, _relative_workspace_dir, _repo_relative_path
from harness_evolution.core.prompts import render_prompt_section


def _copy_skills(config: RuntimeConfig, workspace: Path) -> dict[str, Any]:
    skills = _mapping(config.opencode_evolver, "skills", required=False)
    if not _bool_value(skills, "enabled", False):
        return {"enabled": False}
    source_dir = _repo_relative_path(str(skills.get("source_dir", "skills")))
    workspace_dir = _relative_workspace_dir(skills.get("workspace_dir", "skills"), key="opencode_evolver.skills.workspace_dir")
    target = workspace / workspace_dir
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_dir, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    names = [path.parent.name for path in sorted(target.glob("*/SKILL.md"))]
    return {
        "enabled": True,
        "source_dir": str(source_dir),
        "workspace_dir": workspace_dir,
        "skill_names": names,
    }


_CANDIDATE_REPORT_RELATIVE_PATH = Path(".harness_evolver/candidate-report.json")


def _candidate_requirement(
    *,
    evidence: EvidenceBuildResult,
    skill_copy: dict[str, Any],
    validation_feedback: str,
    memory: MemoryBuildResult | None = None,
) -> str:
    # Fixed text lives in proposer/prompts/candidate.md; this function only
    # decides which optional parts (skills, memory, feedback) are present.
    def section(name: str, **values: object) -> str:
        return render_prompt_section(__package__, "candidate.md", name, **values)

    skill_lines: list[str] = []
    skill_names = skill_copy.get("skill_names")
    workspace_dir = str(
        skill_copy.get("workspace_dir") or ".harness_evolver/skills"
    ).rstrip("/")
    if isinstance(skill_names, list) and skill_names:
        skill_lines = [
            section(
                "skills",
                skill_names=", ".join(str(name) for name in skill_names),
                skill_paths=", ".join(f"{workspace_dir}/{name}/SKILL.md" for name in skill_names),
            )
        ]

    feedback_lines = (
        [section("feedback-heading"), validation_feedback]
        if validation_feedback
        else []
    )
    memory_lines = (
        [memory.requirement_addendum]
        if memory is not None and memory.requirement_addendum
        else []
    )
    sections = [
        section("role"),
        section("scope"),
        *skill_lines,
        *memory_lines,
        evidence.requirement_addendum,
        section("hypotheses"),
        section("change-scope"),
        section("validation"),
        section("report", report_path=_CANDIDATE_REPORT_RELATIVE_PATH.as_posix()),
        *feedback_lines,
    ]
    return "\n".join(text for text in sections if text)


def _load_candidate_report(workspace: Path) -> dict[str, Any]:
    path = workspace / _CANDIDATE_REPORT_RELATIVE_PATH
    record: dict[str, Any] = {"path": str(path)}
    if not path.is_file():
        return {**record, "status": "missing"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - preserve malformed output for audit.
        return {**record, "status": "invalid", "error": repr(exc)}
    if not isinstance(value, dict):
        return {**record, "status": "invalid", "error": "candidate report must be a JSON object"}
    return {**record, "status": "ok", "report": value}
