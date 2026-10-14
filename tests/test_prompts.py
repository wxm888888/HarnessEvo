"""Prompt templates render exactly the recorded prompts.

``fixtures/prompts_golden.json`` was recorded from the code before the prompts
moved into ``*/prompts/*.md``. A failure here means a prompt changed; if that
was intended, regenerate the fixture and review the diff.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_evolution.core.prompts import render_prompt
from harness_evolution.core.types import EvidenceBuildResult, MemoryBuildResult
from harness_evolution.diagnosis.agent_analyzed import AgentAnalyzedEvidence
from harness_evolution.diagnosis.per_trace_then_aggregate import PerTraceThenAggregateEvidence
from harness_evolution.proposer.requirement import _candidate_requirement

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "prompts_golden.json").read_text(encoding="utf-8"))

TRACE_CASES = [
    {"trace_id": "task_000432_f1a8117e::sample-task_000432_f1a8117e-a0", "outcome": "failed", "reward": 0.0, "trace_status": "complete"},
    {"trace_id": "t::b", "outcome": "passed", "reward": 1.0, "trace_status": "complete"},
]
ADDENDUMS = {
    "evidence_per_trace_then_aggregate.md": ("harness_evolution.diagnosis", "pertracethenaggregateevidence"),
    "evidence_agent_analyzed.md": ("harness_evolution.diagnosis", "agentanalyzedevidence"),
    "evidence_instruction_only.md": ("harness_evolution.diagnosis", "instructiononlyevidence"),
    "evidence_sanitized_instruction.md": ("harness_evolution.diagnosis", "sanitizedinstructionevidence"),
}
ROOT = ".harness_evolver/terminal_bench_artifacts"


@pytest.mark.parametrize("index", range(len(TRACE_CASES)))
def test_trace_prompt(index: int) -> None:
    assert PerTraceThenAggregateEvidence._trace_prompt(None, TRACE_CASES[index]) == GOLDEN[f"trace_prompt_{index}"]


@pytest.mark.parametrize("count", [0, 3])
def test_aggregate_and_analysis_prompts(count: int) -> None:
    assert PerTraceThenAggregateEvidence._aggregate_prompt(None, {"records": [{}] * count}) == GOLDEN[f"aggregate_prompt_{count}"]
    assert AgentAnalyzedEvidence._analysis_prompt(None, {"failed_trials": [{}] * count}) == GOLDEN[f"analysis_prompt_{count}"]


@pytest.mark.parametrize("template", sorted(ADDENDUMS))
@pytest.mark.parametrize("coverage", ["full", "empty"])
def test_evidence_addendums(template: str, coverage: str) -> None:
    package, name = ADDENDUMS[template]
    counts = {"diagnosed_trials": 4, "passed_trials": 2, "failed_trials": 2} if coverage == "full" else {}
    values = {"name": name, "root": ROOT}
    if template == "evidence_per_trace_then_aggregate.md":
        values.update({key: counts.get(key, 0) for key in ("diagnosed_trials", "passed_trials", "failed_trials")})
    assert render_prompt(package, template, **values) == GOLDEN[f"{template}:{coverage}"]


def test_memory_addendum() -> None:
    rendered = render_prompt("harness_evolution.memory", "accepted_lineage.md", relative=".harness_evolver/memory/accepted-lineage.jsonl")
    assert rendered == GOLDEN["accepted_lineage.md:full"]


MEMORIES = {
    "none": None,
    "empty": MemoryBuildResult(requirement_addendum="", metadata={}),
    "mem": MemoryBuildResult(requirement_addendum="MEMORY ADDENDUM", metadata={}),
}
SKILLS = {
    "off": {"enabled": False},
    "two": {"enabled": True, "skill_names": ["opencode-architecture-evolution", "b"], "workspace_dir": ".harness_evolver/skills/"},
    "nodir": {"skill_names": ["x"]},
}
FEEDBACK = {"nofb": "", "fb": "attempt 1 failed:\n\ntypecheck error"}


@pytest.mark.parametrize("memory", sorted(MEMORIES))
@pytest.mark.parametrize("skills", sorted(SKILLS))
@pytest.mark.parametrize("feedback", sorted(FEEDBACK))
def test_candidate_requirement(memory: str, skills: str, feedback: str) -> None:
    rendered = _candidate_requirement(
        evidence=EvidenceBuildResult(requirement_addendum="EVIDENCE ADDENDUM\nsecond line", metadata={}),
        skill_copy=SKILLS[skills],
        validation_feedback=FEEDBACK[feedback],
        memory=MEMORIES[memory],
    )
    assert rendered == GOLDEN[f"candidate:{memory}:{skills}:{feedback}"]
