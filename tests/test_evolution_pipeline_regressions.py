from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    openai_module.OpenAI = OpenAI
    sys.modules["openai"] = openai_module
if "tomllib" not in sys.modules:
    try:
        import tomllib  # noqa: F401
    except ImportError:
        sys.modules["tomllib"] = types.ModuleType("tomllib")

from harness_evolution.loop.modular import (  # noqa: E402
    _build_memory_strategy,
    _candidate_requirement,
    _load_candidate_report,
    _memory_section,
    link_shared_node_modules,
)
from harness_evolution.core.types import (  # noqa: E402
    Candidate,
    DiffStats,
    EvidenceBuildResult,
    MemoryBuildResult,
    RuntimeConfig,
    ValidationResult,
)
from harness_evolution.loop.two_sided import (  # noqa: E402
    _ensure_parent_binary,
    _load_completed_rejected_gate,
    _persist_parent_binary,
    _prune_completed_step_artifacts,
    _repeat_counts,
)



def test_dependency_cache_bridges_source_and_worktree_filesystems(tmp_path: Path) -> None:
    source = tmp_path / "image" / "opencode"
    package = source / "packages" / "core"
    dependency = source / "node_modules" / "dep"
    package.mkdir(parents=True)
    dependency.mkdir(parents=True)
    (dependency / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    (source / "node_modules" / "workspace-core").symlink_to(package, target_is_directory=True)

    group = tmp_path / "gpfs" / "run-group"
    workspace = group / "worktrees" / "candidate-1"
    (workspace / "packages" / "core").mkdir(parents=True)

    linked = link_shared_node_modules(workspace, source)

    cache = group / ".node_modules_cache" / "node_modules"
    assert linked == ["node_modules"]
    assert (cache / "dep" / "index.js").read_text(encoding="utf-8") == "module.exports = 1\n"
    assert (workspace / "node_modules" / "dep" / "index.js").stat().st_ino == (
        cache / "dep" / "index.js"
    ).stat().st_ino
    assert (workspace / "node_modules" / "workspace-core").resolve() == (
        workspace / "packages" / "core"
    ).resolve()


def test_candidate_prompt_requires_skill_runtime_change_and_report() -> None:
    prompt = _candidate_requirement(
        evidence=EvidenceBuildResult(
            requirement_addendum="FILTERED DIAGNOSIS VIEW",
            metadata={},
        ),
        skill_copy={
            "workspace_dir": ".harness_evolver/skills",
            "skill_names": ["opencode-architecture-evolution"],
        },
        validation_feedback="",
    )

    assert ".harness_evolver/skills/opencode-architecture-evolution/SKILL.md" in prompt
    assert "If skill loading fails" in prompt
    assert "FILTERED DIAGNOSIS VIEW" in prompt
    assert "one hypothesis or a coherent combination of multiple hypotheses" in prompt
    assert "Problem-solving skills are located in `.opencode/skills/`" in prompt
    assert "You may modify code and skills together" in prompt
    assert "skill_assessment" in prompt
    assert ".harness_evolver/candidate-report.json" in prompt
    assert "The controller will run typecheck and build validation" in prompt


def _runtime_config(tmp_path: Path, *, memory: dict | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        raw={},
        initial_harness_path=str(tmp_path),
        initial_base_commit="base",
        iterations=1,
        rollouts_per_attempt=1,
        workspace_root=tmp_path / "workspaces",
        report_root=tmp_path / "reports",
        evolver_timeout_sec=60,
        terminal_bench={},
        build={},
        opencode_evolver={},
        task_selection={},
        evidence={},
        candidate_selection={},
        validation={},
        memory=memory or {},
    )


def test_memory_none_preserves_prompt_and_creates_no_files(tmp_path: Path) -> None:
    evidence = EvidenceBuildResult(requirement_addendum="EVIDENCE", metadata={})
    baseline_prompt = _candidate_requirement(
        evidence=evidence,
        skill_copy={},
        validation_feedback="",
    )
    disabled_prompt = _candidate_requirement(
        evidence=evidence,
        skill_copy={},
        validation_feedback="",
        memory=MemoryBuildResult(requirement_addendum="", metadata={}),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    strategy = _build_memory_strategy(
        _runtime_config(tmp_path, memory={"method": "none"})
    )

    materialized = strategy.materialize(workspace=workspace)

    assert _memory_section({}) == {}
    assert _memory_section({"memory": "none"}) == {"method": "none"}
    assert strategy.enabled is False
    assert materialized == MemoryBuildResult(requirement_addendum="", metadata={})
    assert baseline_prompt == disabled_prompt
    assert list(workspace.iterdir()) == []


def test_gate_repeats_can_differ_from_held_in_sampling() -> None:
    legacy = types.SimpleNamespace(
        repeats=2,
        sampling_repeats=None,
        gate_repeats=None,
    )
    split = types.SimpleNamespace(
        repeats=1,
        sampling_repeats=1,
        gate_repeats=2,
    )

    assert _repeat_counts(legacy) == (2, 2)
    assert _repeat_counts(split) == (1, 2)


def test_accepted_lineage_memory_materializes_and_deduplicates_resume(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    config = _runtime_config(
        tmp_path,
        memory={"method": "accepted_lineage", "options": {}},
    )
    strategy = _build_memory_strategy(config)
    candidate_report_path = workspace / ".harness_evolver" / "candidate-report.json"
    candidate_report_path.parent.mkdir(parents=True)
    candidate_report_path.write_text(
        json.dumps(
            {
                "behavior_changes": [
                    "Detect repeated exploration and request a concrete artifact."
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = Candidate(
        candidate_idx=2,
        attempt_idx=1,
        rollout_idx=2,
        workspace=workspace,
        branch="candidate-2",
        commit="b" * 40,
        commit_created=True,
        changed_files=["packages/opencode/src/session/prompt.ts"],
        summary="opencode client evolve applied",
        trajectory=[],
        evidence={},
        skill_copy={},
        evolver_metadata={"resumed_partial_step": True},
        validation=ValidationResult(enabled=True, passed=True),
        diff_stats=DiffStats(files=1, added=4, deleted=1, score=1005),
    )

    first = strategy.record_promotion(
        step=3,
        parent_commit="a" * 40,
        winner=candidate,
        baseline_score={"passed_attempts": 27, "total_attempts": 89, "rate": 27 / 89, "complete": True},
        winner_score={"passed_attempts": 29, "total_attempts": 89, "rate": 29 / 89, "complete": True},
    )
    duplicate = strategy.record_promotion(
        step=3,
        parent_commit="a" * 40,
        winner=candidate,
        baseline_score={},
        winner_score={},
    )
    materialized = strategy.materialize(workspace=workspace)

    storage = tmp_path / "reports" / "memory" / "accepted-lineage.jsonl"
    workspace_memory = workspace / ".harness_evolver" / "memory" / "accepted-lineage.jsonl"
    records = [json.loads(line) for line in storage.read_text(encoding="utf-8").splitlines()]
    assert first == duplicate
    assert len(records) == 1
    assert records[0]["summary"] == "Detect repeated exploration and request a concrete artifact."
    assert records[0]["changed_files"] == ["packages/opencode/src/session/prompt.ts"]
    assert records[0]["baseline_score"]["passed_attempts"] == 27
    assert records[0]["winner_score"]["passed_attempts"] == 29
    assert workspace_memory.read_text(encoding="utf-8") == storage.read_text(encoding="utf-8")
    assert materialized.metadata["record_count"] == 1
    assert "git diff <parent_commit>..<winner_commit>" in materialized.requirement_addendum
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".harness_evolver/memory/accepted-lineage.jsonl"],
        cwd=workspace,
        check=False,
    )
    assert ignored.returncode == 0


def test_candidate_report_is_loaded_without_becoming_a_candidate_file(tmp_path: Path) -> None:
    report_path = tmp_path / ".harness_evolver" / "candidate-report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "harness-candidate-report-v1",
                "selected_hypotheses": ["improve verification"],
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_candidate_report(tmp_path)

    assert loaded["status"] == "ok"
    assert loaded["report"]["selected_hypotheses"] == ["improve verification"]


def test_missing_afs_parent_binary_is_rebuilt_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    built = tmp_path / "build" / "opencode"
    built.parent.mkdir()
    built.write_bytes(b"binary")
    built.chmod(0o755)
    monkeypatch.setattr(
        "harness_evolution.loop.two_sided.evolver_build",
        lambda runtime_config, source: built,
    )
    monkeypatch.setattr(
        "harness_evolution.loop.two_sided._build_runtime_config",
        lambda gate_config: object(),
    )

    rebuilt = _ensure_parent_binary(
        {"heldout_gate": {"opensandbox": {"opencode_setup": "afs_bin"}}},
        workspace,
        "a" * 40,
        str(tmp_path / "deleted" / "opencode"),
    )
    durable = Path(_persist_parent_binary(tmp_path / "run", "a" * 40, rebuilt))

    assert durable.read_bytes() == b"binary"
    assert durable.stat().st_mode & 0o111
    assert durable == tmp_path / "run" / "parent-binaries" / ("a" * 40) / "opencode"


def test_retention_keeps_artifacts_containing_active_paths(tmp_path: Path) -> None:
    run = tmp_path / "run"
    active = run / "step-01" / "static-workspace" / "artifacts" / "opencode"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"active")
    disposable = run / "step-01" / "validation-jobs" / "trial.json"
    disposable.parent.mkdir(parents=True)
    disposable.write_text("{}", encoding="utf-8")

    _prune_completed_step_artifacts(
        run,
        completed_step=3,
        retention=2,
        protected_paths=(active,),
    )

    assert active.is_file()
    assert not disposable.exists()


def test_resume_reuses_a_complete_rejected_gate(tmp_path: Path) -> None:
    step = tmp_path / "step-04"
    step.mkdir()
    gate = {
        "accepted": False,
        "winner_commit": "",
        "parent_commit": "parent",
        "held_in_tasks": ["in-1"],
        "held_out_tasks": ["out-1"],
        "baseline": {"held_out": {"complete": True}},
        "candidates": {"candidate": {"held_out": {"complete": True}}},
        "reason": "no candidate passed held-out non-regression gate",
    }
    (step / "gate.json").write_text(json.dumps(gate), encoding="utf-8")

    recovered = _load_completed_rejected_gate(
        step,
        parent_commit="parent",
        held_in=["in-1"],
        held_out=["out-1"],
    )

    assert recovered == gate
