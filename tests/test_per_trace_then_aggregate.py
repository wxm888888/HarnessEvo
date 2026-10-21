from __future__ import annotations

import json
from pathlib import Path

import pytest



from harness_evolution.diagnosis.per_trace_then_aggregate import (  # noqa: E402
    PerTraceThenAggregateEvidence,
    _apply_outcome_pattern_limits,
    _build_aggregate_index,
    _copy_candidate_diagnosis_view,
    _count_jsonl,
    _load_trace_diagnosis,
    _render_aggregate_markdown,
)


def _write_case(
    root: Path,
    number: str,
    *,
    task: str,
    reward: object,
    with_trace: bool = True,
) -> None:
    directory = root / number
    directory.mkdir(parents=True)
    (directory / "sanitized_result.json").write_text(
        json.dumps({"task_name": task, "trial_name": "k01", "reward": reward}),
        encoding="utf-8",
    )
    (directory / "trajectory.json").write_text(
        json.dumps({"events": [{"type": "assistant"}] if with_trace else []}),
        encoding="utf-8",
    )
    (directory / "openai-messages.jsonl").write_text(
        '{"role":"user","content":"run"}\n' if with_trace else "",
        encoding="utf-8",
    )
    (directory / "openai-tools.json").write_text(
        json.dumps([{"type": "function", "function": {"name": "bash"}}])
        if with_trace
        else "",
        encoding="utf-8",
    )
    (directory / "verifier-summary.json").write_text(
        json.dumps({"reward": reward, "status": "ok"}) if with_trace else "",
        encoding="utf-8",
    )


def test_build_cases_and_coverage_marks_missing_outcomes(tmp_path: Path) -> None:
    _write_case(tmp_path, "001", task="passed", reward=1.0)
    _write_case(tmp_path, "002", task="failed", reward=0.0)
    _write_case(tmp_path, "003", task="missing", reward="missing", with_trace=False)
    _write_case(tmp_path, "004", task="failed_no_trace", reward=0.0, with_trace=False)

    evidence = PerTraceThenAggregateEvidence.__new__(PerTraceThenAggregateEvidence)
    evidence.options = {"success_threshold": 1.0}
    cases = evidence._build_cases(tmp_path)
    coverage = evidence._coverage(cases)

    assert [case["outcome"] for case in cases] == [
        "passed",
        "failed",
        "infrastructure_incomplete",
        "failed",
    ]
    assert coverage["total_trials"] == 4
    assert coverage["passed_trials"] == 1
    assert coverage["failed_trials"] == 2
    assert coverage["infrastructure_incomplete_trials"] == 1
    assert coverage["proposer_blocked"] is False
    assert "failed_no_trace::k01" in coverage["missing_trace_tasks"]
    assert "missing::k01" in coverage["incomplete_outcome_tasks"]


def test_trace_diagnosis_selection_can_include_success_and_failure(tmp_path: Path) -> None:
    _write_case(tmp_path, "001", task="passed", reward=1.0)
    _write_case(tmp_path, "002", task="failed", reward=0.0)

    evidence = PerTraceThenAggregateEvidence.__new__(PerTraceThenAggregateEvidence)
    evidence.options = {"analyze_passed": True, "analyze_failed": True}
    cases = evidence._build_cases(tmp_path)

    assert [case["task_name"] for case in evidence._limit_cases(cases)] == ["passed", "failed"]


def test_json_helpers_and_markdown_are_deterministic(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    messages.write_text('{"role":"user"}\n\n{"role":"assistant"}\n', encoding="utf-8")
    assert _count_jsonl(messages) == 2

    markdown = _render_aggregate_markdown(
        {
            "aggregation_status": "ok",
            "confidence": 0.8,
            "shared_patterns": [{"summary": "Repeated retry gap", "supporting_traces": ["a::k01"]}],
            "success_patterns": [],
            "task_specific_patterns": [],
            "harness_change_hypotheses": [],
        }
    )
    assert "# Aggregated terminal-task diagnosis" in markdown
    assert "Repeated retry gap" in markdown
    assert "a::k01" in markdown


def test_trace_diagnosis_is_loaded_from_fixed_json_file(tmp_path: Path) -> None:
    diagnosis = {
        "terminal_cause": "passed",
        "first_divergence_step": "none",
        "agent_mechanism": "verified output",
        "harness_relevance": "low",
        "success_patterns": [],
        "failure_patterns": [],
        "causal_chain": [],
        "evidence_refs": [],
        "confidence": 0.9,
    }
    output = tmp_path / "diagnosis.json"
    output.write_text(json.dumps(diagnosis), encoding="utf-8")
    assert _load_trace_diagnosis(output) == diagnosis

    output.write_text('{"terminal_cause":"incomplete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        _load_trace_diagnosis(output)


def test_trace_prompt_documents_source_inputs_outputs_and_file_protocol() -> None:
    evidence = PerTraceThenAggregateEvidence.__new__(PerTraceThenAggregateEvidence)
    prompt = evidence._trace_prompt(
        {
            "trace_id": "task::trial",
            "outcome": "failed",
            "reward": 0.0,
            "trace_status": "complete",
        }
    )
    assert "OpenCode was used as the harness" in prompt
    assert "do not assume that the task belongs to the Terminal-Bench" not in prompt
    assert "HarnessRL" not in prompt
    assert "verifier.txt" not in prompt
    assert "hidden tests" not in prompt
    assert "expected answers" not in prompt
    assert "checker internals" not in prompt
    assert "instruction.md: the original terminal task" in prompt
    assert "trajectory.json: the normalized chronological OpenCode event timeline" in prompt
    assert "openai-messages.jsonl: the preferred normalized OpenAI-compatible" in prompt
    assert "llm-context.jsonl: the raw provider-native cumulative" in prompt
    assert "openai-tools.json: the exact tool names" in prompt
    assert "verifier-summary.json: a sanitized outcome summary" in prompt
    assert "verifier-feedback.json: sanitized output from the verifier execution" in prompt
    assert "Do not infer a terminal cause that conflicts with this feedback" in prompt
    assert "do not substitute an unrelated warning or timeout" in prompt
    assert "Omit them from every diagnosis field" in prompt
    assert "causal_chain, and evidence_refs" in prompt
    assert "This is intervention relevance, not causal blame" in prompt
    assert "verification strategy" in prompt
    assert "Do not set none merely because the tools functioned" in prompt
    assert "medium means the trace shows a concrete agent mechanism" in prompt
    assert "high means the evidence directly and strongly connects" in prompt
    assert "For a failed trace, default to []" in prompt
    assert "include at most 1 only when the behavior clearly completed" in prompt
    assert "For a passed trace, include only behaviors that causally contributed" in prompt
    assert "include at most 3" in prompt
    assert "For a passed trace, default to []" in prompt
    assert "clearly evidenced near-miss" in prompt
    assert "diagnosis.json" in prompt
    assert "controller ignores JSON in the final assistant message" in prompt
    for field in (
        "terminal_cause (string)",
        "first_divergence_step (object or null)",
        "agent_mechanism (string)",
        "harness_relevance (object)",
        "success_patterns (array of strings)",
        "failure_patterns (array of strings)",
        "causal_chain (array of strings)",
        "evidence_refs (array of objects)",
        "confidence (number)",
    ):
        assert field in prompt


def test_aggregate_prompt_preserves_agent_policy_harness_relevance() -> None:
    evidence = PerTraceThenAggregateEvidence.__new__(PerTraceThenAggregateEvidence)
    prompt = evidence._aggregate_prompt(
        {
            "records": [
                {
                    "trace_id": "task-a::k01",
                    "task_name": "task-a",
                    "diagnosis_status": "ok",
                    "diagnosis_path": "per_trace/task-a.json",
                }
            ]
        }
    )

    assert "strength of evidence for a reusable intervention" in prompt
    assert "not as a claim that the harness caused the outcome" in prompt
    assert "verification, context handling, recovery, tool-use policy" in prompt
    assert "Read each listed diagnosis_path exactly once" in prompt
    assert "Diagnoses with level=none, missing relevance, invalid" in prompt
    assert "systemd/DBus/container-init limitations" in prompt
    assert "Do not copy, summarize, cite, count, or convert such content into any output field" in prompt
    assert "level=none cannot support a shared pattern or harness change" in prompt
    assert "at least two distinct task_name values" in prompt
    assert "Record relevant counterexamples in contradicting_traces" in prompt
    assert "shared_patterns (array, at most 3)" in prompt
    assert "success_patterns (array, at most 3)" in prompt
    assert "harness_change_hypotheses (array, at most 3)" in prompt
    assert "task_specific_patterns are documentation only" in prompt
    assert "verifier.txt" not in prompt
    assert "checker internals" not in prompt


def test_aggregate_index_is_lightweight_and_excludes_unavailable_or_none_diagnoses() -> None:
    coverage = {"diagnosed_trials": 4, "diagnosis_fallbacks": 1}
    index = _build_aggregate_index(
        [
            {
                "trace_id": "task-a::k01",
                "task_name": "task-a",
                "trial_name": "k01",
                "outcome": "failed",
                "reward": 0.0,
                "diagnosis_status": "ok",
                "diagnosis_path": "per_trace/task-a.json",
                "diagnosis": {
                    "terminal_cause": "agent stopped early",
                    "harness_relevance": {"level": "medium", "reason": "reusable"},
                },
                "evidence_refs": {"trajectory": "evidence/001/trajectory.json"},
            },
            {
                "trace_id": "task-b::k01",
                "task_name": "task-b",
                "trial_name": "k01",
                "outcome": "failed",
                "reward": 0.0,
                "diagnosis_status": "llm_diagnosis_unavailable",
                "diagnosis_path": "per_trace/task-b.json",
                "diagnosis": {"terminal_cause": "diagnosis_unavailable"},
            },
            {
                "trace_id": "task-c::k01",
                "task_name": "task-c",
                "trial_name": "k01",
                "outcome": "passed",
                "reward": 1.0,
                "diagnosis_status": "ok",
                "diagnosis_path": "per_trace/task-c.json",
                "diagnosis": {
                    "terminal_cause": "passed",
                    "harness_relevance": {"level": "none", "reason": "task specific"},
                },
            },
            {
                "trace_id": "task-d::k01",
                "task_name": "task-d",
                "trial_name": "k01",
                "outcome": "failed",
                "reward": 0.0,
                "diagnosis_status": "ok",
                "diagnosis_path": "per_trace/task-d.json",
                "diagnosis": {
                    "terminal_cause": "failed",
                    "harness_relevance": "medium",
                },
            },
        ],
        coverage,
    )

    assert index["schema_version"] == "tb21-trace-diagnosis-index-v2"
    assert index["counts"] is coverage
    assert [record["trace_id"] for record in index["records"]] == ["task-a::k01"]
    assert index["records"][0]["diagnosis_path"] == "per_trace/task-a.json"
    assert "diagnosis" not in index["records"][0]
    assert "evidence_refs" not in index["records"][0]


def test_candidate_view_contains_only_allowlisted_diagnoses(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    per_trace = result_dir / "per_trace"
    raw = result_dir / "raw"
    per_trace.mkdir(parents=True)
    raw.mkdir()
    index = {
        "records": [
            {
                "trace_id": "eligible::k01",
                "diagnosis_path": "per_trace/eligible.json",
            }
        ]
    }
    for name, payload in (
        ("index.json", index),
        ("coverage.json", {"diagnosed_trials": 2}),
        ("aggregate.json", {"aggregation_status": "ok"}),
        ("aggregate.md", "duplicate human rendering"),
    ):
        path = result_dir / name
        path.write_text(
            json.dumps(payload) if not isinstance(payload, str) else payload,
            encoding="utf-8",
        )
    (per_trace / "eligible.json").write_text('{"diagnosis_status":"ok"}', encoding="utf-8")
    (per_trace / "excluded-none.json").write_text(
        '{"diagnosis_status":"ok","harness_relevance":"none"}',
        encoding="utf-8",
    )
    (raw / "eligible.log").write_text("raw model output", encoding="utf-8")

    destination = tmp_path / "candidate" / "diagnosis"
    _copy_candidate_diagnosis_view(
        result_dir=result_dir,
        index=index,
        destination=destination,
    )

    assert (destination / "index.json").is_file()
    assert (destination / "coverage.json").is_file()
    assert (destination / "aggregate.json").is_file()
    assert (destination / "per_trace/eligible.json").is_file()
    assert not (destination / "per_trace/excluded-none.json").exists()
    assert not (destination / "raw").exists()
    assert not (destination / "aggregate.md").exists()


def test_pattern_limits_are_enforced_by_outcome() -> None:
    diagnosis = {
        "success_patterns": ["s1", "s2", "s3", "s4"],
        "failure_patterns": ["f1", "f2", "f3", "f4"],
    }

    failed, failed_meta = _apply_outcome_pattern_limits(diagnosis, outcome="failed")
    assert failed["success_patterns"] == ["s1"]
    assert failed["failure_patterns"] == ["f1", "f2", "f3"]
    assert failed_meta["truncated"] is True

    passed, passed_meta = _apply_outcome_pattern_limits(diagnosis, outcome="passed")
    assert passed["success_patterns"] == ["s1", "s2", "s3"]
    assert passed["failure_patterns"] == ["f1"]
    assert passed_meta["counts"]["failure_patterns"]["limit"] == 1


@pytest.mark.parametrize("phase", ["trace", "aggregate"])
def test_opencode_analyzer_preserves_configured_env_in_real_subprocess(tmp_path, phase):
    import sys
    import shlex
    script = tmp_path / "capture.py"
    script.write_text("import os,json; print(json.dumps({k:os.environ.get(k) for k in ['OPENCODE_DISABLE_MODELS_FETCH','OPENCODE_MODELS_PATH','XDG_CACHE_HOME']}))")
    evidence = PerTraceThenAggregateEvidence.__new__(PerTraceThenAggregateEvidence)
    configured = {
        "client_name": "opencode_bun_dev_client",
        "opencode_repo_path": str(tmp_path),
        "dev_run_command_prefix": f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
        "client_init": {"dev_run_env": {
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_MODELS_PATH": str(tmp_path / "models.json"),
            "XDG_CACHE_HOME": "stale-shared-cache",
        }},
    }
    scoped_cache = str(tmp_path / phase / "cache")
    client = evidence._build_analyzer_client(configured, {"XDG_CACHE_HOME": scoped_cache})
    result = client.run_instruction_detailed(workspace=tmp_path, instruction="probe", request_timeout_sec=20)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
        "OPENCODE_MODELS_PATH": str(tmp_path / "models.json"),
        "XDG_CACHE_HOME": scoped_cache,
    }
