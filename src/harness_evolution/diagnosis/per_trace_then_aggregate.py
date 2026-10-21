from __future__ import annotations

import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from harness_evolution.agents import CLIENT_REGISTRY
from harness_evolution.core.prompts import render_prompt
from harness_evolution.core.types import (
    WORKSPACE_ARTIFACT_ROOT,
    CollectionArtifact,
    EvidenceBuildResult,
    RuntimeConfig,
    load_json_object,
    repo_relative_path,
)

from .agent_analyzed import AgentAnalyzedEvidence
from harness_evolution.traces.preflight import filter_collection_artifacts

# Re-exported after the phase-2 split; new code should import from the owning module.
from harness_evolution.diagnosis.aggregate import _TRACE_DIAGNOSIS_FILENAME, _TRACE_DIAGNOSIS_PROTOCOL_VERSION, _TRACE_DIAGNOSIS_REQUIRED_KEYS, _apply_outcome_pattern_limits, _build_aggregate_index, _copy_candidate_diagnosis_view, _count_jsonl, _diagnosis_harness_relevance_level, _load_json_any, _load_json_object_or_empty, _load_trace_diagnosis, _relative_ref, _render_aggregate_markdown, _safe_name, _write_json  # noqa: F401


@dataclass
class _DiagnosisCache:
    cache_key: str
    result_dir: Path
    coverage: dict[str, Any]
    index: dict[str, Any]
    aggregate: dict[str, Any]


class PerTraceThenAggregateEvidence(AgentAnalyzedEvidence):
    """Diagnose each complete trace, then aggregate the structured diagnoses.

    This strategy intentionally reuses the local TB21 evidence writer from
    ``agent_analyzed``.  It does not import or depend on Self-Harness code.
    The analyzer receives one numbered trace at a time; the aggregator only
    receives compact per-trace JSON plus pointers back to the evidence files.
    """

    name = "per_trace_then_aggregate"

    def __init__(self, options: dict[str, Any], config: RuntimeConfig):
        super().__init__(options, config)
        self._diagnosis_cache: _DiagnosisCache | None = None

    def materialize(
        self,
        *,
        artifacts: list[CollectionArtifact],
        workspace: Path,
        task_cache_root: Path,
        current_harness_path: str | Path | None = None,
    ) -> EvidenceBuildResult:
        artifacts, preflight_decisions = filter_collection_artifacts(artifacts)
        if not artifacts:
            raise ValueError(
                "evidence preflight found no complete verifier-backed traces; "
                "environment and infrastructure failures are not evolution evidence"
            )
        root = workspace / WORKSPACE_ARTIFACT_ROOT
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        records = self._write_numbered_evidence(
            artifacts=artifacts,
            root=root,
            task_cache_root=task_cache_root,
            include_trajectory=True,
        )
        cache = self._get_or_run_diagnosis(
            artifacts=artifacts,
            source_evidence_root=root,
            records=records,
            current_harness_path=current_harness_path,
        )
        if cache.aggregate.get("aggregation_status") != "ok":
            raise ValueError(
                "per-trace diagnosis aggregation is unavailable; refusing to generate a candidate. "
                f"See {cache.result_dir / 'aggregate.json'} and "
                f"{cache.result_dir / 'aggregate_error.log'}"
            )
        # The numbered source trajectories are needed to produce the diagnoses,
        # but the candidate should only see the filtered diagnosis view.  Keep
        # the complete source material in the persistent cache for auditing and
        # rebuild the workspace artifact root from the aggregate eligibility
        # index below.
        shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        workspace_diagnosis = root / "diagnosis"
        _copy_candidate_diagnosis_view(
            result_dir=cache.result_dir,
            index=cache.index,
            destination=workspace_diagnosis,
        )

        if self._bool_option("require_complete_failed_traces", True):
            blocked = cache.coverage.get("proposer_blocked", False)
            if blocked:
                raise ValueError(
                    "per-trace diagnosis blocked proposer: "
                    "there are no analyzable complete traces; "
                    f"see {workspace_diagnosis / 'coverage.json'}"
                )

        return EvidenceBuildResult(
            requirement_addendum=render_prompt(
                __package__,
                "evidence_per_trace_then_aggregate.md",
                name=self.name,
                root=WORKSPACE_ARTIFACT_ROOT,
                diagnosed_trials=cache.coverage.get("diagnosed_trials", 0),
                passed_trials=cache.coverage.get("passed_trials", 0),
                failed_trials=cache.coverage.get("failed_trials", 0),
            ),
            metadata={
                "method": self.name,
                "artifact_root": str(root),
                "artifact_count": len(records),
                "records": cache.index.get("records", []),
                "preflight": {
                    "eligible_count": len(artifacts),
                    "excluded_count": sum(not item.eligible for item in preflight_decisions),
                },
                "diagnosis": {
                    "result_dir": str(cache.result_dir),
                    "index_path": str(workspace_diagnosis / "index.json"),
                    "coverage_path": str(workspace_diagnosis / "coverage.json"),
                    "aggregate_path": str(workspace_diagnosis / "aggregate.json"),
                    "aggregate_markdown_path": str(cache.result_dir / "aggregate.md"),
                    "cache_key": cache.cache_key,
                    "coverage": cache.coverage,
                },
            },
        )

    def _get_or_run_diagnosis(
        self,
        *,
        artifacts: list[CollectionArtifact],
        source_evidence_root: Path,
        records: list[dict[str, Any]],
        current_harness_path: str | Path | None,
    ) -> _DiagnosisCache:
        source_cache_key = self._cache_key(
            artifacts, current_harness_path=current_harness_path
        )
        cache_key = sha1(
            f"{source_cache_key}:{_TRACE_DIAGNOSIS_PROTOCOL_VERSION}".encode("utf-8")
        ).hexdigest()
        if self._diagnosis_cache is not None and self._diagnosis_cache.cache_key == cache_key:
            return self._diagnosis_cache

        configured_root = self.options.get("result_root")
        diagnosis_root = (
            repo_relative_path(configured_root)
            if configured_root
            else self.config.report_root / "diagnosis"
        )
        result_dir = diagnosis_root / cache_key[:12]
        if self._bool_option("force_rerun", False) and result_dir.exists():
            shutil.rmtree(result_dir)
        elif (
            (result_dir / "per_trace").is_dir()
            and all(
                (result_dir / name).is_file()
                for name in ("index.json", "coverage.json", "aggregate.json", "aggregate.md")
            )
        ):
            coverage = _load_json_object_or_empty(result_dir / "coverage.json")
            index = _load_json_object_or_empty(result_dir / "index.json")
            aggregate = _load_json_object_or_empty(result_dir / "aggregate.json")
            cache = _DiagnosisCache(
                cache_key=cache_key,
                result_dir=result_dir,
                coverage=coverage,
                index=index,
                aggregate=aggregate,
            )
            self._diagnosis_cache = cache
            return cache
        result_dir.mkdir(parents=True, exist_ok=True)
        persistent_evidence = result_dir / "evidence"
        if persistent_evidence.exists():
            shutil.rmtree(persistent_evidence)
        shutil.copytree(source_evidence_root, persistent_evidence)

        cases = self._build_cases(persistent_evidence)
        coverage = self._coverage(cases)
        _write_json(result_dir / "coverage.json", coverage)

        per_trace_dir = result_dir / "per_trace"
        raw_dir = result_dir / "raw"
        per_trace_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        diagnosis_config = self._resolved_analyzer_config(
            cache_key, current_harness_path=current_harness_path
        )
        env, analyzer_config_path = self._prepare_analyzer_environment(diagnosis_config)
        coverage["analyzer_config_path"] = str(analyzer_config_path)
        _write_json(result_dir / "coverage.json", coverage)

        cases_to_analyze = self._limit_cases(cases)
        diagnoses = self._run_per_trace_diagnoses(
            cases=cases_to_analyze,
            evidence_root=persistent_evidence,
            result_dir=result_dir,
            per_trace_dir=per_trace_dir,
            raw_dir=raw_dir,
            analyzer_config=diagnosis_config,
            shared_env=env,
        )
        coverage["diagnosed_trials"] = len(diagnoses)
        coverage["diagnosis_fallbacks"] = sum(
            item.get("diagnosis_status") != "ok" for item in diagnoses
        )
        coverage["missing_diagnoses"] = [
            case["trace_id"]
            for case in cases_to_analyze
            if case["trace_id"] not in {item["trace_id"] for item in diagnoses}
        ]

        index = _build_aggregate_index(diagnoses, coverage)
        coverage["aggregate_eligible_diagnoses"] = len(index["records"])
        coverage["aggregate_excluded_diagnoses"] = len(diagnoses) - len(index["records"])
        coverage["aggregate_excluded_none_or_invalid_relevance"] = sum(
            item.get("diagnosis_status") == "ok"
            and _diagnosis_harness_relevance_level(item) not in {"low", "medium", "high"}
            for item in diagnoses
        )
        index["counts"] = coverage
        _write_json(result_dir / "index.json", index)
        _write_json(result_dir / "coverage.json", coverage)

        aggregate = self._run_aggregate(
            result_dir=result_dir,
            diagnosis_config=diagnosis_config,
            shared_env=env,
            index=index,
        )
        _write_json(result_dir / "aggregate.json", aggregate)
        (result_dir / "aggregate.md").write_text(
            _render_aggregate_markdown(aggregate), encoding="utf-8"
        )

        cache = _DiagnosisCache(
            cache_key=cache_key,
            result_dir=result_dir,
            coverage=coverage,
            index=index,
            aggregate=aggregate,
        )
        self._diagnosis_cache = cache
        return cache

    def _build_cases(self, evidence_root: Path) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for directory in sorted(evidence_root.glob("[0-9][0-9][0-9]")):
            result = load_json_object(directory / "sanitized_result.json")
            trajectory = _load_json_any(directory / "trajectory.json")
            message_count = _count_jsonl(directory / "openai-messages.jsonl")
            events = trajectory.get("events", []) if isinstance(trajectory, dict) else trajectory
            event_count = len(events) if isinstance(events, list) else 0
            # A legacy ``agent.remote.jsonl``/trajectory alone is useful for
            # debugging, but it is not a complete evolution input.  Require
            # the normalized OpenAI messages, tool schema, and verifier
            # summary alongside the timeline before calling the trace
            # analyzable.
            required_artifacts = {
                name: (directory / name).is_file() and (directory / name).stat().st_size > 0
                for name in (
                    "trajectory.json",
                    "openai-messages.jsonl",
                    "openai-tools.json",
                    "verifier-summary.json",
                )
            }
            trace_status = (
                "complete"
                if (event_count or message_count) and all(required_artifacts.values())
                else "missing"
            )
            reward = result.get("reward")
            try:
                reward_value = float(reward)
            except (TypeError, ValueError):
                reward_value = None
            threshold = float(self.options.get("success_threshold", 1.0))
            if reward_value is None:
                outcome = "infrastructure_incomplete"
            elif reward_value >= threshold:
                outcome = "passed"
            else:
                outcome = "failed"
            task_name = str(result.get("task_name") or directory.name)
            trial_name = str(result.get("trial_name") or "k01")
            trace_id = f"{task_name}::{trial_name}"
            cases.append(
                {
                    "trace_id": trace_id,
                    "trial_dir": directory.name,
                    "task_name": task_name,
                    "trial_name": trial_name,
                    "outcome": outcome,
                    "reward": reward,
                    "trace_status": trace_status,
                    "step_count": event_count,
                    "message_count": message_count,
                    "event_count": event_count,
                    "required_artifacts": required_artifacts,
                    "missing_artifacts": sorted(
                        name for name, present in required_artifacts.items() if not present
                    ),
                    "evidence_dir": str(directory),
                    "trajectory_path": str(directory / "trajectory.json"),
                    "openai_messages_path": str(directory / "openai-messages.jsonl")
                    if (directory / "openai-messages.jsonl").exists()
                    else "",
                    "verifier_summary_path": str(directory / "verifier-summary.json")
                    if (directory / "verifier-summary.json").exists()
                    else "",
                    "verifier_feedback_path": str(directory / "verifier-feedback.json")
                    if (directory / "verifier-feedback.json").exists()
                    else "",
                }
            )
        return cases

    def _coverage(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        failed = [case for case in cases if case["outcome"] == "failed"]
        passed = [case for case in cases if case["outcome"] == "passed"]
        incomplete = [case for case in cases if case["outcome"] == "infrastructure_incomplete"]
        failed_without_trace = [
            case for case in failed if case["trace_status"] != "complete"
        ]
        complete_traces = sum(case["trace_status"] == "complete" for case in cases)
        missing_artifacts = [
            {
                "trace_id": case["trace_id"],
                "files": case.get("missing_artifacts", []),
            }
            for case in cases
            if case.get("missing_artifacts")
        ]
        return {
            "schema_version": "tb21-trace-coverage-v1",
            "total_trials": len(cases),
            "passed_trials": len(passed),
            "failed_trials": len(failed),
            "infrastructure_incomplete_trials": len(incomplete),
            "complete_traces": complete_traces,
            "failed_without_complete_trace": len(failed_without_trace),
            "missing_trace_tasks": [case["trace_id"] for case in failed_without_trace],
            "missing_artifacts": missing_artifacts,
            "incomplete_outcome_tasks": [case["trace_id"] for case in incomplete],
            "diagnosed_trials": 0,
            "diagnosis_fallbacks": 0,
            "missing_diagnoses": [],
            # Missing traces are infrastructure/collection failures, not agent
            # evidence.  Keep them in coverage and in the index as
            # ``not_analyzable`` records, but let the proposer use the other
            # complete passed/failed traces.  Block only when there is no
            # analyzable trace at all; the old any-missing policy turned a few
            # sandbox disconnects into an all-or-nothing no-op candidate.
            "proposer_blocked": complete_traces == 0,
        }

    def _limit_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # The loop filters these before materialization. Keep this second
        # guard so a partially written artifact can never reach an analyzer.
        selected = [
            case
            for case in cases
            if case.get("trace_status") == "complete"
            and case.get("outcome") in {"passed", "failed"}
            and case.get("analysis_eligible", True)
        ]
        if not self._bool_option("analyze_passed", True):
            selected = [case for case in selected if case["outcome"] != "passed"]
        if not self._bool_option("analyze_failed", True):
            selected = [case for case in selected if case["outcome"] != "failed"]
        max_cases = self.options.get("max_trace_diagnoses")
        if max_cases is not None:
            selected = selected[: max(0, int(max_cases))]
        return selected

    def _run_per_trace_diagnoses(
        self,
        *,
        cases: list[dict[str, Any]],
        evidence_root: Path,
        result_dir: Path,
        per_trace_dir: Path,
        raw_dir: Path,
        analyzer_config: dict[str, Any],
        shared_env: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not cases:
            return []
        workers = max(1, int(self.options.get("diagnosis_concurrency", 8)))
        outputs: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as pool:
            futures = {
                pool.submit(
                    self._run_one_trace_diagnosis,
                    case=case,
                    evidence_root=evidence_root,
                    result_dir=result_dir,
                    per_trace_dir=per_trace_dir,
                    raw_dir=raw_dir,
                    analyzer_config=analyzer_config,
                    shared_env=shared_env,
                ): case
                for case in cases
            }
            for future in as_completed(futures):
                outputs.append(future.result())
        return sorted(outputs, key=lambda item: item["trace_id"])

    def _run_one_trace_diagnosis(
        self,
        *,
        case: dict[str, Any],
        evidence_root: Path,
        result_dir: Path,
        per_trace_dir: Path,
        raw_dir: Path,
        analyzer_config: dict[str, Any],
        shared_env: dict[str, str],
    ) -> dict[str, Any]:
        safe_id = _safe_name(case["trace_id"])
        output_path = per_trace_dir / f"{safe_id}.json"
        raw_path = raw_dir / f"{safe_id}.log"
        base = {
            "schema_version": "tb21-trace-diagnosis-v1",
            "trace_id": case["trace_id"],
            "task_name": case["task_name"],
            "trial_name": case["trial_name"],
            "outcome": case["outcome"],
            "reward": case["reward"],
            "trace_status": case["trace_status"],
            "trace_stats": {
                "step_count": case["step_count"],
                "message_count": case["message_count"],
                "event_count": case["event_count"],
            },
            "evidence_refs": {
                "trajectory": _relative_ref(case["trajectory_path"], evidence_root),
                "openai_messages": _relative_ref(case["openai_messages_path"], evidence_root),
                "verifier_summary": _relative_ref(case["verifier_summary_path"], evidence_root),
                "verifier_feedback": _relative_ref(case["verifier_feedback_path"], evidence_root),
            },
            "diagnosis_path": f"per_trace/{safe_id}.json",
            "raw_output_path": f"raw/{safe_id}.log",
        }
        if case["trace_status"] != "complete":
            base.update(
                {
                    "diagnosis_status": "not_analyzable",
                    "diagnosis": {
                        "terminal_cause": "missing_normalized_trace",
                        "harness_relevance": "unknown",
                        "confidence": 0.0,
                    },
                }
            )
            _write_json(output_path, base)
            return base

        prompt = self._trace_prompt(case)
        try:
            run_env = dict(shared_env)
            trace_workspace = result_dir / "trace_inputs" / safe_id
            if trace_workspace.exists() or trace_workspace.is_symlink():
                if trace_workspace.is_dir() and not trace_workspace.is_symlink():
                    shutil.rmtree(trace_workspace)
                else:
                    trace_workspace.unlink()
            trace_workspace.mkdir(parents=True, exist_ok=True)
            source_trace_dir = evidence_root / case["trial_dir"]
            for source in source_trace_dir.iterdir():
                target = trace_workspace / source.name
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            xdg_root = result_dir / "trace_runs" / safe_id / "xdg"
            for key, child in (
                ("XDG_DATA_HOME", "data"),
                ("XDG_STATE_HOME", "state"),
                ("XDG_CACHE_HOME", "cache"),
            ):
                target = xdg_root / child
                target.mkdir(parents=True, exist_ok=True)
                run_env[key] = str(target)
            client = self._build_analyzer_client(analyzer_config, run_env)
            result = client.run_instruction_detailed(
                workspace=trace_workspace,
                instruction=prompt,
                request_timeout_sec=int(
                    self.options.get("diagnosis_timeout_sec", analyzer_config["timeout_sec"])
                ),
                raise_on_error=False,
            )
            raw_path.write_text(result.stdout[-20000:], encoding="utf-8")
            model_output_path = trace_workspace / _TRACE_DIAGNOSIS_FILENAME
            parsed = _load_trace_diagnosis(model_output_path)
            parsed, pattern_normalization = _apply_outcome_pattern_limits(
                parsed, outcome=case["outcome"]
            )
            if pattern_normalization["truncated"]:
                _write_json(model_output_path, parsed)
            base["diagnosis_status"] = "ok"
            base["diagnosis"] = parsed
            base["model_output_path"] = str(
                model_output_path.relative_to(result_dir)
            )
            base["diagnosis_runtime"] = {
                "returncode": result.returncode,
                "trajectory_events": len(result.trajectory),
                "pattern_normalization": pattern_normalization,
            }
        except Exception as exc:  # noqa: BLE001
            existing = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
            failure = traceback.format_exc()[-20000:]
            raw_path.write_text(
                f"{existing}\n\n[diagnosis output error]\n{failure}" if existing else failure,
                encoding="utf-8",
            )
            base["diagnosis_status"] = "llm_diagnosis_unavailable"
            base["diagnosis"] = {
                "terminal_cause": "diagnosis_unavailable",
                "harness_relevance": "unknown",
                "confidence": 0.0,
                "error": str(exc),
            }
        _write_json(output_path, base)
        return base

    def _run_aggregate(
        self,
        *,
        result_dir: Path,
        diagnosis_config: dict[str, Any],
        shared_env: dict[str, str],
        index: dict[str, Any],
    ) -> dict[str, Any]:
        base = {
            "schema_version": "tb21-aggregate-diagnosis-v1",
            "coverage": index["counts"],
            "shared_patterns": [],
            "success_patterns": [],
            "task_specific_patterns": [],
            "harness_change_hypotheses": [],
            "evidence_refs": [],
            "confidence": 0.0,
            "aggregation_status": "unavailable",
        }
        if not self._bool_option("aggregate_enabled", True):
            base["aggregation_status"] = "disabled"
            return base
        prompt = self._aggregate_prompt(index)
        run_env = dict(shared_env)
        xdg_root = result_dir / "aggregate_run" / "xdg"
        for key, child in (
            ("XDG_DATA_HOME", "data"),
            ("XDG_STATE_HOME", "state"),
            ("XDG_CACHE_HOME", "cache"),
        ):
            target = xdg_root / child
            target.mkdir(parents=True, exist_ok=True)
            run_env[key] = str(target)
        aggregate_path = result_dir / "aggregate.json"
        if aggregate_path.exists():
            aggregate_path.unlink()
        try:
            client = self._build_analyzer_client(diagnosis_config, run_env)
            result = client.run_instruction_detailed(
                workspace=result_dir,
                instruction=prompt,
                request_timeout_sec=int(
                    self.options.get("aggregate_timeout_sec", diagnosis_config["timeout_sec"])
                ),
                raise_on_error=False,
            )
            (result_dir / "aggregate_raw.log").write_text(
                result.stdout[-30000:], encoding="utf-8"
            )
            required_keys = {
                "shared_patterns",
                "success_patterns",
                "task_specific_patterns",
                "harness_change_hypotheses",
                "confidence",
            }
            parsed = _load_json_object_or_empty(aggregate_path)
            if not required_keys.issubset(parsed):
                raise ValueError("aggregate model did not write a valid aggregate.json")
            base.update(parsed)
            base["aggregation_status"] = "ok"
            base["aggregation_runtime"] = {
                "returncode": result.returncode,
                "trajectory_events": len(result.trajectory),
            }
        except Exception as exc:  # noqa: BLE001
            (result_dir / "aggregate_error.log").write_text(
                traceback.format_exc()[-30000:], encoding="utf-8"
            )
            base["aggregation_error"] = str(exc)
        return base

    def _trace_prompt(self, case: dict[str, Any]) -> str:
        return render_prompt(
            __package__,
            "trace_diagnosis.md",
            trace_id=case["trace_id"],
            outcome=case["outcome"],
            reward=case["reward"],
            trace_status=case["trace_status"],
            diagnosis_filename=_TRACE_DIAGNOSIS_FILENAME,
        )

    def _resolved_analyzer_config(
        self,
        cache_key: str,
        *,
        current_harness_path: str | Path | None,
    ) -> dict[str, Any]:
        configured = self.options.get("analyzer") or {}
        if not isinstance(configured, dict):
            raise ValueError("evidence.options.analyzer must be a mapping")
        client_name = str(configured.get("client_name") or "opencode_bun_dev_client")
        if client_name != "codex_cli_client":
            resolved = super()._resolved_analyzer_config(
                cache_key,
                current_harness_path=current_harness_path,
            )
            resolved["client_name"] = client_name
            resolved["client_init"] = dict(configured.get("client_init") or {})
            return resolved

        model = str(configured.get("model") or "gpt-5.6-sol")
        client_init = dict(configured.get("client_init") or {})
        client_init.setdefault("model", model)
        client_init.setdefault("reasoning_effort", str(configured.get("reasoning_effort") or "high"))
        client_init.setdefault("sandbox", "workspace-write")
        client_init.setdefault("approve_for_me", False)
        client_init.setdefault("ephemeral", True)
        client_init.setdefault("skip_git_repo_check", True)
        client_init.setdefault("output_log_path", "codex-run.jsonl")
        client_init.setdefault("last_message_path", "codex-last-message.md")
        return {
            "client_name": client_name,
            "client_init": client_init,
            "model": model,
            "configured_model": model,
            "timeout_sec": int(
                configured.get("timeout_sec")
                or self.options.get("timeout_sec")
                or self.config.evolver_timeout_sec
            ),
            "xdg_root": repo_relative_path(
                configured.get("xdg_root")
                or (self.config.report_root / "analyzer" / cache_key[:12] / "xdg")
            ),
        }

    def _prepare_analyzer_environment(
        self, analyzer_config: dict[str, Any]
    ) -> tuple[dict[str, str], Path]:
        if analyzer_config.get("client_name") != "codex_cli_client":
            return self._prepare_opencode_config(analyzer_config)
        config_path = Path(analyzer_config["xdg_root"]) / "codex-analyzer.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            config_path,
            {
                "client_name": analyzer_config["client_name"],
                "model": analyzer_config["model"],
                "timeout_sec": analyzer_config["timeout_sec"],
            },
        )
        return {}, config_path

    def _build_analyzer_client(
        self,
        analyzer_config: dict[str, Any],
        run_env: dict[str, str],
    ) -> Any:
        client_name = str(analyzer_config.get("client_name") or "opencode_bun_dev_client")
        client_init = dict(analyzer_config.get("client_init") or {})
        if client_name == "opencode_bun_dev_client":
            client_init.update(
                {
                    "opencode_repo_path": str(analyzer_config["opencode_repo_path"]),
                    "dev_run_command_prefix": str(analyzer_config["dev_run_command_prefix"]),
                    "dev_run_add_dir_arg": True,
                    "dev_run_env": {**dict(client_init.get("dev_run_env") or {}), **run_env},
                }
            )
        elif client_name == "codex_cli_client":
            client_init["env"] = {**dict(client_init.get("env") or {}), **run_env}
        return CLIENT_REGISTRY.build(client_name, client_init)

    def _aggregate_prompt(self, index: dict[str, Any]) -> str:
        return render_prompt(__package__, "aggregate.md", record_count=len(index.get("records", [])))

    def _bool_option(self, key: str, default: bool) -> bool:
        value = self.options.get(key, default)
        if isinstance(value, str):
            return value.lower() not in {"0", "false", "no", "off"}
        return bool(value)


