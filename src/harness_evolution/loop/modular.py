from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from harness_evolution.core.paths import REPO_ROOT
from harness_evolution.selection.candidates import CANDIDATE_SELECTION_REGISTRY
from harness_evolution.core.types import (
    DEFAULT_TASK_CACHE_ROOT,
    Candidate,
    CandidateSelectionContext,
    CandidateSelectionDecision,
    CollectionArtifact,
    CollectionResult,
    RuntimeConfig,
    TaskSelectionContext,
)
from harness_evolution.diagnosis import EVIDENCE_REGISTRY
from harness_evolution.tasks.selection import TASK_SELECTION_REGISTRY

# Re-exported after the phase-2 split; new code should import from the owning module.
from harness_evolution.core.config import _bool_value, _int_value, _load_yaml_config, _mapping, _memory_section, _optional_int_value, _relative_workspace_dir, _repo_relative_path, _str_list, _str_value, _tail_text, _write_json, load_runtime_config  # noqa: F401
from harness_evolution.core.strategies import _build_memory_strategy, _build_strategy  # noqa: F401
from harness_evolution.harnesses.opencode.build import _NODE_MODULES_CACHE_LOCK, _command_env, _configured_bun_bin, _copy_node_modules_tree, _current_binary_name, _git_head_commit, _relink_workspace_members, _remove_partial_tree, _workspace_member_links, build_opencode_binary_from_source, ensure_bun_install, link_shared_node_modules, register_built_binary, resolve_local_opencode_binary  # noqa: F401
from harness_evolution.harnesses.opencode.validate import _run_validation_command, _validation_log_path, validate_candidate  # noqa: F401
from harness_evolution.proposer.workspace import _git_diff_stats, commit_changes_or_adopt_head  # noqa: F401
from harness_evolution.proposer.requirement import _CANDIDATE_REPORT_RELATIVE_PATH, _candidate_requirement, _copy_skills, _load_candidate_report  # noqa: F401
from harness_evolution.proposer.batch import _candidate_to_json, _codex_client_env, _prepare_opencode_evolver_environment, _validation_feedback, generate_candidate_batch  # noqa: F401


# -----------------------------------------------------------------------------
# Small config helpers
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular OpenCode Terminal-Bench evolver loop.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Runtime config
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# OpenCode build, validation, and rollout
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Terminal-Bench collection
# -----------------------------------------------------------------------------


def _harbor_task_filter(task_name: str) -> str:
    return task_name if "/" in task_name else f"terminal-bench/{task_name}"


def run_terminal_bench_collection(
    *,
    config: RuntimeConfig,
    current_harness_path: str,
    run_id: str,
    job_name: str,
    selected_task_names: list[str],
    task_selection_metadata: dict[str, Any],
    trials: int,
    parallelism: int,
    max_tasks: int | None,
    stage: str,
) -> CollectionResult:
    tb = config.terminal_bench
    agent_import_path = _str_value(
        tb,
        "agent_import_path",
        "examples.terminal_bench.harbor_noinstall_opencode:NoInstallOpenCode",
    )
    uses_local_opencode = "harbor_noinstall_opencode" in agent_import_path or "NoInstallOpenCode" in agent_import_path
    local_opencode_path = resolve_local_opencode_binary(config, current_harness_path) if uses_local_opencode else None
    jobs_root = _repo_relative_path(_str_value(tb, "jobs_root"))
    command = ["harbor", "run", "-d", _str_value(tb, "dataset"), "--jobs-dir", str(jobs_root)]
    environment_import_path = tb.get("environment_import_path")
    if environment_import_path:
        command.extend(["--environment-import-path", str(environment_import_path)])
    else:
        command.extend(["-e", _str_value(tb, "environment_type", "docker")])
    for task_name in selected_task_names:
        command.extend(["--include-task-name", _harbor_task_filter(task_name)])
    command.extend(
        [
            "--agent-import-path",
            agent_import_path,
            "-m",
            _str_value(tb, "model"),
            "--n-attempts",
            str(trials),
            "--n-concurrent",
            str(parallelism),
            "--ae",
            f"OPENAI_API_KEY={_str_value(tb, 'openai_api_key')}",
            "--ae",
            f"OPENAI_BASE_URL={_str_value(tb, 'openai_base_url')}",
            "--job-name",
            job_name,
        ]
    )
    if _bool_value(tb, "debug", True):
        command.append("--debug")
    if _bool_value(tb, "no_delete", True):
        command.append("--no-delete")
    if max_tasks is not None:
        command.extend(["--n-tasks", str(max_tasks)])
    for raw_agent_kwarg in tb.get("agent_kwargs") or []:
        command.extend(["--ak", str(raw_agent_kwarg)])

    env = os.environ.copy()
    process_env = tb.get("process_env") or {}
    if process_env:
        if not isinstance(process_env, dict):
            raise ValueError("terminal_bench.process_env must be a mapping")
        for key, value in process_env.items():
            if value is not None:
                env[str(key)] = str(value)
    for config_key, env_key in {
        "daytona_api_key": "DAYTONA_API_KEY",
        "e2b_api_key": "E2B_API_KEY",
        "e2b_domain": "E2B_DOMAIN",
        "e2b_api_url": "E2B_API_URL",
        "harbor_e2b_template": "HARBOR_E2B_TEMPLATE",
        "harbor_e2b_image_rewrite": "HARBOR_E2B_IMAGE_REWRITE",
        "harbor_e2b_envd_port": "HARBOR_E2B_ENVD_PORT",
        "harbor_e2b_fs_user": "HARBOR_E2B_FS_USER",
        "harbor_use_package_mirrors": "HARBOR_USE_PACKAGE_MIRRORS",
    }.items():
        if tb.get(config_key):
            env[env_key] = str(tb[config_key])
    if local_opencode_path is not None:
        env["LOCAL_OPENCODE_PATH"] = str(local_opencode_path)
    package_src = REPO_ROOT / "src"
    env["PYTHONPATH"] = f"{package_src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(package_src)

    if local_opencode_path is not None:
        print(f"collecting Terminal-Bench trajectories ({stage}) with {local_opencode_path}", file=sys.stderr)
    else:
        print(f"collecting Terminal-Bench trajectories ({stage}) with agent {agent_import_path}", file=sys.stderr)
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    job_dir = jobs_root / job_name
    if not job_dir.exists():
        raise RuntimeError(f"harbor did not create expected job directory: {job_dir}\n{result.stdout}")
    log_path = config.workspace_root / "artifacts" / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout or "", encoding="utf-8")
    trajectory_paths = sorted(job_dir.rglob("agent/trajectory.json"))
    if not trajectory_paths:
        raise RuntimeError(f"no trajectory.json files found under {job_dir}; log: {log_path}")
    artifacts = []
    for trajectory_path in trajectory_paths:
        context_path = trajectory_path.parent / "llm-context.jsonl"
        openai_messages_path = trajectory_path.parent / "openai-messages.jsonl"
        artifacts.append(
            CollectionArtifact(
                trajectory_path=trajectory_path,
                result_path=trajectory_path.parents[1] / "result.json",
                context_path=context_path if context_path.is_file() else None,
                openai_messages_path=openai_messages_path
                if openai_messages_path.is_file()
                else None,
            )
        )
    return CollectionResult(
        job_name=job_name,
        job_dir=job_dir,
        log_path=log_path,
        returncode=result.returncode,
        artifacts=artifacts,
        metadata={
            "run_id": run_id,
            "stage": stage,
            "job_name": job_name,
            "job_dir": str(job_dir),
            "command": command,
            "returncode": result.returncode,
            "collection_log_path": str(log_path),
            "local_opencode_path": str(local_opencode_path) if local_opencode_path is not None else None,
            "task_selection": task_selection_metadata,
            "task_names": selected_task_names,
            "trials": trials,
            "parallelism": parallelism,
            "max_tasks": max_tasks,
            "trajectory_paths": [str(path) for path in trajectory_paths],
            "artifact_paths": [
                {"trajectory_path": str(artifact.trajectory_path), "result_path": str(artifact.result_path)}
                for artifact in artifacts
            ],
        },
    )


# -----------------------------------------------------------------------------
# Reports and loop
# -----------------------------------------------------------------------------


def _write_iteration_report(config: RuntimeConfig, iteration_record: dict[str, Any]) -> Path:
    config.report_root.mkdir(parents=True, exist_ok=True)
    iteration = int(iteration_record["iteration"])
    path = config.report_root / f"modular_iteration_{iteration:03d}.md"
    lines = [
        f"# Modular Iteration {iteration:03d}",
        "",
        f"- input harness: `{iteration_record.get('input_harness_path')}`",
        f"- output harness: `{iteration_record.get('output_harness_path')}`",
        f"- accepted candidate: `{iteration_record.get('accepted_candidate_idx')}`",
        f"- fallback reason: `{iteration_record.get('fallback_reason')}`",
        f"- task selection method: `{(iteration_record.get('task_selection') or {}).get('method')}`",
        f"- evidence method: `{iteration_record.get('evidence_method')}`",
        f"- candidate selection method: `{(iteration_record.get('candidate_selection') or {}).get('method')}`",
        "",
        "## Candidates",
        "",
    ]
    for candidate in iteration_record.get("candidates", []):
        validation = candidate.get("validation") or {}
        diff_stats = candidate.get("diff_stats") or {}
        lines.extend(
            [
                f"### Candidate {candidate.get('candidate_idx')}",
                "",
                f"- attempt: `{candidate.get('attempt_idx')}`",
                f"- commit: `{candidate.get('commit')}`",
                f"- workspace: `{candidate.get('workspace')}`",
                f"- validation passed: `{validation.get('passed')}`",
                f"- diff score: `{diff_stats.get('score')}`",
                f"- changed files: `{', '.join(candidate.get('changed_files') or [])}`",
                "",
            ]
        )
        for command in validation.get("commands") or []:
            lines.extend(
                [
                    f"- validation `{command.get('command')}`:",
                    f"  - passed: `{command.get('passed')}`",
                    f"  - return code: `{command.get('returncode')}`",
                    f"  - log: `{command.get('log_path')}`",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_summary(config: RuntimeConfig, metadata: dict[str, Any], output_path: Path) -> Path:
    path = config.report_root / "modular_summary.md"
    lines = [
        "# Modular Experiment Summary",
        "",
        f"- metadata: `{output_path}`",
        f"- final harness: `{metadata.get('final_harness_path')}`",
        f"- final commit: `{metadata.get('final_commit')}`",
        "",
    ]
    for iteration in metadata.get("iterations", []):
        lines.extend(
            [
                f"- iteration {iteration.get('iteration')}:",
                f"  - report: `{iteration.get('report_path')}`",
                f"  - accepted candidate: `{iteration.get('accepted_candidate_idx')}`",
                f"  - fallback reason: `{iteration.get('fallback_reason')}`",
            ]
        )
    final_eval = metadata.get("final_terminal_bench_evaluation")
    lines.extend(["", "## Final Evaluation", ""])
    if not isinstance(final_eval, dict):
        lines.append("- no final evaluation metadata recorded")
    elif final_eval.get("skipped"):
        lines.extend(
            [
                "- status: `skipped`",
                f"- reason: `{final_eval.get('reason')}`",
            ]
        )
    elif final_eval.get("error"):
        error = final_eval.get("error") or {}
        lines.extend(
            [
                "- status: `failed`",
                f"- error type: `{error.get('type')}`",
                f"- error message: `{error.get('message')}`",
                f"- traceback: `{error.get('traceback_path')}`",
            ]
        )
    else:
        lines.extend(
            [
                "- status: `completed`",
                f"- job dir: `{final_eval.get('job_dir')}`",
                f"- collection log: `{final_eval.get('collection_log_path')}`",
                f"- return code: `{final_eval.get('returncode')}`",
                f"- trial count: `{final_eval.get('trials')}`",
                f"- parallelism: `{final_eval.get('parallelism')}`",
                f"- task filters: `{', '.join(final_eval.get('task_names') or [])}`",
                f"- trajectory count: `{len(final_eval.get('trajectory_paths') or [])}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _task_cache_root(config: RuntimeConfig) -> Path:
    if config.opencode_evolver.get("task_cache_root"):
        return _repo_relative_path(str(config.opencode_evolver["task_cache_root"]))
    evidence_options = _mapping(config.evidence, "options", required=False)
    if evidence_options.get("task_cache_root"):
        return _repo_relative_path(str(evidence_options["task_cache_root"]))
    return DEFAULT_TASK_CACHE_ROOT


def _max_attempts(config: RuntimeConfig) -> int:
    return _int_value(config.validation, "max_attempts", 3) if _bool_value(config.validation, "enabled", True) else 1


def run_final_terminal_bench_evaluation(
    *,
    config: RuntimeConfig,
    current_harness_path: str,
    current_base_commit: str,
    run_id: str,
) -> dict[str, Any]:
    final_config = _mapping(config.terminal_bench, "final_evaluation", required=False)
    trials = _optional_int_value(final_config, "trials") or _int_value(config.terminal_bench, "trials_per_iteration")
    parallelism = _optional_int_value(final_config, "parallelism") or _int_value(config.terminal_bench, "parallelism")
    force_full_dataset = _bool_value(final_config, "force_full_dataset", True)
    if force_full_dataset:
        selected_task_names: list[str] = []
        max_tasks: int | None = None
        selection_strategy = "full_dataset"
    else:
        selected_task_names = _str_list(config.terminal_bench, "task_names", allow_empty=True)
        final_max_tasks = _optional_int_value(final_config, "max_tasks")
        max_tasks = final_max_tasks if final_max_tasks is not None else _optional_int_value(config.terminal_bench, "max_tasks")
        selection_strategy = "static_config"

    task_selection_metadata = {
        "method": "final_evaluation",
        "strategy": f"final_evaluation_{selection_strategy}",
        "force_full_dataset": force_full_dataset,
    }
    collection = run_terminal_bench_collection(
        config=config,
        current_harness_path=current_harness_path,
        run_id=run_id,
        job_name=f"tb-opencode-modular-loop-{run_id}-final-eval",
        selected_task_names=selected_task_names,
        task_selection_metadata=task_selection_metadata,
        trials=trials,
        parallelism=parallelism,
        max_tasks=max_tasks,
        stage="final_evaluation",
    )
    return {
        **collection.metadata,
        "input_harness_path": current_harness_path,
        "input_base_commit": current_base_commit,
        "config": {
            "enabled": _bool_value(final_config, "enabled", True),
            "force_full_dataset": force_full_dataset,
            "trials": _optional_int_value(final_config, "trials"),
            "parallelism": _optional_int_value(final_config, "parallelism"),
            "max_tasks": _optional_int_value(final_config, "max_tasks"),
        },
    }


def main() -> None:
    args = _parse_args()
    config = load_runtime_config(args.config)
    run_id = datetime.now(timezone.utc).strftime("opencode_harness_modular_loop_%Y%m%d_%H%M%S_%f")
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    config.report_root.mkdir(parents=True, exist_ok=True)

    default_task_method = "history_scheduler" if _mapping(config.terminal_bench, "task_scheduler", required=False) else "static_list"
    task_strategy = _build_strategy(TASK_SELECTION_REGISTRY, config.task_selection, default_task_method, config)
    evidence_strategy = _build_strategy(EVIDENCE_REGISTRY, config.evidence, "sanitized_instruction", config)
    candidate_selector = _build_strategy(CANDIDATE_SELECTION_REGISTRY, config.candidate_selection, "first_valid", config)
    memory_strategy = _build_memory_strategy(config)
    evolver_env, evolver_env_meta = _prepare_opencode_evolver_environment(config, run_id)

    tb = config.terminal_bench
    current_harness_path = config.initial_harness_path
    current_base_commit = config.initial_base_commit
    iterations = []
    consecutive_exhausted = 0

    for iteration_idx in range(config.iterations):
        selection = task_strategy.select(
            TaskSelectionContext(
                iteration_idx=iteration_idx,
                static_task_names=_str_list(tb, "task_names", allow_empty=True),
                jobs_root=_repo_relative_path(_str_value(tb, "jobs_root")),
            )
        )
        collection = run_terminal_bench_collection(
            config=config,
            current_harness_path=current_harness_path,
            run_id=run_id,
            job_name=f"tb-opencode-modular-loop-{run_id}-iter-{iteration_idx + 1:03d}",
            selected_task_names=selection.task_names,
            task_selection_metadata=selection.metadata,
            trials=_int_value(tb, "trials_per_iteration"),
            parallelism=_int_value(tb, "parallelism"),
            max_tasks=_optional_int_value(tb, "max_tasks"),
            stage="iteration_collection",
        )
        history_update = task_strategy.update_after_collection(
            iteration_idx=iteration_idx,
            selected_tasks=selection.task_names,
            job_dir=collection.job_dir,
        )

        all_candidates: list[Candidate] = []
        trajectories: list[dict[str, list[dict[str, Any]]]] = []
        attempt_meta = []
        feedback = ""
        decision = CandidateSelectionDecision(False, None, "no_candidates", {"method": candidate_selector.name})
        for attempt_idx in range(_max_attempts(config)):
            candidates, attempt_trajectories, rollout_meta = generate_candidate_batch(
                config=config,
                evidence_strategy=evidence_strategy,
                artifacts=collection.artifacts,
                task_cache_root=_task_cache_root(config),
                current_harness_path=current_harness_path,
                current_base_commit=current_base_commit,
                iteration_idx=iteration_idx,
                attempt_idx=attempt_idx,
                candidate_offset=len(all_candidates),
                evolver_env=evolver_env,
                validation_feedback=feedback,
            )
            all_candidates.extend(candidates)
            trajectories.extend(attempt_trajectories)
            attempt_meta.append(rollout_meta)
            decision = candidate_selector.select(
                CandidateSelectionContext(
                    candidates=all_candidates,
                    options=_mapping(config.candidate_selection, "options", required=False),
                )
            )
            if decision.accepted:
                break
            feedback = _validation_feedback(candidates, config)

        accepted_candidate = next(
            (candidate for candidate in all_candidates if candidate.candidate_idx == decision.candidate_idx),
            None,
        )
        memory_record: dict[str, Any] | None = None
        if accepted_candidate is not None:
            next_harness_path = str(accepted_candidate.workspace)
            next_base_commit = accepted_candidate.commit
            memory_record = memory_strategy.record_promotion(
                step=iteration_idx + 1,
                parent_commit=current_base_commit,
                winner=accepted_candidate,
                baseline_score=None,
                winner_score=None,
            )
            consecutive_exhausted = 0
        else:
            next_harness_path = current_harness_path
            next_base_commit = current_base_commit
            consecutive_exhausted += 1

        iteration_record = {
            "iteration": iteration_idx + 1,
            "input_harness_path": current_harness_path,
            "input_base_commit": current_base_commit,
            "output_harness_path": next_harness_path,
            "output_commit": next_base_commit,
            "accepted_candidate_idx": decision.candidate_idx,
            "fallback_reason": decision.fallback_reason,
            "task_selection": selection.metadata,
            "task_scheduler_history_update": history_update,
            "terminal_bench_collection": collection.metadata,
            "evidence_method": evidence_strategy.name,
            "candidate_selection": decision.metadata,
            "attempts": attempt_meta,
            "candidates": [_candidate_to_json(candidate) for candidate in all_candidates],
            "trajectory": trajectories,
        }
        if memory_record is not None:
            iteration_record["memory"] = memory_record
        trajectory_path = config.workspace_root / "artifacts" / f"{run_id}-iter-{iteration_idx + 1:03d}-modular-trajectory.json"
        _write_json(trajectory_path, iteration_record)
        iteration_record["trajectory_path"] = str(trajectory_path)
        report_path = _write_iteration_report(config, iteration_record)
        iteration_record["report_path"] = str(report_path)
        iterations.append(iteration_record)

        current_harness_path = next_harness_path
        current_base_commit = next_base_commit
        if (
            not decision.accepted
            and consecutive_exhausted >= _int_value(config.validation, "max_consecutive_exhausted_iterations", 2)
        ):
            print("stopping modular loop: validation attempts exhausted repeatedly", file=sys.stderr)
            break

    final_evaluation_meta_data: dict[str, Any] | None
    final_config = _mapping(tb, "final_evaluation", required=False)
    if _bool_value(final_config, "enabled", True):
        try:
            final_evaluation_meta_data = run_final_terminal_bench_evaluation(
                config=config,
                current_harness_path=current_harness_path,
                current_base_commit=current_base_commit,
                run_id=run_id,
            )
            print(f"final_evaluation_job_dir: {final_evaluation_meta_data.get('job_dir')}")
        except Exception as exc:
            trace_path = config.report_root / "modular_final_evaluation_error_traceback.txt"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(traceback.format_exc(), encoding="utf-8")
            final_evaluation_meta_data = {
                "stage": "final_evaluation",
                "input_harness_path": current_harness_path,
                "input_base_commit": current_base_commit,
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback_path": str(trace_path),
                },
            }
            print(f"final evaluation failed; traceback: {trace_path}", file=sys.stderr)
    else:
        final_evaluation_meta_data = {
            "stage": "final_evaluation",
            "skipped": True,
            "reason": "terminal_bench.final_evaluation.enabled=false",
            "input_harness_path": current_harness_path,
            "input_base_commit": current_base_commit,
        }

    metadata = {
        "run_id": run_id,
        "initial_harness_path": config.initial_harness_path,
        "initial_base_commit": config.initial_base_commit,
        "final_harness_path": current_harness_path,
        "final_commit": current_base_commit,
        "strategies": {
            "task_selection": task_strategy.name,
            "evidence": evidence_strategy.name,
            "candidate_selection": candidate_selector.name,
        },
        "opencode_evolver": evolver_env_meta,
        "final_terminal_bench_evaluation": final_evaluation_meta_data,
        "iterations": iterations,
    }
    output_path = config.report_root / "modular_metadata.json"
    _write_json(output_path, metadata)
    summary_path = _write_summary(config, metadata, output_path)
    print(f"modular_metadata_path: {output_path}")
    print(f"modular_summary_path: {summary_path}")


if __name__ == "__main__":
    main()
