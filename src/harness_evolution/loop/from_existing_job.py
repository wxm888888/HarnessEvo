from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from harness_evolution.loop.modular import (
    _build_strategy,
    _bool_value,
    _candidate_to_json,
    _int_value,
    _mapping,
    _repo_relative_path,
    _str_value,
    _task_cache_root,
    _validation_feedback,
    _write_iteration_report,
    _write_json,
    generate_candidate_batch,
    load_runtime_config,
    _prepare_opencode_evolver_environment,
)
from harness_evolution.selection.candidates import CANDIDATE_SELECTION_REGISTRY
from harness_evolution.core.types import Candidate, CandidateSelectionContext, CollectionArtifact
from harness_evolution.core.paths import REPO_ROOT
from harness_evolution.diagnosis import EVIDENCE_REGISTRY


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one modular OpenCode harness evolve step from an existing Terminal-Bench job directory."
    )
    parser.add_argument("--config", required=True, help="YAML config used by the modular loop.")
    parser.add_argument(
        "--job-dir",
        help="Existing Terminal-Bench job directory. If omitted, find latest job for --iteration.",
    )
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number used when auto-finding a job dir.")
    parser.add_argument("--harness-path", help="Override initial.harness_path from the YAML.")
    parser.add_argument("--base-commit", help="Override initial.base_commit from the YAML.")
    parser.add_argument("--rollouts", type=int, help="Override loop.rollouts_per_attempt from the YAML.")
    parser.add_argument("--workspace-root", help="Override loop.workspace_root from the YAML.")
    parser.add_argument("--report-root", help="Override loop.report_root from the YAML.")
    parser.add_argument("--max-attempts", type=int, help="Override validation.max_attempts from the YAML.")
    return parser.parse_args()


def _resolve_existing_job_dir(*, jobs_root: Path, iteration: int, explicit_job_dir: str | None) -> Path:
    if explicit_job_dir:
        job_dir = Path(explicit_job_dir).expanduser()
        if not job_dir.is_absolute():
            job_dir = REPO_ROOT / job_dir
        job_dir = job_dir.resolve()
        if not job_dir.exists():
            raise ValueError(f"job dir does not exist: {job_dir}")
        return job_dir

    patterns = [
        f"tb-opencode-modular-loop-*-iter-{iteration:03d}",
        f"tb-opencode-harness-loop-*-iter-{iteration:03d}",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(path for path in jobs_root.glob(pattern) if path.is_dir())
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        searched = ", ".join(str(jobs_root / pattern) for pattern in patterns)
        raise ValueError(f"no existing job dir matched: {searched}")
    return candidates[0].resolve()


def _collect_existing_artifacts(job_dir: Path) -> tuple[list[CollectionArtifact], dict]:
    trajectory_paths = sorted(job_dir.rglob("agent/trajectory.json"))
    artifacts = []
    for trajectory_path in trajectory_paths:
        result_path = trajectory_path.parents[1] / "result.json"
        if not result_path.exists():
            raise ValueError(f"missing result.json for trajectory: {trajectory_path}")
        context_path = trajectory_path.parent / "llm-context.jsonl"
        openai_messages_path = trajectory_path.parent / "openai-messages.jsonl"
        artifacts.append(
            CollectionArtifact(
                trajectory_path=trajectory_path,
                result_path=result_path,
                context_path=context_path if context_path.is_file() else None,
                openai_messages_path=openai_messages_path
                if openai_messages_path.is_file()
                else None,
            )
        )
    if not artifacts:
        raise ValueError(f"no agent/trajectory.json files found under {job_dir}")
    return artifacts, {
        "job_dir": str(job_dir),
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "artifact_paths": [
            {
                "trajectory_path": str(artifact.trajectory_path),
                "result_path": str(artifact.result_path),
            }
            for artifact in artifacts
        ],
    }


def main() -> None:
    args = _parse_args()
    config = load_runtime_config(args.config)
    if args.harness_path:
        config.initial_harness_path = str(_repo_relative_path(args.harness_path))
    if args.base_commit:
        config.initial_base_commit = args.base_commit
    if args.rollouts is not None:
        config.rollouts_per_attempt = int(args.rollouts)
    if args.workspace_root:
        config.workspace_root = _repo_relative_path(args.workspace_root)
    if args.report_root:
        config.report_root = _repo_relative_path(args.report_root)
    if args.max_attempts is not None:
        config.validation["max_attempts"] = int(args.max_attempts)

    tb_config = config.terminal_bench
    jobs_root = _repo_relative_path(_str_value(tb_config, "jobs_root"))
    job_dir = _resolve_existing_job_dir(
        jobs_root=jobs_root,
        iteration=int(args.iteration),
        explicit_job_dir=args.job_dir,
    )
    artifacts, tb_metadata = _collect_existing_artifacts(job_dir)

    run_id = datetime.now(timezone.utc).strftime("opencode_harness_existing_tb_modular_%Y%m%d_%H%M%S_%f")
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    config.report_root.mkdir(parents=True, exist_ok=True)
    evolver_env, evolver_env_meta = _prepare_opencode_evolver_environment(config, run_id)
    evidence_strategy = _build_strategy(
        EVIDENCE_REGISTRY,
        config.evidence,
        _str_value(config.evidence, "method", "agent_analyzed"),
        config,
    )
    candidate_selector = _build_strategy(CANDIDATE_SELECTION_REGISTRY, config.candidate_selection, "first_valid", config)

    all_candidates: list[Candidate] = []
    trajectories = []
    attempt_meta = []
    feedback = ""
    decision = None
    max_attempts = _int_value(config.validation, "max_attempts", 3) if _bool_value(config.validation, "enabled", True) else 1
    for attempt_idx in range(max_attempts):
        candidates, attempt_trajectories, rollout_meta = generate_candidate_batch(
            config=config,
            evidence_strategy=evidence_strategy,
            artifacts=artifacts,
            task_cache_root=_task_cache_root(config),
            current_harness_path=config.initial_harness_path,
            current_base_commit=config.initial_base_commit,
            iteration_idx=0,
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
        # candidate_selector ("first_valid" by default) is only used to decide
        # *whether to retry the batch*: as soon as this attempt's batch has
        # produced at least one validated candidate, we stop -- all validated
        # candidates from this run (potentially several, generated
        # concurrently) go on to the gate together, not just the first one.
        if decision.accepted:
            break
        feedback = _validation_feedback(candidates, config)

    valid_candidates = [candidate for candidate in all_candidates if candidate.validation.passed]

    # Best-effort single "accepted_candidate" fields kept for backward
    # compatibility with tooling/logs that only look at one output commit.
    accepted_candidate = next(
        (candidate for candidate in all_candidates if decision and candidate.candidate_idx == decision.candidate_idx),
        None,
    )
    output_harness_path = str(accepted_candidate.workspace) if accepted_candidate else config.initial_harness_path
    output_commit = accepted_candidate.commit if accepted_candidate else config.initial_base_commit

    iteration_record = {
        "run_id": run_id,
        "source_job_dir": str(job_dir),
        "input_harness_path": config.initial_harness_path,
        "input_base_commit": config.initial_base_commit,
        "output_harness_path": output_harness_path,
        "output_commit": output_commit,
        "accepted_candidate_idx": decision.candidate_idx if decision else None,
        "fallback_reason": decision.fallback_reason if decision else "no_decision",
        # All validated candidates from this run -- the gate evaluates every
        # one of these against a single shared baseline run and promotes
        # whichever scores best (see harness_evolution_pipeline.py).
        "accepted_candidates": [
            {
                "candidate_idx": candidate.candidate_idx,
                "workspace": str(candidate.workspace),
                "commit": candidate.commit,
            }
            for candidate in valid_candidates
        ],
        "task_selection": {
            "method": "existing_tb_job",
            "source_job_dir": str(job_dir),
        },
        "terminal_bench_collection": tb_metadata,
        "evidence_method": evidence_strategy.name,
        "candidate_selection": decision.metadata if decision else {"method": candidate_selector.name},
        "attempts": attempt_meta,
        "candidates": [_candidate_to_json(candidate) for candidate in all_candidates],
        "trajectory": trajectories,
        "opencode_evolver": evolver_env_meta,
    }

    artifact_root = config.workspace_root / "artifacts"
    metadata_path = artifact_root / f"{run_id}.json"
    trajectory_path = artifact_root / f"{run_id}-evolver-trajectory.json"
    _write_json(metadata_path, iteration_record)
    _write_json(
        trajectory_path,
        {
            "run_id": run_id,
            "source_job_dir": str(job_dir),
            "trajectory": trajectories,
            "attempts": attempt_meta,
            "candidates": [_candidate_to_json(candidate) for candidate in all_candidates],
        },
    )
    report_path = _write_iteration_report(config, {**iteration_record, "iteration": 1})

    print(f"source_job_dir: {job_dir}")
    print(f"output_harness_path: {output_harness_path}")
    print(f"output_commit: {output_commit}")
    print(f"accepted_candidate_idx: {iteration_record['accepted_candidate_idx']}")
    print(f"accepted_candidates: {len(valid_candidates)} of {len(all_candidates)} generated")
    print(f"metadata_path: {metadata_path}")
    print(f"evolver_trajectory_path: {trajectory_path}")
    print(f"report_path: {report_path}")


if __name__ == "__main__":
    main()
