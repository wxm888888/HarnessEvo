"""Held-in sampling on OpenSandbox and loading the resulting trial artifacts."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from harness_evolution.evaluation.backends.opensandbox import _promote_final_trial_artifacts, _run_trials, custom_afs_binary_path, expand_trials, opensandbox_settings, publish_side_binaries, resolve_task_specs
from harness_evolution.core.types import CollectionArtifact, CollectionResult
from harness_evolution.traces.preflight import filter_collection_artifacts
from harness_evolution.traces.materialize import materialize as materialize_trace_artifacts


def _load_collection_artifacts(job_dir: Path) -> list[CollectionArtifact]:
    paths = sorted(job_dir.rglob("agent/trajectory.json"))
    if not paths:
        raise RuntimeError(f"no agent/trajectory.json files found under {job_dir}")
    artifacts: list[CollectionArtifact] = []
    for trajectory in paths:
        result = trajectory.parents[1] / "result.json"
        if not result.is_file():
            raise RuntimeError(f"missing result.json for {trajectory}")
        context = trajectory.parent / "llm-context.jsonl"
        openai_messages = trajectory.parent / "openai-messages.jsonl"
        artifacts.append(
            CollectionArtifact(
                trajectory_path=trajectory,
                result_path=result,
                context_path=context if context.is_file() else None,
                openai_messages_path=openai_messages if openai_messages.is_file() else None,
            )
        )
    eligible, decisions = filter_collection_artifacts(artifacts)
    (job_dir / "analysis-preflight.json").write_text(
        json.dumps(
            {
                "schema_version": "harnessrl-analysis-preflight-v1",
                "total_artifacts": len(artifacts),
                "eligible_artifacts": len(eligible),
                "excluded_artifacts": len(artifacts) - len(eligible),
                "decisions": [decision.as_dict() for decision in decisions],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    excluded = [decision for decision in decisions if not decision.eligible]
    if excluded:
        reasons: dict[str, int] = {}
        for decision in excluded:
            reasons[decision.reason_code] = reasons.get(decision.reason_code, 0) + 1
        print(
            f"analysis preflight: total={len(artifacts)} eligible={len(eligible)} "
            f"excluded={len(excluded)} reasons={reasons}",
            flush=True,
        )
    if not eligible:
        raise RuntimeError(
            f"analysis preflight found no eligible traces under {job_dir}; "
            "environment/infrastructure failures are excluded and cannot produce a candidate"
        )
    return eligible


def _collection_artifacts_for_tasks(
    collection: CollectionResult,
    task_names: list[str],
) -> list[CollectionArtifact]:
    """Select trajectory evidence by task while retaining one merged run.

    Sampling can evaluate both splits in one OpenSandbox pool.  The evolver
    still receives only held-in evidence, so held-out remains a gate-only
    split even though its trials share the same collection job.
    """
    wanted = set(task_names)
    selected = _select_artifacts_for_tasks(collection.artifacts, task_names)
    seen = {
        str(json.loads(artifact.result_path.read_text(encoding="utf-8")).get("task_name") or "")
        for artifact in selected
    }
    missing = sorted(wanted - seen)
    if missing:
        raise RuntimeError(
            f"merged sampling did not produce held-in evidence under {collection.job_dir}: "
            f"missing={missing[:8]} selected={len(selected)}"
        )
    eligible, decisions = filter_collection_artifacts(selected)
    (collection.job_dir / "analysis-preflight.json").write_text(
        json.dumps(
            {
                "schema_version": "harnessrl-analysis-preflight-v1",
                "total_artifacts": len(selected),
                "eligible_artifacts": len(eligible),
                "excluded_artifacts": len(selected) - len(eligible),
                "held_in_tasks": list(task_names),
                "decisions": [decision.as_dict() for decision in decisions],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    excluded = [decision for decision in decisions if not decision.eligible]
    reasons: dict[str, int] = {}
    for decision in excluded:
        reasons[decision.reason_code] = reasons.get(decision.reason_code, 0) + 1
    print(
        f"merged sampling: total_trials={len(collection.artifacts)} "
        f"held_in_evidence={len(eligible)} held_out_excluded={len(collection.artifacts) - len(selected)} "
        f"analysis_excluded={len(excluded)} reasons={reasons}",
        flush=True,
    )
    if not eligible:
        raise RuntimeError(
            f"analysis preflight found no eligible held-in traces under {collection.job_dir}; "
            "environment/infrastructure failures are excluded and cannot produce a candidate"
        )
    return eligible


def _select_artifacts_for_tasks(
    artifacts: list[CollectionArtifact],
    task_names: list[str],
) -> list[CollectionArtifact]:
    """Keep only artifacts belonging to the requested task split."""
    wanted = set(task_names)
    selected: list[CollectionArtifact] = []
    for artifact in artifacts:
        payload = json.loads(artifact.result_path.read_text(encoding="utf-8"))
        task = str(payload.get("task_name") or "")
        if task in wanted:
            selected.append(artifact)
    return selected


def _run_opensandbox_sampling(
    *,
    gate_config: dict[str, Any],
    task_names: list[str],
    task_catalog: dict[str, dict[str, str]],
    commit: str,
    binary: str,
    repeats: int,
    parallelism: int,
    jobs_root: Path,
    job_name: str,
    stage: str,
) -> CollectionResult:
    """Collect evolver trajectories through the same OpenSandbox path as the gate.

    The older modular collector shells out to ``harbor run`` and therefore
    requires a local Docker daemon.  TB2.1 evaluation in this project uses
    OpenSandbox directly, so sampling must use the same runtime and upload the
    active candidate binary into each sandbox.
    """
    if not binary or not Path(binary).is_file():
        raise RuntimeError(f"OpenSandbox sampling needs a local binary: {binary!r}")
    config = copy.deepcopy(gate_config)
    heldout = config.setdefault("heldout_gate", {})
    osb = heldout.setdefault("opensandbox", {})
    heldout["backend"] = "opensandbox"
    configured_setup = str(
        os.environ.get("HARNESSRL_OPENCODE_SETUP")
        or osb.get("opencode_setup")
        or heldout.get("opencode_setup")
        or os.environ.get("OPENCODE_SETUP")
        or "afs_bin"
    ).strip().lower()
    osb["opencode_setup"] = configured_setup
    osb["agent_timeout_sec"] = int(osb.get("agent_timeout_sec") or 1800)
    osb["verifier_timeout_sec"] = int(osb.get("verifier_timeout_sec") or 300)
    settings = opensandbox_settings(heldout)
    specs = resolve_task_specs(heldout, task_names, task_catalog=task_catalog)
    sample_side: tuple[str, str, str] | tuple[str, str, str, str] = ("sample", commit, binary)
    if settings["opencode_setup"] != "upload":
        # Sampling used to publish to the legacy /{commit[:7]}/opencode path.
        # That path is immutable and can already contain a different binary
        # from an earlier loop, so every sampling batch gets its own AFS path.
        afs_root = str(
            osb.get("opencode_afs_root")
            or heldout.get("opencode_afs_root")
            or os.environ.get("OPENCODE_AFS_ROOT")
            or "/sandbox-data/duchenyu/opencode"
        )
        sample_afs = custom_afs_binary_path(
            afs_root,
            run_tag=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{job_name}",
            side_label=f"sampling-{stage}",
            commit=commit,
        )
        sample_side = ("sample", commit, binary, sample_afs)
        publish_side_binaries([sample_side], env_file=settings["env_file"])
    job_dir = jobs_root / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    trials = expand_trials(
        sides=[sample_side],
        task_names=task_names,
        specs=specs,
        n_attempts=repeats,
    )
    started = time.monotonic()
    outcomes = asyncio.run(
        _run_trials(
            trials,
            settings=settings,
            jobs_dir=job_dir,
            n_concurrent=max(1, int(parallelism)),
        )
    )

    artifacts: list[CollectionArtifact] = []
    outcome_by_id = {item.trial_id: item for item in outcomes}
    for trial in trials:
        session_dir = job_dir / trial.side / trial.trial_id
        session_dir.mkdir(parents=True, exist_ok=True)
        _promote_final_trial_artifacts(session_dir)
        remote = session_dir / "agent.remote.jsonl"
        trajectory = session_dir / "agent" / "trajectory.json"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        events: list[dict[str, Any]] = []
        if remote.is_file():
            for raw in remote.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)
        trajectory.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outcome = outcome_by_id.get(trial.trial_id)
        reward_value: float | None = None if outcome and outcome.error else (
            1.0 if outcome and outcome.passed else 0.0
        )
        result = {
            "id": trial.trial_id,
            "task_name": trial.task.name,
            "trial_name": trial.trial_id,
            "task_id": {"name": f"terminal-bench/{trial.task.name}", "ref": ""},
            "source": "opensandbox",
            # A trial-level runtime/control-plane error is incomplete evidence,
            # not an agent failure.  Preserve it as a missing reward so the
            # diagnosis coverage report can exclude it from pass/fail claims.
            "verifier_result": {"rewards": {"reward": reward_value}},
            "exception_info": {
                "exception_type": "OpenSandboxTrialError" if outcome and outcome.error else None,
                "exception_message": outcome.detail if outcome and outcome.error else None,
            },
            "agent_result": {},
            "config": {"backend": "opensandbox", "stage": stage},
        }
        result_path = session_dir / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # TaskSpec.instruction is instruction.md for Terminal-Bench tasks and
        # the problem statement for MiMo tasks.
        if trial.task.instruction:
            (session_dir / "instruction.md").write_text(trial.task.instruction, encoding="utf-8")
        trace_meta = materialize_trace_artifacts(
            session_dir,
            reward="missing" if outcome and outcome.error else ("1" if outcome and outcome.passed else "0"),
            status="ok" if outcome and not outcome.error else "error",
            error=outcome.detail if outcome and outcome.error else None,
            extra_meta={"verifier_rc": 0 if outcome and not outcome.error else None},
        )
        # Keep a small machine-readable completeness record next to each
        # trial.  The evidence stage uses it to distinguish a real wrong
        # answer from an infrastructure trial that never emitted a full trace.
        (session_dir / "artifact-status.json").write_text(
            json.dumps(trace_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if events:
            (trajectory.parent / "trajectory.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events),
                encoding="utf-8",
            )
        context = trajectory.parent / "llm-context.jsonl"
        openai_messages = trajectory.parent / "openai-messages.jsonl"
        artifacts.append(
            CollectionArtifact(
                trajectory_path=trajectory,
                result_path=result_path,
                context_path=context if context.is_file() else None,
                openai_messages_path=openai_messages if openai_messages.is_file() else None,
            )
        )

    log_path = jobs_root / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"backend=opensandbox stage={stage} trials={len(trials)}\n"
        f"duration_sec={time.monotonic() - started:.3f}\n",
        encoding="utf-8",
    )
    return CollectionResult(
        job_name=job_name,
        job_dir=job_dir,
        log_path=log_path,
        returncode=0 if all(not item.error for item in outcomes) else 1,
        artifacts=artifacts,
        metadata={
            "run_id": job_name,
            "stage": stage,
            "backend": "opensandbox",
            "job_name": job_name,
            "job_dir": str(job_dir),
            "collection_log_path": str(log_path),
            "task_names": task_names,
            "trials": repeats,
            "parallelism": parallelism,
            "trajectory_paths": [str(item.trajectory_path) for item in artifacts],
            "outcomes": [asdict(item) for item in outcomes],
        },
    )
