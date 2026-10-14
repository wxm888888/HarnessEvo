"""Synchronous harness co-evolution pipeline (one invocation per pause).

Called by slime's train.py at every HARNESS_EVOLVE_INTERVAL step boundary
(training is paused while this runs). One invocation does:

  1. select recent trajectories over the last `window` steps -- grouped by
     prompt, picking pure_fail_tasks task-diverse failures, mixed_tasks
     tasks where the same prompt both won and lost (one success + one
     failure each, for direct contrast), and pure_success_tasks successes --
     and convert them to a TB-style evidence job dir
  2. run the evolver against the active harness commit to produce a
     candidate worktree
  3. run the configured gate against this round's evidence tasks. The default
     is static + held-out A/B on OpenSandbox; ``gate_mode: static_only`` keeps
     only local typecheck/build/smoke checks. The gate promotes/rejects in the
     version state file. OpenSandbox training reads that file on the next
     sandbox start and copies {OPENCODE_AFS_ROOT}/{commit7}/opencode (ACCEPT
     also publishes the binary to AFS via harness_dist.afs).

Exit code 0 covers ACCEPT, REJECT, and "no candidate produced" -- those are
all normal outcomes. Non-zero means the pipeline infrastructure itself
broke; train.py logs that and continues on the current active harness.

Usage:
  python3 harness_evolution_pipeline.py --config pipeline.yaml [--rollout-id N]

The rollout id defaults to $HARNESS_EVOLVE_ROLLOUT_ID (set by train.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from harness_evolution.core.paths import REPO_ROOT, ensure_external_paths

# Shared harness-distribution kernel (version state) from the sibling harness_dist.
ensure_external_paths()

from harness_evolution.integrations.slime.reward_bridge import (
    build_harness_change_manifest,
    run_rubric_reward,
    write_harness_change,
)
from harness_evolution.evaluation.gate import (
    load_gate_config,
    run_full_gate_multi,
    run_static_gate_multi,
)

from harness_dist import state as version_state


def _log(message: str) -> None:
    print(f"[evolution-pipeline {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _module_command(module: str) -> list[str]:
    """Run a package module in a child interpreter (``python -m``)."""
    return [sys.executable, "-m", f"harness_evolution.{module}"]


def _child_env() -> dict[str, str]:
    """Child interpreters must import this package even when it is not installed."""
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    return env


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# Per-run/per-lineage identity knobs that would otherwise only be settable by
# hand-copying the whole pipeline YAML (pipeline_<lineage>.yaml). Letting a
# launch script override them via env vars means ONE shared pipeline YAML is
# enough for every lineage -- no more copy-paste drift between lineages'
# config files. This is the root-cause fix for the 2026-08-26 incident where
# two lineages' hand-copied YAMLs both ended up pointing at the same
# `version_state` file, so one lineage's evolution result silently clobbered
# the other's.
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "trajectory_root": ("HARNESS_PIPELINE_TRAJECTORY_ROOT", str),
    "version_state": ("HARNESS_PIPELINE_VERSION_STATE", str),
    "stage_dir": ("HARNESS_PIPELINE_STAGE_DIR", str),
    "work_root": ("HARNESS_PIPELINE_WORK_ROOT", str),
    "evolve_config": ("HARNESS_PIPELINE_EVOLVE_CONFIG", str),
    "gate_config": ("HARNESS_PIPELINE_GATE_CONFIG", str),
    "window": ("HARNESS_PIPELINE_WINDOW", int),
    "pure_fail_tasks": ("HARNESS_PIPELINE_PURE_FAIL_TASKS", int),
    "mixed_tasks": ("HARNESS_PIPELINE_MIXED_TASKS", int),
    "pure_success_tasks": ("HARNESS_PIPELINE_PURE_SUCCESS_TASKS", int),
    "mixed_task_attempts": ("HARNESS_PIPELINE_MIXED_TASK_ATTEMPTS", int),
    "rollouts_per_attempt": ("HARNESS_PIPELINE_ROLLOUTS_PER_ATTEMPT", int),
    "evolver_timeout_sec": ("HARNESS_PIPELINE_EVOLVER_TIMEOUT_SEC", int),
    "gate_mode": ("HARNESS_PIPELINE_GATE_MODE", str),
}


def _apply_env_overrides(config: dict) -> dict:
    """Apply HARNESS_PIPELINE_* env var overrides on top of the loaded YAML.
    Env vars win when set (and non-empty); otherwise the YAML value (or its
    own default deeper in the pipeline) is left untouched.
    """
    for key, (env_name, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            config[key] = caster(raw)
    return config


def _select_trajectories(config: dict, rollout_id: int, job_dir: Path) -> None:
    trajectory_root = config.get("trajectory_root") or os.environ.get("POLAR_TRAJECTORY_DIR", "")
    if not trajectory_root:
        raise RuntimeError("set trajectory_root in the pipeline config or POLAR_TRAJECTORY_DIR")
    cmd = [
        *_module_command("integrations.slime.select_trajectories"),
        "--trajectory-root", trajectory_root,
        "--end-rollout-id", str(rollout_id),
        "--window", str(config.get("window", 10)),
        "--pure-fail-tasks", str(config.get("pure_fail_tasks", 10)),
        "--mixed-tasks", str(config.get("mixed_tasks", 6)),
        "--pure-success-tasks", str(config.get("pure_success_tasks", 3)),
        "--output-dir", str(job_dir),
    ]
    subprocess.run(cmd, check=True, env=_child_env())


def _resolve_evidence_task_ids(config: dict, gate_config: dict, job_dir: Path) -> None:
    """Best-effort: fill in dataset_task_id/dataset_task_dir/dataset_image_tag
    on job_dir's conversion_manifest.json by matching each evidence session's
    instruction text against `heldout_gate.tasks_dir`'s task instructions
    (see integrations/slime/resolve_task_ids.py -- the trajectory dump itself never
    carries these fields, so this recovers them purely from on-disk text).

    Failures here are non-fatal: _evidence_task_names() below just sees
    fewer/no resolved tasks and the held-out gate falls back to its fixed
    canary set, same as if this step didn't run at all.
    """
    configured_tasks_dir = str(
        (gate_config.get("heldout_gate") or {}).get("tasks_dir") or ""
    ).strip()
    if configured_tasks_dir.startswith("__") and configured_tasks_dir.endswith("__"):
        configured_tasks_dir = ""
    tasks_dir = configured_tasks_dir or os.environ.get("HARNESS_GATE_TASKS_DIR", "").strip()
    if not tasks_dir:
        _log(
            "skipping dataset_task_id resolution: no concrete heldout_gate.tasks_dir "
            "or HARNESS_GATE_TASKS_DIR"
        )
        return
    # The index this builds is keyed on tasks_dir's contents, so it's safe (and
    # desirable) to share across lineages/runs that point at the same tasks_dir.
    # Default lives next to work_root rather than a hardcoded machine path.
    default_cache_dir = Path(config.get("work_root") or (REPO_ROOT / "evolution-runs"))
    cache_file = Path(
        os.environ.get("HARNESS_PIPELINE_TASK_CACHE_FILE")
        or config.get("task_instruction_cache_file")
        or (default_cache_dir / "task_instruction_index_cache.json")
    )
    cmd = [
        *_module_command("integrations.slime.resolve_task_ids"),
        "--tasks-dir", str(tasks_dir),
        "--job-dir", str(job_dir),
        "--cache-file", str(cache_file),
    ]
    _log(f"resolving dataset_task_id for evidence sessions in {job_dir}")
    result = subprocess.run(cmd, check=False, env=_child_env())
    if result.returncode != 0:
        _log(
            "WARNING: dataset_task_id resolution failed (non-fatal); held-out gate "
            "will fall back to the fixed canary set for this round"
        )


def _evidence_task_names(job_dir: Path, mixed_task_attempts: int = 1) -> list[str]:
    """Harbor task names actually behind this round's evidence trajectories
    (dataset_task_id, resolved by _resolve_evidence_task_ids above from
    on-disk instruction text). Sessions that couldn't be matched have
    dataset_task_id == None and are skipped; if none are usable, the caller
    falls back to the gate config's fixed held-out canary set.

    Tasks whose evidence came from the "mixed" selection bucket (same
    prompt won and lost across training's n_samples_per_prompt rollouts --
    i.e. proven noisy/borderline for this model) are repeated
    `mixed_task_attempts` times in the returned list. harbor has no
    per-task attempts knob (n_attempts is a single job-level value applied
    to every task), but duplicate {"path": ...} entries for the same task
    in one job's `tasks` list produce independent trials that
    run_heldout_gate_multi's _parse_task_passes already aggregates by task
    name -- so this is a drop-in way to give noisy tasks a finer-grained
    (out of N instead of out of 1) win/loss signal while pure_fail/
    pure_success tasks -- which already showed a consistent outcome across
    8 training samples -- stay at a single attempt. Attempts of the same
    task run concurrently (bounded by heldout_gate.n_concurrent), so this
    adds sandboxes, not wall-clock time, as long as concurrency headroom
    exists.
    """
    manifest_path = job_dir / "conversion_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    buckets: dict[str, set[str]] = {}
    order: list[str] = []
    for record in manifest.get("records", []):
        task_id = record.get("dataset_task_id")
        if not task_id:
            continue
        if task_id not in buckets:
            order.append(task_id)
        buckets.setdefault(task_id, set()).add(record.get("selection_bucket") or "")

    result: list[str] = []
    for task_id in order:
        weight = mixed_task_attempts if "mixed" in buckets[task_id] else 1
        result.extend([task_id] * max(1, weight))
    return result


def _evidence_task_catalog(job_dir: Path) -> dict[str, dict[str, str]]:
    """task_id -> {task_dir, image_tag} from conversion_manifest.json."""
    manifest_path = job_dir / "conversion_manifest.json"
    if not manifest_path.exists():
        return {}
    catalog: dict[str, dict[str, str]] = {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("records", []):
        task_id = record.get("dataset_task_id")
        if not task_id:
            continue
        entry = catalog.setdefault(str(task_id), {})
        task_dir = str(record.get("dataset_task_dir") or "").strip()
        image = str(record.get("dataset_image_tag") or "").strip()
        if task_dir and not entry.get("task_dir"):
            entry["task_dir"] = task_dir
        if image and not entry.get("image_tag"):
            entry["image_tag"] = image
    return catalog


def _resolve_active(config: dict, gate_config: dict) -> dict:
    """Active harness version; initializes the state file from the gate
    baseline on the very first run."""
    state_path = config["version_state"]
    if not Path(state_path).exists():
        baseline_commit = gate_config["paths"]["baseline_commit"]
        staged = Path(config.get("stage_dir") or (REPO_ROOT / "harness_sync_stage"))
        binary = staged / "binaries" / baseline_commit[:7] / "opencode"
        _log(f"version state missing; initializing at baseline {baseline_commit[:12]}")
        version_state.init_state(state_path, baseline_commit, str(binary))
    return version_state.read_active(state_path)


def _run_evolver(config: dict, run_dir: Path, job_dir: Path, base_commit: str) -> dict:
    workspaces = run_dir / "workspaces"
    cmd = [
        *_module_command("loop.from_existing_job"),
        "--config", config["evolve_config"],
        "--job-dir", str(job_dir),
        "--base-commit", base_commit,
        "--workspace-root", str(workspaces),
        "--report-root", str(run_dir / "report"),
    ]
    rollouts_per_attempt = config.get("rollouts_per_attempt")
    if rollouts_per_attempt is not None:
        if int(rollouts_per_attempt) <= 0:
            raise ValueError("rollouts_per_attempt must be positive")
        cmd.extend(["--rollouts", str(int(rollouts_per_attempt))])
    timeout = int(config.get("evolver_timeout_sec", 3600))
    _log(f"running evolver from base {base_commit[:12]} (timeout {timeout}s)")
    subprocess.run(cmd, check=True, timeout=timeout, env=_child_env())

    artifacts = [
        p for p in (workspaces / "artifacts").glob("*.json")
        if not p.name.endswith("-evolver-trajectory.json")
    ]
    if not artifacts:
        raise RuntimeError(f"evolver produced no artifact metadata under {workspaces / 'artifacts'}")
    return json.loads(max(artifacts, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))


def _run_reward_side(
    config: dict,
    *,
    run_dir: Path,
    job_dir: Path,
    active: dict,
    evolver_meta: dict,
    verdict: dict,
) -> dict[str, Any] | None:
    """Publish the accepted Harness change and optionally run reward design."""

    co_config = dict(config.get("co_evolution") or {})
    if not bool(co_config.get("enabled", False)):
        return None
    manifest = build_harness_change_manifest(
        evolver_meta,
        verdict,
        active_parent=active,
        job_dir=job_dir,
        config=config,
    )
    manifest_path = write_harness_change(run_dir / "harness_change.json", manifest)
    _log(f"published Harness/reward contract: {manifest_path}")
    reward_config = co_config.get("reward_config")
    reward_config_path = co_config.get("reward_config_path")
    if reward_config_path:
        path = Path(str(reward_config_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"co_evolution.reward_config_path does not exist: {path}")
        reward_config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reward_config, dict):
        raise TypeError(
            "co_evolution.reward_config or reward_config_path must resolve to a mapping "
            "when co_evolution.enabled=true"
        )
    reward_result = run_rubric_reward(
        reward_config=reward_config,
        harness_gate_manifest=co_config.get("harness_gate_manifest")
        or reward_config.get("harness_gate_manifest"),
        run_dir=run_dir,
    )
    write_harness_change(run_dir / "reward_cycle_result.json", reward_result)
    _log(
        "reward side completed: "
        f"trajectories={reward_result.get('summary', {}).get('trajectory_count', 0)} "
        f"output={run_dir / 'reward_cycle_result.json'}"
    )
    return reward_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pipeline YAML config")
    parser.add_argument("--rollout-id", type=int, default=None)
    args = parser.parse_args()

    config = _apply_env_overrides(_load_config(args.config))
    rollout_id = args.rollout_id
    if rollout_id is None:
        raw = os.environ.get("HARNESS_EVOLVE_ROLLOUT_ID", "")
        if not raw:
            parser.error("--rollout-id or HARNESS_EVOLVE_ROLLOUT_ID is required")
        rollout_id = int(raw)

    gate_config = load_gate_config(config["gate_config"])
    # The gate must promote/reject into the same state file the training-side
    # image prep reads.
    gate_config.setdefault("paths", {})["version_state"] = config["version_state"]

    run_dir = (
        Path(config.get("work_root") or (REPO_ROOT / "evolution-runs"))
        / f"step{rollout_id + 1:06d}_{datetime.now().strftime('%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"step {rollout_id + 1}: run dir {run_dir}")
    started = time.monotonic()

    try:
        job_dir = run_dir / "tb_job"
        _select_trajectories(config, rollout_id, job_dir)
        _resolve_evidence_task_ids(config, gate_config, job_dir)

        active = _resolve_active(config, gate_config)
        _log(f"active harness: {active['commit_short']} ({active['commit']})")

        evolver_meta = _run_evolver(config, run_dir, job_dir, active["commit"])
        accepted_candidates = evolver_meta.get("accepted_candidates") or []
        if not accepted_candidates:
            candidates = evolver_meta.get("candidates") or []
            if not candidates or all((c.get("evolver_metadata") or {}).get("error") for c in candidates):
                raise RuntimeError("evolver produced no completed candidates; see candidate errors in artifacts")
            _log(
                "no accepted candidate from evolver "
                f"(fallback_reason={evolver_meta.get('fallback_reason')}); keeping {active['commit_short']}"
            )
            return 0
        candidate_workspaces = [Path(str(c["workspace"])) for c in accepted_candidates]
        commit_list = ", ".join(str(c["commit"])[:12] for c in accepted_candidates)
        _log(f"evolver produced {len(candidate_workspaces)} candidate(s): {commit_list}")

        gate_mode = str(config.get("gate_mode", "dynamic")).strip().lower()
        mixed_task_attempts = int(config.get("mixed_task_attempts", 1))
        evidence_tasks = _evidence_task_names(job_dir, mixed_task_attempts=mixed_task_attempts)
        evidence_catalog = _evidence_task_catalog(job_dir)
        if gate_mode in {"static_only", "static-only", "static"}:
            _log("Harness gate mode: static-only (held-out dynamic gate disabled)")
        elif evidence_tasks:
            n_distinct = len(set(evidence_tasks))
            _log(
                f"held-out gate: using {n_distinct} distinct task(s) from this round's evidence "
                f"({len(evidence_tasks)} total task-attempt entries after mixed_task_attempts={mixed_task_attempts}x "
                f"weighting) instead of the fixed canary set: {', '.join(sorted(set(evidence_tasks))[:5])}"
                + (", ..." if n_distinct > 5 else "")
            )
        else:
            _log("held-out gate: no dataset_task_id in evidence (old trajectory dump); falling back to fixed canary set")
        if gate_mode in {"static_only", "static-only", "static"}:
            verdict = run_static_gate_multi(gate_config, candidate_workspaces)
        elif gate_mode in {"dynamic", "full", "heldout"}:
            verdict = run_full_gate_multi(
                gate_config,
                candidate_workspaces,
                task_names=evidence_tasks or None,
                task_catalog=evidence_catalog or None,
            )
        else:
            raise ValueError(
                "gate_mode must be one of dynamic, static_only; "
                f"got {config.get('gate_mode')!r}"
            )
        outcome = "ACCEPT" if verdict.get("accepted") else "REJECT"
        winner = verdict.get("winner_commit") or "none"
        _log(
            f"gate verdict: {outcome} winner={winner[:12] if winner != 'none' else winner} "
            f"({verdict.get('num_static_gate_survivors')}/{verdict.get('num_candidates')} passed static gate; "
            f"report: {verdict.get('report_path')})"
        )
        if verdict.get("accepted"):
            _run_reward_side(
                config,
                run_dir=run_dir,
                job_dir=job_dir,
                active=active,
                evolver_meta=evolver_meta,
                verdict=verdict,
            )
        _log(f"pipeline done in {time.monotonic() - started:.0f}s")
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        _log("pipeline FAILED; training continues on the current active harness")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
