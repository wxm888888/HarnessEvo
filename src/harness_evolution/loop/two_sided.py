#!/usr/bin/env python3
"""Run a two-sided, resumable OpenCode self-harness loop.

Each step samples only held-in tasks, generates a configurable number of
OpenCode candidates, evaluates every candidate on the fixed held-out task set
using one global OpenSandbox pool, and promotes the highest-scoring candidate
that passes the held-out non-regression gate.  The parent baseline for the
fixed held-out split is cached once per parent commit; only the rotating
held-in parent sample is refreshed between steps.
When a full held-in catalog is supplied, each step consumes a fresh disjoint
batch from that catalog, including when no candidate is promoted.

The winning candidate is sampled on held-in before the next step.  Therefore
the next diagnosis sees trajectories produced by the harness that actually
won the previous gate; held-out trajectories are never sent to the evolver.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


from harness_evolution.core.paths import default_output_root, ensure_external_paths, harnessrl_root

# A relative --run-dir is resolved below this root ($HARNESSRL_OUTPUT_ROOT);
# source repositories, model endpoints, task catalogs, and skills remain
# explicit input paths.
DEFAULT_EXPERIMENT_ROOT = default_output_root()
# The OpenSandbox gate imports the local Polar runtime.
ensure_external_paths()

from harness_evolution.loop.modular import (  # noqa: E402
    _build_memory_strategy,
    _build_strategy,
    _candidate_to_json,
    _prepare_opencode_evolver_environment,
    _task_cache_root,
    _write_json,
    generate_candidate_batch,
    load_runtime_config,
)
from harness_evolution.evaluation.gate import (  # noqa: E402
    StaticGateResult,
    _build_runtime_config,
    _resolve_baseline,
    load_gate_config,
    run_static_gate,
)
from harness_evolution.evaluation.backends.opensandbox import (  # noqa: E402
    SideOutcome,
    _promote_final_trial_artifacts,
    custom_afs_binary_path,
    run_heldout_opensandbox,
)
from harness_evolution.core.types import (  # noqa: E402
    Candidate,
)
from harness_evolution.diagnosis import EVIDENCE_REGISTRY  # noqa: E402

# Re-exported after the phase-2 split; new code should import from the owning module.
from harness_evolution.tasks.splits import _build_held_in_schedule, _load_task_catalog, _read_catalog_task_ids, _read_split, _sha256  # noqa: F401
from harness_evolution.evaluation.sampling import _collection_artifacts_for_tasks, _load_collection_artifacts, _run_opensandbox_sampling, _select_artifacts_for_tasks  # noqa: F401
from harness_evolution.evaluation.scoring import _merge_side_outcomes, _not_run_score, _score_side, _side_for_tasks, _side_from_collection, _side_from_trial_jsons, _side_is_complete  # noqa: F401
from harness_evolution.evaluation.baseline_cache import _baseline_cache_path, _baseline_cache_signature, _find_completed_resume_baseline, _find_prior_heldout_baseline, _read_baseline_cache, _run_heldout_baseline, _write_baseline_cache  # noqa: F401
from harness_evolution.selection.promotion import _candidate_score  # noqa: F401
from harness_evolution.loop.resume import _find_partial_validation_job, _load_completed_rejected_gate, _load_partial_step_candidates, _path_is_within, _prune_completed_step_artifacts, _write_candidate_checkpoint  # noqa: F401


def _run_two_sided_gate(
    *,
    gate_config: dict[str, Any],
    candidates: list[Candidate],
    parent_commit: str,
    parent_binary: str,
    held_in: list[str],
    held_out: list[str],
    repeats: int,
    pool_size: int,
    run_dir: Path,
    task_catalog: dict[str, dict[str, str]] | None = None,
    baseline_side: SideOutcome | None = None,
    prebuilt_binaries: dict[str, str] | None = None,
    resume_jobs_dir: Path | None = None,
) -> dict[str, Any]:
    """Static gate then one shared global OpenSandbox pool for held-out.

    The parent baseline can be supplied from the pre-evolution cache.  In
    that case this call spends the validation pool on candidates only.
    """
    candidate_workspaces = [candidate.workspace for candidate in candidates]
    config = copy.deepcopy(gate_config)
    paths = config.setdefault("paths", {})
    # The gate compares against the active parent for this step.  Do not let a
    # stale global version_state silently replace it. Keep gate artifacts next
    # to this step so the user's local state directory does not grow.
    paths["version_state"] = ""
    paths["baseline_commit"] = parent_commit
    paths["baseline_binary"] = parent_binary
    paths["baseline_repo"] = str(candidates[0].workspace.parent) if candidates else str(run_dir)
    paths["workspace_root"] = str(run_dir / "static-workspace")
    paths["report_root"] = str(run_dir / "gate-reports")
    # Bun typecheck/build is memory-heavy. Keep these two local checks
    # sequential to avoid the SIGKILLs seen when several candidates build at
    # once; the expensive sandbox validation below still uses one global pool.
    if prebuilt_binaries:
        static_results = [
            StaticGateResult(
                passed=True,
                candidate_commit=candidate.commit,
                changed_files=list(candidate.changed_files),
                binary_path=str(prebuilt_binaries[candidate.commit]),
            )
            for candidate in candidates
            if candidate.commit in prebuilt_binaries and Path(prebuilt_binaries[candidate.commit]).is_file()
        ]
    else:
        static_results = (
            [run_static_gate(config, workspace) for workspace in candidate_workspaces]
            if candidate_workspaces
            else []
        )
    static_by_commit = {
        result.candidate_commit: result
        for result in static_results
        if result.passed and result.binary_path
    }
    candidate_pairs = [
        (candidate.commit, static_by_commit[candidate.commit].binary_path)
        for candidate in candidates
        if candidate.commit in static_by_commit
    ]

    report: dict[str, Any] = {
        "parent_commit": parent_commit,
        "held_in_tasks": held_in,
        "held_out_tasks": held_out,
        "repeats": repeats,
        "pool_size": pool_size,
        "static_results": [asdict(item) for item in static_results],
        "static_reused": bool(prebuilt_binaries),
        "candidates": {},
        "accepted": False,
        "winner_commit": "",
    }
    if not candidate_pairs:
        report["reason"] = "no candidate passed static gate"
        _write_json(run_dir / "gate.json", report)
        return report

    heldout = config.setdefault("heldout_gate", {})
    if str(heldout.get("backend", "opensandbox")).lower() != "opensandbox":
        raise ValueError("opencode_self_harness_loop currently requires heldout_gate.backend=opensandbox")
    heldout["n_attempts"] = repeats
    heldout["n_concurrent"] = pool_size
    heldout["jobs_dir"] = str(run_dir / "validation-jobs")

    # Held-in is sampled separately after promotion and supplies the next
    # diagnosis/evolution input. It is intentionally not part of candidate
    # validation because the batch rotates every step.
    validation_tasks = list(dict.fromkeys(held_out))
    opensandbox = heldout.get("opensandbox") or {}
    setup = str(
        os.environ.get("HARNESSRL_OPENCODE_SETUP")
        or opensandbox.get("opencode_setup")
        or heldout.get("opencode_setup")
        or os.environ.get("OPENCODE_SETUP")
        or "afs_bin"
    ).strip().lower()
    afs_paths: dict[str, str] = {}
    if setup == "afs_bin":
        afs_root = str(
            opensandbox.get("opencode_afs_root")
            or heldout.get("opencode_afs_root")
            or os.environ.get("OPENCODE_AFS_ROOT")
            or "/sandbox-data/duchenyu/opencode"
        )
        run_tag = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_dir.name}"
        for index, candidate in enumerate(candidates, 1):
            afs_paths[candidate.commit] = custom_afs_binary_path(
                afs_root,
                run_tag=run_tag,
                side_label=f"candidate-{index:02d}",
                commit=candidate.commit,
            )
        if baseline_side is None:
            afs_paths[parent_commit] = custom_afs_binary_path(
                afs_root,
                run_tag=run_tag,
                side_label="baseline",
                commit=parent_commit,
            )
        _write_json(run_dir / "afs-paths.json", afs_paths)
    raw = run_heldout_opensandbox(
        config,
        candidate_pairs,
        task_names=validation_tasks,
        task_catalog=task_catalog,
        include_baseline=baseline_side is None,
        afs_paths=afs_paths,
        resume_jobs_dir=resume_jobs_dir,
    )
    if baseline_side is None:
        baseline_side = raw["baseline"]
    baseline_out = _score_side(baseline_side, held_out, repeats)
    report["validation_tasks"] = validation_tasks
    report["held_in_evaluated"] = False
    report["baseline"] = {
        "held_in": _not_run_score(held_in, reason="held_in_not_run_in_gate"),
        "held_out": baseline_out,
    }
    if not baseline_out["complete"]:
        report["reason"] = "parent held-out baseline evaluation incomplete"
        _write_json(run_dir / "gate.json", report)
        return report

    eligible: list[tuple[Candidate, dict[str, Any]]] = []
    candidate_by_commit = {candidate.commit: candidate for candidate in candidates}
    for commit, side in raw["candidates"].items():
        score = {
            "held_in": _not_run_score(held_in, reason="held_in_not_run_in_gate"),
            "held_out": _score_side(side, held_out, repeats),
        }
        # Promotion is decided only on the fixed 89-task held-out suite.
        # Held-in is intentionally rotating TMAX evidence for the next
        # evolution step, so comparing its score across steps is invalid.
        out_ok = score["held_out"]["complete"] and score["held_out"]["rate"] >= baseline_out["rate"]
        strict = score["held_out"]["rate"] > baseline_out["rate"]
        accepted = bool(out_ok and strict)
        score["accepted"] = accepted
        score["promotion_basis"] = "held_out_only"
        score["score"] = _candidate_score(score)
        score["commit"] = commit
        report["candidates"][commit] = score
        if accepted:
            eligible.append((candidate_by_commit[commit], score))

    winner_side: SideOutcome | None = None
    if eligible:
        winner, winner_score = max(eligible, key=lambda item: _candidate_score(item[1]))
        report["accepted"] = True
        report["winner_commit"] = winner.commit
        report["winner_candidate_idx"] = winner.candidate_idx
        report["winner_score"] = winner_score
        # Reuse the winner's held-out trials as the next parent baseline
        # instead of launching another 89-task baseline run.
        winner_side = raw["candidates"].get(winner.commit)
    else:
        report["reason"] = "no candidate passed held-out non-regression gate"
    _write_json(run_dir / "gate.json", report)
    # Keep the in-memory side out of gate.json; it contains dataclass objects
    # and is only needed by the loop to seed the next baseline cache.
    if winner_side is not None:
        report["_winner_side"] = winner_side
    return report


def _ensure_parent_binary(gate_config: dict[str, Any], workspace: Path, commit: str, binary: str) -> str:
    if binary and Path(binary).is_file():
        return str(Path(binary).resolve())
    # AFS avoids uploading a binary into every evaluation sandbox, but the
    # loop still needs a local binary when it publishes the next sampling
    # batch. Rebuild a missing artifact for every setup mode.
    print(
        f"[binary] local parent binary is unavailable for {commit[:12]}; "
        f"rebuilding from {workspace}",
        flush=True,
    )
    runtime_config = _build_runtime_config(gate_config)
    built = evolver_build(runtime_config, workspace)
    return str(built)


def _persist_parent_binary(run_dir: Path, commit: str, binary: str) -> str:
    """Copy an active parent binary outside disposable step directories."""
    source = Path(binary).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"cannot persist missing parent binary: {source}")
    destination = run_dir / "parent-binaries" / commit / "opencode"
    if destination.is_file():
        destination.chmod(destination.stat().st_mode | 0o111)
        return str(destination.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".opencode.tmp-{os.getpid()}"
    try:
        shutil.copy2(source, temporary)
        temporary.chmod(temporary.stat().st_mode | 0o111)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    _write_json(
        destination.parent / "metadata.json",
        {
            "schema": "opencode_self_harness_loop.parent_binary.v1",
            "commit": commit,
            "source": str(source),
            "binary": str(destination),
        },
    )
    print(f"[binary] persisted parent {commit[:12]} at {destination}", flush=True)
    return str(destination.resolve())


def evolver_build(runtime_config: Any, workspace: Path) -> Path:
    # Kept in a tiny wrapper so baseline build errors point at this loop.
    from harness_evolution.loop.modular import build_opencode_binary_from_source

    return build_opencode_binary_from_source(runtime_config, workspace)


def _bind_loop_output_roots(config: Any, run_dir: Path) -> None:
    """Bind generated loop artifacts below ``run_dir``.

    Older YAMLs contained absolute output paths at the user's GPFS2 home
    root. Keeping those values scattered reports, task caches, XDG state, and
    sampling jobs across several directories. This binding changes generated
    state only; model/source/task input paths remain untouched.
    """
    workspace_root = run_dir / "workspaces"
    report_root = run_dir / "reports"
    task_cache_root = run_dir / "task-cache"
    jobs_root = run_dir / "sampling-jobs"
    xdg_root = run_dir / "xdg"
    evolver_xdg_root = run_dir / "opencode-evolver-xdg"
    diagnosis_root = run_dir / "diagnosis-cache"

    config.workspace_root = workspace_root
    config.report_root = report_root
    config.terminal_bench["jobs_root"] = str(jobs_root)

    # The original smoke YAMLs pinned OpenCode to a dated source worktree.
    # That worktree is disposable and may be removed between resumes.  The
    # evidence analyzer and evolver both receive the live parent worktree at
    # call time, so remove stale explicit pins and let those call sites use
    # current_harness_path instead.
    if isinstance(config.opencode_evolver, dict):
        client_init = config.opencode_evolver.get("client_init")
        if isinstance(client_init, dict):
            client_init.pop("opencode_repo_path", None)
    if isinstance(config.evidence, dict):
        options = config.evidence.get("options")
        if isinstance(options, dict):
            analyzer = options.get("analyzer")
            if isinstance(analyzer, dict):
                analyzer.pop("opencode_repo_path", None)

    if isinstance(config.build, dict):
        config.build["xdg_root"] = str(xdg_root)
    if isinstance(config.opencode_evolver, dict):
        config.opencode_evolver["task_cache_root"] = str(task_cache_root)
        config.opencode_evolver["xdg_root"] = str(evolver_xdg_root)
    if isinstance(config.evidence, dict):
        options = config.evidence.setdefault("options", {})
        if isinstance(options, dict):
            options["result_root"] = str(diagnosis_root)
            analyzer = options.get("analyzer")
            if isinstance(analyzer, dict):
                analyzer["xdg_root"] = str(diagnosis_root / "analyzer-xdg")

    # Keep the raw mapping consistent for strategies that inspect it.
    raw = getattr(config, "raw", None)
    if not isinstance(raw, dict):
        return
    loop = raw.setdefault("loop", {})
    if isinstance(loop, dict):
        loop["workspace_root"] = str(workspace_root)
        loop["report_root"] = str(report_root)
    tb = raw.setdefault("terminal_bench", {})
    if isinstance(tb, dict):
        tb["jobs_root"] = str(jobs_root)
    build = raw.setdefault("build", {})
    if isinstance(build, dict):
        build["xdg_root"] = str(xdg_root)
    evolver = raw.setdefault("opencode_evolver", {})
    if isinstance(evolver, dict):
        evolver["task_cache_root"] = str(task_cache_root)
        evolver["xdg_root"] = str(evolver_xdg_root)
        client_init = evolver.get("client_init")
        if isinstance(client_init, dict):
            client_init.pop("opencode_repo_path", None)
    evidence = raw.setdefault("evidence", {})
    if isinstance(evidence, dict):
        options = evidence.setdefault("options", {})
        if isinstance(options, dict):
            options["result_root"] = str(diagnosis_root)
            analyzer = options.get("analyzer")
            if isinstance(analyzer, dict):
                analyzer["xdg_root"] = str(diagnosis_root / "analyzer-xdg")
                analyzer.pop("opencode_repo_path", None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="modular OpenCode evolver YAML")
    parser.add_argument("--gate-config", required=True, type=Path, help="OpenSandbox gate YAML")
    parser.add_argument("--held-in", required=False, type=Path, help="fixed held-in JSON split (legacy mode)")
    parser.add_argument(
        "--held-in-catalog",
        type=Path,
        help="full catalog from which a fresh held-in batch is drawn each step",
    )
    parser.add_argument("--held-in-size", type=int, default=32, help="tasks per held-in batch")
    parser.add_argument("--held-in-seed", type=int, default=0, help="deterministic seed for held-in batches")
    parser.add_argument("--held-out", required=True, type=Path)
    parser.add_argument(
        "--task-catalog",
        type=Path,
        action="append",
        default=None,
        help="JSONL catalog carrying task_dir and image_tag metadata; repeat for multiple datasets",
    )
    parser.add_argument(
        "--gate-model",
        default=None,
        help="override gate model; comma-separated replica model names are distributed across trials",
    )
    parser.add_argument("--gate-gateway", default=None, help="override gate gateway; defaults to evolver endpoint")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="root for relative --run-dir values (default: %(default)s)",
    )
    parser.add_argument("--steps", type=int, default=10, help="evolution steps (default: 10)")
    parser.add_argument("--candidates", type=int, default=None, help="candidate count per step (default: YAML or 2)")
    parser.add_argument(
        "--trajectory-retention-steps",
        type=int,
        default=2,
        help="keep raw artifacts for this many completed steps; default 2 (delete step n after n+2)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="legacy default for both sampling and gate attempts per task",
    )
    parser.add_argument(
        "--sampling-repeats",
        type=int,
        default=None,
        help="held-in sampling attempts per task; defaults to --repeats",
    )
    parser.add_argument(
        "--gate-repeats",
        type=int,
        default=None,
        help="held-out baseline/candidate attempts per task; defaults to --repeats",
    )
    parser.add_argument("--validation-pool-size", type=int, default=8, help="one global pool across candidates/splits")
    parser.add_argument("--sampling-concurrency", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _repeat_counts(args: argparse.Namespace) -> tuple[int, int]:
    legacy = int(args.repeats)
    sampling = legacy if args.sampling_repeats is None else int(args.sampling_repeats)
    gate = legacy if args.gate_repeats is None else int(args.gate_repeats)
    return sampling, gate


def _prepare_terminal_bench(config: Any, held_in: list[str], args: argparse.Namespace) -> None:
    tb = config.terminal_bench
    tb["task_names"] = held_in
    tb.setdefault("trials_per_iteration", 1)
    tb.setdefault("parallelism", args.sampling_concurrency or 4)
    tb.setdefault("model", config.opencode_evolver.get("model", ""))
    tb.setdefault("openai_api_key", config.opencode_evolver.get("api_key", "EMPTY"))
    tb.setdefault("openai_base_url", config.opencode_evolver.get("openai_base_url", ""))
    tb["backend"] = "opensandbox"


def main() -> int:
    args = _parse_args()
    sampling_repeats, gate_repeats = _repeat_counts(args)
    if sampling_repeats < 1 or gate_repeats < 1 or args.validation_pool_size < 1:
        raise SystemExit("repeat counts and --validation-pool-size must be positive")
    if args.trajectory_retention_steps < 0:
        raise SystemExit("--trajectory-retention-steps must be non-negative")
    if args.held_in is None and args.held_in_catalog is None:
        raise SystemExit("provide either --held-in or --held-in-catalog")
    if args.held_in is not None and args.held_in_catalog is not None:
        raise SystemExit("--held-in and --held-in-catalog are mutually exclusive")
    if args.held_in_catalog is not None and sampling_repeats != 1:
        raise SystemExit("rotating held-in catalogs require --sampling-repeats 1")
    config = load_runtime_config(args.config)
    gate_config = load_gate_config(args.gate_config)
    gate_osb = gate_config.setdefault("heldout_gate", {}).setdefault("opensandbox", {})
    gate_model = args.gate_model or os.environ.get("AGENT_MODEL_NAME") or config.opencode_evolver.get("model", "")
    gate_models = [item.strip() for item in gate_model.split(",") if item.strip()]
    gate_models = [item.split("/", 1)[1] if item.startswith("openai-compatible/") else item for item in gate_models]
    evolver_client_name = str(
        config.opencode_evolver.get("client_name") or "opencode_bun_dev_client"
    )
    if gate_models:
        gate_osb["agent_model_name"] = gate_models[0]
        gate_osb["agent_model_names"] = gate_models
        # Reuse the replica pool for Bun evolver rollouts.  Previously only
        # opencode_evolver.model (usually r0) was passed to every Bun process,
        # even when --gate-model listed all replicas.
        if evolver_client_name == "opencode_bun_dev_client":
            config.opencode_evolver["models"] = [
                f"openai-compatible/{model_name}" for model_name in gate_models
            ]
    gate_gateway = args.gate_gateway or os.environ.get("AGENT_GATEWAY_URL") or config.opencode_evolver.get("openai_base_url", "")
    if gate_gateway.endswith("/v1"):
        gate_gateway = gate_gateway[:-3]
    if gate_gateway:
        gate_osb["agent_gateway_url"] = gate_gateway
    held_out = _read_split(args.held_out.expanduser().resolve())
    held_in_path = args.held_in.expanduser().resolve() if args.held_in is not None else None
    held_in_catalog_path = (
        args.held_in_catalog.expanduser().resolve() if args.held_in_catalog is not None else None
    )
    if held_in_path is not None and not held_in_path.is_file():
        raise SystemExit(f"held-in split does not exist: {held_in_path}")
    if held_in_catalog_path is not None and not held_in_catalog_path.is_file():
        raise SystemExit(f"held-in catalog does not exist: {held_in_catalog_path}")
    default_catalog = Path(
        os.environ.get(
            "PROMPT_DATA",
            str(harnessrl_root().parent / "tb21-eval-qwen36-fixed/data/tb21_polar_pjlab_images.jsonl"),
        )
    )
    catalog_paths = [path.expanduser().resolve() for path in (args.task_catalog or [default_catalog])]
    if held_in_catalog_path is not None and held_in_catalog_path not in catalog_paths:
        catalog_paths.insert(0, held_in_catalog_path)
    for catalog_path in catalog_paths:
        if not catalog_path.is_file():
            raise SystemExit(f"task catalog does not exist: {catalog_path}")
    task_catalog: dict[str, dict[str, Any]] = {}
    for catalog_path in catalog_paths:
        task_catalog.update(_load_task_catalog(catalog_path))
    source_task_ids = (
        _read_catalog_task_ids(held_in_catalog_path)
        if held_in_catalog_path is not None
        else _read_split(held_in_path)
    )
    missing_catalog = [
        task
        for task in [*held_out, *source_task_ids]
        if task not in task_catalog
    ]
    if missing_catalog:
        raise SystemExit(
            f"task catalog is missing {len(missing_catalog)} split tasks; "
            f"first entries: {', '.join(missing_catalog[:8])}"
        )
    steps = int(args.steps)
    configured_candidates = (config.raw.get("loop") or {}).get("candidates_per_step")
    candidates_per_step = int(args.candidates if args.candidates is not None else configured_candidates or 2)
    if steps < 1 or candidates_per_step < 1:
        raise SystemExit("steps and candidates must be positive")

    run_dir = args.run_dir.expanduser()
    if not run_dir.is_absolute():
        run_dir = args.output_root.expanduser().resolve() / run_dir
    run_dir = run_dir.resolve()
    _bind_loop_output_roots(config, run_dir)
    # Gate YAMLs from older runs contain absolute generated-state paths. Keep
    # build caches and gate reports isolated when a new run directory is used.
    gate_paths = gate_config.setdefault("paths", {})
    gate_paths["workspace_root"] = str(run_dir / "gate-workspaces")
    gate_paths["report_root"] = str(run_dir / "gate-reports")
    gate_paths["version_state"] = str(run_dir / "version-state.json")
    gate_config.setdefault("build", {})["xdg_root"] = str(run_dir / "gate-xdg")
    gate_config.setdefault("heldout_gate", {})["jobs_dir"] = str(run_dir / "validation-jobs")
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = run_dir / "held-in-schedule.json"
    if args.resume and schedule_path.is_file():
        schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        held_in_schedule = schedule_payload.get("batches")
        if not isinstance(held_in_schedule, list) or len(held_in_schedule) < steps:
            raise SystemExit(f"held-in schedule is missing batches for {steps} steps: {schedule_path}")
        held_in_schedule = [list(map(str, batch)) for batch in held_in_schedule[:steps]]
    else:
        held_in_schedule = _build_held_in_schedule(
            held_in_path=held_in_path,
            held_in_catalog=held_in_catalog_path,
            held_in_size=args.held_in_size,
            held_in_seed=args.held_in_seed,
            steps=steps,
        )
        _write_json(
            schedule_path,
            {
                "schema": "opencode_self_harness_loop.held_in_schedule.v1",
                "source": str(held_in_catalog_path or held_in_path),
                "seed": args.held_in_seed,
                "batch_size": args.held_in_size,
                "batches": held_in_schedule,
            },
        )
    schedule_tasks = {task for batch in held_in_schedule for task in batch}
    overlap = sorted(schedule_tasks & set(held_out))
    if overlap:
        raise SystemExit(f"held-in and held-out overlap: {', '.join(overlap[:10])}")
    # The existing modular YAMLs often describe a one-candidate smoke run.
    # This loop has its own explicit candidate-count/step controls so an old
    # config cannot silently turn a requested two-candidate, ten-step run into
    # a different experiment.
    config.rollouts_per_attempt = candidates_per_step
    config.iterations = steps
    current_held_in = list(held_in_schedule[0])
    _prepare_terminal_bench(config, current_held_in, args)
    # Keep sampling jobs with the durable loop artifacts.  This prevents an
    # old YAML's small local jobs directory from filling while the loop itself
    # is correctly writing workspaces to GPFS2.
    config.terminal_bench["jobs_root"] = str(run_dir / "sampling-jobs")
    _write_json(
        run_dir / "manifest.json",
        {
            "schema": "opencode_self_harness_loop.v1",
            "held_in": str(held_in_path) if held_in_path else None,
            "held_in_catalog": str(held_in_catalog_path) if held_in_catalog_path else None,
            "held_in_schedule": str(schedule_path),
            "held_in_size": args.held_in_size,
            "held_in_seed": args.held_in_seed,
            "held_out": str(args.held_out.resolve()),
            "held_in_sha256": _sha256(held_in_path) if held_in_path else None,
            "held_in_catalog_sha256": _sha256(held_in_catalog_path) if held_in_catalog_path else None,
            "held_out_sha256": _sha256(args.held_out.resolve()),
            "task_catalog": [str(path) for path in catalog_paths],
            "steps": steps,
            "candidates_per_step": candidates_per_step,
            "trajectory_retention_steps": args.trajectory_retention_steps,
            "repeats": args.repeats,
            "validation_pool_size": args.validation_pool_size,
            "baseline_cache_mode": "held_out_per_parent_commit",
            "sampling_repeats": sampling_repeats,
            "gate_repeats": gate_repeats,
        },
    )

    state_path = run_dir / "state.json"
    # The fixed held-out baseline is cached by parent commit.  The rotating
    # held-in baseline is kept separately and replaced after each sampling
    # batch.  This is the key distinction that prevents 89 held-out tasks
    # from being rerun merely because the next 32 held-in tasks changed.
    baseline_signature = _baseline_cache_signature(
        held_out=held_out,
        repeats=gate_repeats,
        gate_config=gate_config,
        task_catalog=task_catalog,
    )
    baseline_cache: dict[str, SideOutcome] = {}
    baseline_for_step: SideOutcome | None = None
    resume_sampling_needed = False
    if args.resume:
        if not state_path.is_file():
            raise SystemExit(f"--resume requested but state does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        current_workspace = Path(state["current_workspace"]).resolve()
        current_commit = str(state["current_commit"])
        current_binary = str(state.get("current_binary") or "")
        current_job_dir = Path(state["current_job_dir"]).resolve()
        start_step = int(state["next_step"])
        if start_step < 1 or start_step > steps + 1:
            raise SystemExit(f"invalid next_step in state: {start_step}")
        current_held_in = list(
            state.get("current_held_in")
            or (held_in_schedule[start_step - 1] if start_step <= steps else held_in_schedule[-1])
        )
        history = list(state.get("history", []))
        # A prior run may have sampled the next held-in batch before being
        # interrupted.  Promote any completed retry artifacts before loading
        # those trajectories for the resumed step.
        for retry_history in current_job_dir.glob("*/*/retry_history.json"):
            _promote_final_trial_artifacts(retry_history.parent)
        # A previous run can fail after all trial results were written but
        # before trace materialization (for example when raw proxy wire was
        # disabled).  Do not abort resume while loading that incomplete
        # collection: the current held-in batch can be sampled again below.
        try:
            loaded_artifacts = _load_collection_artifacts(current_job_dir)
            artifacts = _select_artifacts_for_tasks(loaded_artifacts, current_held_in)
            print(
                f"[resume] restricted evolution evidence to held-in tasks: "
                f"{len(artifacts)} eligible artifacts from {len(loaded_artifacts)} total eligible",
                flush=True,
            )
            resume_sampling_needed = False
        except RuntimeError as exc:
            print(
                f"[resume] current held-in collection is incomplete; "
                f"will resample it: {exc}",
                flush=True,
            )
            artifacts = []
            resume_sampling_needed = True
        resume_gate_config = copy.deepcopy(gate_config)
        resume_gate_config.setdefault("paths", {})["baseline_commit"] = current_commit
        resume_gate_config["paths"]["baseline_repo"] = str(current_workspace)
        current_binary = _ensure_parent_binary(
            resume_gate_config, current_workspace, current_commit, current_binary
        )
        current_binary = _persist_parent_binary(run_dir, current_commit, current_binary)
        if state.get("current_binary") != current_binary:
            state["current_binary"] = current_binary
            _write_json(state_path, state)
        baseline_heldout = _read_baseline_cache(
            run_dir,
            commit=current_commit,
            held_out=held_out,
            repeats=gate_repeats,
            signature=baseline_signature,
        )
        if baseline_heldout is None:
            # A process can be interrupted after the initial merged sampling
            # finished but before step 1 wrote the explicit cache file.  That
            # sampling already contains the fixed held-out results, so split
            # them out before considering another OpenSandbox run.
            initial_sample = _side_from_trial_jsons(
                current_job_dir / "sample", current_commit, label="baseline"
            )
            initial_heldout = _side_for_tasks(
                initial_sample, held_out, label="baseline", commit=current_commit
            )
            if all(item.commit == current_commit for item in initial_heldout.trials) and _side_is_complete(
                initial_heldout, held_out, gate_repeats
            ):
                baseline_heldout = initial_heldout
                _write_baseline_cache(
                    run_dir,
                    side=baseline_heldout,
                    commit=current_commit,
                    held_out=held_out,
                    repeats=gate_repeats,
                    signature=baseline_signature,
                )
        if baseline_heldout is None:
            baseline_heldout = _find_prior_heldout_baseline(
                run_dir,
                commit=current_commit,
                held_out=held_out,
                repeats=gate_repeats,
            )
            if baseline_heldout is not None:
                _write_baseline_cache(
                    run_dir,
                    side=baseline_heldout,
                    commit=current_commit,
                    held_out=held_out,
                    repeats=gate_repeats,
                    signature=baseline_signature,
                )
        if baseline_heldout is None:
            print("[resume] no complete cached 89-task baseline; running held-out baseline", flush=True)
            baseline_heldout = _run_heldout_baseline(
                gate_config=resume_gate_config,
                commit=current_commit,
                binary=current_binary,
                held_out=held_out,
                repeats=gate_repeats,
                pool_size=args.validation_pool_size,
                run_dir=run_dir / "resume-baseline",
                task_catalog=task_catalog,
            )
            _write_baseline_cache(
                run_dir,
                side=baseline_heldout,
                commit=current_commit,
                held_out=held_out,
                repeats=gate_repeats,
                signature=baseline_signature,
            )
        else:
            print(
                f"[resume] reusing held-out baseline from {baseline_heldout.job_dir} "
                f"({len(baseline_heldout.trials)} trials, errors={baseline_heldout.errors})",
                flush=True,
            )
        baseline_cache[current_commit] = baseline_heldout

        # The resumed step's current 32-task parent sample is already stored
        # in current_job_dir.  It is independent of the fixed held-out cache.
        heldin_job = current_job_dir / "sample"
        # The initial step-00 collection contains both the rotating held-in
        # tasks and the fixed held-out baseline.  On resume, keep only the
        # current held-in tasks before merging with the separately cached
        # held-out side; otherwise every held-out trial is duplicated and the
        # completeness check correctly rejects the parent baseline.
        heldin_side = _side_for_tasks(
            _side_from_trial_jsons(heldin_job, current_commit, label="baseline"),
            current_held_in,
            label="baseline",
            commit=current_commit,
        )
        if (
            not _side_is_complete(heldin_side, current_held_in, sampling_repeats)
            or resume_sampling_needed
        ):
            # A recovered candidate gate evaluates only the fixed held-out
            # suite.  Older runs may have already pruned the rotating
            # held-in sample by the time the gate is resumed; do not spend a
            # new 32-task sampling job merely to reconstruct evidence that
            # cannot affect this gate.
            recovered_for_gate = (
                _load_partial_step_candidates(
                    run_dir / f"step-{start_step:02d}",
                    parent_commit=current_commit,
                    expected_count=candidates_per_step,
                )
                if start_step <= steps and not resume_sampling_needed
                else None
            )
            if recovered_for_gate is not None:
                print(
                    "[resume] held-in sample is unavailable, but recovered candidates are ready; "
                    "skipping held-in reconstruction because the gate is held-out-only",
                    flush=True,
                )
                heldin_side = SideOutcome(label="baseline", commit=current_commit)
            else:
                print(
                    "[resume] current held-in evidence is incomplete; "
                    "sampling only the current 32 tasks",
                    flush=True,
                )
                # This must be the sampling path, rather than the held-out
                # gate helper: candidate generation needs the complete
                # trajectory artifacts produced by the sampling collector.
                resume_sampling_config = copy.deepcopy(resume_gate_config)
                resume_sampling_config.setdefault("paths", {})["baseline_commit"] = current_commit
                resume_sampling_config["paths"]["baseline_repo"] = str(current_workspace)
                collection = _run_opensandbox_sampling(
                    gate_config=resume_sampling_config,
                    task_names=current_held_in,
                    task_catalog=task_catalog,
                    commit=current_commit,
                    binary=current_binary,
                    repeats=sampling_repeats,
                    parallelism=args.validation_pool_size,
                    jobs_root=run_dir / f"step-{start_step:02d}" / "resume-sampling",
                    job_name=f"harnessRL-{run_dir.name}-step-{start_step:02d}-resume-held-in",
                    stage="resume_held_in_sampling",
                )
                current_job_dir = collection.job_dir
                artifacts = _collection_artifacts_for_tasks(collection, current_held_in)
                heldin_side = _side_from_collection(collection, current_commit)
                resume_sampling_needed = False
        baseline_for_step = _merge_side_outcomes(heldin_side, baseline_heldout, commit=current_commit)
    else:
        current_workspace = Path(config.initial_harness_path).expanduser().resolve()
        current_commit = config.initial_base_commit
        baseline = _resolve_baseline(gate_config)
        current_binary = str(baseline.get("binary_path") or "") if baseline.get("commit") == current_commit else ""
        initial_dir = run_dir / "step-00" / "sampling"
        # Preserve the original one-pool startup when both counts match. If
        # held-out needs more repeats, collect the 32 held-in trajectories
        # once and run a separate held-out baseline with the gate count.
        merge_initial_splits = sampling_repeats == gate_repeats
        initial_sampling_tasks = (
            list(dict.fromkeys([*current_held_in, *held_out]))
            if merge_initial_splits
            else list(current_held_in)
        )
        initial_gate_config = copy.deepcopy(gate_config)
        initial_gate_config.setdefault("paths", {})["baseline_commit"] = current_commit
        initial_gate_config["paths"]["baseline_repo"] = str(current_workspace)
        current_binary = _ensure_parent_binary(
            initial_gate_config, current_workspace, current_commit, current_binary
        )
        current_binary = _persist_parent_binary(run_dir, current_commit, current_binary)
        collection = _run_opensandbox_sampling(
            gate_config=initial_gate_config,
            task_names=initial_sampling_tasks,
            task_catalog=task_catalog,
            commit=current_commit,
            binary=current_binary,
            repeats=sampling_repeats,
            parallelism=args.sampling_concurrency,
            jobs_root=initial_dir,
            job_name=f"harnessRL-{run_dir.name}-step-00",
            stage="initial_sampling",
        )
        current_job_dir = collection.job_dir
        # Only held-in artifacts feed the evolver. When repeat counts match,
        # the same collection also contains the reusable held-out baseline.
        artifacts = _collection_artifacts_for_tasks(collection, current_held_in)
        initial_side = _side_from_collection(collection, current_commit)
        baseline_heldin = _side_for_tasks(
            initial_side, current_held_in, label="baseline", commit=current_commit
        )
        baseline_heldout = (
            _side_for_tasks(initial_side, held_out, label="baseline", commit=current_commit)
            if merge_initial_splits
            else _run_heldout_baseline(
                gate_config=initial_gate_config,
                commit=current_commit,
                binary=current_binary,
                held_out=held_out,
                repeats=gate_repeats,
                pool_size=args.validation_pool_size,
                run_dir=run_dir / "step-00" / "baseline",
                task_catalog=task_catalog,
            )
        )
        if not _side_is_complete(baseline_heldout, held_out, gate_repeats):
            raise RuntimeError("initial sampling did not produce a complete 89-task parent baseline")
        baseline_cache[current_commit] = baseline_heldout
        _write_baseline_cache(
            run_dir,
            side=baseline_heldout,
            commit=current_commit,
            held_out=held_out,
            repeats=gate_repeats,
            signature=baseline_signature,
        )
        baseline_for_step = _merge_side_outcomes(baseline_heldin, baseline_heldout, commit=current_commit)
        start_step = 1
        history = []
        _write_json(
            state_path,
            {
                "status": "running",
                "next_step": start_step,
                "current_workspace": str(current_workspace),
                "current_commit": current_commit,
                "current_binary": current_binary,
                "current_job_dir": str(current_job_dir),
                "current_held_in": current_held_in,
                "baseline_cache_mode": "held_out_per_parent_commit",
                "baseline_cache_signature": baseline_signature,
                "history": history,
            },
        )

    evolver_env, evolver_env_meta = _prepare_opencode_evolver_environment(config, run_dir.name)
    evidence_strategy = _build_strategy(EVIDENCE_REGISTRY, config.evidence, "agent_analyzed", config)
    memory_strategy = _build_memory_strategy(config)
    for step in range(start_step, steps + 1):
        # The evidence used to propose this step always belongs to the
        # currently selected held-in batch.  In rotating-catalog mode this is
        # a fresh batch every step, including after a rejected candidate.
        config.terminal_bench["task_names"] = list(current_held_in)
        step_dir = run_dir / f"step-{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        prebuilt_binaries: dict[str, str] | None = None
        resume_jobs_dir: Path | None = None
        completed_rejected_gate = (
            _load_completed_rejected_gate(
                step_dir,
                parent_commit=current_commit,
                held_in=current_held_in,
                held_out=held_out,
            )
            if args.resume and step == start_step and not resume_sampling_needed
            else None
        )
        recovered = (
            _load_partial_step_candidates(
                step_dir,
                parent_commit=current_commit,
                expected_count=candidates_per_step,
            )
            if completed_rejected_gate is None
            and args.resume
            and step == start_step
            and not resume_sampling_needed
            else None
        )
        if completed_rejected_gate is not None:
            candidates = []
            trajectories = []
            attempt_meta = {
                "resumed_completed_gate": True,
                "step": step,
                "gate_path": str(step_dir / "gate.json"),
            }
            print(
                f"[resume] reusing completed rejected gate for step-{step:02d}; "
                "continuing with the next held-in sample",
                flush=True,
            )
        elif recovered is not None:
            candidates, prebuilt_binaries = recovered
            resume_jobs_dir = _find_partial_validation_job(step_dir, candidates)
            trajectories = [{candidate.commit: []} for candidate in candidates]
            attempt_meta = {
                "resumed_partial_step": True,
                "step": step,
                "candidates": [_candidate_to_json(candidate) for candidate in candidates],
            }
            print(
                f"[resume] reusing {len(candidates)} candidates already built for step-{step:02d}; "
                "starting dynamic gate on the 89-task held-out suite",
                flush=True,
            )
            if resume_jobs_dir is not None:
                print(f"[resume] reusing partial validation job {resume_jobs_dir}", flush=True)
        else:
            candidates, trajectories, attempt_meta = generate_candidate_batch(
                config=config,
                evidence_strategy=evidence_strategy,
                artifacts=artifacts,
                task_cache_root=_task_cache_root(config),
                current_harness_path=str(current_workspace),
                current_base_commit=current_commit,
                iteration_idx=step - 1,
                attempt_idx=0,
                candidate_offset=0,
                evolver_env=evolver_env,
                validation_feedback="",
            )
            candidates = candidates[:candidates_per_step]
            checkpoint_path = _write_candidate_checkpoint(
                step_dir,
                parent_commit=current_commit,
                candidates=candidates,
            )
            print(
                f"candidate checkpoint: {checkpoint_path} ({len(candidates)} candidates)",
                flush=True,
            )
        gate_config_step = copy.deepcopy(gate_config)
        gate_config_step.setdefault("paths", {})["baseline_commit"] = current_commit
        gate_config_step["paths"]["baseline_repo"] = str(current_workspace)
        gate_config_step["paths"]["workspace_root"] = str(step_dir / "static-workspace")
        gate_config_step["paths"]["report_root"] = str(step_dir / "gate-reports")
        current_binary = _ensure_parent_binary(gate_config_step, current_workspace, current_commit, current_binary)
        current_binary = _persist_parent_binary(run_dir, current_commit, current_binary)
        gate = completed_rejected_gate or _run_two_sided_gate(
            gate_config=gate_config_step,
            candidates=candidates,
            parent_commit=current_commit,
            parent_binary=current_binary,
            held_in=current_held_in,
            held_out=held_out,
            repeats=gate_repeats,
            pool_size=args.validation_pool_size,
            run_dir=step_dir,
            task_catalog=task_catalog,
            baseline_side=baseline_for_step,
            prebuilt_binaries=prebuilt_binaries,
            resume_jobs_dir=resume_jobs_dir,
        )
        winner_side = gate.pop("_winner_side", None)
        winner_commit = str(gate.get("winner_commit") or "")
        winner = next((candidate for candidate in candidates if candidate.commit == winner_commit), None)
        memory_record: dict[str, Any] | None = None
        if winner is not None:
            promoted_parent_commit = current_commit
            # The winner has just completed the fixed held-out suite as a
            # candidate. Promote those exact outcomes to its parent baseline;
            # never launch a second baseline run for the same commit.
            if winner_side is not None:
                winner_heldout = _side_for_tasks(
                    winner_side,
                    held_out,
                    label="baseline",
                    commit=winner.commit,
                )
                if _side_is_complete(winner_heldout, held_out, gate_repeats):
                    baseline_cache[winner.commit] = winner_heldout
                    _write_baseline_cache(
                        run_dir,
                        side=winner_heldout,
                        commit=winner.commit,
                        held_out=held_out,
                        repeats=gate_repeats,
                        signature=baseline_signature,
                    )
            current_workspace = winner.workspace.resolve()
            current_commit = winner.commit
            static = next((item for item in gate.get("static_results", []) if item.get("candidate_commit") == winner.commit), {})
            current_binary = str(static.get("binary_path") or "")
            current_binary = _persist_parent_binary(run_dir, current_commit, current_binary)
            baseline_gate_score = gate.get("baseline")
            baseline_held_out_score = (
                baseline_gate_score.get("held_out")
                if isinstance(baseline_gate_score, dict)
                else None
            )
            winner_gate_score = gate.get("winner_score")
            winner_held_out_score = (
                winner_gate_score.get("held_out")
                if isinstance(winner_gate_score, dict)
                else None
            )
            memory_record = memory_strategy.record_promotion(
                step=step,
                parent_commit=promoted_parent_commit,
                winner=winner,
                baseline_score=baseline_held_out_score,
                winner_score=winner_held_out_score,
            )
        next_held_in = list(held_in_schedule[step]) if step < steps else None
        if next_held_in is not None:
            # Whether the candidate was promoted or rejected, move to the
            # next disjoint held-in batch.  Only this 32-task sample is fed to
            # the next diagnosis/proposal stage; the 89-task TB2.1 suite is a
            # gate-only evaluation set.
            sampling_gate_config = copy.deepcopy(gate_config_step)
            sampling_gate_config.setdefault("paths", {})["baseline_commit"] = current_commit
            sampling_gate_config["paths"]["baseline_repo"] = str(current_workspace)
            collection = _run_opensandbox_sampling(
                gate_config=sampling_gate_config,
                task_names=next_held_in,
                task_catalog=task_catalog,
                commit=current_commit,
                binary=current_binary,
                repeats=sampling_repeats,
                parallelism=args.sampling_concurrency,
                jobs_root=step_dir / "sampling",
                job_name=f"harnessRL-{run_dir.name}-step-{step:02d}-next-held-in",
                stage="next_held_in_sampling",
            )
            current_job_dir = collection.job_dir
            artifacts = _collection_artifacts_for_tasks(collection, next_held_in)
            current_held_in = next_held_in
            heldin_side = _side_from_collection(collection, current_commit)
            baseline_heldout = baseline_cache.get(current_commit)
            if baseline_heldout is None:
                baseline_heldout = _read_baseline_cache(
                    run_dir,
                    commit=current_commit,
                    held_out=held_out,
                    repeats=gate_repeats,
                    signature=baseline_signature,
                )
                if baseline_heldout is not None:
                    baseline_cache[current_commit] = baseline_heldout
            if baseline_heldout is None or not _side_is_complete(baseline_heldout, held_out, gate_repeats):
                raise RuntimeError(
                    f"missing complete held-out baseline cache for parent {current_commit[:12]} "
                    "after gate; refusing to rerun the 89-task baseline implicitly"
                )
            baseline_for_step = _merge_side_outcomes(
                _side_for_tasks(heldin_side, next_held_in, label="baseline", commit=current_commit),
                baseline_heldout,
                commit=current_commit,
            )
            trajectory_source = str(current_job_dir)
        else:
            trajectory_source = str(current_job_dir)

        record = {
            "step": step,
            "parent_commit": gate.get("parent_commit", current_commit),
            "winner_commit": winner_commit,
            "accepted": bool(winner is not None),
            "held_in_tasks": list(gate.get("held_in_tasks") or current_held_in),
            "next_held_in_tasks": next_held_in,
            "gate": gate,
            "trajectory_source_for_next_step": trajectory_source,
            "candidates": [_candidate_to_json(candidate) for candidate in candidates],
            "evolver_attempt": attempt_meta,
            "evolver_environment": evolver_env_meta,
            "duration_sec": round(time.monotonic() - started, 1),
        }
        if memory_record is not None:
            record["memory"] = memory_record
        _write_json(step_dir / "step.json", record)
        history.append({
            "step": step,
            "accepted": bool(winner is not None),
            "parent_commit": record["parent_commit"],
            "winner_commit": winner_commit,
            "winner_score": gate.get("winner_score"),
        })
        _write_json(
            state_path,
            {
                "status": "running" if step < steps else "complete",
                "next_step": step + 1,
                "current_workspace": str(current_workspace),
                "current_commit": current_commit,
                "current_binary": current_binary,
                "current_job_dir": str(current_job_dir),
                "current_held_in": current_held_in,
                "baseline_cache_mode": "held_out_per_parent_commit",
                "baseline_cache_signature": baseline_signature,
                "history": history,
            },
        )
        _prune_completed_step_artifacts(
            run_dir,
            completed_step=step,
            retention=args.trajectory_retention_steps,
            protected_paths=(Path(current_binary), current_workspace),
        )
        print(
            f"step={step}/{steps} accepted={bool(winner)} "
            f"winner={winner_commit[:12] if winner_commit else 'none'} "
            f"trajectory={trajectory_source}",
            flush=True,
        )

    _write_json(
        run_dir / "summary.json",
        {
            "status": "complete",
            "steps": steps,
            "candidates_per_step": candidates_per_step,
            "repeats": args.repeats,
            "sampling_repeats": sampling_repeats,
            "gate_repeats": gate_repeats,
            "validation_pool_size": args.validation_pool_size,
            "final_workspace": str(current_workspace),
            "final_commit": current_commit,
            "history": history,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
