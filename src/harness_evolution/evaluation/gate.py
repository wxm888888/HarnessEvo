"""Gate module for evolved OpenCode harness candidates.

Two gate stages, run in order of increasing cost:

1. Static gate (seconds to minutes, no sandboxes):
     no-op check -> optional diff-size limit -> workspace typecheck ->
     build (binary registered per commit) -> smoke test of the built binary.

2. Held-out gate (sandbox evaluation, paired A/B):
     Bake the candidate and baseline binaries onto the held-out tasks' images
     (crane mount+append relayed through an SSH host with registry access),
     run the same held-out task set with both harnesses under the *same*
     policy model, then compare per-task: a task is a "win" when the candidate
     passes it strictly more often than the baseline, a "loss" for the
     reverse. Accept only when wins - losses >= accept_net_wins.

     A paired comparison is used instead of a fixed pass-rate threshold
     because the policy model itself keeps changing across RL steps: a fixed
     bar cannot distinguish "harness got better" from "model got better".

On rejection the caller keeps the current active harness; nothing in the
training loop changes. Every gate run writes a full JSON report.

Usage (library):
    from harness_evolution.evaluation.gate import load_gate_config, run_full_gate
    verdict = run_full_gate(config, candidate_workspace)

Usage (CLI):
    python3 opencode_harness_gate.py --config gate.yaml --workspace <worktree>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from harness_evolution.core.paths import ensure_external_paths

# Shared harness-distribution kernel (version state + image relay) lives in the
# sibling harness_dist project.
ensure_external_paths()

# Reused for the build step so gate builds land in the same commit-keyed
# binary registry the evolver loop uses.
from harness_dist import afs as harness_afs  # noqa: E402
from harness_dist import state as version_state  # noqa: E402

# Re-exported after the phase-2 split; new code should import from the owning module.
from harness_evolution.evaluation.gate_common import _git, _log, _paths, _report_root, _resolve_baseline, load_gate_config  # noqa: F401
from harness_evolution.evaluation.static_gate import GateCheck, StaticGateResult, _BINARY_DIFF_NOISE_PATTERNS, _binary_content_fingerprint, _build_runtime_config, run_static_gate, run_static_gate_batch  # noqa: F401
from harness_evolution.evaluation.backends.harbor import _attach_one_image, _cleanup_tags, _harbor_env, _image_prep_cfg, _load_env_file, _open_ssh_master, _parse_task_passes, _prebuild_side, _relay, _resolve_task_images, _run_harbor, _stage_binary, _start_filebridge_watcher, _stop_filebridge_watcher, _write_job_config  # noqa: F401


# -----------------------------------------------------------------------------
# Report dataclasses
# -----------------------------------------------------------------------------


@dataclass
class TaskComparison:
    task: str
    baseline_passes: int
    candidate_passes: int
    n_attempts: int
    outcome: str  # "win" | "loss" | "tie"


@dataclass
class HeldoutGateResult:
    passed: bool
    reason: str = ""
    wins: int = 0
    losses: int = 0
    ties: int = 0
    net: int = 0
    accept_net_wins: int = 0
    tasks: list[TaskComparison] = field(default_factory=list)
    baseline_job_dir: str = ""
    candidate_job_dir: str = ""
    baseline_errors: int = 0
    candidate_errors: int = 0
    duration_sec: float = 0.0
    candidate_commit: str = ""
    candidate_passes_total: int = 0


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def _publish_accepted_binary(commit: str, binary_path: str) -> None:
    """Push an ACCEPTed binary to OpenSandbox AFS so the next train step can copy it.

    Harbor image-bake is not the OpenSandbox consumption path. `should_publish()`
    is auto-on when HARNESS_VERSION_STATE is set (or HARNESS_AFS_PUBLISH=1).
    """
    if not harness_afs.should_publish():
        return
    if not commit or not binary_path:
        raise RuntimeError("cannot publish accepted candidate without commit and binary path")
    dest = harness_afs.publish_binary(binary_path, commit, log=_log)
    _log(f"AFS publish: {commit[:12]} -> {dest}")


def _publish_for_promotion(commit: str, binary_path: str, verdict: dict[str, Any], report_path: Path) -> None:
    try:
        _publish_accepted_binary(commit, binary_path)
    except Exception as exc:
        # Passing the quality gate is insufficient if the next rollout cannot
        # install the winner. Preserve that distinction in the durable report.
        verdict.update(accepted=False, gate_passed=True, publication_error=str(exc))
        report_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        raise


def _persist_candidate_commit(config: dict[str, Any], workspace: Path, commit: str) -> None:
    """Pull a promoted candidate's commit into the persistent baseline repo.

    Candidate commits are created inside an evolution run's ephemeral
    `shared_repo` clone (see GitHarnessManager.prepare_shared_workspace),
    which lives under evolution-runs/<run>/ and is thrown away once that run
    ends. `paths.baseline_repo` is the one persistent repo every future run
    clones from -- so unless a promoted commit is fetched back into it here,
    the next evolution round (which resolves its base commit from this
    version_state) will `git worktree add <base_commit>` in a *fresh* clone
    that never saw that commit and fail with "invalid reference". Tagging it
    keeps the object reachable (safe from gc) and tags are fetched by every
    plain `git clone`, so the object travels into every future shared_repo.
    """
    baseline_repo = Path(_paths(config)["baseline_repo"]).expanduser().resolve()
    tag = f"harness-evolved-{commit[:12]}"
    fetch = _git(baseline_repo, "fetch", str(workspace), commit)
    if fetch.returncode != 0:
        raise RuntimeError(f"could not persist candidate {commit[:12]} into baseline repo (fetch failed): {fetch.stdout.strip()}")
    tag_result = _git(baseline_repo, "tag", "-f", tag, commit)
    if tag_result.returncode != 0:
        raise RuntimeError(f"could not persist candidate {commit[:12]} into baseline repo (tag failed): {tag_result.stdout.strip()}")
    _log(f"persisted candidate {commit[:12]} into baseline repo {baseline_repo} as tag {tag}")


def _prepare_for_promotion(
    config: dict[str, Any], workspace: Path | None, commit: str,
    binary_path: str, verdict: dict[str, Any], report_path: Path,
) -> None:
    """A winner must remain Git-reachable before it can become active."""
    try:
        if workspace is None:
            raise RuntimeError(f"no workspace found for winner {commit[:12]}")
        _persist_candidate_commit(config, workspace, commit)
    except Exception as exc:
        verdict.update(accepted=False, gate_passed=True, persistence_error=str(exc))
        report_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        raise
    _publish_for_promotion(commit, binary_path, verdict, report_path)


# -----------------------------------------------------------------------------
# Static gate
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Held-out gate: image prep helpers (delegated to harness_dist.relay;
# empty/absent ssh_host in the config means direct local registry access)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Held-out gate: harbor run + paired comparison
# -----------------------------------------------------------------------------


def _score_heldout_sides(
    *,
    candidates: list[tuple[str, str]],
    distinct_task_names: list[str],
    per_task_attempts: dict[str, int],
    baseline_passes: dict[str, int],
    baseline_errors: int,
    baseline_job_dir: str,
    candidate_results: dict[str, tuple[dict[str, int], int, str]],
    accept_net_wins: int,
    duration_sec: float = 0.0,
) -> dict[str, HeldoutGateResult]:
    """Paired per-task comparison, shared by Harbor and OpenSandbox runners."""
    results: dict[str, HeldoutGateResult] = {}
    for commit, _binary in candidates:
        passes, errors, job_dir = candidate_results[commit]
        result = HeldoutGateResult(
            passed=False,
            accept_net_wins=accept_net_wins,
            candidate_commit=commit,
            baseline_job_dir=baseline_job_dir,
            candidate_job_dir=job_dir,
            baseline_errors=baseline_errors,
            candidate_errors=errors,
            candidate_passes_total=sum(passes.values()),
            duration_sec=duration_sec,
        )
        for task in distinct_task_names:
            baseline_n = baseline_passes.get(task, 0)
            candidate_n = passes.get(task, 0)
            outcome = "win" if candidate_n > baseline_n else ("loss" if candidate_n < baseline_n else "tie")
            result.tasks.append(
                TaskComparison(task, baseline_n, candidate_n, per_task_attempts.get(task, 1), outcome)
            )
            if outcome == "win":
                result.wins += 1
            elif outcome == "loss":
                result.losses += 1
            else:
                result.ties += 1
        result.net = result.wins - result.losses
        result.passed = result.net >= result.accept_net_wins
        result.reason = (
            f"wins={result.wins} losses={result.losses} ties={result.ties} "
            f"net={result.net} (accept threshold: net >= {result.accept_net_wins})"
        )
        _log(f"heldout gate: candidate {commit[:12]} {'PASSED' if result.passed else 'REJECT'} ({result.reason})")
        results[commit] = result
    return results


def run_heldout_gate(
    config: dict[str, Any],
    candidate_commit: str,
    candidate_binary: str,
    task_names: list[str] | None = None,
    task_catalog: dict[str, dict[str, str]] | None = None,
) -> HeldoutGateResult:
    """Single-candidate held-out gate (kept for backward compatibility with
    run_full_gate/run_gate_e2e.py). Delegates to the multi-candidate path."""
    results = run_heldout_gate_multi(
        config,
        [(candidate_commit, candidate_binary)],
        task_names=task_names,
        task_catalog=task_catalog,
    )
    return results[candidate_commit]


def _run_heldout_opensandbox(
    config: dict[str, Any],
    candidates: list[tuple[str, str]],
    task_names: list[str] | None,
    task_catalog: dict[str, dict[str, str]] | None,
    accept_net_wins: int,
) -> dict[str, HeldoutGateResult]:
    from harness_evolution.evaluation.backends.opensandbox import run_heldout_opensandbox

    t_start = time.monotonic()
    raw = run_heldout_opensandbox(
        config, candidates, task_names=task_names, task_catalog=task_catalog
    )
    baseline = raw["baseline"]
    return _score_heldout_sides(
        candidates=candidates,
        distinct_task_names=raw["distinct_task_names"],
        per_task_attempts=raw["per_task_attempts"],
        baseline_passes=baseline.passes,
        baseline_errors=baseline.errors,
        baseline_job_dir=baseline.job_dir,
        candidate_results={
            commit: (side.passes, side.errors, side.job_dir)
            for commit, side in raw["candidates"].items()
        },
        accept_net_wins=accept_net_wins,
        duration_sec=round(time.monotonic() - t_start, 1),
    )


def run_heldout_gate_multi(
    config: dict[str, Any],
    candidates: list[tuple[str, str]],
    task_names: list[str] | None = None,
    task_catalog: dict[str, dict[str, str]] | None = None,
) -> dict[str, HeldoutGateResult]:
    """Held-out gate for N candidates sharing a single baseline run.

    `candidates` is a list of (candidate_commit, candidate_binary_path).
    Baseline is evaluated exactly once; every candidate is evaluated against
    that shared baseline. Default backend is OpenSandbox (AFS binary copy).
    Set ``heldout_gate.backend: harbor`` to use the leftover Harbor/E2B path.

    `task_names`, when given, overrides `heldout_gate.task_names` from the
    config for this call only -- used by the evolution pipeline to evaluate
    against the exact tasks that produced this round's evidence instead of
    the fixed held-out canary set. May contain duplicate task names so a
    task repeated k times gets k independent trials (mixed_task_attempts).
    `task_catalog` optionally maps task_id -> {task_dir, image_tag}.
    """
    heldout = config["heldout_gate"]
    accept_net_wins = int(heldout.get("accept_net_wins", 1))
    if not heldout.get("enabled", True):
        return {
            commit: HeldoutGateResult(
                passed=True, reason="heldout gate disabled in config",
                accept_net_wins=accept_net_wins, candidate_commit=commit,
            )
            for commit, _binary in candidates
        }
    if not candidates:
        return {}

    from harness_evolution.evaluation.backends.opensandbox import heldout_backend

    if heldout_backend(heldout) == "opensandbox":
        return _run_heldout_opensandbox(
            config, candidates, task_names, task_catalog, accept_net_wins
        )

    t_start = time.monotonic()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    baseline = _resolve_baseline(config)
    # Derived image tags embed the commit string verbatim, and every existing
    # tag (and the baseline carrier) was created with the 7-char short form.
    baseline_short = baseline["commit_short"]
    # Dedupe by short commit: with several candidates generated concurrently
    # a short-hash collision is astronomically unlikely, but guard anyway so
    # a collision fails loudly instead of silently merging two candidates.
    candidate_sides: list[tuple[str, str, str]] = []  # (side_label, short_commit, binary)
    seen_short: dict[str, str] = {}
    for commit, binary in candidates:
        short = commit[:7]
        if short in seen_short and seen_short[short] != commit:
            raise RuntimeError(f"short-commit collision between {seen_short[short]} and {commit}")
        seen_short[short] = commit
    for i, (commit, binary) in enumerate(candidates):
        candidate_sides.append((f"candidate-{i + 1:02d}", commit[:7], binary))

    # task_names may contain duplicate entries -- callers (e.g. the evolution
    # pipeline's mixed_task_attempts weighting) use that to give specific
    # tasks more attempts than the job-level default without harbor
    # supporting a native per-task n_attempts. distinct_task_names is what
    # actually gets baked/imaged (baking a task twice is wasted work);
    # task_names (with duplicates) is what gets sent to harbor's job config
    # so it produces the right number of independent trials per task.
    task_names = list(task_names) if task_names is not None else list(heldout["task_names"])
    distinct_task_names = list(dict.fromkeys(task_names))
    n_attempts = int(heldout.get("n_attempts", 1))
    per_task_attempts = Counter(task_names)
    per_task_attempts = {task: count * n_attempts for task, count in per_task_attempts.items()}
    _log(
        f"heldout gate: {len(distinct_task_names)} distinct task(s) "
        f"({len(task_names)} task-attempt entries) x {n_attempts} attempt(s)/entry, "
        f"baseline {baseline_short} vs {len(candidate_sides)} candidate(s): "
        f"{', '.join(short for _label, short, _binary in candidate_sides)}"
    )

    images = _resolve_task_images(config, distinct_task_names)
    base_images = sorted(set(images.values()))
    tag_log = f"{_image_prep_cfg(config)['stage_dir']}/gate_tags_{run_id}.log"

    _open_ssh_master(config)
    staged_candidates = [
        (label, short, _stage_binary(config, short, binary)) for label, short, binary in candidate_sides
    ]
    # The baseline carrier normally already lives in the registry
    # (ensure_carrier short-circuits on it); staging the baseline binary too,
    # when we have it, lets the relay host recreate the carrier if needed.
    baseline_binary = baseline.get("binary_path", "")
    if baseline_binary and Path(baseline_binary).exists():
        baseline_stage = _stage_binary(config, baseline_short, baseline_binary)
    else:
        baseline_stage = str(heldout.get("baseline_binary", "/tmp/gate_baseline_unused"))

    all_sides: list[tuple[str, str, str]] = [("baseline", baseline_short, baseline_stage), *staged_candidates]
    results: dict[str, HeldoutGateResult] = {}

    try:
        # Bake images for every side (baseline + all candidates) concurrently;
        # each side still internally caps at image_prep.concurrency.
        with ThreadPoolExecutor(max_workers=len(all_sides)) as ex:
            futures = {
                ex.submit(_prebuild_side, config, label, short, stage_path, base_images, tag_log): label
                for label, short, stage_path in all_sides
            }
            errors = []
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{label}: {exc}")
            if errors:
                raise RuntimeError("image prebuild failed for: " + "; ".join(errors))

        # Run every side's harbor job concurrently against one shared
        # FileBridge watcher (it rglobs the whole jobs_dir for sandbox
        # markers, so one watcher covers all job subdirectories).
        filebridge = heldout.get("filebridge") or {}
        shared_watcher: subprocess.Popen | None = None
        if filebridge.get("enabled"):
            watcher_env = _harbor_env(config, baseline_short)
            shared_watcher = _start_filebridge_watcher(filebridge, Path(heldout["jobs_dir"]), run_id, watcher_env)

        side_job_dirs: dict[str, Path] = {}
        try:
            job_configs = {
                label: _write_job_config(config, run_id, label, task_names) for label, _short, _stage in all_sides
            }
            with ThreadPoolExecutor(max_workers=len(all_sides)) as ex:
                futures = {
                    ex.submit(
                        _run_harbor, config, job_configs[label][0], job_configs[label][1], short,
                        shared_watcher=shared_watcher,
                    ): label
                    for label, short, _stage in all_sides
                }
                for fut in as_completed(futures):
                    side_job_dirs[futures[fut]] = fut.result()
        finally:
            _stop_filebridge_watcher(shared_watcher)

        baseline_passes, baseline_errors = _parse_task_passes(side_job_dirs["baseline"])
        candidate_results = {}
        for commit, _binary in candidates:
            label = next(lbl for lbl, short, _stage in staged_candidates if short == commit[:7])
            candidate_passes, candidate_errors = _parse_task_passes(side_job_dirs[label])
            candidate_results[commit] = (candidate_passes, candidate_errors, str(side_job_dirs[label]))
        results = _score_heldout_sides(
            candidates=candidates,
            distinct_task_names=distinct_task_names,
            per_task_attempts=per_task_attempts,
            baseline_passes=baseline_passes,
            baseline_errors=baseline_errors,
            baseline_job_dir=str(side_job_dirs["baseline"]),
            candidate_results=candidate_results,
            accept_net_wins=accept_net_wins,
        )
    finally:
        if heldout.get("cleanup_tags", True):
            try:
                _cleanup_tags(config, tag_log)
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: tag cleanup failed (tags listed in {tag_log}): {exc}")

    duration = round(time.monotonic() - t_start, 1)
    for result in results.values():
        result.duration_sec = duration
    return results


# -----------------------------------------------------------------------------
# Full gate
# -----------------------------------------------------------------------------


def run_full_gate(
    config: dict[str, Any],
    workspace: Path,
    task_names: list[str] | None = None,
    task_catalog: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    static_result = run_static_gate(config, workspace)

    heldout_result: HeldoutGateResult | None = None
    if static_result.passed:
        heldout_result = run_heldout_gate(
            config,
            static_result.candidate_commit,
            static_result.binary_path,
            task_names=task_names,
            task_catalog=task_catalog,
        )

    accepted = static_result.passed and heldout_result is not None and heldout_result.passed
    verdict = {
        "accepted": accepted,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "candidate_commit": static_result.candidate_commit,
        "baseline_commit": _resolve_baseline(config)["commit"],
        "static_gate": asdict(static_result),
        "heldout_gate": asdict(heldout_result) if heldout_result is not None else None,
    }
    report_path = (
        _report_root(config)
        / f"gate_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{static_result.candidate_commit[:12] or 'unknown'}.json"
    )
    report_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    verdict["report_path"] = str(report_path)
    _log(f"verdict: {'ACCEPT' if accepted else 'REJECT'} -> {report_path}")

    # Persisting the commit and publishing the binary are what make an accepted
    # candidate *reachable*; neither depends on who records the promotion. A
    # caller that owns its own version state (the RL loop promotes from the
    # returned verdict) passes no `version_state` and must still end up with a
    # fetchable commit and an installed binary.
    if accepted:
        _prepare_for_promotion(
            config, workspace, static_result.candidate_commit,
            static_result.binary_path, verdict, report_path,
        )

    state_path = _paths(config).get("version_state")
    if state_path:
        if accepted:
            version_state.promote(
                state_path,
                static_result.candidate_commit,
                static_result.binary_path,
                gate_report_path=str(report_path),
            )
            _log(f"version state: promoted {static_result.candidate_commit[:12]} to active")
        else:
            reason = heldout_result.reason if (static_result.passed and heldout_result) else "static gate failed"
            version_state.record_rejection(
                state_path,
                static_result.candidate_commit,
                gate_report_path=str(report_path),
                reason=reason,
            )
            streak = version_state.consecutive_rejections(state_path)
            _log(f"version state: rejection recorded (consecutive rejections: {streak})")
    return verdict


def run_full_gate_multi(
    config: dict[str, Any],
    workspaces: list[Path],
    task_names: list[str] | None = None,
    task_catalog: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Best-of-N gate: run the static gate on every candidate workspace, send
    every static-gate survivor into ONE shared-baseline held-out gate run,
    and promote whichever survivor scores best -- provided it still clears
    the existing acceptance bar (net wins >= accept_net_wins). If nothing
    clears the bar, nothing is promoted, same as the single-candidate gate.

    `task_names`, when given, overrides the config's fixed held-out canary
    list for this run -- see run_heldout_gate_multi.
    """
    started = datetime.now(timezone.utc).isoformat()
    static_results = run_static_gate_batch(config, workspaces)
    workspace_by_commit = {
        sr.candidate_commit: workspace for workspace, sr in zip(workspaces, static_results) if sr.candidate_commit
    }
    for workspace, static_result in zip(workspaces, static_results):
        if not static_result.passed:
            _log(f"static gate: REJECT for {workspace} (candidate {static_result.candidate_commit[:12] or 'unknown'})")

    survivors = [sr for sr in static_results if sr.passed]
    heldout_results: dict[str, HeldoutGateResult] = {}
    winner: StaticGateResult | None = None
    if survivors:
        heldout_results = run_heldout_gate_multi(
            config,
            [(sr.candidate_commit, sr.binary_path) for sr in survivors],
            task_names=task_names,
            task_catalog=task_catalog,
        )
        passing = [sr for sr in survivors if heldout_results[sr.candidate_commit].passed]

        def _score(sr: StaticGateResult) -> tuple[int, int, int]:
            r = heldout_results[sr.candidate_commit]
            return (r.net, r.candidate_passes_total, -r.candidate_errors)

        if passing:
            winner = max(passing, key=_score)

    accepted = winner is not None
    verdict = {
        "accepted": accepted,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "num_candidates": len(workspaces),
        "num_static_gate_survivors": len(survivors),
        "baseline_commit": _resolve_baseline(config)["commit"],
        "winner_commit": winner.candidate_commit if winner else "",
        # Promoting the winner needs its binary, and a caller holding only the
        # verdict would otherwise have to find it among per-candidate records.
        "winner_binary_path": winner.binary_path if winner else "",
        "winner_workspace": str(workspace_by_commit.get(winner.candidate_commit, "")) if winner else "",
        "candidates": [
            {
                "workspace": str(workspace),
                "static_gate": asdict(sr),
                "heldout_gate": asdict(heldout_results[sr.candidate_commit]) if sr.candidate_commit in heldout_results else None,
            }
            for workspace, sr in zip(workspaces, static_results)
        ],
    }
    report_path = (
        _report_root(config)
        / f"gate_multi_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{len(workspaces)}cand.json"
    )
    report_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    verdict["report_path"] = str(report_path)
    if accepted:
        _log(f"verdict: ACCEPT winner={winner.candidate_commit[:12]} -> {report_path}")
    else:
        _log(f"verdict: REJECT (0/{len(workspaces)} candidates cleared the gate) -> {report_path}")

    # See run_full_gate: reachability of the winner is not conditional on
    # recording the promotion here.
    if winner is not None:
        winner_workspace = workspace_by_commit.get(winner.candidate_commit)
        _prepare_for_promotion(
            config, winner_workspace, winner.candidate_commit,
            winner.binary_path, verdict, report_path,
        )

    state_path = _paths(config).get("version_state")
    if state_path:
        if winner is not None:
            version_state.promote(
                state_path,
                winner.candidate_commit,
                winner.binary_path,
                gate_report_path=str(report_path),
                note=f"best of {len(workspaces)} concurrent candidates",
            )
            _log(f"version state: promoted {winner.candidate_commit[:12]} to active")
        else:
            # Record one representative rejection: the best-scoring candidate
            # that reached the held-out gate (even though it didn't clear the
            # bar), or the first candidate if none got that far. Full
            # per-candidate detail lives in the JSON report.
            if survivors and heldout_results:
                best_effort = max(survivors, key=_score)
                reason = (
                    f"best of {len(survivors)}/{len(workspaces)} candidates: "
                    f"{heldout_results[best_effort.candidate_commit].reason}"
                )
                rep_commit = best_effort.candidate_commit
            else:
                reason = f"all {len(workspaces)} candidates failed the static gate"
                rep_commit = static_results[0].candidate_commit if static_results else ""
            version_state.record_rejection(
                state_path,
                rep_commit,
                gate_report_path=str(report_path),
                reason=reason,
            )
            streak = version_state.consecutive_rejections(state_path)
            _log(f"version state: rejection recorded (consecutive rejections: {streak})")
    return verdict


def run_static_gate_multi(
    config: dict[str, Any],
    workspaces: list[Path],
) -> dict[str, Any]:
    """Promote the first statically valid candidate without held-out A/B.

    This mode is intended for Harness -> Reward experiments where the user
    explicitly wants to disable the dynamic/held-out gate.  It still runs the
    complete local safety checks (typecheck, build, and smoke when enabled),
    writes a durable report, and updates the version state in exactly the same
    way as the normal gate.  Candidate order is the evolver's stable order, so
    this does not introduce a hidden score-based selection criterion.
    """
    started = datetime.now(timezone.utc).isoformat()
    static_results = run_static_gate_batch(config, workspaces)
    workspace_by_commit = {
        sr.candidate_commit: workspace
        for workspace, sr in zip(workspaces, static_results)
        if sr.candidate_commit
    }
    survivors = [sr for sr in static_results if sr.passed]
    winner = survivors[0] if survivors else None
    verdict: dict[str, Any] = {
        "accepted": winner is not None,
        "gate_mode": "static_only",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "num_candidates": len(workspaces),
        "num_static_gate_survivors": len(survivors),
        "baseline_commit": _resolve_baseline(config)["commit"],
        "winner_commit": winner.candidate_commit if winner else "",
        "winner_binary_path": winner.binary_path if winner else "",
        "winner_workspace": str(workspace_by_commit.get(winner.candidate_commit, "")) if winner else "",
        "candidates": [
            {
                "workspace": str(workspace),
                "static_gate": asdict(sr),
                "heldout_gate": None,
            }
            for workspace, sr in zip(workspaces, static_results)
        ],
    }
    report_path = (
        _report_root(config)
        / f"gate_static_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{len(workspaces)}cand.json"
    )
    verdict["report_path"] = str(report_path)
    report_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    # See run_full_gate: reachability of the winner is not conditional on
    # recording the promotion here.
    if winner is not None:
        winner_workspace = workspace_by_commit.get(winner.candidate_commit)
        _prepare_for_promotion(
            config, winner_workspace, winner.candidate_commit,
            winner.binary_path, verdict, report_path,
        )

    state_path = _paths(config).get("version_state")
    if state_path:
        if winner is not None:
            version_state.promote(
                state_path,
                winner.candidate_commit,
                winner.binary_path,
                gate_report_path=str(report_path),
                note=f"static-only gate; first of {len(workspaces)} candidates passing local checks",
            )
            _log(f"version state: promoted {winner.candidate_commit[:12]} to active (static-only)")
        else:
            rep_commit = static_results[0].candidate_commit if static_results else ""
            version_state.record_rejection(
                state_path,
                rep_commit,
                gate_report_path=str(report_path),
                reason=f"all {len(workspaces)} candidates failed the static gate",
            )
            streak = version_state.consecutive_rejections(state_path)
            _log(f"version state: rejection recorded (consecutive rejections: {streak})")

    outcome = "ACCEPT" if winner is not None else "REJECT"
    _log(
        f"static-only verdict: {outcome} "
        f"({len(survivors)}/{len(workspaces)} candidates passed; report: {report_path})"
    )
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="gate YAML config path")
    parser.add_argument("--workspace", required=True, help="candidate worktree (committed changes)")
    parser.add_argument("--static-only", action="store_true", help="run only the static gate")
    args = parser.parse_args()

    config = load_gate_config(args.config)
    workspace = Path(args.workspace).expanduser().resolve()
    if args.static_only:
        result = run_static_gate(config, workspace)
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.passed else 1
    verdict = run_full_gate(config, workspace)
    return 0 if verdict["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
