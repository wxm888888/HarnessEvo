"""One evolution round: evidence in, a gated harness commit out.

:func:`evolve_once` is the whole contract the RL trainer needs. Give it an
evidence directory and the commit training currently runs, and it assembles
candidate workspaces, lets the evolver edit the harness, gates the results,
records what was promoted into evolution memory, and returns the commit to run
next -- the winner on ACCEPT, the parent otherwise, so the caller never has to
branch on the verdict to know what to install.

Two things stay deliberately outside:

* **The version state file.** Sandboxes re-read it on every start, so it is
  the trainer's to write; this function only returns what to write. The gate
  would otherwise promote into a file the caller also owns.
* **Nothing about memory.** Which memory strategy runs, where its file lives
  and when a promotion is recorded are all decided here from the evolve
  config. The trainer neither configures nor sees it.

The evolver itself runs in a child interpreter. That is not incidental: it
needs Codex and this package's own dependencies, and a crash inside it must
not take down the training process that called in.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from harness_evolution.core.paths import REPO_ROOT, ensure_external_paths

ensure_external_paths()

from harness_evolution.core.config import load_runtime_config  # noqa: E402
from harness_evolution.core.strategies import _build_memory_strategy  # noqa: E402
from harness_evolution.core.types import (  # noqa: E402
    Candidate,
    DiffStats,
    RuntimeConfig,
    ValidationResult,
)
from harness_evolution.evaluation.gate_common import load_gate_config  # noqa: E402

from harness_dist import afs as harness_afs  # noqa: E402

_STATIC_ONLY_MODES = {"static", "static_only", "static-only"}
_DYNAMIC_MODES = {"dynamic", "full", "heldout"}


def _log(message: str) -> None:
    print(f"[evolve {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# The gate implementations transitively import the whole evolver loop, so they
# are pulled in only when a round actually reaches the gate. That keeps the
# config and bookkeeping half of this module importable on its own -- it is
# what :mod:`harness_evolution.rl` re-exports next to the selection helpers.


def _full_gate(
    gate_config: dict[str, Any],
    workspaces: list[Path],
    *,
    task_names: list[str] | None,
    task_catalog: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    from harness_evolution.evaluation.gate import run_full_gate_multi

    return run_full_gate_multi(
        gate_config, workspaces, task_names=task_names, task_catalog=task_catalog
    )


def _static_gate(gate_config: dict[str, Any], workspaces: list[Path]) -> dict[str, Any]:
    from harness_evolution.evaluation.gate import run_static_gate_multi

    return run_static_gate_multi(gate_config, workspaces)


@dataclass(frozen=True)
class EvolveConfig:
    """What one round needs beyond its evidence and parent commit."""

    evolve_config: Path
    """The evolver (loop) YAML: diagnosis strategy, evolver client, memory."""

    gate_config: Path
    """The gate YAML: static checks, held-out tasks, build settings."""

    gate_mode: str = "dynamic"
    """``dynamic`` for static + held-out A/B, ``static_only`` for local checks."""

    rollouts_per_attempt: int | None = None
    """Concurrent candidates per attempt; ``None`` keeps the evolver YAML's."""

    evolver_timeout_sec: int = 3600

    require_publish: bool = True
    """Refuse to start when an ACCEPT could not be published to AFS."""

    evolver: str = ""
    """Which ``evolver_presets`` entry of the evolve YAML to apply."""

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, base: Path) -> "EvolveConfig":
        """Build from an already-loaded mapping; ``base`` resolves relative paths."""
        if not isinstance(raw, dict):
            raise ValueError("evolve config root must be a mapping")

        def _path(key: str) -> Path:
            value = str(raw.get(key) or "").strip()
            if not value:
                raise ValueError(f"evolve config is missing '{key}'")
            # Relative paths resolve next to this config, so one directory can
            # be copied between lineages without rewriting every key.
            resolved = Path(value).expanduser()
            return resolved if resolved.is_absolute() else (base / resolved).resolve()

        rollouts = raw.get("rollouts_per_attempt")
        if rollouts is not None and int(rollouts) <= 0:
            raise ValueError(f"rollouts_per_attempt must be positive: {rollouts!r}")
        return cls(
            evolve_config=_path("evolve_config"),
            gate_config=_path("gate_config"),
            gate_mode=str(raw.get("gate_mode") or "dynamic").strip().lower(),
            rollouts_per_attempt=None if rollouts is None else int(rollouts),
            evolver_timeout_sec=int(raw.get("evolver_timeout_sec") or 3600),
            require_publish=bool(raw.get("require_publish", True)),
            evolver=str(raw.get("evolver") or "").strip(),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "EvolveConfig":
        resolved = Path(path).expanduser().resolve()
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"evolve config root must be a mapping: {path}")
        return cls.from_mapping(raw, base=resolved.parent)


@dataclass(frozen=True)
class EvolveResult:
    """What the trainer acts on."""

    accepted: bool
    commit: str
    """The harness to run next: the winner on ACCEPT, else ``parent_commit``."""

    parent_commit: str
    binary_path: str
    """The winner's locally built binary, to record alongside the commit."""

    workspace: str
    report_path: str
    reason: str
    """Why this round did not accept anything; empty on ACCEPT."""

    candidate_commit: str
    """A candidate this round produced -- the winner, or a representative of
    the rejected ones, so a caller recording the outcome has something to name
    other than the parent. Empty when the evolver produced nothing to gate."""

    round_index: int
    memory_recorded: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _child_env() -> dict[str, str]:
    """Child interpreters must import this package even when it is not installed."""
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    return env


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Overlay onto base, recursing into mappings and replacing everything else."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_evolve_config(config: EvolveConfig, run_dir: Path) -> Path:
    """Apply the chosen evolver preset and return the config this round runs.

    One evolve YAML carries every evolver setup under ``evolver_presets``,
    because they differ only in the diagnosis analyzer and the evolver client
    -- the policy's own sglang-served OpenCode on one side, the Codex CLI on
    the other. Keeping a separate near-identical file per setup is how the
    shared sections (initial harness, loop, validation, build) drift apart.

    The merged result is written into the round directory, so what actually ran
    stays recoverable; the launcher already renders its configs this way.
    """
    raw = yaml.safe_load(config.evolve_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"evolve config root must be a mapping: {config.evolve_config}")
    presets = raw.pop("evolver_presets", None) or {}
    if not presets:
        if config.evolver:
            raise ValueError(
                f"evolver is {config.evolver!r} but {config.evolve_config} defines no "
                "evolver_presets"
            )
        return config.evolve_config
    if not config.evolver:
        raise ValueError(
            f"{config.evolve_config} defines evolver_presets {sorted(presets)}; "
            "set `evolver` to one of them"
        )
    if config.evolver not in presets:
        raise ValueError(
            f"unknown evolver {config.evolver!r}; {config.evolve_config} offers "
            f"{sorted(presets)}"
        )
    merged = _deep_merge(raw, presets[config.evolver] or {})
    resolved_path = run_dir / "evolve.resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _log(f"evolver preset {config.evolver!r} -> {resolved_path}")
    return resolved_path


def _check_memory_is_stable(runtime_config: RuntimeConfig) -> None:
    """Refuse a memory strategy whose file would be new every round.

    ``memory.options.storage_path`` defaults to ``report_root/memory/...`` and
    every round is given its own report root, so an unset path means the
    accepted lineage restarts empty each time -- memory that silently never
    remembers. Demanding an absolute path also keeps the parent (which records
    the promotion) and the evolver child (which reads the lineage back) on the
    same file.
    """
    section = runtime_config.memory or {}
    method = str(section.get("method") or "none").strip()
    if method in {"", "none"}:
        return
    options = section.get("options") or {}
    storage = str(options.get("storage_path") or "").strip()
    if not storage or not Path(storage).expanduser().is_absolute():
        raise ValueError(
            f"memory.method is '{method}', so memory.options.storage_path must be an "
            "absolute path. It otherwise defaults to report_root/memory/, which is "
            "per-round, and the accepted lineage would start empty every round."
        )


def _prepare_gate_config(
    config: EvolveConfig, *, parent_commit: str, parent_binary: str
) -> dict[str, Any]:
    gate_config = load_gate_config(config.gate_config)
    paths = gate_config.setdefault("paths", {})
    # The caller owns the version state file (see the module docstring), so the
    # gate must not promote into it. Dropping it also means _resolve_baseline
    # falls through to the explicit baseline below instead of reading an active
    # harness that may not be the parent this round was asked to evolve.
    paths.pop("version_state", None)
    paths["baseline_commit"] = parent_commit
    if parent_binary:
        paths["baseline_binary"] = parent_binary
    return gate_config


def _resolve_evidence_task_ids(gate_config: dict[str, Any], evidence_dir: Path, run_dir: Path) -> None:
    """Best effort: recover each evidence session's dataset task id.

    The trajectory dump does not carry one, so this matches instruction text
    against the gate's tasks_dir. Failing is fine -- the held-out gate then
    falls back to its fixed canary set, exactly as if this never ran.
    """
    tasks_dir = str((gate_config.get("heldout_gate") or {}).get("tasks_dir") or "").strip()
    if tasks_dir.startswith("__") and tasks_dir.endswith("__"):
        tasks_dir = ""
    tasks_dir = tasks_dir or os.environ.get("HARNESS_GATE_TASKS_DIR", "").strip()
    if not tasks_dir:
        _log("skipping task id resolution: no heldout_gate.tasks_dir")
        return
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "harness_evolution.integrations.slime.resolve_task_ids",
            "--tasks-dir",
            tasks_dir,
            "--job-dir",
            str(evidence_dir),
            # The index is keyed on tasks_dir's contents, so it is shared
            # across rounds rather than rebuilt per round dir.
            "--cache-file",
            str(run_dir.parent / "task_instruction_index_cache.json"),
        ],
        check=False,
        env=_child_env(),
    )
    if result.returncode != 0:
        _log("WARNING: task id resolution failed; held-out gate falls back to its canary set")


def _evidence_task_names(evidence_dir: Path) -> list[str]:
    """The distinct dataset tasks behind this round's evidence."""
    manifest_path = evidence_dir / "conversion_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for record in manifest.get("records", []):
        task_id = str(record.get("dataset_task_id") or "").strip()
        if task_id and task_id not in names:
            names.append(task_id)
    return names


def _evidence_task_catalog(evidence_dir: Path) -> dict[str, dict[str, str]]:
    """Task metadata needed to reconstruct the exact held-out task."""
    manifest_path = evidence_dir / "conversion_manifest.json"
    if not manifest_path.is_file():
        return {}
    catalog: dict[str, dict[str, str]] = {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("records", []):
        task_id = str(record.get("dataset_task_id") or "").strip()
        if not task_id:
            continue
        entry = catalog.setdefault(task_id, {})
        task_dir = str(record.get("dataset_task_dir") or "").strip()
        task_file = str(record.get("dataset_task_file") or "").strip()
        task_sha256 = str(record.get("dataset_task_sha256") or "").strip()
        mimo_dataset_type = str(record.get("dataset_mimo_dataset_type") or "").strip()
        image = str(record.get("dataset_image_tag") or "").strip()
        if task_dir and not entry.get("task_dir"):
            entry["task_dir"] = task_dir
        if task_file and not entry.get("task_file"):
            entry["task_file"] = task_file
        if task_sha256 and not entry.get("task_sha256"):
            entry["task_sha256"] = task_sha256
        if mimo_dataset_type and not entry.get("mimo_dataset_type"):
            entry["mimo_dataset_type"] = mimo_dataset_type
        if image and not entry.get("image_tag"):
            entry["image_tag"] = image
    return catalog


def _run_evolver(
    config: EvolveConfig,
    *,
    run_dir: Path,
    evidence_dir: Path,
    parent_commit: str,
    evolve_config: Path,
) -> dict[str, Any]:
    """Produce candidate worktrees from the evidence; returns the run metadata."""
    workspace_root = run_dir / "workspaces"
    command = [
        sys.executable,
        "-m",
        "harness_evolution.loop.from_existing_job",
        "--config",
        str(evolve_config),
        "--job-dir",
        str(evidence_dir),
        "--base-commit",
        parent_commit,
        "--workspace-root",
        str(workspace_root),
        "--report-root",
        str(run_dir / "report"),
    ]
    if config.rollouts_per_attempt is not None:
        command.extend(["--rollouts", str(config.rollouts_per_attempt)])
    _log(f"evolving from {parent_commit[:12]} (timeout {config.evolver_timeout_sec}s)")
    subprocess.run(
        command, check=True, timeout=config.evolver_timeout_sec, env=_child_env()
    )

    artifact_root = workspace_root / "artifacts"
    artifacts = [
        path
        for path in artifact_root.glob("*.json")
        if not path.name.endswith("-evolver-trajectory.json")
    ]
    if not artifacts:
        raise RuntimeError(f"evolver produced no run metadata under {artifact_root}")
    newest = max(artifacts, key=lambda path: path.stat().st_mtime)
    return json.loads(newest.read_text(encoding="utf-8"))


def _candidate_from_metadata(record: dict[str, Any]) -> Candidate:
    """Rebuild the winner as a Candidate so memory can summarise it.

    The evolver ran in a child process, so its in-memory candidates are gone;
    everything the memory record needs was serialised into the run metadata,
    and the candidate report it quotes still sits in the worktree.
    """
    validation = record.get("validation") or {}
    diff_stats = record.get("diff_stats") or {}
    known = {field: diff_stats[field] for field in ("files", "added", "deleted", "score") if field in diff_stats}
    return Candidate(
        candidate_idx=int(record.get("candidate_idx") or 0),
        attempt_idx=int(record.get("attempt_idx") or 0),
        rollout_idx=int(record.get("rollout_idx") or 0),
        workspace=Path(str(record.get("workspace") or "")),
        branch=str(record.get("branch") or ""),
        commit=str(record.get("commit") or ""),
        commit_created=bool(record.get("commit_created")),
        changed_files=[str(item) for item in (record.get("changed_files") or [])],
        summary=str(record.get("summary") or ""),
        trajectory=[],
        evidence=record.get("evidence") or {},
        skill_copy=record.get("skill_copy") or {},
        evolver_metadata=record.get("evolver_metadata") or {},
        validation=ValidationResult(
            enabled=bool(validation.get("enabled")), passed=bool(validation.get("passed"))
        ),
        diff_stats=DiffStats(**known),
    )


def _heldout_sides(verdict: dict[str, Any], winner_commit: str) -> tuple[dict | None, dict | None]:
    """Baseline and winner pass counts over the gate's held-out tasks.

    Memory keeps a compact score per promotion so a later round can see how
    much an ancestor actually won by. The per-candidate held-out record stores
    the comparison task by task, which is where these totals come from; a
    static-only round has no held-out side and records none.
    """
    for candidate in verdict.get("candidates", []):
        static_gate = candidate.get("static_gate") or {}
        if static_gate.get("candidate_commit") != winner_commit:
            continue
        heldout = candidate.get("heldout_gate")
        if not isinstance(heldout, dict):
            return None, None
        tasks = heldout.get("tasks") or []
        attempts = sum(int(task.get("n_attempts") or 0) for task in tasks)
        if not attempts:
            return None, None
        baseline_passes = sum(int(task.get("baseline_passes") or 0) for task in tasks)
        candidate_passes = sum(int(task.get("candidate_passes") or 0) for task in tasks)
        return (
            {
                "passed_attempts": baseline_passes,
                "total_attempts": attempts,
                "rate": baseline_passes / attempts,
                "complete": True,
            },
            {
                "passed_attempts": candidate_passes,
                "total_attempts": attempts,
                "rate": candidate_passes / attempts,
                "complete": True,
            },
        )
    return None, None


def _record_memory(
    runtime_config: RuntimeConfig,
    *,
    round_index: int,
    parent_commit: str,
    winner_commit: str,
    evolver_meta: dict[str, Any],
    verdict: dict[str, Any],
) -> bool:
    """Append this promotion to evolution memory. Returns whether it landed."""
    strategy = _build_memory_strategy(runtime_config)
    if not getattr(strategy, "enabled", False):
        return False
    record = next(
        (
            item
            for item in (evolver_meta.get("candidates") or [])
            if str(item.get("commit") or "") == winner_commit
        ),
        None,
    )
    if record is None:
        # The gate promoted a commit the evolver did not report. Memory is an
        # aid, not the verdict, so say so and keep the accepted commit.
        _log(f"WARNING: no evolver record for winner {winner_commit[:12]}; memory not updated")
        return False
    baseline_score, winner_score = _heldout_sides(verdict, winner_commit)
    strategy.record_promotion(
        step=round_index,
        parent_commit=parent_commit,
        winner=_candidate_from_metadata(record),
        baseline_score=baseline_score,
        winner_score=winner_score,
    )
    return True


def evolve_once(
    evidence_dir: Path | str,
    parent_commit: str,
    *,
    config: EvolveConfig,
    run_dir: Path | str,
    round_index: int,
    parent_binary: str = "",
) -> EvolveResult:
    """Run one evolution round and return the harness to train on next.

    ``parent_commit`` is the harness training currently runs; ``parent_binary``
    is the binary recorded alongside it, which the held-out gate needs to run
    the baseline side of the A/B (empty keeps the gate config's own baseline).
    ``round_index`` only labels the round in reports and memory.

    Raises when the round could not be carried out -- a missing config, an
    evolver crash, a gate that cannot run, a winner that cannot be published.
    A round that simply found nothing worth promoting is not a failure: it
    returns ``accepted=False`` with ``commit == parent_commit``.
    """
    evidence_dir = Path(evidence_dir).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    parent_commit = str(parent_commit or "").strip()
    if not parent_commit:
        raise ValueError("parent_commit is required: a round evolves from a known harness")
    if not evidence_dir.is_dir():
        raise ValueError(f"evidence directory does not exist: {evidence_dir}")
    if config.gate_mode not in _STATIC_ONLY_MODES | _DYNAMIC_MODES:
        raise ValueError(
            "gate_mode must be one of dynamic, static_only; got "
            f"{config.gate_mode!r}"
        )
    # An accepted commit no sandbox can install is worse than no round at all,
    # and publication is decided by the environment, so check it before
    # spending an evolver run rather than after.
    if config.require_publish and not harness_afs.should_publish():
        raise RuntimeError(
            "AFS publication is disabled, so an accepted harness would not be "
            "installable by any sandbox: set HARNESS_AFS_PUBLISH=1 (or "
            "require_publish: false to evolve without publishing)"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    evolve_config = resolve_evolve_config(config, run_dir)
    runtime_config = load_runtime_config(evolve_config)
    _check_memory_is_stable(runtime_config)
    gate_config = _prepare_gate_config(
        config, parent_commit=parent_commit, parent_binary=parent_binary
    )

    _resolve_evidence_task_ids(gate_config, evidence_dir, run_dir)
    evolver_meta = _run_evolver(
        config,
        run_dir=run_dir,
        evidence_dir=evidence_dir,
        parent_commit=parent_commit,
        evolve_config=evolve_config,
    )

    accepted_candidates = evolver_meta.get("accepted_candidates") or []
    if not accepted_candidates:
        candidates = evolver_meta.get("candidates") or []
        if not candidates or all(
            (candidate.get("evolver_metadata") or {}).get("error") for candidate in candidates
        ):
            raise RuntimeError(
                "evolver produced no completed candidate; see the candidate errors in "
                f"{run_dir / 'workspaces' / 'artifacts'}"
            )
        reason = str(evolver_meta.get("fallback_reason") or "no candidate passed validation")
        _log(f"no candidate to gate ({reason}); keeping {parent_commit[:12]}")
        return EvolveResult(
            accepted=False,
            commit=parent_commit,
            parent_commit=parent_commit,
            binary_path="",
            workspace="",
            report_path="",
            reason=reason,
            candidate_commit="",
            round_index=round_index,
            memory_recorded=False,
        )

    workspaces = [Path(str(candidate["workspace"])) for candidate in accepted_candidates]
    _log(
        f"gating {len(workspaces)} candidate(s): "
        + ", ".join(str(candidate["commit"])[:12] for candidate in accepted_candidates)
    )

    if config.gate_mode in _STATIC_ONLY_MODES:
        _log("gate mode: static-only (no held-out A/B)")
        verdict = _static_gate(gate_config, workspaces)
    else:
        task_names = _evidence_task_names(evidence_dir)
        if task_names:
            _log(f"held-out gate on this round's {len(task_names)} evidence task(s)")
        else:
            _log("held-out gate: no dataset task id in the evidence; using the canary set")
        verdict = _full_gate(
            gate_config,
            workspaces,
            task_names=task_names or None,
            task_catalog=_evidence_task_catalog(evidence_dir) or None,
        )

    report_path = str(verdict.get("report_path") or "")
    if not verdict.get("accepted"):
        reason = (
            f"{verdict.get('num_static_gate_survivors')}/{verdict.get('num_candidates')} "
            "candidates cleared the static gate, none cleared the gate"
        )
        _log(f"REJECT: keeping {parent_commit[:12]} (report: {report_path})")
        return EvolveResult(
            accepted=False,
            commit=parent_commit,
            parent_commit=parent_commit,
            binary_path="",
            workspace="",
            report_path=report_path,
            reason=reason,
            candidate_commit=str(accepted_candidates[0].get("commit") or ""),
            round_index=round_index,
            memory_recorded=False,
        )

    winner_commit = str(verdict.get("winner_commit") or "")
    binary_path = str(verdict.get("winner_binary_path") or "")
    if not winner_commit or not binary_path:
        # The gate accepted without naming what it accepted; promoting that
        # would point training at a harness nobody can fetch.
        raise RuntimeError(
            f"gate accepted but reported no winner commit/binary; see {report_path}"
        )
    memory_recorded = _record_memory(
        runtime_config,
        round_index=round_index,
        parent_commit=parent_commit,
        winner_commit=winner_commit,
        evolver_meta=evolver_meta,
        verdict=verdict,
    )
    _log(f"ACCEPT: {winner_commit[:12]} (report: {report_path})")
    return EvolveResult(
        accepted=True,
        commit=winner_commit,
        parent_commit=parent_commit,
        binary_path=binary_path,
        workspace=str(verdict.get("winner_workspace") or ""),
        report_path=report_path,
        reason="",
        candidate_commit=winner_commit,
        round_index=round_index,
        memory_recorded=memory_recorded,
    )
