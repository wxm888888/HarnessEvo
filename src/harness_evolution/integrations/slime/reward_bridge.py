"""Publish an accepted Harness candidate for rubric-based reward design.

The Harness evolver already produces a rich metadata artifact.  This module
turns that artifact into the small, stable ``harness_change.v1`` contract used
by reward design.  The reward side consumes an explicit
``harness_gate_pairs.v1`` manifest containing baseline/candidate trajectory
pairs; keeping that handoff explicit prevents a single unpaired trajectory
directory from silently producing a misleading rubric.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _git_changed_files(workspace: Path, parent: str, candidate: str) -> list[str]:
    """Return files changed between the Harness parent and candidate commit."""

    if not workspace.exists():
        return []
    try:
        completed = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--name-only", parent, candidate],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})


def _trajectory_paths(job_dir: str | Path) -> list[str]:
    root = Path(job_dir).expanduser().resolve()
    return [str(path.resolve()) for path in sorted(root.rglob("agent/trajectory.json"))]


def build_harness_change_manifest(
    evolver_meta: Mapping[str, Any],
    gate_verdict: Mapping[str, Any],
    *,
    active_parent: Mapping[str, Any],
    job_dir: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert an accepted Harness pipeline result to ``harness_change.v1``.

    ``config.co_evolution`` supplies semantic event names because only the
    Harness implementation knows what its new trace fields mean.  Requiring
    these names prevents a successful code diff from silently becoming an
    unusable reward signal.
    """

    co_config = dict(config.get("co_evolution") or {})
    accepted = bool(gate_verdict.get("accepted"))
    if not accepted:
        raise ValueError("cannot publish a rejected Harness candidate")

    winner = str(gate_verdict.get("winner_commit") or "").strip()
    candidates = evolver_meta.get("accepted_candidates") or []
    candidate: Mapping[str, Any] | None = None
    for item in candidates:
        if isinstance(item, Mapping) and str(item.get("commit") or "") == winner:
            candidate = item
            break
    if candidate is None and candidates and isinstance(candidates[0], Mapping):
        candidate = candidates[0]
    candidate = candidate or {}

    parent = str(
        co_config.get("parent_version")
        or active_parent.get("commit")
        or evolver_meta.get("input_base_commit")
        or ""
    ).strip()
    version = str(winner or candidate.get("commit") or evolver_meta.get("output_commit") or "").strip()
    workspace = Path(str(candidate.get("workspace") or evolver_meta.get("output_harness_path") or ""))
    changed_files = [str(item) for item in (candidate.get("changed_files") or []) if str(item).strip()]
    if not changed_files:
        changed_files = _git_changed_files(workspace, parent, version)
    if not changed_files:
        raise ValueError("accepted Harness candidate has no discoverable changed_files")

    new_capabilities = [str(item) for item in (co_config.get("new_capabilities") or []) if str(item).strip()]
    new_events = [str(item) for item in (co_config.get("new_events") or []) if str(item).strip()]
    if not new_capabilities or not new_events:
        raise ValueError(
            "co_evolution.new_capabilities and co_evolution.new_events are required "
            "to publish a Harness/reward contract"
        )

    return {
        "schema": "harness_change.v1",
        "parent_version": parent,
        "candidate_version": version,
        "changed_files": sorted(set(changed_files)),
        "new_capabilities": new_capabilities,
        "new_events": new_events,
        "event_schema_version": str(co_config.get("event_schema_version") or "harness_events.v1"),
        "expected_behavior": str(
            co_config.get("expected_behavior")
            or "make Harness verification progress and rollback observable to reward design"
        ),
        "source": {
            "evolver_run_id": evolver_meta.get("run_id"),
            "gate_report": gate_verdict.get("report_path"),
            "job_dir": str(Path(job_dir).expanduser().resolve()),
            "trajectory_paths": _trajectory_paths(job_dir),
            "candidate_workspace": str(workspace.resolve()) if str(workspace) else "",
        },
    }


def write_harness_change(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def run_rubric_reward(
    *,
    reward_config: Mapping[str, Any],
    harness_gate_manifest: str | Path | None,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Invoke the rubric pipeline using a baseline/candidate pair manifest."""

    # Import lazily so the Harness pipeline remains usable without reward
    # module dependencies when co-evolution is disabled.
    import sys

    from harness_evolution.core.paths import reward_design_root

    reward_root = reward_design_root()
    if str(reward_root) not in sys.path:
        sys.path.insert(0, str(reward_root))
    from reward_design.rubric_pipeline import run_rubric_pipeline

    config = dict(reward_config)
    co_config = dict(config.get("co_evolution") or {})
    manifest = (
        harness_gate_manifest
        or config.get("harness_gate_manifest")
        or co_config.get("harness_gate_manifest")
    )
    endpoint = config.get("endpoint") or co_config.get("endpoint")
    model = config.get("model") or co_config.get("model")
    if not manifest:
        raise ValueError(
            "rubric reward design requires co_evolution.harness_gate_manifest "
            "with paired baseline/candidate trajectories"
        )
    if not endpoint or not model:
        raise ValueError("rubric reward design requires endpoint and model")
    manifest_path = str(Path(manifest).expanduser().resolve())
    output = config.get("output") or co_config.get("output")
    if not output:
        output = Path(run_dir).expanduser().resolve() / "rubric_reward_run.json"
    contrastive_provider = str(
        config.get("contrastive_provider")
        or co_config.get("contrastive_provider")
        or "inline"
    )
    contrastive_workspace = (
        config.get("contrastive_workspace")
        or co_config.get("contrastive_workspace")
    )
    opencode_binary = str(
        config.get("opencode_binary")
        or co_config.get("opencode_binary")
        or "opencode"
    )
    return run_rubric_pipeline(
        manifest=manifest_path,
        endpoint=str(endpoint),
        model=str(model),
        output=output,
        api_key=str(config.get("api_key", co_config.get("api_key", "EMPTY"))),
        shaping_weight=float(
            config.get("shaping_weight", co_config.get("shaping_weight", 0.25))
        ),
        contrastive_provider=contrastive_provider,
        contrastive_workspace=contrastive_workspace,
        opencode_binary=opencode_binary,
    )

