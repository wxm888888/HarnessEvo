"""One evolution round, as a command the training loop can call.

slime's train.py pauses at every ``HARNESS_EVOLVE_INTERVAL`` step boundary and
runs this once:

    python -m harness_evolution.rl --config rl_evolve.yaml

Everything else comes from the environment the launcher already sets --
``POLAR_TRAJECTORY_DIR``, ``HARNESS_EVOLVE_ROLLOUT_ID``,
``HARNESS_VERSION_STATE``, ``HARNESS_BASELINE_COMMIT``/``_BINARY`` -- so the
training side owns no evolution logic at all. The round is:

  1. read the harness training currently runs from the version state file
     (initialising it from the configured baseline on the very first round)
  2. select this round's evidence from the recent rollouts
  3. evolve, gate, and record what was promoted
  4. write the accepted commit back into the version state file

Step 4 is the only thing the next rollout actually consumes: every sandbox
start re-reads that file and copies ``{OPENCODE_AFS_ROOT}/{commit7}/opencode``.
Nothing is baked into an image.

Exit code 0 covers ACCEPT, REJECT and "nothing worth gating" -- all normal
outcomes for a round. Non-zero means the round itself broke, which train.py
logs before continuing on the current harness (or raises, under
``HARNESS_EVOLVE_STRICT=1``). The last stdout line is always the result as
JSON, for a caller that wants more than the exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from harness_evolution.core.paths import ensure_external_paths

ensure_external_paths()

from harness_evolution.rl.evolve import EvolveConfig, evolve_once  # noqa: E402
from harness_evolution.rl.select import DEFAULT_WINDOW, select_evidence  # noqa: E402

from harness_dist import state as version_state  # noqa: E402

# Per-run values a launcher has to vary without editing the shared YAML. This
# replaces the fourteen HARNESS_PIPELINE_* overrides the previous pipeline
# carried: what is left is the run's own identity (where it writes, which
# rendered configs it reads) plus the round's size.
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "evolve_config": ("HARNESS_EVOLVE_EVOLVE_CONFIG", str),
    "gate_config": ("HARNESS_EVOLVE_GATE_CONFIG", str),
    "gate_mode": ("HARNESS_EVOLVE_GATE_MODE", str),
    "evolver": ("HARNESS_EVOLVE_EVOLVER", str),
    "work_root": ("HARNESS_EVOLVE_WORK_ROOT", str),
    "stage_dir": ("HARNESS_EVOLVE_STAGE_DIR", str),
    "window": ("HARNESS_EVOLVE_WINDOW", int),
    "n_fail": ("HARNESS_EVOLVE_N_FAIL", int),
    "n_success": ("HARNESS_EVOLVE_N_SUCCESS", int),
}


def _log(message: str) -> None:
    print(f"[evolve-round {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class RoundSettings:
    """How much evidence one round takes, and where it works."""

    n_fail: int
    n_success: int
    window: int
    work_root: Path
    stage_dir: Path | None


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    merged = dict(raw)
    for key, (env_name, caster) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is not None and value != "":
            merged[key] = caster(value)
    return merged


def _round_settings(raw: dict[str, Any], *, base: Path) -> RoundSettings:
    def _dir(key: str) -> Path | None:
        value = str(raw.get(key) or "").strip()
        if not value:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base / path).resolve()

    work_root = _dir("work_root")
    if work_root is None:
        raise ValueError("set work_root in the config or HARNESS_EVOLVE_WORK_ROOT")
    n_fail = int(raw.get("n_fail", 0) or 0)
    n_success = int(raw.get("n_success", 0) or 0)
    if n_fail + n_success <= 0:
        raise ValueError("set n_fail and/or n_success: a round with no evidence is empty")
    return RoundSettings(
        n_fail=n_fail,
        n_success=n_success,
        window=int(raw.get("window") or DEFAULT_WINDOW),
        work_root=work_root,
        stage_dir=_dir("stage_dir"),
    )


def _resolve_parent(state_path: Path, settings: RoundSettings) -> dict[str, str]:
    """The harness training runs now, initialising the state file if needed.

    A fresh lineage has no state file yet. Its baseline must be the same one
    the training side bootstraps from, which is why both read
    HARNESS_BASELINE_COMMIT/_BINARY -- see integrations/slime/resolve_baseline.
    """
    if not state_path.exists():
        commit = os.environ.get("HARNESS_BASELINE_COMMIT", "").strip()
        if not commit:
            raise RuntimeError(
                f"version state {state_path} does not exist yet and "
                "HARNESS_BASELINE_COMMIT is unset, so this round has no parent harness"
            )
        binary = os.environ.get("HARNESS_BASELINE_BINARY", "").strip()
        if not binary and settings.stage_dir is not None:
            binary = str(settings.stage_dir / "binaries" / commit[:7] / "opencode")
        _log(f"version state missing; initialising at baseline {commit[:12]}")
        version_state.init_state(state_path, commit, binary)
    return version_state.read_active(state_path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one harness evolution round.")
    parser.add_argument("--config", required=True, help="RL evolve YAML")
    parser.add_argument(
        "--rollout-id",
        type=int,
        default=None,
        help="step boundary this round targets (default: $HARNESS_EVOLVE_ROLLOUT_ID)",
    )
    parser.add_argument(
        "--rollout-root",
        default="",
        help="per-session trajectory dumps (default: $POLAR_TRAJECTORY_DIR)",
    )
    parser.add_argument(
        "--version-state",
        default="",
        help="harness version state file (default: $HARNESS_VERSION_STATE)",
    )
    parser.add_argument("--run-dir", default="", help="round directory (default: under work_root)")
    # Flags for running a round by hand; each one otherwise comes from the
    # config, or from the launcher's HARNESS_EVOLVE_* override of it.
    parser.add_argument("--n-fail", type=int, default=None)
    parser.add_argument("--n-success", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--evolver", default="")
    parser.add_argument("--gate-mode", default="")
    parser.add_argument("--parent-commit", default="")
    parser.add_argument("--parent-binary", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print(f"evolve config root must be a mapping: {config_path}", file=sys.stderr)
        return 1
    raw = _apply_env_overrides(raw)
    for key, value in (
        ("n_fail", args.n_fail),
        ("n_success", args.n_success),
        ("window", args.window),
        ("evolver", args.evolver),
        ("gate_mode", args.gate_mode),
    ):
        if value not in (None, ""):
            raw[key] = value

    rollout_id = args.rollout_id
    if rollout_id is None:
        value = os.environ.get("HARNESS_EVOLVE_ROLLOUT_ID", "").strip()
        if not value:
            print("--rollout-id or HARNESS_EVOLVE_ROLLOUT_ID is required", file=sys.stderr)
            return 1
        rollout_id = int(value)

    rollout_root = args.rollout_root or os.environ.get("POLAR_TRAJECTORY_DIR", "").strip()
    if not rollout_root:
        print("--rollout-root or POLAR_TRAJECTORY_DIR is required", file=sys.stderr)
        return 1
    state_path = args.version_state or os.environ.get("HARNESS_VERSION_STATE", "").strip()
    if not state_path:
        print("--version-state or HARNESS_VERSION_STATE is required", file=sys.stderr)
        return 1

    started = time.monotonic()
    try:
        settings = _round_settings(raw, base=config_path.parent)
        config = EvolveConfig.from_mapping(raw, base=config_path.parent)
        run_dir = Path(args.run_dir).expanduser() if args.run_dir else (
            settings.work_root / f"step{rollout_id + 1:06d}_{datetime.now().strftime('%m%d-%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        _log(f"step {rollout_id + 1}: {run_dir}")

        active = _resolve_parent(Path(state_path), settings)
        parent_commit = args.parent_commit or str(active["commit"])
        parent_binary = args.parent_binary or str(active.get("binary_path") or "")
        _log(f"parent harness {parent_commit[:12]}")

        manifest = select_evidence(
            rollout_root,
            n_fail=settings.n_fail,
            n_success=settings.n_success,
            output_dir=run_dir / "tb_job",
            end_rollout_id=rollout_id,
            window=settings.window,
        )
        selection = manifest.get("selection") or {}
        picked = selection.get("picked") or {}
        _log(
            f"evidence: {picked.get('n_fail')} failure(s) + {picked.get('n_success')} success(es) "
            f"over {len(selection.get('scanned_rollout_dirs') or [])} rollout dir(s)"
            + (" -- SHORT of the requested quota" if selection.get("short") else "")
        )

        result = evolve_once(
            run_dir / "tb_job",
            parent_commit,
            config=config,
            run_dir=run_dir,
            round_index=rollout_id + 1,
            parent_binary=parent_binary,
        )

        # The only thing the next rollout reads. Writing it here, rather than
        # inside the gate, keeps one writer for the file the sandboxes poll.
        if result.accepted:
            version_state.promote(
                state_path,
                result.commit,
                result.binary_path,
                gate_report_path=result.report_path,
                note=f"evolution round at step {rollout_id + 1}",
            )
            _log(f"promoted {result.commit[:12]} to active")
        else:
            version_state.record_rejection(
                state_path,
                result.candidate_commit,
                gate_report_path=result.report_path,
                reason=result.reason,
            )
            streak = version_state.consecutive_rejections(state_path)
            _log(f"kept {result.commit[:12]} (consecutive rejections: {streak})")

        payload = result.to_json()
        payload["selection"] = selection
        payload["run_dir"] = str(run_dir)
        (run_dir / "round.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _log(f"round done in {time.monotonic() - started:.0f}s")
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        _log("round FAILED; the caller keeps training on the current harness")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
