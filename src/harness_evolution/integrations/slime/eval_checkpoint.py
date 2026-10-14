"""Evaluate one (checkpoint, opencode-harness-commit) pair on the held-out
terminal-bench task set, reusing the gate's held-out-eval machinery:

  1. publish the harness binary to AFS (OpenSandbox consume path);
  2. run the OpenSandbox held-out A/B against a policy model you point at
     via AGENT_GATEWAY_URL (normally LLM-Proxy in front of a sglang server
     serving the checkpoint -- this script does not start it);
  3. parse per-task pass counts from the OpenSandbox trial outcomes.

  Harbor/crane image-bake remains available only if gate yaml sets
  heldout_gate.backend: harbor.

A "baseline" side (paths.baseline_commit/baseline_repo in the config) is
always run alongside the candidate for a free reference point -- same
sglang endpoint, so any gap is attributable to the harness, not the model.

Usage:
    python3 -m harness_evolution.integrations.slime.eval_checkpoint \\
        --config /root/opencode-baseline/gate_1179_step79_eval.yaml \\
        --commit bc46c40a766acca14457f30b6721b9b2b2ac7b55 \\
        --binary /root/opencode-baseline/gate-run/artifacts/binaries/bc46c40.../opencode \\
        --label step79

Before running: edit the config's heldout_gate.filebridge.sglang_base_url to
point at the sglang server serving the checkpoint under test (see
launch_eval_sglang.sh), and confirm it answers on /v1/models.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from harness_evolution.evaluation.gate import load_gate_config, run_heldout_gate_multi, _log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="gate YAML (heldout_gate.enabled must be true)")
    parser.add_argument("--commit", required=True, help="candidate opencode commit (full hash)")
    parser.add_argument("--binary", required=True, help="path to the built candidate opencode binary")
    parser.add_argument("--label", default=None, help="free-text label for the report file name (e.g. step79)")
    parser.add_argument(
        "--task-names", default=None,
        help="comma-separated task names to override config's heldout_gate.task_names",
    )
    parser.add_argument("--keep-tags", action="store_true", help="skip deleting derived image tags afterward")
    parser.add_argument("--out", default=None, help="path to write the full JSON report (default: alongside config)")
    args = parser.parse_args()

    config = load_gate_config(args.config)
    if not config.get("heldout_gate", {}).get("enabled", True):
        _log("WARNING: heldout_gate.enabled is false in this config; forcing it on for this run")
        config["heldout_gate"]["enabled"] = True
    if args.keep_tags:
        config["heldout_gate"]["cleanup_tags"] = False

    task_names = args.task_names.split(",") if args.task_names else None
    binary = str(Path(args.binary).expanduser().resolve())
    if not Path(binary).is_file():
        raise SystemExit(f"binary not found: {binary}")

    sglang_url = config.get("heldout_gate", {}).get("filebridge", {}).get("sglang_base_url", "<unset>")
    _log(f"policy model endpoint: {sglang_url}")
    _log(f"candidate: {args.commit[:12]} ({binary})")

    results = run_heldout_gate_multi(config, [(args.commit, binary)], task_names=task_names)
    result = results[args.commit]

    n_tasks = len(result.tasks)
    print()
    print("=" * 70)
    print(f"CHECKPOINT/HARNESS EVAL{' -- ' + args.label if args.label else ''}")
    print(f"policy model: {sglang_url}")
    print(f"candidate harness: {args.commit[:12]}")
    print(f"tasks: {n_tasks}  wins={result.wins} losses={result.losses} ties={result.ties} net={result.net}")
    print(
        f"candidate pass total: {result.candidate_passes_total}/{n_tasks}  "
        f"errors={result.candidate_errors}"
    )
    print(f"baseline pass total (reference, same model): errors={result.baseline_errors}")
    for t in result.tasks:
        print(f"  {t.task:<30} baseline {t.baseline_passes}/{t.n_attempts}  candidate {t.candidate_passes}/{t.n_attempts}  -> {t.outcome}")
    print(f"baseline job dir:   {result.baseline_job_dir}")
    print(f"candidate job dir:  {result.candidate_job_dir}")
    print("=" * 70)

    label = args.label or args.commit[:12]
    out_path = Path(args.out) if args.out else Path(args.config).with_name(f"eval_report_{label}.json")
    out_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    _log(f"full report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
