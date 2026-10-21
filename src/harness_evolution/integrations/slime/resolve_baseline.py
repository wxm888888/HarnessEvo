#!/usr/bin/env python3
"""Print the harness baseline commit + its staged binary path (two lines),
for the launcher to export as HARNESS_BASELINE_COMMIT/_BINARY.

Resolution order: explicit HARNESS_BASELINE_COMMIT/_BINARY env vars win
outright; otherwise derived from the gate config's paths.baseline_commit plus
the RL evolve config's stage_dir. Both the training side
(slime_bridge.rollout) and the evolution round's own bootstrap of a fresh
HARNESS_VERSION_STATE read those two env vars, so a brand-new lineage cannot
start from two different commits depending on which side got there first.

Usage: python -m harness_evolution.integrations.slime.resolve_baseline \
           <gate_config.yaml> <rl_evolve_config.yaml>
"""
from __future__ import annotations

import os
import sys

import yaml


def main() -> None:
    gate_path, evolve_path = sys.argv[1], sys.argv[2]

    commit = os.environ.get("HARNESS_BASELINE_COMMIT", "")
    binary = os.environ.get("HARNESS_BASELINE_BINARY", "")
    if not (commit and binary):
        # The launcher renders both configs into the run dir before training
        # starts; prefer the rendered copies, which is what the round reads.
        gate = yaml.safe_load(open(os.environ.get("HARNESS_EVOLVE_GATE_CONFIG") or gate_path))
        evolve = yaml.safe_load(open(evolve_path))
        commit = commit or gate["paths"]["baseline_commit"]
        stage_dir = os.environ.get("HARNESS_EVOLVE_STAGE_DIR") or (evolve or {}).get("stage_dir") or ""
        binary = binary or (f"{stage_dir}/binaries/{commit[:7]}/opencode" if stage_dir else "")

    print(commit)
    print(binary)


if __name__ == "__main__":
    main()
