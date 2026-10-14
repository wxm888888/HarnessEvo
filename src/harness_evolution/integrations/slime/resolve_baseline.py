#!/usr/bin/env python3
"""Print the harness baseline commit + its staged binary path (two lines),
for launch_training.sh to export as HARNESS_BASELINE_COMMIT/_BINARY.

Resolution order: explicit HARNESS_BASELINE_COMMIT/_BINARY env vars win
outright; otherwise derived from the gate config's paths.baseline_commit
(the same value harness_evolution_pipeline.py's own bootstrap fallback
reads -- see its _resolve_active()) plus the pipeline config's stage_dir,
so the training-side (slime_bridge.rollout) and pipeline-side bootstraps of
a fresh HARNESS_VERSION_STATE can never disagree on which commit/binary a
brand-new lineage starts from.

Usage: resolve_harness_baseline.py <gate_config.yaml> <pipeline_config.yaml>
"""
from __future__ import annotations

import os
import sys

import yaml


def main() -> None:
    gate_path, pipeline_path = sys.argv[1], sys.argv[2]

    commit = os.environ.get("HARNESS_BASELINE_COMMIT", "")
    binary = os.environ.get("HARNESS_BASELINE_BINARY", "")
    if not (commit and binary):
        gate = yaml.safe_load(open(os.environ.get("HARNESS_PIPELINE_GATE_CONFIG") or gate_path))
        pipeline = yaml.safe_load(open(pipeline_path))
        commit = commit or gate["paths"]["baseline_commit"]
        stage_dir = os.environ.get("HARNESS_PIPELINE_STAGE_DIR") or pipeline.get("stage_dir") or ""
        binary = binary or (f"{stage_dir}/binaries/{commit[:7]}/opencode" if stage_dir else "")

    print(commit)
    print(binary)


if __name__ == "__main__":
    main()
