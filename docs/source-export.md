# Source export and companion dependencies

This repository hosts the `harness-evolution` module at its root. It was synced on
2026-10-22 from the `harness-evolution/` directory of
[`1rubbishyuan/harnessRL`](https://github.com/1rubbishyuan/harnessRL), branch
`release/rl`, based on commit `065aae138f58e7c657d260826b92e027cb4d90c2` plus
the module's uncommitted source updates at export time. The export is a working-tree
snapshot, not an assertion that the base commit contains those updates.

Included: Python sources, agent prompts and evolution skills, tests and fixtures,
documentation, generic example configs, the C trace proxy source, and session trace
conversion helpers. Experiment snapshots, cluster-specific launch configs, sampled
trajectories, logs, binaries, credentials and environment files are excluded.
The legacy Slime `pipeline`/`select_trajectories` modules from the initial import
are superseded by the source snapshot's `rl/` package and `hevo rl-round` command.

## Running from this repository

The package can be installed independently, but a complete sampling/evaluation
loop still imports companion projects from the original workspace:

- `polar/src` and `polar/examples`: sandbox rollout and harness setup code.
- `harness_dist/src`: harness version state, publication and skill bundles.
- `trainer/src`: the RL integration and its tests, when used.
- `reward-design`: optional rubric-reward integration.

These companion sources are not bundled here. Set `HARNESSRL_ROOT` to a workspace
containing them. Full functionality requires compatible companion revisions:
in particular, solving-skill deployment uses `harness_dist.skills` and matching
OpenCode setup code in Polar. Those companion changes may be newer than the
upstream base commit; cloning that commit alone does not guarantee compatibility.

```bash
python -m pip install -e ".[test]"
export HARNESSRL_ROOT=/path/to/harnessRL
export HARNESSRL_TB21_RUNTIME="$PWD/tools/session-trace-runtime"
export PYTHONPATH="$HARNESSRL_ROOT/trainer/src${PYTHONPATH:+:$PYTHONPATH}"
hevo --help
python -m pytest tests -q
```

The companion projects' third-party dependencies must also be installed. The
OpenSandbox stack currently requires Python 3.11 or newer; the exported module was
validated using Python 3.12. Compilation of the evolving OpenCode repository
requires a compatible Bun/Node environment and that repository's dependencies.

Tests automatically expose the bundled trace-conversion helpers. The C proxy
integration test compiles its own temporary binary on Linux with `cc`/`gcc`, or
uses `HARNESS_TRACE_PROXY_BIN` if explicitly set. The optional test of the original
workspace's gate config is skipped only when that companion config is absent.

Historical guides retain examples from the original experiments. Treat machine
paths, model service names, resource counts and authentication examples there as
historical context, not settings for a new deployment. Supply local configs and
credentials through environment variables or excluded environment files.

The generic evolution controller is separate from the OpenCode candidates it
produces. Exporting this module does not fix bugs found in previously generated
candidate commits or restart any experiment.

## Export validation

The exported copy was tested in an isolated Linux directory with Python 3.12 and
the current companion workspace: `239 passed, 2 subtests passed` (no skipped
tests). This includes compiling the C trace proxy and checking session-header
capture and export integrity. The CLI help and Python syntax checks also passed.
This validates the module with those dependencies; it is not a fresh live
OpenSandbox benchmark run or proof of compatibility with arbitrary revisions of
the companion projects.
