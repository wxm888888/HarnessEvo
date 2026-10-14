# Restructure plan

The code in this repository was moved from the old `harness-evolution` checkout
(`scripts/` plus the `harness-evolver` library). The old checkout stays untouched
so running experiments and their launch scripts keep working.

## Phases

| Phase | Scope | Status |
|---|---|---|
| P1 | Create the package, move whole files, rewrite imports, remove per-file `sys.path` hacks, port tests | done |
| P2 | Split the four ~1400–2500 line files by function into their stage packages | done |
| P3a | Prompts as `.md` templates, `hevo` CLI, `run.yaml` run config | done |
| P3b | Break up the long `main()` and gate functions; drop the P2 re-exports and leading underscores | deferred until the held-out gate can run again |
| P4 | Toy harness and fake evaluation backend, English docs, CI, license | planned |

## P3a: prompts, CLI and run config

- **Prompts.** Every agent-facing prompt is a template under `<stage>/prompts/` and is
  rendered by `harness_evolution.core.prompts`. The templates were generated from the old
  code by calling it with `${name}` placeholders, and `tests/test_prompts.py` checks 34
  rendered variants byte-for-byte against prompts recorded from the code before the move.
- **CLI.** `hevo` (`harness_evolution.cli`) wraps the existing entry points; see the README.
- **Run config.** `run.yaml` (`harness_evolution.core.run_config`) replaces the part of the
  launch scripts that froze `config.yaml`/`gate.yaml` into the run directory, exported the
  loop's environment knobs and spelled out the loop flags. Modules still read those knobs from
  the environment; `run.yaml` is the single place that sets them. A `run.yaml` equivalent of
  the 2026-10-11 launch produced the same parsed loop arguments (fresh and `--resume`) and the
  same 18 exported variables as the shell driver. Cluster-specific steps (rjob submission,
  Codex login and network preflight, proxy setup) remain the launcher's job.
- **Smoke.** Before P3a, `hevo evolve` on 4 existing held-in traces ran diagnosis,
  aggregation, one candidate and typecheck/build end to end without GPUs.

## Where the old modules went (P1)

| Old module | New module |
|---|---|
| `scripts/opencode_loop_common.py` | `harness_evolution.core.types` |
| `harness_evolver.harness_git` | `harness_evolution.core.git` |
| `harness_evolver.utils.{path,serialization}` | `harness_evolution.core.utils` |
| `scripts/opencode_trace_artifacts.py` | `harness_evolution.traces.materialize` |
| `scripts/verifier_feedback.py` | `harness_evolution.traces.verifier_feedback` |
| `scripts/opencode_loop_evidence/preflight.py` | `harness_evolution.traces.preflight` |
| `scripts/opencode_loop_evidence/common.py` | `harness_evolution.traces.sanitize` |
| `scripts/convert_polar_rollout_to_tb_job.py` | `harness_evolution.traces.converters.polar_rollout` |
| `scripts/convert_tb21_run_to_existing_tb_job.py` | `harness_evolution.traces.converters.tb21_run` |
| `scripts/opencode_loop_evidence/*` (strategies) | `harness_evolution.diagnosis.*` |
| `harness_evolver.clients` | `harness_evolution.agents` |
| `harness_evolver.evolvers` | `harness_evolution.proposer.evolvers` |
| `scripts/opencode_loop_memory` | `harness_evolution.memory` |
| `scripts/opencode_loop_task_selection` | `harness_evolution.tasks.selection` |
| `scripts/terminal_bench_task_scheduler.py` | `harness_evolution.tasks.scheduler` |
| `scripts/opencode_loop_candidate_selection` | `harness_evolution.selection.candidates` |
| `scripts/opencode_harness_gate.py` | `harness_evolution.evaluation.gate` |
| `scripts/opencode_harness_gate_opensandbox.py` | `harness_evolution.evaluation.backends.opensandbox` |
| `scripts/gate_filebridge_agent.py` | `harness_evolution.evaluation.backends.harbor_filebridge` |
| `scripts/opencode_self_harness_loop.py` | `harness_evolution.loop.two_sided` |
| `scripts/opencode_harness_evolver_loop_modular.py` | `harness_evolution.loop.modular` |
| `scripts/opencode_harness_evolve_from_existing_tb_modular.py` | `harness_evolution.loop.from_existing_job` |
| `scripts/harness_evolution_pipeline.py` | `harness_evolution.integrations.slime.pipeline` |
| `scripts/harness_reward_bridge.py` | `harness_evolution.integrations.slime.reward_bridge` |
| `scripts/select_trajectories_for_evolution.py` | `harness_evolution.integrations.slime.select_trajectories` |
| `scripts/resolve_evidence_task_ids.py` | `harness_evolution.integrations.slime.resolve_task_ids` |
| `scripts/resolve_harness_baseline.py` | `harness_evolution.integrations.slime.resolve_baseline` |
| `scripts/eval_checkpoint_terminal_bench.py` | `harness_evolution.integrations.slime.eval_checkpoint` |
| `skills/opencode-architecture-evolution` | `harness_evolution/harnesses/opencode/skills/` |

## Function-level split (P2)

Definitions were moved verbatim (checked by comparing the AST of every moved
definition with the original). The original module re-exports each moved name,
so `from harness_evolution.loop.two_sided import _score_side` still works; new
code should import from the owning module.

| From | Moved to |
|---|---|
| `loop.two_sided` | `tasks.splits` (held-in schedule, split files), `evaluation.sampling` (held-in sampling), `evaluation.scoring` (side scores), `evaluation.baseline_cache`, `selection.promotion` (`_candidate_score`), `loop.resume` (checkpoints, partial steps, pruning) |
| `loop.modular` | `core.config` (YAML accessors, `load_runtime_config`), `core.strategies` (`_build_strategy`, `_build_memory_strategy`), `harnesses.opencode.build` (Bun build, node_modules, binary registry), `harnesses.opencode.validate` (typecheck/build commands), `proposer.workspace` (commit, diff stats), `proposer.requirement` (evolver instruction, candidate report), `proposer.batch` (`generate_candidate_batch`) |
| `evaluation.gate` | `evaluation.gate_common` (config, log, git helpers), `evaluation.static_gate`, `evaluation.backends.harbor` (legacy image bake + harbor run) |
| `evaluation.backends.opensandbox` | `tasks.catalog` (`TaskSpec`, catalog loading), `evaluation.trials` (`Trial`, `TrialOutcome`, `SideOutcome`), `evaluation.backends.opensandbox_settings`, `evaluation.backends.opensandbox_artifacts` (trace downloads) |
| `diagnosis.per_trace_then_aggregate` | `diagnosis.aggregate` (diagnosis loading, aggregate index, markdown rendering) |

Still large and left for P3, because splitting them needs real refactoring
rather than moves: `loop.two_sided.main` (~500 lines) and `_run_two_sided_gate`,
`evaluation.gate.run_heldout_gate_multi` / `run_full_gate_multi`, and the
`PerTraceThenAggregateEvidence` / `AgentAnalyzedEvidence` classes, which are
mostly prompt text.

## Behavior changes in P1

- Paths that were derived from the script location now come from `harness_evolution.core.paths`.
- The default root for a relative `--run-dir` is `$HARNESSRL_OUTPUT_ROOT`, else `<repo>/runs`
  (previously a hard-coded GPFS directory). Launchers that pass an absolute run dir are unaffected.
- The slime pipeline starts its helper steps with `python -m harness_evolution...` instead of script paths.
- The Harbor gate's filebridge agent is now imported as
  `harness_evolution.evaluation.backends.harbor_filebridge`; Harbor configs that named
  `gate_filebridge_agent:...` need the new module path.

## Not migrated

Launch, submit, repair, cleanup and monitor shell scripts; one-off scripts with hard-coded
paths (`run_qwen36_baseline_89x5_20261009.py`, `retry_missing_gate_task.py`); experiment
templates and history notes; the older `self-harness/` workflow; the unused
`harness_evolver.{loop,bench,evaluators}` modules.
