# harness-evolution

中文使用说明见 [docs/guide.zh.md](docs/guide.zh.md)。

Evolve an agent harness (currently OpenCode) from its own trajectories. Each step:

1. **traces** – sample tasks with the current harness and normalize every trial into a fixed set of files
   (`instruction.md`, `trajectory.json`, `openai-messages.jsonl`, `openai-tools.json`,
   `verifier-summary.json`, `verifier-feedback.json`).
2. **diagnosis** – a coding agent analyzes each trace on its own, then aggregates the per-trace findings.
3. **proposer** – the agent edits the harness source in isolated git worktrees, producing N candidates.
4. **evaluation** – candidates pass static checks (typecheck, build), then run on a fixed held-out task set.
5. **selection** – the best candidate that beats the parent becomes the next parent.

> Status: restructuring in progress (phases 1–2 and part of 3 done). Code was moved
> from the `harness-evolution/scripts` + `harness-evolver` layout into this package with
> behavior unchanged. See `docs/restructure.md` for the plan.

## Layout

```
src/harness_evolution/
  core/          shared types, config helpers, paths, git worktrees
  traces/        raw sandbox output -> standard per-trial files; preflight checks
  diagnosis/     evidence strategies (per-trace diagnosis, then aggregation)
  agents/        wrappers around Codex CLI, Claude Code CLI, OpenCode, OpenAI chat
  proposer/      evolvers that turn evidence into candidate commits
  memory/        what later candidates learn about earlier promotions
  tasks/         task selection and scheduling
  harnesses/     adapters for the harness being evolved (OpenCode skills today)
  evaluation/    static gate and held-out evaluation backends (OpenSandbox, legacy Harbor)
  selection/     candidate selection strategies
  loop/          step orchestration and resume
  integrations/slime/   co-evolution with RL training
tests/
docs/
tools/trace-proxy/
```

## Running

```bash
pip install -e ".[test]"
pytest
```

Everything goes through the `hevo` command:

| Command | What it does |
|---|---|
| `hevo run run.yaml [--resume] [--dry-run]` | the full two-sided loop described by one `run.yaml` (see `configs/examples/run.yaml`) |
| `hevo evolve --config evolve.yaml --job-dir JOB` | diagnose an existing trajectory job and propose candidates; no sampling, no gate |
| `hevo status RUN_DIR` | summarize a run's `state.json` |
| `hevo loop ...`, `hevo gate ...`, `hevo slime-pipeline ...` | the underlying modules with their raw flags |

`run.yaml` names the evolve and gate configs, the task splits, the loop settings, the gate
model, and the environment the run exports. `hevo run --dry-run` prints the resulting
loop arguments and environment (values from `env_files` are masked) without touching
the run directory.

## Prompts

The prompts the analysis and evolution agents receive are Markdown templates next to the
code that uses them:

| File | Used for |
|---|---|
| `diagnosis/prompts/trace_diagnosis.md` | diagnosing one trajectory |
| `diagnosis/prompts/aggregate.md` | aggregating the per-trace diagnoses |
| `diagnosis/prompts/evidence_*.md` | how each evidence strategy describes its files to the evolver |
| `memory/prompts/accepted_lineage.md` | how promotion history is presented |
| `proposer/prompts/candidate.md` | the evolver's main instruction, in sections |

Templates use `${name}` placeholders. `tests/test_prompts.py` pins the rendered text;
after an intended prompt change, update `tests/fixtures/prompts_golden.json`.

### Environment

| Variable | Meaning | Default |
|---|---|---|
| `HARNESSRL_ROOT` | workspace with the sibling `polar/`, `harness_dist/`, `reward-design/` sources | parent of this checkout |
| `HARNESSRL_OUTPUT_ROOT` | root for relative `--run-dir` values | `<repo>/runs` |
| `HARNESSRL_TB21_RUNTIME` | directory providing the `agent_artifacts` trace converter | `$HARNESSRL_ROOT/../tb21-eval-qwen36-fixed/runtime` |
| `HARNESSRL_REWARD_DESIGN_ROOT` | reward-design sources (slime integration only) | `$HARNESSRL_ROOT/reward-design` |
