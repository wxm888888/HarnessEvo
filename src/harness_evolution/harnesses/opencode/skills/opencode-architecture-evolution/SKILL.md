---
name: opencode-architecture-evolution
description: Map the OpenCode runtime and design evidence-driven, substantive harness improvements instead of prompt-only changes
---

# OpenCode architecture evolution

Use this skill when evolving the OpenCode repository from Terminal-Bench trajectories, verifier failures, or other agent-run evidence. The goal is a general runtime improvement that helps future tasks; do not encode a solution for one benchmark task.

## Repository map

Treat the repository root as the OpenCode checkout. Confirm the paths in the checkout before editing because this map describes the 1.17.x layout.

### Request loop and agent orchestration

- `packages/opencode/src/session/prompt.ts` assembles a user turn, loads context, selects the agent, and drives the prompt/run path.
- `packages/opencode/src/session/processor.ts` processes model output and tool calls across a turn.
- `packages/opencode/src/session/llm.ts` and `packages/opencode/src/session/llm/` adapt model requests, streaming, retries, and response parsing.
- `packages/opencode/src/agent/agent.ts` defines built-in/configured agents, model selection, modes, step limits, and permissions.
- `packages/core/src/session/runner/` contains the lower-level run coordinator, model runner, LLM conversion, and max-step handling.

### Memory, history, and durable state

There is no single `memory/` module. Separate these concerns before changing them:

- `packages/opencode/src/session/session.ts`, `message-v2.ts`, `message.ts`, `storage/`, and `run-state.ts` hold the application-facing session and message state.
- `packages/core/src/session/store.ts` persists session data; `history.ts` reconstructs history; `projector.ts` derives the usable view; `context-epoch.ts` tracks context boundaries; `message.ts` and `schema.ts` define durable shapes.
- `packages/opencode/src/session/todo.ts` and `packages/core/src/session/todo.ts` track explicit task state, which is different from conversational memory.

When evidence says “the model forgot something”, first determine whether the loss happened in persistence/history projection, context selection, compaction, or the model prompt. Do not blindly add a memory prompt.

### Context, instructions, skills, and compaction

- `packages/opencode/src/session/instruction.ts` discovers root/global `AGENTS.md`, `CLAUDE.md`, `CONTEXT.md`, configured instruction paths, and URL instructions.
- `packages/opencode/src/session/system.ts` builds the environment and system-level context exposed to the model.
- `packages/opencode/src/agent/prompt/` contains agent prompt templates; prompt text is only one layer of the runtime.
- `packages/opencode/src/skill/index.ts` discovers `skills/**/SKILL.md` and configured skill paths; `.opencode/skills/` contains repository-local skills.
- `packages/opencode/src/session/compaction.ts`, `overflow.ts`, `summary.ts`, and `message-v2.ts` govern overflow, pruning, summaries, and context rebuilding. The corresponding lower-level logic is in `packages/core/src/session/compaction.ts`.

### Tools and subagents

- `packages/opencode/src/tool/registry.ts` assembles the tools visible to an agent.
- Tool entry points live in `packages/opencode/src/tool/`: `read.ts`, `write.ts`, `edit.ts`, `apply_patch.ts`, `shell.ts`, `glob.ts`, `grep.ts`, `lsp.ts`, `skill.ts`, `task.ts`, `todo.ts`, `question.ts`, `plan.ts`, and the web/MCP tools.
- `packages/core/src/tool/` contains shared tool implementations and schemas used by the runtime.
- `packages/opencode/src/tool/task.ts` creates and coordinates subagent work; `packages/opencode/src/agent/subagent-permissions.ts` and `agent.ts` control what subagents may do.
- `packages/opencode/src/session/tools.ts` and `session/todo.ts` connect tool calls and explicit task state to a session.

### Permissions, providers, and project configuration

- `packages/opencode/src/permission/evaluate.ts`, `permission/index.ts`, and agent permission merging decide whether a requested action is allowed.
- `packages/opencode/src/provider/provider.ts` and `provider/transform.ts` select and normalize providers; `packages/opencode/src/session/llm.ts` performs the session-side request.
- `packages/opencode/src/config/` parses project/user configuration, paths, agents, and markdown config.
- `packages/opencode/src/project/` and `packages/opencode/src/snapshot/` own project identity and filesystem snapshots used by session workflows.

Read the nearest `AGENTS.md` before editing. In particular, `packages/opencode/AGENTS.md` contains package validation commands; run typecheck from that package directory rather than assuming the repository root command is valid.

## Evidence-to-surface workflow

1. Classify the failure from the trajectory: planning, context discovery, tool selection, tool execution, verification, compaction/context loss, subagent coordination, permission, or finalization.
2. Locate the first incorrect state transition, not only the last failing command. Record the observed behavior, the expected behavior, and the evidence that separates a runtime defect from a task-specific mistake.
3. Map the transition to at least one runtime owner and one state/context boundary. For example, a lost instruction may involve `instruction.ts` plus history/compaction; a bad shell action may involve `tool/registry.ts` plus `tool/shell.ts` and permission evaluation.
4. Read the owning source, its tests/fixtures, and the nearest `AGENTS.md`. Search for existing helpers and call sites before adding a parallel mechanism.
5. Write a short hypothesis and an observable regression case before editing. The regression should fail on the base checkout when practical and pass after the change.
6. Implement the smallest coherent cross-layer fix. Prefer a runtime source change under `packages/opencode/src/` or `packages/core/src/` plus a focused test or fixture. A candidate whose only changes are prompt text, `AGENTS.md`, or `.opencode/skills/` is insufficient unless the evidence directly proves prompt/skill discovery is the root cause; in that exception, add a regression test and explain the causal evidence.
7. Keep the change focused within one subsystem. Two to four related files are often a useful shape, but do not add unrelated edits merely to increase the diff.
8. Validate the changed path with the package's targeted test/typecheck command, then run `git diff --check` and inspect the final diff and stat. Report skipped validation explicitly.

## Suggested routing by failure pattern

Use these as starting points, then verify the actual call graph:

- **No self-verification or premature success:** inspect `session/processor.ts`, `session/run-state.ts`, `core/session/runner/`, and the relevant agent step/termination logic; add a focused regression around the stop condition.
- **Wrong or missing tool call:** inspect `tool/registry.ts`, the selected tool implementation/schema, and `packages/core/src/tool/`; test tool selection/execution semantics rather than only rewriting descriptions.
- **Context lost after many turns:** inspect `session/compaction.ts`, `overflow.ts`, `summary.ts`, `message-v2.ts`, and `core/session/compaction.ts`; add a context-boundary or compaction fixture.
- **History or state is forgotten/replayed incorrectly:** inspect `core/session/store.ts`, `history.ts`, `projector.ts`, `context-epoch.ts`, and the application session storage; test persistence and reconstruction.
- **Instructions or skills not applied:** inspect `session/instruction.ts`, `session/system.ts`, `skill/index.ts`, and configuration path resolution; test discovery, precedence, and deduplication.
- **Permission or external-directory failure:** inspect `permission/evaluate.ts`, `permission/index.ts`, agent permission merging, and the owning tool; test allow/deny and path-boundary behavior.
- **Subagent failure or missing result:** inspect `tool/task.ts`, `agent.ts`, `agent/subagent-permissions.ts`, and session tool/state integration; test propagation of status, errors, and results.
- **Provider/retry/streaming failure:** inspect `provider/provider.ts`, `provider/transform.ts`, `session/llm.ts`, and `session/retry.ts`; test retry boundaries and partial responses.

## Candidate quality contract

Before presenting a candidate, ensure the summary states:

- **Hypothesis:** which observed failure and state transition motivated the change.
- **Architecture:** which subsystem owns the behavior and why the selected files are the right boundary.
- **Behavior change:** what a future agent run now does differently.
- **Validation:** exact commands and results, including any limitation.
- **Generality:** why the fix applies beyond the triggering Terminal-Bench task.
- **Risk:** compatibility, performance, or untested paths that remain.

Do not claim a larger diff is better by itself. Prefer a tested runtime behavior change that addresses the failure mechanism, and reject a broad refactor when the trajectory does not justify it.
