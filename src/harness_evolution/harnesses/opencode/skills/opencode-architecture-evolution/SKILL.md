---
name: opencode-architecture-evolution
description: Map the OpenCode runtime and design evidence-driven, substantive harness improvements instead of prompt-only changes
---

# OpenCode architecture evolution

Use this skill when evolving the OpenCode repository from agent trajectories, verifier failures, or other agent-run evidence. The goal is a general runtime improvement that helps future tasks; do not encode a solution for one benchmark task.

## How the evidence was produced

Every trajectory you are given comes from one non-interactive `opencode run` inside a sandbox: one user turn, no human follow-up, a per-task time budget, then a hidden verifier decides pass or fail. Two task families appear:

- **Code repair** (most held-in tasks): the sandbox holds a real repository at its own working directory (`/testbed`, `/workspace/repo`, `/app`). The agent is asked to fix a bug or add a feature. It cannot see the hidden tests or the reference fix. Verification restores the test paths, applies a hidden test patch, and runs that project's own command (`pytest`, `go test`, `bun test`, …); exit 0 passes.
- **Terminal tasks**: a command-line goal verified by a `tests/test.sh` script.

Both reward the same behaviors: discover the real contract in an unfamiliar repository before editing, and verify the edit against that contract — not against self-authored checks — before stopping.

## Execution path used by evaluation runs

Confirm reachability before editing. A change on a path the evaluation never executes builds, typechecks, passes its unit test, and still changes nothing.

The live path is:

`opencode run` → server `session.prompt` → **`packages/opencode/src/session/prompt.ts`** (`SessionPrompt`, `export class Service … "@opencode/SessionPrompt"`) → `session/processor.ts`, `session/llm.ts` → tools from `tool/registry.ts`.

- **`packages/core/src/session/runner/` is not on this path.** The only thing `packages/opencode/src` imports from it is the `MAX_STEPS_PROMPT` constant (`prompt.ts` line 19). Candidates that added system-prompt text or turn logic inside `core/session/runner/llm.ts` produced no observable change across ~180 evaluation traces. Treat that directory as the V2 runner used by `packages/core`'s own code and tests, not as the loop under evaluation.
- **The loop and its exit condition** live in the `SessionPrompt` loop in `prompt.ts`. Each pass re-reads the messages and breaks when the last assistant message finished with a reason other than `tool-calls`, has no pending tool parts, and `lastUser.id < lastAssistant.id`.
- **To add a turn**, change that exit condition, or create a *new* user message so the `lastUser.id < lastAssistant.id` comparison no longer holds. In either case, trace execution all the way to the next provider call: persisting a reminder followed by the original `break` or `return` still exits; the active control flow must actually continue. Appending a synthetic part to the existing last user message does **not** reopen the loop: its id is unchanged, so the next pass takes the same break and the text is never sent. A candidate made exactly this mistake; its reminder appeared in zero of 43 traces that had edited files.

### Runtime tool names

The model sees these ids, which differ from the file names. Match on the id, never the file stem:

| runtime id | implementation |
|---|---|
| `bash` | `tool/shell.ts` |
| `edit`, `write`, `read` | `tool/edit.ts`, `write.ts`, `read.ts` |
| `glob`, `grep` | `tool/glob.ts`, `grep.ts` |
| `task`, `todowrite`, `skill`, `webfetch` | `tool/task.ts`, `todo.ts`, `skill.ts`, web tools |

A typical evaluation run exposes exactly: `bash, edit, glob, grep, read, skill, task, todowrite, webfetch, write`. Read `openai-tools.json` in the trajectory to confirm for the run you are diagnosing. There is no `shell` tool id at runtime.

## Repository map

Treat the repository root as the OpenCode checkout. Confirm the paths in the checkout before editing because this map describes the 1.17.x layout.

### Request loop and agent orchestration

- `packages/opencode/src/session/prompt.ts` assembles a user turn, loads context, selects the agent, and drives the prompt/run path.
- `packages/opencode/src/session/processor.ts` processes model output and tool calls across a turn.
- `packages/opencode/src/session/llm.ts` and `packages/opencode/src/session/llm/` adapt model requests, streaming, retries, and response parsing.
- `packages/opencode/src/agent/agent.ts` defines built-in/configured agents, model selection, modes, step limits, and permissions.
- `packages/core/src/session/runner/` contains a separate V2 run coordinator, model runner, and LLM conversion. Evaluation runs do not execute it (see “Execution path” above); do not place behavior changes here.

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
- Tool entry points live in `packages/opencode/src/tool/`: `read.ts`, `write.ts`, `edit.ts`, `apply_patch.ts`, `shell.ts`, `glob.ts`, `grep.ts`, `lsp.ts`, `skill.ts`, `task.ts`, `todo.ts`, `question.ts`, `plan.ts`, and the web/MCP tools. File name and runtime id differ — see the table above, and note that not every file here is exposed in a given run.
- `packages/core/src/tool/` contains shared tool implementations and schemas. Confirm a specific symbol is reached from `packages/opencode/src` before relying on it.
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
3. Map the transition to at least one runtime owner and one state/context boundary. For example, a lost instruction may involve `instruction.ts` plus history/compaction; a bad `bash` action may involve `tool/registry.ts` plus `tool/shell.ts` and permission evaluation.
4. Prove the owner is reachable from `session/prompt.ts` before editing it: follow the imports and call sites from the `SessionPrompt` loop down to the function you intend to change. If you cannot trace that chain, you have the wrong file.
5. Read the owning source, its tests/fixtures, and the nearest `AGENTS.md`. Search for existing helpers and call sites before adding a parallel mechanism.
6. Write a short hypothesis and an observable regression case before editing. The regression should fail on the base checkout when practical and pass after the change.
7. Implement the smallest coherent cross-layer fix. Based on the evidence, choose code changes, skill changes, or a combination of both. Problem-solving skills are located in `.opencode/skills/` and are part of the harness. Assess whether the diagnosis evidence contains reusable problem-solving lessons worth capturing in skills. If you do not modify any problem-solving skill, explain why in the candidate report. You may modify code and skills together; avoid adding duplicate skills when existing skills already cover the relevant lessons.
8. Keep the change focused within one subsystem. Two to four related files are often a useful shape, but do not add unrelated edits merely to increase the diff.
9. Validate the changed path with the package's targeted test/typecheck command, then run `git diff --check` and inspect the final diff and stat. Report skipped validation explicitly.
10. Demonstrate the change actually fires. A unit test over your new helper is not enough — it proves the function works, not that the runtime calls it. Exercise the behavior through the `SessionPrompt` loop (a session-level test or fixture that drives a turn to completion), or state plainly which evidence you could not obtain and why. A candidate that cannot show its new behavior occurring in a driven session should be reported as unverified rather than as validated.

### When changing control flow

- Use the real message and tool-state shapes from the owning code and existing tests. Tool execution can be in an earlier `tool-calls` message while the final `stop` message contains only text. Verify ordering, exit metadata, and later edits; command-name matches or tool completion alone do not prove verification succeeded or output was reviewed.
- Define when the intervention triggers, when it must not trigger, and what prevents repeated or stale triggers. Scope state to the intended user turn and session; consider child sessions and structured output where relevant.
- For added turns, drive the actual session loop and assert the additional provider call receives the intervention and can terminate. Include a normal non-trigger case and a repeated-exit case. Run the regression against the base and candidate when feasible. A persisted message, helper-only test, filtered typecheck, or written explanation does not establish that the state transition works; report it as unverified if the execution evidence is missing.

## Suggested routing by failure pattern

Use these as starting points, then verify the actual call graph:

- **No self-verification or premature success:** inspect the `SessionPrompt` loop and its exit condition in `session/prompt.ts`, plus `session/processor.ts` and `session/run-state.ts`. Re-read “Execution path” first: the stop condition under evaluation is in `prompt.ts`, not in `core/session/runner/`, and reopening the loop requires a new user message or a changed exit condition.
- **Wrong or missing tool call:** inspect `tool/registry.ts`, the selected tool implementation/schema, and `packages/core/src/tool/`; test tool selection/execution semantics rather than only rewriting descriptions. Match tools by runtime id (`bash`, not `shell`).
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
- **Reachability:** the import/call chain from the `SessionPrompt` loop to the changed code, and the evidence that the new behavior fires in a driven session.
- **Behavior change:** what a future agent run now does differently.
- **Validation:** exact commands and results, including any limitation.
- **Generality:** why the fix applies beyond the one task that triggered it.
- **Risk:** compatibility, performance, or untested paths that remain.

Do not claim a larger diff is better by itself. Prefer a tested runtime behavior change that addresses the failure mechanism, and reject a broad refactor when the trajectory does not justify it.
