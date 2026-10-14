You are analyzing Terminal-Bench harness failures for a harness evolver.
The current directory contains numbered evidence directories. Each directory may contain trajectory.json, openai-messages.jsonl, openai-tools.json, verifier-summary.json, verifier-feedback.json, sanitized_result.json, and instruction.md. openai-messages.jsonl is the preferred OpenAI-compatible incremental conversation: it contains the initial prompts, then only newly appended user/tool messages and the assistant response for each round. llm-context.jsonl is the raw cumulative fallback containing the exact model messages, roles, tools and responses for each request. Use verifier-summary.json for outcome categories and verifier-feedback.json for the sanitized failed check and assertion output. Trace that concrete failure back to agent behavior. failed_trials.json lists the failed subset.
Analyze up to the ${failed_count} failed trajectories listed in failed_trials.json.
Write your final analysis to analysis/trajectory_analysis.md. Do not modify any numbered evidence files.
Focus on generic harness defects and recurring failure modes. Do not propose task-specific solution files.
Environment/setup failures are out of scope and must not become harness findings: python/python3 not found, missing commands or packages, apt/pip/conda installation, network/DNS, permissions, GPU/CUDA, memory/disk, ports, sandbox/control-plane or trace-proxy errors, verifier dependencies/tests not executed, trace capture failures, and timeouts. If a record mentions one, label it environment_only and do not analyze it or propose a harness fix. Analyze only behavior after a valid verifier execution.
Use this markdown structure:
# Failed trajectory analysis
## High-confidence patterns
## Candidate harness fixes
## Evidence notes
Keep the analysis concise, concrete, and tied to trial directory names.

