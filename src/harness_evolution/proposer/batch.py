"""Generate one batch of candidate harness changes from the evidence of a step."""
from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from harness_evolution.proposer.evolvers import EVOLVER_REGISTRY, EvolverRequest, EvolverTarget
from harness_evolution.core.git import GitHarnessManager
from harness_evolution.core.types import Candidate, CollectionArtifact, CommandValidationResult, DiffStats, EvidenceStrategy, RuntimeConfig, ValidationResult
from harness_evolution.core.config import _bool_value, _int_value, _mapping, _relative_workspace_dir, _repo_relative_path, _str_value, _tail_text
from harness_evolution.core.strategies import _build_memory_strategy
from harness_evolution.harnesses.opencode.build import _command_env, link_shared_node_modules
from harness_evolution.harnesses.opencode.validate import _validation_log_path, validate_candidate
from harness_evolution.proposer.requirement import _candidate_requirement, _copy_skills, _load_candidate_report
from harness_evolution.proposer.workspace import _git_diff_stats, commit_changes_or_adopt_head


def _codex_client_env(
    config: RuntimeConfig,
    existing: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Give Codex the same Bun-backed Node environment as static checks.

    Codex executes some tool commands through ``bash -lc``.  The CPU rjob's
    login profile replaces PATH, so the environment inherited by the Codex
    process alone is insufficient.  BASH_ENV is sourced by non-interactive
    Bash after its startup files and restores the prepared PATH for both
    login and ordinary command shells.

    The script must merge into the PATH the shell already has instead of
    overwriting it: Bun prepends ``node_modules/.bin`` when it runs a package
    script, and a plain ``export PATH=...`` would drop it (``tsgo: command not
    found``, exit 127) whenever Bash decides to honor BASH_ENV.
    """
    prepared = _command_env(config)
    env = {str(key): str(value) for key, value in (existing or {}).items()}
    for key in ("BUN_BIN", "PATH"):
        if prepared.get(key):
            env[key] = prepared[key]
    if env.get("BUN_BIN") and env.get("PATH"):
        shell_env_dir = Path(os.environ.get("TMPDIR") or "/tmp") / "harnessrl-bun-node-shim"
        shell_env_dir.mkdir(parents=True, exist_ok=True)
        shell_env = shell_env_dir / "codex-bash-env.sh"
        temporary = shell_env.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(
            f"export BUN_BIN={shlex.quote(env['BUN_BIN'])}\n"
            f"__hr_path={shlex.quote(env['PATH'])}\n"
            "__hr_ifs=$IFS; IFS=:\n"
            "for __hr_dir in $PATH; do\n"
            "  case \":$__hr_path:\" in\n"
            "    *\":$__hr_dir:\"*) ;;\n"
            "    *) [ -n \"$__hr_dir\" ] && __hr_path=\"$__hr_path:$__hr_dir\" ;;\n"
            "  esac\n"
            "done\n"
            "IFS=$__hr_ifs\n"
            "export PATH=$__hr_path\n"
            "unset __hr_path __hr_ifs __hr_dir\n",
            encoding="utf-8",
        )
        temporary.replace(shell_env)
        env["BASH_ENV"] = str(shell_env)
    return env


def _prepare_opencode_evolver_environment(config: RuntimeConfig, run_id: str) -> tuple[dict[str, str], dict[str, Any]]:
    evolver_config = config.opencode_evolver
    if not evolver_config:
        return {}, {"configured": False}
    model = _str_value(evolver_config, "model")
    configured_models = evolver_config.get("models")
    if configured_models in (None, ""):
        configured_models = evolver_config.get("model_names")
    if isinstance(configured_models, str):
        configured_models = [item.strip() for item in configured_models.split(",") if item.strip()]
    if not isinstance(configured_models, (list, tuple)):
        configured_models = []
    model_pool = [str(item).strip() for item in configured_models if str(item).strip()]
    if model and model not in model_pool:
        model_pool.insert(0, model)
    if not model_pool and model:
        model_pool = [model]
    if not model and model_pool:
        model = model_pool[0]
    client_name = _str_value(evolver_config, "client_name", "opencode_bun_dev_client")
    if client_name == "codex_cli_client":
        return {}, {
            "configured": True,
            "client_name": client_name,
            "model": model,
            "models": model_pool or [model],
            "config_path": None,
            "xdg_root": None,
            "env": {},
        }
    provider_id, _, model_id = model.partition("/")
    provider_id = _str_value(evolver_config, "provider_id", provider_id)
    model_id = _str_value(evolver_config, "model_id", model_id)
    openai_base_url = _str_value(evolver_config, "openai_base_url")
    api_key_env = str(evolver_config.get("api_key_env") or "").strip()
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"required API key environment variable is not set: {api_key_env}")
    else:
        api_key = _str_value(evolver_config, "api_key", "EMPTY")
    xdg_root = _repo_relative_path(evolver_config.get("xdg_root") or (config.workspace_root / "opencode_evolver_xdg" / run_id))
    xdg_paths = {
        "XDG_CONFIG_HOME": xdg_root / "config",
        "XDG_DATA_HOME": xdg_root / "data",
        "XDG_CACHE_HOME": xdg_root / "cache",
        "XDG_STATE_HOME": xdg_root / "state",
    }
    for path in xdg_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    # Keep every replica model in the generated provider config.  Each
    # concurrent Bun rollout selects one model from this pool on its command
    # line; a config containing only r0 makes the other replicas unusable.
    provider_models: dict[str, dict[str, Any]] = {}
    for pool_model in model_pool or [model]:
        pool_provider, separator, pool_model_id = pool_model.partition("/")
        if separator and pool_provider != provider_id:
            continue
        provider_models[pool_model_id if separator else pool_provider] = {
            "name": evolver_config.get("model_name", pool_model_id if separator else pool_provider),
            "provider": {
                "api": openai_base_url,
                "npm": _str_value(evolver_config, "npm", "@ai-sdk/openai-compatible"),
            },
        }
    if not provider_models:
        provider_models[model_id] = {
            "name": evolver_config.get("model_name", model_id),
            "provider": {
                "api": openai_base_url,
                "npm": _str_value(evolver_config, "npm", "@ai-sdk/openai-compatible"),
            },
        }
    generated = {
        "permission": {"external_directory": "deny"},
        "experimental": {"continue_loop_on_deny": True},
        "model": model,
        "enabled_providers": [provider_id],
        "provider": {
            provider_id: {
                "name": evolver_config.get("provider_name", provider_id),
                "api": openai_base_url,
                "npm": _str_value(evolver_config, "npm", "@ai-sdk/openai-compatible"),
                "options": {
                    "apiKey": api_key,
                    "timeout": _int_value(evolver_config, "provider_timeout_ms", 300000),
                },
                "models": provider_models,
            },
        },
    }
    if evolver_config.get("chunk_timeout_ms") is not None:
        generated["provider"][provider_id]["options"]["chunkTimeout"] = int(evolver_config["chunk_timeout_ms"])
    skills = _mapping(evolver_config, "skills", required=False)
    if _bool_value(skills, "enabled", False):
        generated["skills"] = {"paths": [_relative_workspace_dir(skills.get("workspace_dir", "skills"), key="skills.workspace_dir")]}
    config_path = xdg_paths["XDG_CONFIG_HOME"] / "opencode" / "opencode.jsonc"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = {key: str(value) for key, value in xdg_paths.items()}
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    return env, {
        "configured": True,
        "model": model,
        "models": model_pool or [model],
        "provider_id": provider_id,
        "model_id": model_id,
        "config_path": str(config_path),
        "xdg_root": str(xdg_root),
        "env": env,
    }


def _validation_feedback(candidates: list[Candidate], config: RuntimeConfig) -> str:
    if not _bool_value(config.validation, "feed_errors_to_evolver", True):
        return ""
    failed = [candidate for candidate in candidates if not candidate.validation.passed]
    if not failed:
        return ""
    lines = [
        "Previous candidate harnesses failed validation. Generate a new patch from the original base commit, "
        "not from failed candidates. Preserve existing public exports and keep changes minimal.",
        "",
    ]
    for candidate in failed[:5]:
        failed_command = next((command for command in candidate.validation.commands if not command.passed), None)
        if failed_command is None:
            continue
        lines.extend(
            [
                f"Candidate {candidate.candidate_idx} failed `{failed_command.command}`.",
                "Output tail:",
                "```text",
                _tail_text(failed_command.output_tail, 1600),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def generate_candidate_batch(
    *,
    config: RuntimeConfig,
    evidence_strategy: EvidenceStrategy,
    artifacts: list[CollectionArtifact],
    task_cache_root: Path,
    current_harness_path: str,
    current_base_commit: str,
    iteration_idx: int,
    attempt_idx: int,
    candidate_offset: int,
    evolver_env: dict[str, str],
    validation_feedback: str,
) -> tuple[list[Candidate], list[dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    evolver_config = config.opencode_evolver
    configured_models = evolver_config.get("models")
    if configured_models in (None, ""):
        configured_models = evolver_config.get("model_names")
    if isinstance(configured_models, str):
        configured_models = [item.strip() for item in configured_models.split(",") if item.strip()]
    if not isinstance(configured_models, (list, tuple)):
        configured_models = []
    model_pool = [str(item).strip() for item in configured_models if str(item).strip()]
    base_model = _str_value(evolver_config, "model")
    if base_model and base_model not in model_pool:
        model_pool.insert(0, base_model)
    if not model_pool:
        model_pool = [base_model]
    evolver_name = _str_value(evolver_config, "evolver_name", "opencode_bun_dev")
    client_name = _str_value(evolver_config, "client_name", "opencode_bun_dev_client")
    configured_client_init = dict(_mapping(evolver_config, "client_init", required=False))
    # The loop deliberately removes stale absolute repo pins before each
    # step.  Always bind the required client argument to the live parent
    # worktree, even when the YAML already contains dev_run_env: the old
    # conditional only set this default in the no-dev_run_env branch.
    if client_name == "opencode_bun_dev_client":
        configured_client_init.setdefault(
            "opencode_repo_path",
            str(Path(current_harness_path).expanduser().resolve()),
        )
        configured_client_init.setdefault("dev_run_command_prefix", "bun dev -- run")
        configured_client_init.setdefault("dev_run_add_dir_arg", True)
    elif client_name == "codex_cli_client":
        existing_env = configured_client_init.get("env")
        configured_client_init["env"] = _codex_client_env(
            config,
            existing_env if isinstance(existing_env, dict) else None,
        )
    if "dev_run_env" in configured_client_init and isinstance(configured_client_init["dev_run_env"], dict):
        configured_client_init["dev_run_env"] = {
            **configured_client_init["dev_run_env"],
            **evolver_env,
        }
    elif client_name == "opencode_bun_dev_client":
        configured_client_init["dev_run_env"] = evolver_env

    def _command_for_model(command_prefix: str, model_name: str) -> str:
        """Return a Bun command prefix pinned to one replica model."""
        tokens = shlex.split(command_prefix)
        for index, token in enumerate(tokens):
            if token == "--model" and index + 1 < len(tokens):
                tokens[index + 1] = model_name
                return shlex.join(tokens)
        tokens.extend(["--model", model_name])
        return shlex.join(tokens)

    def _build_rollout_evolver(rollout_idx: int):
        """One evolver (and opencode process env) per rollout.

        opencode keeps its session store in XDG_DATA_HOME. Concurrent rollouts
        sharing one data dir race on the first-run sqlite migration --
        'PRAGMA journal_mode = WAL' fails for the losers and those opencode
        processes exit non-zero -- so each rollout gets its own data/state dir.
        Config (the generated provider/model jsonc) and cache stay shared:
        config is written once up front and only read afterwards.
        """
        client_init = dict(configured_client_init)
        # Rotate the starting point by step as well as by rollout.  This lets
        # a two-candidate run use r0/r1 on one step and r2/r3 on the next,
        # while a four-candidate run uses all replicas concurrently.
        pool_index = (
            iteration_idx * max(1, int(config.rollouts_per_attempt)) + rollout_idx
        ) % len(model_pool)
        rollout_model = model_pool[pool_index]
        print(
            f"evolver rollout={rollout_idx + 1} model={rollout_model}",
            flush=True,
        )
        if client_name == "codex_cli_client":
            client_init["model"] = rollout_model
        else:
            command_prefix = str(client_init.get("dev_run_command_prefix") or "bun dev -- run")
            client_init["dev_run_command_prefix"] = _command_for_model(command_prefix, rollout_model)
            run_env = dict(client_init.get("dev_run_env") or {})
            data_home = run_env.get("XDG_DATA_HOME")
            if data_home:
                rollout_root = Path(data_home).parent / f"rollout_{rollout_idx + 1:03d}"
                for key, child in (("XDG_DATA_HOME", "data"), ("XDG_STATE_HOME", "state")):
                    target = rollout_root / child
                    target.mkdir(parents=True, exist_ok=True)
                    run_env[key] = str(target)
                client_init["dev_run_env"] = run_env
        return EVOLVER_REGISTRY.build(
            evolver_name,
            {
                "client_name": client_name,
                "client_init": client_init,
                "request_timeout_sec": max(1, int(config.evolver_timeout_sec)),
            },
        )

    manager = GitHarnessManager(workspace_root=str(config.workspace_root))
    memory_strategy = _build_memory_strategy(config)
    rollout_run_id = datetime.now(timezone.utc).strftime("rl_rollout_opencode_modular_%Y%m%d_%H%M%S_%f")
    shared_repo = manager.prepare_shared_workspace(
        repo_path=str(Path(current_harness_path).expanduser().resolve()),
        workspace_group=rollout_run_id,
    )
    trajectories: list[dict[str, list[dict[str, Any]]]] = []
    candidates: list[Candidate] = []
    meta = {
        "run_id": rollout_run_id,
        "attempt_idx": attempt_idx + 1,
        "base_commit": current_base_commit,
        "rollouts": [],
        "model_pool": model_pool,
    }

    # `git worktree add`/commit touch shared administrative state under the
    # same shared_repo's .git dir; serialize just those calls so N rollouts
    # can still run their (much more expensive) opencode agent calls fully
    # concurrently.
    git_lock = threading.Lock()
    # Evidence materialization is content-keyed, so every rollout in a batch
    # resolves to the SAME analyzer result_dir -- which force_rerun rmtree's
    # on entry. Concurrent calls would delete each other's analysis mid-write
    # and run N identical analyzer LLM calls. Serializing means the first
    # rollout runs the analyzer once and the rest hit its in-memory cache.
    evidence_lock = threading.Lock()
    # Worktrees share Git administrative state. The accepted-lineage strategy
    # adds one local exclude rule while materializing its read-only artifact.
    memory_lock = threading.Lock()

    def _run_one_rollout(rollout_idx: int) -> tuple[Candidate, dict[str, list[dict[str, Any]]], dict[str, Any]]:
        worktree_name = f"attempt_{attempt_idx + 1:03d}_rollout_{rollout_idx + 1:03d}"
        branch_name = f"rl-rollout/opencode/{rollout_run_id}/{worktree_name}"
        try:
            with git_lock:
                workspace = manager.create_worktree_from_commit(
                    shared_repo=shared_repo,
                    worktree_name=worktree_name,
                    base_commit=current_base_commit,
                    branch_name=branch_name,
                )
            linked_node_modules = link_shared_node_modules(
                workspace, Path(current_harness_path).expanduser().resolve()
            )
            if linked_node_modules:
                print(
                    f"linked shared node_modules into {workspace}: {', '.join(linked_node_modules)}",
                    file=sys.stderr,
                )
            with evidence_lock:
                evidence = evidence_strategy.materialize(
                    artifacts=artifacts,
                    workspace=workspace,
                    task_cache_root=task_cache_root,
                    current_harness_path=current_harness_path,
                )
            skill_copy = _copy_skills(config, workspace)
            with memory_lock:
                memory = memory_strategy.materialize(workspace=workspace)
            request = EvolverRequest(
                workspace=workspace,
                requirement=_candidate_requirement(
                    evidence=evidence,
                    skill_copy=skill_copy,
                    validation_feedback=validation_feedback,
                    memory=memory,
                ),
                target=EvolverTarget(scope="repo"),
                context={
                    "iteration_idx": iteration_idx,
                    "attempt_idx": attempt_idx,
                    "rollout_idx": rollout_idx,
                    "base_commit": current_base_commit,
                    "evidence_method": evidence.metadata.get("method"),
                    **(
                        {"memory_method": memory.metadata.get("method")}
                        if memory.metadata
                        else {}
                    ),
                },
            )
            result = _build_rollout_evolver(rollout_idx).evolve(request)
            candidate_report = _load_candidate_report(workspace)
            if result.changed_files:
                with git_lock:
                    commit, commit_created = commit_changes_or_adopt_head(
                        manager,
                        workspace,
                        result.changed_files,
                        (
                            "chore(rl): opencode modular evolver rollout\n\n"
                            f"iteration_idx: {iteration_idx}\n"
                            f"attempt_idx: {attempt_idx}\n"
                            f"rollout_idx: {rollout_idx}\n"
                            f"changed_files: {', '.join(result.changed_files)}\n"
                            f"generated_at: {datetime.now(timezone.utc).isoformat()}"
                        ),
                    )
            else:
                commit = manager.current_head(workspace)
                commit_created = False
            agent_trajectory = result.metadata.get("agent_trajectory", [])
            if not isinstance(agent_trajectory, list):
                agent_trajectory = [agent_trajectory]
            validation = validate_candidate(
                config=config,
                workspace=workspace,
                iteration_idx=iteration_idx,
                attempt_idx=attempt_idx,
                rollout_idx=rollout_idx,
                changed_files=list(result.changed_files),
            )
            candidate = Candidate(
                candidate_idx=candidate_offset + rollout_idx + 1,
                attempt_idx=attempt_idx + 1,
                rollout_idx=rollout_idx + 1,
                workspace=workspace,
                branch=branch_name,
                commit=commit,
                commit_created=commit_created,
                changed_files=list(result.changed_files),
                summary=result.summary,
                trajectory=agent_trajectory,
                evidence=evidence.metadata,
                skill_copy=skill_copy,
                evolver_metadata={
                    **{
                        key: value
                        for key, value in result.metadata.items()
                        if key != "agent_trajectory"
                    },
                    "candidate_report": candidate_report,
                    **({"memory": memory.metadata} if memory.metadata else {}),
                },
                validation=validation,
                diff_stats=_git_diff_stats(workspace, current_base_commit, commit),
            )
        except Exception:  # noqa: BLE001
            # One rollout blowing up (opencode crash/timeout, etc.) must not
            # take the other concurrent rollouts in this batch down with it.
            error_text = traceback.format_exc()
            log_path = _validation_log_path(config, iteration_idx, attempt_idx, rollout_idx, "rollout_error")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(error_text, encoding="utf-8")
            print(f"rollout {rollout_idx} failed: {_tail_text(error_text, 500)}", file=sys.stderr)
            candidate = Candidate(
                candidate_idx=candidate_offset + rollout_idx + 1,
                attempt_idx=attempt_idx + 1,
                rollout_idx=rollout_idx + 1,
                workspace=Path(current_harness_path),
                branch=branch_name,
                commit=current_base_commit,
                commit_created=False,
                changed_files=[],
                summary="rollout crashed before producing a candidate",
                trajectory=[],
                evidence={},
                skill_copy={},
                evolver_metadata={"error": _tail_text(error_text, 2000)},
                validation=ValidationResult(
                    enabled=True,
                    passed=False,
                    commands=[
                        CommandValidationResult(
                            command="rollout_error",
                            passed=False,
                            returncode=1,
                            duration_sec=0.0,
                            log_path=str(log_path),
                            output_tail=_tail_text(error_text, 2000),
                        )
                    ],
                ),
                diff_stats=DiffStats(),
            )
            agent_trajectory = []
        return candidate, {candidate.commit: agent_trajectory}, _candidate_to_json(candidate)

    results_by_idx = _collect_rollout_results(_run_one_rollout, config.rollouts_per_attempt)

    for rollout_idx in range(config.rollouts_per_attempt):
        candidate, trajectory_entry, rollout_json = results_by_idx[rollout_idx]
        candidates.append(candidate)
        trajectories.append(trajectory_entry)
        meta["rollouts"].append(rollout_json)
    return candidates, trajectories, meta


def _collect_rollout_results(run_one, rollout_count: int) -> dict[int, Any]:
    """Run all candidates with a separate, optionally bounded worker count."""
    concurrency = int(os.environ.get("HARNESSRL_EVOLVER_CONCURRENCY", str(max(1, rollout_count))))
    if concurrency < 1:
        raise ValueError("HARNESSRL_EVOLVER_CONCURRENCY must be positive")
    max_workers = min(max(1, rollout_count), concurrency)
    print(f"[proposer] candidates={rollout_count} concurrency={max_workers}", flush=True)
    results: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, index): index for index in range(rollout_count)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_idx": candidate.candidate_idx,
        "attempt_idx": candidate.attempt_idx,
        "rollout_idx": candidate.rollout_idx,
        "workspace": str(candidate.workspace),
        "branch": candidate.branch,
        "commit": candidate.commit,
        "commit_created": candidate.commit_created,
        "changed_files": candidate.changed_files,
        "summary": candidate.summary,
        "evidence": candidate.evidence,
        "skill_copy": candidate.skill_copy,
        "evolver_metadata": candidate.evolver_metadata,
        "validation": {
            "enabled": candidate.validation.enabled,
            "passed": candidate.validation.passed,
            "commands": [asdict(command) for command in candidate.validation.commands],
        },
        "diff_stats": asdict(candidate.diff_stats),
    }
