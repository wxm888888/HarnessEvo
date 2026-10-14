from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from harness_evolution.agents.opencode_bun_dev import OpenCodeBunDevClient
from harness_evolution.core.prompts import render_prompt
from harness_evolution.core.types import (
    REPO_ROOT,
    WORKSPACE_ARTIFACT_ROOT,
    CollectionArtifact,
    EvidenceBuildResult,
    RuntimeConfig,
    load_json_object,
    repo_relative_path,
    safe_mapping,
    tail_text,
)
from harness_evolution.traces.sanitize import copy_instruction_if_available, write_sanitized_result
from harness_evolution.traces.preflight import filter_collection_artifacts
from harness_evolution.traces.verifier_feedback import materialize_verifier_feedback


@dataclass
class _AnalysisCacheEntry:
    cache_key: str
    result_dir: Path
    analysis_path: Path
    failed_count: int
    analyzer_ran: bool
    metadata: dict[str, Any]


class AgentAnalyzedEvidence:
    name = "agent_analyzed"

    def __init__(self, options: dict[str, Any], config: RuntimeConfig):
        self.options = options
        self.config = config
        self._cache: _AnalysisCacheEntry | None = None

    def materialize(
        self,
        *,
        artifacts: list[CollectionArtifact],
        workspace: Path,
        task_cache_root: Path,
        current_harness_path: str | Path | None = None,
    ) -> EvidenceBuildResult:
        # Keep the strategy safe when called outside the main loop as well.
        # Excluded artifacts are never copied into the proposer workspace.
        artifacts, _decisions = filter_collection_artifacts(artifacts)
        if not artifacts:
            raise ValueError(
                "evidence preflight found no complete verifier-backed traces; "
                "environment and infrastructure failures are not evolution evidence"
            )
        root = workspace / WORKSPACE_ARTIFACT_ROOT
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        records = self._write_numbered_evidence(
            artifacts=artifacts,
            root=root,
            task_cache_root=task_cache_root,
            include_trajectory=True,
        )
        analysis = self._get_or_run_analysis(
            artifacts=artifacts,
            task_cache_root=task_cache_root,
            current_harness_path=current_harness_path,
        )
        workspace_analysis_dir = root / "analysis"
        workspace_analysis_dir.mkdir(parents=True, exist_ok=True)
        workspace_analysis_path = workspace_analysis_dir / "trajectory_analysis.md"
        shutil.copy2(analysis.analysis_path, workspace_analysis_path)

        return EvidenceBuildResult(
            requirement_addendum=render_prompt(
                __package__, "evidence_agent_analyzed.md", name=self.name, root=WORKSPACE_ARTIFACT_ROOT
            ),
            metadata={
                "method": self.name,
                "artifact_root": str(root),
                "artifact_count": len(records),
                "records": records,
                "analysis": {
                    "workspace_analysis_path": str(workspace_analysis_path),
                    "source_analysis_path": str(analysis.analysis_path),
                    "result_dir": str(analysis.result_dir),
                    "cache_key": analysis.cache_key,
                    "failed_count": analysis.failed_count,
                    "analyzer_ran": analysis.analyzer_ran,
                    "metadata": analysis.metadata,
                },
            },
        )

    def _get_or_run_analysis(
        self,
        *,
        artifacts: list[CollectionArtifact],
        task_cache_root: Path,
        current_harness_path: str | Path | None,
    ) -> _AnalysisCacheEntry:
        cache_key = self._cache_key(artifacts, current_harness_path=current_harness_path)
        if self._cache is not None and self._cache.cache_key == cache_key:
            return self._cache

        result_dir = self._analysis_result_dir(cache_key)
        evidence_root = result_dir / "evidence"
        analysis_path = result_dir / "trajectory_analysis.md"
        force = _bool_option(self.options, "force_rerun", False)

        if force and result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        records = self._write_numbered_evidence(
            artifacts=artifacts,
            root=evidence_root,
            task_cache_root=task_cache_root,
            include_trajectory=True,
        )
        failed_trials = self._select_failed_trials(evidence_root)
        _write_json(result_dir / "failed_trials.json", failed_trials)
        _write_json(evidence_root / "failed_trials.json", failed_trials)
        _write_json(result_dir / "source_records.json", records)

        analyzer_ran = False
        run_metadata: dict[str, Any] = {}
        evidence_analysis_path = evidence_root / "analysis" / "trajectory_analysis.md"
        evidence_analysis_path.parent.mkdir(parents=True, exist_ok=True)
        if not failed_trials["failed_trials"]:
            evidence_analysis_path.write_text(
                "# Failed trajectory analysis\n\nNo failed trials were found under the configured threshold.\n",
                encoding="utf-8",
            )
        elif not analysis_path.exists() or force:
            analyzer_ran = True
            run_metadata = self._run_analyzer(
                evidence_root=evidence_root,
                result_dir=result_dir,
                failed_trials=failed_trials,
                cache_key=cache_key,
                current_harness_path=current_harness_path,
            )
        elif evidence_analysis_path.exists():
            run_metadata = {"reused_existing_analysis": True}
        else:
            shutil.copy2(analysis_path, evidence_analysis_path)
            run_metadata = {"reused_existing_analysis": True}

        if not evidence_analysis_path.exists():
            fallback_text = self._fallback_analysis_text(result_dir=result_dir, run_metadata=run_metadata)
            evidence_analysis_path.write_text(fallback_text, encoding="utf-8")

        shutil.copy2(evidence_analysis_path, analysis_path)
        metadata = {
            "result_dir": str(result_dir),
            "evidence_root": str(evidence_root),
            "analysis_path": str(analysis_path),
            "failed_trials_path": str(result_dir / "failed_trials.json"),
            "source_records_path": str(result_dir / "source_records.json"),
            **run_metadata,
        }
        _write_json(result_dir / "analysis_metadata.json", metadata)

        entry = _AnalysisCacheEntry(
            cache_key=cache_key,
            result_dir=result_dir,
            analysis_path=analysis_path,
            failed_count=len(failed_trials["failed_trials"]),
            analyzer_ran=analyzer_ran,
            metadata=metadata,
        )
        self._cache = entry
        return entry

    def _write_numbered_evidence(
        self,
        *,
        artifacts: list[CollectionArtifact],
        root: Path,
        task_cache_root: Path,
        include_trajectory: bool,
    ) -> list[dict[str, Any]]:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        records = []
        for idx, artifact in enumerate(artifacts, start=1):
            target_dir = root / f"{idx:03d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = target_dir / "trajectory.json"
            result_path = target_dir / "sanitized_result.json"
            if include_trajectory:
                shutil.copy2(artifact.trajectory_path, trajectory_path)
            context_source = artifact.context_path
            if context_source is None:
                for candidate in (
                    artifact.trajectory_path.parent / "llm-context.jsonl",
                    artifact.result_path.parent / "llm-context.jsonl",
                    artifact.result_path.parent / "agent" / "llm-context.jsonl",
                ):
                    if candidate.is_file():
                        context_source = candidate
                        break
            context_path = target_dir / "llm-context.jsonl"
            if context_source is not None and context_source.is_file():
                shutil.copy2(context_source, context_path)
            openai_source = artifact.openai_messages_path
            if openai_source is None:
                for candidate in (
                    artifact.trajectory_path.parent / "openai-messages.jsonl",
                    artifact.result_path.parent / "openai-messages.jsonl",
                    artifact.result_path.parent / "agent" / "openai-messages.jsonl",
                ):
                    if candidate.is_file():
                        openai_source = candidate
                        break
            openai_path = target_dir / "openai-messages.jsonl"
            if openai_source is not None and openai_source.is_file():
                shutil.copy2(openai_source, openai_path)
                openai_meta_source = openai_source.with_name("openai-messages.meta.json")
                if openai_meta_source.is_file():
                    shutil.copy2(openai_meta_source, target_dir / openai_meta_source.name)
            # The tool schema and sanitized outcome are useful evolution input.
            # Do not copy verifier.txt: it can expose checker details and
            # encourage task-specific reward hacking.
            for name in (
                "openai-tools.json",
                "openai-request.meta.json",
                "verifier-summary.json",
            ):
                source_extra = next(
                    (
                        candidate
                        for candidate in (
                            artifact.trajectory_path.parent / name,
                            artifact.result_path.parent / name,
                            artifact.result_path.parent / "agent" / name,
                        )
                        if candidate.is_file()
                    ),
                    None,
                )
                if source_extra is not None:
                    shutil.copy2(source_extra, target_dir / name)
            verifier_feedback_path = materialize_verifier_feedback(
                trajectory_path=artifact.trajectory_path,
                result_path=artifact.result_path,
                target_path=target_dir / "verifier-feedback.json",
            )
            write_sanitized_result(artifact.result_path, result_path)
            instruction_path = copy_instruction_if_available(
                result_path=artifact.result_path,
                task_cache_root=task_cache_root,
                target_dir=target_dir,
            )
            records.append(
                {
                    "source_trajectory_path": str(artifact.trajectory_path),
                    "source_result_path": str(artifact.result_path),
                    "workspace_trajectory_path": str(trajectory_path) if include_trajectory else "",
                    "workspace_result_path": str(result_path),
                    "workspace_instruction_path": str(instruction_path) if instruction_path else "",
                    "source_context_path": str(context_source) if context_source else "",
                    "workspace_context_path": str(context_path) if context_path.exists() else "",
                    "source_openai_messages_path": str(openai_source) if openai_source else "",
                    "workspace_openai_messages_path": str(openai_path)
                    if openai_path.exists()
                    else "",
                    "workspace_openai_tools_path": str(target_dir / "openai-tools.json")
                    if (target_dir / "openai-tools.json").exists()
                    else "",
                    "workspace_request_meta_path": str(target_dir / "openai-request.meta.json")
                    if (target_dir / "openai-request.meta.json").exists()
                    else "",
                    "workspace_verifier_summary_path": str(target_dir / "verifier-summary.json")
                    if (target_dir / "verifier-summary.json").exists()
                    else "",
                    "workspace_verifier_feedback_path": str(verifier_feedback_path)
                    if verifier_feedback_path
                    else "",
                }
            )
        return records

    def _select_failed_trials(self, evidence_root: Path) -> dict[str, Any]:
        threshold = float(self.options.get("success_threshold", 1.0))
        max_failed = int(self.options.get("max_failed_trials", 20))
        failed = []
        for result_path in sorted(evidence_root.glob("*/sanitized_result.json")):
            trial_dir = result_path.parent
            result = load_json_object(result_path)
            exception = safe_mapping(result.get("exception"))
            reward = result.get("reward")
            has_exception = bool(exception.get("type") or exception.get("message"))
            try:
                reward_value = None if reward is None else float(reward)
            except (TypeError, ValueError):
                reward_value = None
            is_failed = has_exception or reward_value is None or reward_value < threshold
            if not is_failed:
                continue
            failed.append(
                {
                    "trial_dir": trial_dir.name,
                    "task_name": result.get("task_name"),
                    "trial_name": result.get("trial_name"),
                    "reward": reward,
                    "exception": result.get("exception"),
                    "trajectory_path": f"{trial_dir.name}/trajectory.json",
                    "sanitized_result_path": f"{trial_dir.name}/sanitized_result.json",
                    "instruction_path": f"{trial_dir.name}/instruction.md"
                    if (trial_dir / "instruction.md").exists()
                    else "",
                }
            )
            if len(failed) >= max_failed:
                break
        return {
            "success_threshold": threshold,
            "max_failed_trials": max_failed,
            "failed_trials": failed,
        }

    def _run_analyzer(
        self,
        *,
        evidence_root: Path,
        result_dir: Path,
        failed_trials: dict[str, Any],
        cache_key: str,
        current_harness_path: str | Path | None,
    ) -> dict[str, Any]:
        analyzer_config = self._resolved_analyzer_config(cache_key, current_harness_path=current_harness_path)
        env, opencode_config_path = self._prepare_opencode_config(analyzer_config)
        opencode_repo_path = repo_relative_path(analyzer_config["opencode_repo_path"])
        timeout_sec = int(analyzer_config["timeout_sec"])
        prompt = self._analysis_prompt(failed_trials)

        raw_output_path = result_dir / "raw_output.log"
        trajectory_path = result_dir / "analyzer_trajectory.json"
        started_at = datetime.now(timezone.utc)
        output = ""
        returncode: int | None = None
        timed_out = False
        error: str | None = None
        error_traceback: str | None = None
        trajectory: list[dict[str, Any]] = []
        command: list[str] = []
        run_env = {**os.environ, **env}
        launch_metadata = {
            "command_prefix": analyzer_config["dev_run_command_prefix"],
            "cwd": str(opencode_repo_path),
            "timeout_sec": timeout_sec,
            "evidence_root": str(evidence_root),
            "opencode_config_path": str(opencode_config_path),
            "xdg_root": str(analyzer_config["xdg_root"]),
            "env": {key: run_env.get(key) for key in sorted(env)},
            "model": analyzer_config["model"],
            "provider_id": analyzer_config["provider_id"],
            "model_id": analyzer_config["model_id"],
        }
        _write_json(result_dir / "analyzer_launch.json", launch_metadata)
        try:
            client = OpenCodeBunDevClient(
                opencode_repo_path=str(opencode_repo_path),
                dev_run_command_prefix=str(analyzer_config["dev_run_command_prefix"]),
                dev_run_add_dir_arg=True,
                dev_run_env=env,
            )
            result = client.run_instruction_detailed(
                workspace=evidence_root,
                instruction=prompt,
                request_timeout_sec=timeout_sec,
                raise_on_error=False,
            )
            output = result.stdout
            returncode = result.returncode
            trajectory = result.trajectory
            command = result.command
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            output = _coerce_output(getattr(exc, "output", None))
            command = [str(item) for item in exc.cmd] if isinstance(exc.cmd, list) else [str(exc.cmd)]
            trajectory = _build_agent_trajectory(prompt=prompt, output=output or "")
            error = f"analyzer timed out after {timeout_sec}s"
        except Exception as exc:  # pragma: no cover - keeps result artifacts for debugging.
            error = repr(exc)
            error_traceback = traceback.format_exc()
            trajectory = _build_agent_trajectory(prompt=prompt, output=output or "")

        raw_output_path.write_text(output or "", encoding="utf-8")
        if not trajectory:
            trajectory = _build_agent_trajectory(prompt=prompt, output=output or "")
        _write_json(trajectory_path, trajectory)
        opencode_log = _copy_opencode_dev_log(xdg_root=Path(analyzer_config["xdg_root"]), result_dir=result_dir)

        if returncode not in (0, None) and error is None:
            error = f"analyzer exited with code {returncode}"

        analysis_path = evidence_root / "analysis" / "trajectory_analysis.md"
        if not analysis_path.exists():
            assistant_text = _last_assistant_text(trajectory)
            if assistant_text:
                analysis_path.write_text(assistant_text.strip() + "\n", encoding="utf-8")

        finished_at = datetime.now(timezone.utc)
        run_metadata = {
            "analyzer": {
                "command": _redacted_command(command),
                "cwd": str(opencode_repo_path),
                "returncode": returncode,
                "timed_out": timed_out,
                "error": error,
                "error_traceback": error_traceback,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_sec": (finished_at - started_at).total_seconds(),
                "raw_output_path": str(raw_output_path),
                "raw_output_tail": tail_text(output or "", 4000),
                "trajectory_path": str(trajectory_path),
                "opencode_config_path": str(opencode_config_path),
                "opencode_dev_log_path": str(opencode_log["path"]) if opencode_log else None,
                "opencode_dev_log_tail": opencode_log["tail"] if opencode_log else "",
                "xdg_root": str(analyzer_config["xdg_root"]),
                "model": analyzer_config["model"],
                "provider_id": analyzer_config["provider_id"],
                "model_id": analyzer_config["model_id"],
            }
        }
        if returncode != 0 or timed_out or error_traceback:
            _write_json(result_dir / "analyzer_failure.json", run_metadata["analyzer"])
        return run_metadata

    def _resolved_analyzer_config(
        self,
        cache_key: str,
        *,
        current_harness_path: str | Path | None,
    ) -> dict[str, Any]:
        configured = self.options.get("analyzer") or {}
        if not isinstance(configured, dict):
            raise ValueError("evidence.options.analyzer must be a mapping")
        base = self.config.opencode_evolver or {}
        merged = {**base, **configured}
        required = ["model", "openai_base_url"]
        missing = [key for key in required if not merged.get(key)]
        if missing:
            raise ValueError(
                "agent_analyzed evidence requires evidence.options.analyzer or opencode_evolver "
                f"to provide: {', '.join(missing)}"
            )

        model = str(merged["model"])
        provider_id, _, model_id = model.partition("/")
        provider_id = str(merged.get("provider_id") or provider_id)
        model_id = str(merged.get("model_id") or model_id)
        resolved_model = f"{provider_id}/{model_id}"
        xdg_root = repo_relative_path(configured.get("xdg_root") or (self.config.report_root / "analyzer" / cache_key[:12] / "xdg"))
        opencode_repo_path = str(
            configured.get("opencode_repo_path")
            or merged.get("opencode_repo_path")
            or current_harness_path
            or self.config.initial_harness_path
        )
        return {
            "model": resolved_model,
            "configured_model": model,
            "provider_id": provider_id,
            "model_id": model_id,
            "openai_base_url": str(merged["openai_base_url"]),
            "api_key": str(merged.get("api_key") or "EMPTY"),
            "provider_name": str(merged.get("provider_name") or provider_id),
            "model_name": str(merged.get("model_name") or model_id),
            "npm": str(merged.get("npm") or "@ai-sdk/openai-compatible"),
            "provider_timeout_ms": int(merged.get("provider_timeout_ms") or 300000),
            "chunk_timeout_ms": int(merged.get("chunk_timeout_ms") or 300000),
            "timeout_sec": int(merged.get("timeout_sec") or self.options.get("timeout_sec") or self.config.evolver_timeout_sec),
            "xdg_root": xdg_root,
            "opencode_repo_path": opencode_repo_path,
            "dev_run_command_prefix": str(merged.get("dev_run_command_prefix") or "bun dev -- run"),
        }

    def _prepare_opencode_config(self, analyzer_config: dict[str, Any]) -> tuple[dict[str, str], Path]:
        xdg_root = Path(analyzer_config["xdg_root"])
        xdg_paths = {
            "XDG_CONFIG_HOME": xdg_root / "config",
            "XDG_DATA_HOME": xdg_root / "data",
            "XDG_CACHE_HOME": xdg_root / "cache",
            "XDG_STATE_HOME": xdg_root / "state",
        }
        for path in xdg_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        provider_id = analyzer_config["provider_id"]
        model_id = analyzer_config["model_id"]
        openai_base_url = analyzer_config["openai_base_url"]
        npm = analyzer_config["npm"]
        generated = {
            "model": analyzer_config["model"],
            "enabled_providers": [provider_id],
            "provider": {
                provider_id: {
                    "name": analyzer_config["provider_name"],
                    "api": openai_base_url,
                    "npm": npm,
                    "options": {
                        "apiKey": analyzer_config["api_key"],
                        "timeout": analyzer_config["provider_timeout_ms"],
                        "chunkTimeout": analyzer_config["chunk_timeout_ms"],
                    },
                    "models": {
                        model_id: {
                            "name": analyzer_config["model_name"],
                            "provider": {
                                "api": openai_base_url,
                                "npm": npm,
                            },
                        },
                    },
                },
            },
        }
        config_path = xdg_paths["XDG_CONFIG_HOME"] / "opencode" / "opencode.jsonc"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        env = {key: str(value) for key, value in xdg_paths.items()}
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
        return env, config_path

    def _analysis_prompt(self, failed_trials: dict[str, Any]) -> str:
        failed_count = len(failed_trials.get("failed_trials") or [])
        return render_prompt(__package__, "failed_trajectory_analysis.md", failed_count=failed_count)

    def _fallback_analysis_text(self, *, result_dir: Path, run_metadata: dict[str, Any]) -> str:
        analyzer = safe_mapping(run_metadata.get("analyzer"))
        return (
            "# Failed trajectory analysis\n\n"
            "The analyzer did not produce analysis/trajectory_analysis.md.\n\n"
            "## Analyzer run\n\n"
            f"- result_dir: {result_dir}\n"
            f"- returncode: {analyzer.get('returncode')}\n"
            f"- timed_out: {analyzer.get('timed_out')}\n"
            f"- error: {analyzer.get('error')}\n"
            f"- raw_output_path: {analyzer.get('raw_output_path')}\n"
            f"- trajectory_path: {analyzer.get('trajectory_path')}\n"
            f"- opencode_dev_log_path: {analyzer.get('opencode_dev_log_path')}\n"
        )

    def _analysis_result_dir(self, cache_key: str) -> Path:
        configured = self.options.get("result_root")
        if configured:
            return repo_relative_path(configured) / cache_key[:12]
        return self.config.report_root / "analyzer" / cache_key[:12]

    def _cache_key(
        self,
        artifacts: list[CollectionArtifact],
        *,
        current_harness_path: str | Path | None,
    ) -> str:
        payload: dict[str, Any] = {
            "strategy": self.name,
            "current_harness_path": str(current_harness_path or self.config.initial_harness_path),
            "options": {
                key: value
                for key, value in self.options.items()
                if key not in {"result_root", "force_rerun"}
            },
            "artifacts": [],
        }
        for artifact in artifacts:
            payload["artifacts"].append(
                {
                    "trajectory_path": str(artifact.trajectory_path),
                    "trajectory_stat": _stat_fingerprint(artifact.trajectory_path),
                    "result_path": str(artifact.result_path),
                    "result_stat": _stat_fingerprint(artifact.result_path),
                    "context_path": str(artifact.context_path) if artifact.context_path else "",
                    "context_stat": _stat_fingerprint(artifact.context_path)
                    if artifact.context_path and artifact.context_path.exists()
                    else None,
                    "openai_messages_path": str(artifact.openai_messages_path)
                    if artifact.openai_messages_path
                    else "",
                    "openai_messages_stat": _stat_fingerprint(artifact.openai_messages_path)
                    if artifact.openai_messages_path and artifact.openai_messages_path.exists()
                    else None,
                }
            )
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return sha1(encoded).hexdigest()


def _stat_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bool_option(options: dict[str, Any], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _has_cli_option(command: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(token == option or token.startswith(prefix) for token in command)


def _coerce_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _copy_opencode_dev_log(*, xdg_root: Path, result_dir: Path) -> dict[str, Any] | None:
    source = xdg_root / "data" / "opencode" / "log" / "dev.log"
    if not source.exists():
        return None
    target = result_dir / "opencode_dev.log"
    shutil.copy2(source, target)
    text = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": target,
        "source_path": source,
        "tail": tail_text(text, 8000),
    }


def _build_agent_trajectory(*, prompt: str, output: str) -> list[dict[str, Any]]:
    events = _parse_json_events(output)
    trajectory: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    pending_assistant: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}

    for event in events:
        event_type = event.get("type")
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if event_type == "step_start":
            pending_assistant = _flush_assistant_message(trajectory, pending_assistant)
            continue
        if event_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                pending_assistant["content"] = f"{pending_assistant['content'] or ''}\n{text}".strip()
            continue
        if event_type == "tool_use":
            tool_call, tool_message = _convert_tool_event(part)
            if tool_call:
                pending_assistant.setdefault("tool_calls", []).append(tool_call)
            if tool_message:
                pending_assistant = _flush_assistant_message(trajectory, pending_assistant)
                trajectory.append(tool_message)
            continue
        if event_type == "step_finish":
            pending_assistant = _flush_assistant_message(trajectory, pending_assistant)
            continue

    _flush_assistant_message(trajectory, pending_assistant)
    return trajectory


def _parse_json_events(output: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in output.splitlines():
        raw = line.strip()
        if not raw.startswith("{"):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            events.append(value)
    return events


def _flush_assistant_message(trajectory: list[dict[str, Any]], message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    if content is not None or tool_calls:
        normalized = {"role": "assistant", "content": content}
        if tool_calls:
            normalized["tool_calls"] = tool_calls
        trajectory.append(normalized)
    return {"role": "assistant", "content": None, "tool_calls": []}


def _convert_tool_event(part: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    state = part.get("state") if isinstance(part.get("state"), dict) else {}
    call_id = part.get("callID")
    tool_name = part.get("tool")
    tool_call = None
    if isinstance(call_id, str) and isinstance(tool_name, str):
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(state.get("input", {}), ensure_ascii=False),
            },
        }

    tool_message = None
    if state.get("status") == "completed" or "output" in state:
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": state.get("output", ""),
        }
    elif state.get("status") == "error" or "error" in state:
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": state.get("error", ""),
            "error": True,
        }
    return tool_call, tool_message


def _last_assistant_text(trajectory: list[dict[str, Any]]) -> str:
    for message in reversed(trajectory):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _redacted_command(command: list[str]) -> list[str]:
    redacted = []
    for token in command:
        if str(REPO_ROOT) in token:
            redacted.append(token.replace(str(REPO_ROOT), "$REPO_ROOT"))
        else:
            redacted.append(token)
    return redacted
