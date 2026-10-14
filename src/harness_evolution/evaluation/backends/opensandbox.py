"""OpenSandbox held-out A/B runner for the harness gate.

Replaces the Harbor/E2B + crane image-bake path. Each trial:

  1. publishes the side's opencode binary to AFS (unless already skipped)
  2. creates an OpenSandbox with the task image
  3. pins ``opencode_commit`` so start() copies ``{AFS}/{commit7}/opencode``
  4. uploads ``tests/``, runs OpenCodeCompatHarness, then the TB verifier

The policy model is the live training one: sandboxes talk to
``AGENT_GATEWAY_URL`` (LLM-Proxy), same as Polar rollouts.

This module does not import the Harbor gate or the evolver.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from harness_evolution.core.paths import ensure_external_paths

ensure_external_paths()

from polar.agent.models import AgentSpec
from polar.runtime.models import ExecResult, RuntimeSpec
from polar.trajectory.models import Trajectory
from terminal_bench_opensandbox.opencode_compat import OpenCodeCompatHarness
from terminal_bench_opensandbox.opensandbox_runtime import OpenSandboxRuntime
from terminal_bench_opensandbox.verifier import OpenSandboxTBVerifierEvaluator

from harness_evolution.evaluation.backends import mimo as mimo_backend
from harness_evolution.traces.materialize import materialize as materialize_trace_artifacts

# Re-exported after the phase-2 split; new code should import from the owning module.
from harness_evolution.tasks.catalog import TaskSpec, _concrete, _first_nonempty, _image_from_task_dir, _tasks_index_path, load_jsonl_catalog, resolve_task_spec, resolve_task_specs  # noqa: F401
from harness_evolution.evaluation.trials import SideOutcome, Trial, TrialOutcome, _aggregate, _trial_outcome_from_json, _write_trial_outcome, expand_trials  # noqa: F401

# Re-exported after the phase-2 split; new code should import from the owning module.
from harness_evolution.evaluation.backends.opensandbox_artifacts import _download_raw_trace_for_materialization, _download_trace_bundle, _download_trace_wire_files, _keep_raw_trace_wire, _promote_final_trial_artifacts  # noqa: F401
from harness_evolution.evaluation.backends.opensandbox_settings import _log, _should_publish_for_eval, custom_afs_binary_path, heldout_backend, opensandbox_settings, publish_side_binaries  # noqa: F401


def _gate_timeout_requested(settings: Mapping[str, Any]) -> bool:
    event = settings.get("_gate_timeout_event")
    return bool(event is not None and event.is_set())


def default_runtime_factory(spec: RuntimeSpec, session_id: str, session_dir: Path) -> OpenSandboxRuntime:
    return OpenSandboxRuntime(spec, session_id, session_dir)


def _runtime_spec(trial: Trial, settings: dict[str, Any]) -> RuntimeSpec:
    gateway = settings["agent_gateway_url"].rstrip("/")
    if not gateway:
        raise RuntimeError("held-out OpenSandbox gate needs AGENT_GATEWAY_URL / opensandbox.agent_gateway_url")
    if not settings["env_file"]:
        raise RuntimeError("held-out OpenSandbox gate needs OPEN_SANDBOX_SECRET_ENV / opensandbox.env_file")
    model = settings["agent_model_name"] or "qwen3.5-9b"
    workdir = trial.task.workdir or settings["workdir"]
    # MiMo tasks work inside a repository; keep OpenCode's $HOME config out of it.
    home = mimo_backend.MIMO_HOME if trial.task.is_mimo else workdir
    agent_timeout = int(trial.task.agent_timeout_sec or settings["agent_timeout_sec"])
    verifier_timeout = int(trial.task.verifier_timeout_sec or settings["verifier_timeout_sec"])
    # Keep the sandbox alive for the full task budget.  The old fixed 900s
    # TTL could delete a sandbox while a metadata-budget task was still
    # running (some TB2.1 tasks allow 3600–12000s).
    ttl_seconds = max(int(settings["ttl_seconds"]), agent_timeout + 180 + 600, verifier_timeout + 600)
    return RuntimeSpec(
        backend="docker",
        image=trial.task.image,
        workdir=workdir,
        env={
            "HOME": home,
            "TASK_WORKSPACE": workdir,
            "ANTHROPIC_BASE_URL": gateway,
            "OPENAI_BASE_URL": f"{gateway}/v1",
            "GOOGLE_API_URL": gateway,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "dummy",
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY") or "dummy",
        },
        kwargs={
            "env_file": settings["env_file"],
            "base_url": settings["base_url"],
            "project": settings["project"],
            "docker_registry": settings["docker_registry"],
            "opencode_setup": settings["opencode_setup"],
            "opencode_afs_root": settings["opencode_afs_root"],
            "opencode_afs_bin": trial.afs_bin,
            "opencode_commit": trial.commit,  # pin this side's harness
            "opencode_upload_path": trial.binary,
            "volumes": settings["volumes"],
            "ttl_seconds": ttl_seconds,
            "cpu": trial.task.cpu or settings["cpu"],
            "memory": trial.task.memory or settings["memory"],
            "create_timeout": settings["create_timeout"],
            "create_rate": settings["create_rate"],
            "create_burst": settings["create_burst"],
            "create_retries": settings["create_retries"],
            "exec_timeout": max(agent_timeout, verifier_timeout) + 60,
            "metadata": {
                "purpose": "harness-heldout-gate",
                "gate_side": trial.side,
                "gate_task": trial.task.name,
                "opencode_commit": trial.commit,
            },
        },
        import_path="terminal_bench_opensandbox.opensandbox_runtime:OpenSandboxRuntime",
    )


def _agent_model_for_trial(trial: Trial, settings: dict[str, Any]) -> str:
    names = [str(item).strip() for item in settings.get("agent_model_names") or [] if str(item).strip()]
    if not names:
        names = [str(settings.get("agent_model_name") or "qwen3.5-9b")]
    digest = hashlib.sha256(trial.trial_id.encode("utf-8")).digest()
    return names[int.from_bytes(digest[:8], "big") % len(names)]


def _agent_spec(settings: dict[str, Any], trial: Trial) -> AgentSpec:
    model = _agent_model_for_trial(trial, settings)
    if "/" not in model:
        model = f"openai-compatible/{model}"
    return AgentSpec(
        import_path="terminal_bench_opensandbox.opencode_compat:OpenCodeCompatHarness",
        model_name=model,
        settings={
            "opencode_context_limit": settings["context_limit"],
            "opencode_output_limit": settings["output_limit"],
            "opencode_base_url": f"{settings['agent_gateway_url'].rstrip('/')}/v1",
        },
    )


async def _run_trial_once(
    trial: Trial,
    *,
    settings: dict[str, Any],
    session_dir: Path,
    runtime_factory: Callable[[RuntimeSpec, str, Path], Any] | None = None,
) -> TrialOutcome:
    factory = runtime_factory or default_runtime_factory
    runtime = factory(_runtime_spec(trial, settings), trial.trial_id, session_dir)
    active_runtimes = settings.get("_active_runtimes")
    if isinstance(active_runtimes, dict):
        active_runtimes[trial.trial_id] = runtime
    workdir = trial.task.workdir or settings["workdir"]
    detail = ""
    agent_stdout: list[str] = []
    agent_stderr: list[str] = []
    passed = False
    verifier_ran = False
    mimo_task: dict[str, Any] | None = None
    mimo_state: dict[str, Any] = {}
    try:
        await runtime.start()
        runtime_id = getattr(runtime, "runtime_id", trial.trial_id)
        _log(f"ready {trial.trial_id} runtime={runtime_id} (harness installed)")
        if trial.task.mimo_task_file is not None:
            mimo_task = mimo_backend.load_task(trial.task.mimo_task_file)
            mimo_state = await mimo_backend.prepare(runtime, mimo_task, session_dir=session_dir)
            _log(f"mimo {trial.trial_id} prepared ({mimo_task['dataset_type']})")
        else:
            await runtime.upload_dir(str(trial.task.tests_dir), "/tests")
            _log(f"tests {trial.trial_id} uploaded")
        harness = OpenCodeCompatHarness(_agent_spec(settings, trial))
        await harness.setup(runtime)
        agent_timeout = float(trial.task.agent_timeout_sec or settings["agent_timeout_sec"])
        for step in harness.run_steps(trial.task.instruction):
            # Keep a small cancellation grace outside the runtime's HTTP
            # timeout.  This prevents a stuck transport from holding the
            # whole gate after the command budget has expired.
            result: ExecResult = await asyncio.wait_for(
                runtime.exec(
                    step.command,
                    cwd=step.cwd or workdir,
                    env=step.env,
                    timeout_sec=agent_timeout,
                ),
                timeout=agent_timeout + 45.0,
            )
            if result.stdout:
                agent_stdout.append(result.stdout)
            if result.stderr:
                agent_stderr.append(result.stderr)
            if result.return_code not in (0, None):
                detail = f"agent exit {result.return_code}"
        (session_dir / "agent.stdout.log").write_text("\n".join(agent_stdout), encoding="utf-8")
        (session_dir / "agent.stderr.log").write_text("\n".join(agent_stderr), encoding="utf-8")
        _log(
            f"agent {trial.trial_id} completed"
            f"{f' ({detail})' if detail else ''}"
        )
        verifier_timeout = int(
            trial.task.verifier_timeout_sec or settings["verifier_timeout_sec"]
        )
        if mimo_task is not None:
            await mimo_backend.capture_model_diff(runtime, mimo_task, mimo_state, session_dir=session_dir)
            verification = await asyncio.wait_for(
                mimo_backend.verify(runtime, mimo_task, mimo_state, session_dir=session_dir),
                # Patch restore/apply and uploads run before the verifier itself.
                timeout=verifier_timeout + 300.0,
            )
            passed = verification.passed
            verifier_stdout, verifier_stderr = verification.stdout, verification.stderr
        else:
            evaluator = OpenSandboxTBVerifierEvaluator(command="bash /tests/test.sh", cwd=workdir)
            eval_result = await asyncio.wait_for(
                evaluator.evaluate(
                    Trajectory(status="COMPLETED"),
                    runtime=runtime,
                    timeout_seconds=verifier_timeout,
                ),
                timeout=verifier_timeout + 45.0,
            )
            passed = float(eval_result.outcome_reward or 0.0) >= 0.5
            verifier_metadata = eval_result.metadata or {}
            verifier_stdout = str(verifier_metadata.get("verifier_stdout") or "")
            verifier_stderr = str(verifier_metadata.get("verifier_stderr") or "")
        verifier_ran = True
        (session_dir / "verifier.stdout.log").write_text(verifier_stdout, encoding="utf-8")
        (session_dir / "verifier.stderr.log").write_text(verifier_stderr, encoding="utf-8")
        _log(f"verify {trial.trial_id} {'PASS' if passed else 'FAIL'}")
        return TrialOutcome(
            trial_id=trial.trial_id,
            side=trial.side,
            commit=trial.commit,
            task=trial.task.name,
            passed=passed,
            error=False,
            detail=detail or ("pass" if passed else "fail"),
        )
    except Exception as exc:  # noqa: BLE001
        if _gate_timeout_requested(settings):
            raise asyncio.CancelledError("gate deadline exceeded") from exc
        _log(f"trial {trial.trial_id} ERROR: {exc}")
        return TrialOutcome(
            trial_id=trial.trial_id,
            side=trial.side,
            commit=trial.commit,
            task=trial.task.name,
            passed=False,
            error=True,
            detail=f"{type(exc).__name__}: {exc}",
        )
    finally:
        gate_timed_out = _gate_timeout_requested(settings)
        # The sandbox is deleted below, so preserve the tee'd OpenCode stream.
        # This is especially useful when runtime.exec raises on timeout and no
        # ExecResult is available to persist above. Artifact collection must
        # never turn an otherwise valid trial into an infrastructure error.
        download_file = getattr(runtime, "download_file", None)
        if callable(download_file) and not gate_timed_out:
            agent_dir = session_dir / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            try:
                await download_file(
                    "/logs/agent/opencode.txt", str(session_dir / "agent.remote.jsonl")
                )
            except Exception as exc:  # noqa: BLE001
                _log(f"artifact {trial.trial_id} agent log unavailable: {exc}")
            try:
                debug_errors: list[str] = []
                for remote_debug in (
                    "/polar/session/logs/agent/trace-debug.txt",
                    "/logs/agent/trace-debug.txt",
                ):
                    try:
                        await download_file(remote_debug, str(agent_dir / "trace-debug.txt"))
                        break
                    except Exception as debug_exc:  # noqa: BLE001
                        debug_errors.append(f"{remote_debug}: {debug_exc}")
                else:
                    raise RuntimeError("; ".join(debug_errors))
            except Exception as exc:  # noqa: BLE001
                _log(f"artifact {trial.trial_id} trace debug unavailable: {exc}")
            try:
                for remote_log in (
                    "/logs/agent/trace-proxy.log",
                    "/logs/agent/trace-proxy-status.json",
                ):
                    try:
                        await download_file(remote_log, str(agent_dir / Path(remote_log).name))
                    except Exception as log_exc:  # noqa: BLE001
                        _log(f"artifact {trial.trial_id} trace proxy log unavailable: {log_exc}")
                raw_wire_dir = agent_dir / "llm-reqs"
                keep_raw_wire = _keep_raw_trace_wire()
                context_downloaded = False
                context_errors: list[str] = []
                for remote_context in (
                    "/tmp/polar-opencode-trace/llm-context.jsonl",
                    "/polar/session/logs/agent/llm-context.jsonl",
                    "/logs/agent/llm-context.jsonl",
                ):
                    try:
                        await download_file(remote_context, str(agent_dir / "llm-context.jsonl"))
                        context_downloaded = True
                        break
                    except Exception as context_exc:  # noqa: BLE001
                        context_errors.append(f"{remote_context}: {context_exc}")
                # Most images do not publish the normalized context file. In
                # that case, fetch the proxy wire temporarily so the local
                # materializer can reconstruct it from request metadata and
                # streamed responses. Explicit debug mode keeps the same raw
                # files after materialization; normal runs remove them below.
                raw_wire_available = False
                if keep_raw_wire or not context_downloaded:
                    if not context_downloaded:
                        _log(
                            f"artifact {trial.trial_id} normalized context unavailable; "
                            "temporarily downloading raw trace wire for materialization"
                        )
                    raw_wire_available = await _download_raw_trace_for_materialization(
                        runtime,
                        session_dir,
                        raw_wire_dir,
                    )
                if not context_downloaded and not raw_wire_available:
                    # A sandbox/control-plane failure can prevent both the
                    # normalized context and the temporary raw fallback from
                    # being downloaded. Keep the trial result for operations,
                    # but let materialize_trace_artifacts mark the evidence
                    # incomplete so preflight excludes it from evolution.
                    _log(
                        f"artifact {trial.trial_id} normalized context unavailable; "
                        f"recording incomplete trace ({'; '.join(context_errors)})"
                    )
                try:
                    materialize_trace_artifacts(
                        session_dir,
                        reward="missing" if (not verifier_ran and detail) else ("1" if passed else "0"),
                        status="ok" if verifier_ran else "error",
                        error=detail or None,
                        extra_meta={
                            "verifier_rc": 0 if verifier_ran else None,
                            "trace_capture_mode": (
                                "portable-proxy-context"
                                if context_downloaded
                                else ("portable-proxy-raw-wire" if raw_wire_available else "unavailable")
                            ),
                        },
                    )
                finally:
                    # The raw wire is only a transport-level fallback.  Keep
                    # the normalized evidence and remove request/response
                    # payloads even when materialization itself fails.
                    if raw_wire_dir.is_dir() and not keep_raw_wire:
                        shutil.rmtree(raw_wire_dir, ignore_errors=True)
                        _log(f"artifact {trial.trial_id} removed temporary raw trace wire")
            except Exception as exc:  # noqa: BLE001
                _log(f"artifact {trial.trial_id} full context unavailable: {exc}")
        if gate_timed_out:
            _log(f"gate timeout {trial.trial_id}: skipping artifact collection; outer gate is stopping runtime")
        else:
            try:
                await runtime.stop()
            except Exception as exc:  # noqa: BLE001
                _log(f"trial {trial.trial_id} stop failed: {exc}")


def _infrastructure_error_type(detail: str) -> str:
    """Return a stable label for errors eligible for zero-score fallback."""
    text = str(detail or "").lower()
    labels = (
        ("verifierinfrastructureerror", "verifier_infrastructure"),
        ("mimoinfrastructureerror", "verifier_infrastructure"),
        ("verifier tests did not execute", "verifier_infrastructure"),
        ("route not active", "sandbox_route"),
        ("route not found", "sandbox_route"),
        ("command result unknown", "sandbox_disconnect"),
        ("command termination unconfirmed", "sandbox_disconnect"),
        ("connecttimeout", "sandbox_connectivity"),
        ("readtimeout", "sandbox_connectivity"),
        ("all connection attempts failed", "sandbox_connectivity"),
        ("transport error", "sandbox_connectivity"),
        ("peer closed connection", "sandbox_disconnect"),
    )
    for marker, label in labels:
        if marker in text:
            return label
    return ""


async def run_trial(
    trial: Trial,
    *,
    settings: dict[str, Any],
    session_dir: Path,
    runtime_factory: Callable[[RuntimeSpec, str, Path], Any] | None = None,
) -> TrialOutcome:
    """Run one trial, retrying terminal failures in fresh sandboxes."""
    max_attempts = max(1, int(settings.get("trial_retries") or 1))
    history: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        if _gate_timeout_requested(settings):
            raise asyncio.CancelledError("gate deadline exceeded")
        attempt_dir = session_dir if attempt == 1 else session_dir.parent / f"{session_dir.name}.retry-{attempt - 1}"
        outcome = await _run_trial_once(
            trial,
            settings=settings,
            session_dir=attempt_dir,
            runtime_factory=runtime_factory,
        )
        if _gate_timeout_requested(settings):
            raise asyncio.CancelledError("gate deadline exceeded")
        history.append(
            {
                "attempt": attempt,
                "error": outcome.error,
                "passed": outcome.passed,
                "detail": outcome.detail,
            }
        )
        # Every failed attempt gets the configured number of fresh-sandbox
        # retries.  The gate does not maintain an error-type allowlist: once
        # the retry budget is exhausted, the durable terminal error is scored
        # as reward zero.
        if not outcome.error or attempt >= max_attempts:
            if outcome.error and attempt >= max_attempts:
                # A terminal trial has a durable record even when the final
                # failure is a timeout or another non-retryable exception.
                # Gate scoring treats every such record as reward zero; the
                # raw error remains visible in trial.json and reports.
                outcome = replace(
                    outcome,
                    imputed_zero=True,
                    retry_exhausted=True,
                    error_type=_infrastructure_error_type(outcome.detail) or "terminal_error",
                )
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "retry_history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return outcome
        delay = min(20.0, float(2 ** (attempt - 1)))
        _log(
            f"retry {trial.trial_id} after trial error "
            f"(attempt {attempt + 1}/{max_attempts}) in {delay:.0f}s: {outcome.detail}"
        )
        if _gate_timeout_requested(settings):
            raise asyncio.CancelledError("gate deadline exceeded")
        await asyncio.sleep(delay)
    raise AssertionError("trial retry loop did not return")


async def _run_trials(
    trials: list[Trial],
    *,
    settings: dict[str, Any],
    jobs_dir: Path,
    n_concurrent: int,
    runtime_factory: Callable[[RuntimeSpec, str, Path], Any] | None = None,
    trial_runner: Callable[..., Any] | None = None,
    existing_outcomes: Mapping[str, TrialOutcome] | None = None,
    overall_timeout_sec: int | float = 0,
) -> list[TrialOutcome]:
    runner = trial_runner or run_trial
    semaphore = asyncio.Semaphore(max(1, n_concurrent))
    gate_timeout_event = asyncio.Event()
    active_runtimes: dict[str, Any] = {}
    runner_settings = dict(settings)
    runner_settings["_gate_timeout_event"] = gate_timeout_event
    runner_settings["_active_runtimes"] = active_runtimes
    cleanup_grace = max(0.0, float(settings.get("gate_cleanup_grace_sec", 60)))

    async def _one(trial: Trial) -> TrialOutcome:
        async with semaphore:
            session_dir = jobs_dir / trial.side / trial.trial_id
            session_dir.mkdir(parents=True, exist_ok=True)
            if existing_outcomes and trial.trial_id in existing_outcomes:
                outcome = existing_outcomes[trial.trial_id]
                _log(f"reuse {trial.trial_id} {'PASS' if outcome.passed else ('ERROR' if outcome.error else 'FAIL')}")
                return outcome
            _log(f"start {trial.trial_id} commit={trial.commit[:12]} image={trial.task.image}")
            try:
                outcome = await runner(
                    trial,
                    settings=runner_settings,
                    session_dir=session_dir,
                    runtime_factory=runtime_factory,
                )
            except asyncio.CancelledError as exc:
                # A cancellation from one runtime must become a durable
                # task-level infrastructure error, not cancel the whole gate.
                outcome = TrialOutcome(
                    trial_id=trial.trial_id,
                    side=trial.side,
                    commit=trial.commit,
                    task=trial.task.name,
                    passed=False,
                    error=True,
                    detail=f"CancelledError: {exc}",
                    imputed_zero=True,
                    retry_exhausted=True,
                    error_type="task_cancelled",
                )
                _log(f"trial {trial.trial_id} cancelled; recording task-level error")
            except Exception as exc:  # noqa: BLE001
                # Keep an unexpected trial failure isolated to this task.
                outcome = TrialOutcome(
                    trial_id=trial.trial_id,
                    side=trial.side,
                    commit=trial.commit,
                    task=trial.task.name,
                    passed=False,
                    error=True,
                    detail=f"{type(exc).__name__}: {exc}",
                    imputed_zero=True,
                    retry_exhausted=True,
                    error_type="trial_runner_exception",
                )
                _log(f"trial {trial.trial_id} uncaught runner exception: {exc}")
            _write_trial_outcome(session_dir / "trial.json", outcome)
            _log(f"done  {trial.trial_id} {'PASS' if outcome.passed else ('ERROR' if outcome.error else 'FAIL')}")
            return outcome

    # Keep one task per trial so a gate-wide deadline can cancel all active
    # sandboxes together.  Per-trial timeouts remain enforced inside
    # ``_run_trial_once``; this is only the outer deadline for the whole
    # baseline/candidate batch.
    tasks = [asyncio.create_task(_one(trial), name=trial.trial_id) for trial in trials]
    gate_timeout = float(overall_timeout_sec or 0)
    if gate_timeout > 0:
        done, pending = await asyncio.wait(tasks, timeout=gate_timeout)
    else:
        done, pending = await asyncio.wait(tasks)

    timed_out = set(pending)
    if pending:
        gate_timeout_event.set()
        _log(
            f"overall gate timeout after {gate_timeout:.0f}s; "
            f"hard-stopping {len(pending)} unfinished trial(s) "
            f"with {cleanup_grace:.0f}s cleanup grace"
        )
        for task in pending:
            task.cancel()
        cleanup_started = asyncio.get_running_loop().time()
        stop_timeout = min(30.0, cleanup_grace)

        async def _stop_runtime(trial_id: str, runtime: Any) -> None:
            try:
                await asyncio.wait_for(runtime.stop(), timeout=stop_timeout)
            except asyncio.TimeoutError:
                _log(f"gate timeout {trial_id}: runtime.stop exceeded {stop_timeout:.0f}s")
            except Exception as exc:  # noqa: BLE001
                _log(f"gate timeout {trial_id}: runtime.stop failed: {exc}")

        if stop_timeout > 0:
            stop_tasks = []
            for task in pending:
                trial_id = task.get_name()
                runtime = active_runtimes.get(trial_id)
                if runtime is not None:
                    stop_tasks.append(asyncio.create_task(_stop_runtime(trial_id, runtime)))
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)

        remaining_grace = max(
            0.0,
            cleanup_grace - (asyncio.get_running_loop().time() - cleanup_started),
        )
        _, cleanup_pending = await asyncio.wait(pending, timeout=remaining_grace)
        if cleanup_pending:
            _log(
                f"gate cleanup grace exhausted; force-cancelling "
                f"{len(cleanup_pending)} remaining trial task(s)"
            )
            for task in cleanup_pending:
                task.cancel()
            # Deliver the second cancellation to a runtime adapter that
            # swallowed the first while converting it to a transport error.
            await asyncio.wait(cleanup_pending, timeout=min(1.0, max(0.05, cleanup_grace)))

    # A task can finish while cancellation cleanup is running.  It still did
    # not finish before the gate deadline, so keep it scored as a timeout.
    gathered: list[TrialOutcome | BaseException] = []
    for trial, task in zip(trials, tasks):
        if task in timed_out:
            outcome = TrialOutcome(
                trial_id=trial.trial_id,
                side=trial.side,
                commit=trial.commit,
                task=trial.task.name,
                passed=False,
                error=True,
                detail=f"overall gate timeout after {gate_timeout:.0f}s",
                imputed_zero=True,
                retry_exhausted=True,
                error_type="gate_timeout",
            )
            _write_trial_outcome(
                jobs_dir / trial.side / trial.trial_id / "trial.json", outcome
            )
            gathered.append(outcome)
            continue
        if task.cancelled():
            gathered.append(asyncio.CancelledError())
            continue
        try:
            gathered.append(task.result())
        except BaseException as exc:  # noqa: BLE001
            gathered.append(exc)

    # return_exceptions protects the gate from a future BaseException escaping
    # a runtime adapter.  Normal task exceptions are already converted above;
    # this is the final containment boundary for the whole validation batch.
    outcomes: list[TrialOutcome] = []
    for trial, value in zip(trials, gathered):
        if isinstance(value, TrialOutcome):
            outcomes.append(value)
            continue
        detail = f"{type(value).__name__}: {value}"
        outcome = TrialOutcome(
            trial_id=trial.trial_id,
            side=trial.side,
            commit=trial.commit,
            task=trial.task.name,
            passed=False,
            error=True,
            detail=detail,
            imputed_zero=True,
            retry_exhausted=True,
            error_type="trial_task_exception",
        )
        _write_trial_outcome(jobs_dir / trial.side / trial.trial_id / "trial.json", outcome)
        _log(f"trial {trial.trial_id} isolated exception: {detail}")
        outcomes.append(outcome)
    return outcomes


def run_heldout_opensandbox(
    config: dict[str, Any],
    candidates: list[tuple[str, str]],
    *,
    task_names: list[str] | None = None,
    task_catalog: dict[str, dict[str, Any]] | None = None,
    include_baseline: bool = True,
    runtime_factory: Callable[[RuntimeSpec, str, Path], Any] | None = None,
    publish: Callable[..., str] | None = None,
    trial_runner: Callable[..., Any] | None = None,
    afs_paths: Mapping[str, str] | None = None,
    resume_jobs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run baseline + candidates on OpenSandbox. Returns side outcomes.

    Shape::

        {
          "baseline": SideOutcome,
          "candidates": {full_commit: SideOutcome},
          "distinct_task_names": [...],
          "per_task_attempts": {task: n},
        }
    """
    heldout = config["heldout_gate"]
    settings = opensandbox_settings(heldout)
    names = list(task_names) if task_names is not None else list(heldout.get("task_names") or [])
    if not names:
        raise RuntimeError("held-out OpenSandbox gate has no task_names")
    specs = resolve_task_specs(heldout, names, task_catalog=task_catalog)
    n_attempts = int(heldout.get("n_attempts", 1))
    n_concurrent = int(heldout.get("n_concurrent") or 16)

    baseline = _baseline_from_config(config)
    baseline_short = baseline["commit"]
    custom_afs_paths = dict(afs_paths or {})
    sides: list[tuple[str, str, str, str]] = []
    seen_short: dict[str, str] = {}
    if include_baseline:
        sides.append(
            (
                "baseline",
                baseline_short,
                str(baseline.get("binary_path") or ""),
                custom_afs_paths.get(baseline_short, ""),
            )
        )
        seen_short[baseline_short[:7]] = baseline_short
    for i, (commit, binary) in enumerate(candidates):
        short = commit[:7]
        if short in seen_short and seen_short[short] != commit:
            raise RuntimeError(f"short-commit collision between {seen_short[short]} and {commit}")
        seen_short[short] = commit
        sides.append((f"candidate-{i + 1:02d}", commit, binary, custom_afs_paths.get(commit, "")))

    _log(
        f"{len(dict.fromkeys(names))} distinct task(s) ({len(names)} entries) x {n_attempts} "
        f"attempt(s)/entry, {len(sides)} sides, n_concurrent={n_concurrent}, "
        f"gate_timeout={settings['gate_timeout_sec'] or 'disabled'}s, "
        f"gate_cleanup_grace={settings['gate_cleanup_grace_sec']:.0f}s"
    )
    if settings["opencode_setup"] == "upload":
        missing = [
            f"{label}:{binary}"
            for label, _commit, binary, _afs_bin in sides
            if not binary or not Path(binary).is_file()
        ]
        if missing:
            raise RuntimeError(f"OpenSandbox direct upload needs every side's local binary: {', '.join(missing)}")
        _log("using direct binary upload to each sandbox; skipping RJob-to-AFS publish")
    else:
        publish_side_binaries(sides, publish=publish, env_file=settings["env_file"])

    jobs_root = Path(
        heldout.get("jobs_dir")
        or (Path(config["paths"]["report_root"]) / "heldout_jobs")
    )
    jobs_dir = Path(resume_jobs_dir) if resume_jobs_dir else jobs_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    jobs_dir.mkdir(parents=True, exist_ok=True)

    trials = expand_trials(
        sides=sides,
        task_names=names,
        specs=specs,
        n_attempts=n_attempts,
    )
    existing_outcomes: dict[str, TrialOutcome] = {}
    if resume_jobs_dir:
        for trial in trials:
            outcome = _trial_outcome_from_json(
                jobs_dir / trial.side / trial.trial_id / "trial.json", trial
            )
            if outcome is not None:
                existing_outcomes[trial.trial_id] = outcome
        _log(
            f"resuming validation job {jobs_dir}: "
            f"reusing {len(existing_outcomes)}/{len(trials)} completed trial(s)"
        )

    outcomes = asyncio.run(
        _run_trials(
            trials,
            settings=settings,
            jobs_dir=jobs_dir,
            n_concurrent=n_concurrent,
            runtime_factory=runtime_factory,
            trial_runner=trial_runner,
            existing_outcomes=existing_outcomes,
            overall_timeout_sec=settings["gate_timeout_sec"],
        )
    )

    aggregated = _aggregate(trials, outcomes, jobs_dir)
    per_task_attempts = {
        task: count * n_attempts for task, count in Counter(names).items()
    }
    candidate_map: dict[str, SideOutcome] = {}
    for commit, _binary in candidates:
        label = next(lbl for lbl, cmt, _b, _afs_bin in sides if cmt == commit)
        candidate_map[commit] = aggregated[label]
    return {
        "baseline": aggregated.get("baseline"),
        "candidates": candidate_map,
        "distinct_task_names": list(dict.fromkeys(names)),
        "per_task_attempts": per_task_attempts,
        "gate_timeout_sec": settings["gate_timeout_sec"],
        "timed_out_trials": [
            outcome.trial_id
            for outcome in outcomes
            if outcome.error_type == "gate_timeout"
        ],
        "jobs_dir": str(jobs_dir),
    }


def _baseline_from_config(config: dict[str, Any]) -> dict[str, str]:
    from harness_dist import state as version_state

    paths = config.get("paths") or {}
    state_path = paths.get("version_state")
    if state_path and Path(state_path).exists():
        return version_state.read_active(state_path)
    commit = str(paths.get("baseline_commit") or "")
    # Standalone two-sided loops keep the active parent in their own state
    # file and pass its binary explicitly.  Older callers do not set
    # baseline_binary, so preserve the previous empty default.
    binary = str(paths.get("baseline_binary") or "")
    return {"commit": commit, "commit_short": commit[:7], "binary_path": binary}
