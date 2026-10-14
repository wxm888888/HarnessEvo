"""Run the configured validation commands (typecheck, build) on a candidate workspace."""
from __future__ import annotations

import subprocess
import time
import traceback
from pathlib import Path
from harness_evolution.core.types import CommandValidationResult, RuntimeConfig, ValidationResult
from harness_evolution.core.config import _bool_value, _str_list, _tail_text
from harness_evolution.harnesses.opencode.build import _command_env, build_opencode_binary_from_source


def _validation_log_path(config: RuntimeConfig, iteration_idx: int, attempt_idx: int, rollout_idx: int, command: str) -> Path:
    return (
        config.report_root
        / "validation"
        / f"iteration_{iteration_idx + 1:03d}"
        / f"attempt_{attempt_idx + 1:03d}"
        / f"rollout_{rollout_idx + 1:03d}_{command}.log"
    )


def _run_validation_command(
    *,
    config: RuntimeConfig,
    workspace: Path,
    iteration_idx: int,
    attempt_idx: int,
    rollout_idx: int,
    command: str,
) -> CommandValidationResult:
    started = time.monotonic()
    log_path = _validation_log_path(config, iteration_idx, attempt_idx, rollout_idx, command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if command == "typecheck":
        # Scope to the opencode package plus its internal dependency closure
        # (which includes packages/core, the earlier blind spot). The full
        # workspace is deliberately NOT typechecked: desktop/app UI packages
        # fail tsc in worktrees with symlinked node_modules even on a pristine
        # baseline, and none of them ship into the harness binary.
        result = subprocess.run(
            ["bun", "run", "typecheck", "--filter=opencode..."],
            cwd=str(workspace),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_command_env(
                config,
                {"HUSKY": "0", "TURBO_TELEMETRY_DISABLED": "1"},
            ),
            check=False,
        )
        output = result.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        return CommandValidationResult(
            command=command,
            passed=result.returncode == 0,
            returncode=result.returncode,
            duration_sec=round(time.monotonic() - started, 3),
            log_path=str(log_path),
            output_tail=_tail_text(output),
        )
    if command == "build":
        try:
            binary = build_opencode_binary_from_source(config, workspace)
        except Exception:
            output = traceback.format_exc()
            log_path.write_text(output, encoding="utf-8")
            return CommandValidationResult(
                command=command,
                passed=False,
                returncode=1,
                duration_sec=round(time.monotonic() - started, 3),
                log_path=str(log_path),
                output_tail=_tail_text(output),
            )
        output = f"build validation passed\nbinary_path: {binary}\n"
        log_path.write_text(output, encoding="utf-8")
        return CommandValidationResult(
            command=command,
            passed=True,
            returncode=0,
            duration_sec=round(time.monotonic() - started, 3),
            log_path=str(log_path),
            output_tail=output,
        )
    raise ValueError(f"unsupported validation command: {command}")


def validate_candidate(
    *,
    config: RuntimeConfig,
    workspace: Path,
    iteration_idx: int,
    attempt_idx: int,
    rollout_idx: int,
    changed_files: list[str],
) -> ValidationResult:
    if not _bool_value(config.validation, "enabled", True):
        return ValidationResult(enabled=False, passed=True)
    if not changed_files:
        # A candidate that touched nothing passes every static check, so without
        # this it gets selected as the "first valid" one and the iteration is
        # reported as a successful evolution that changed nothing.
        log_path = _validation_log_path(config, iteration_idx, attempt_idx, rollout_idx, "no_op")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output = "candidate changed no files\n"
        log_path.write_text(output, encoding="utf-8")
        return ValidationResult(
            enabled=True,
            passed=False,
            commands=[
                CommandValidationResult(
                    command="no_op",
                    passed=False,
                    returncode=1,
                    duration_sec=0.0,
                    log_path=str(log_path),
                    output_tail=output,
                )
            ],
        )
    commands = _str_list(config.validation, "commands") if "commands" in config.validation else ["typecheck"]
    allowed = {"typecheck", "build"}
    unknown = [command for command in commands if command not in allowed]
    if unknown:
        raise ValueError(f"unsupported validation command(s): {', '.join(unknown)}")
    results = []
    for command in commands:
        result = _run_validation_command(
            config=config,
            workspace=workspace,
            iteration_idx=iteration_idx,
            attempt_idx=attempt_idx,
            rollout_idx=rollout_idx,
            command=command,
        )
        results.append(result)
        if not result.passed:
            break
    return ValidationResult(enabled=True, passed=all(result.passed for result in results), commands=results)
