"""Static gate: no-op check, diff limits, typecheck, build and smoke test, without sandboxes."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from harness_evolution.loop import modular as evolver
from harness_evolution.core.types import RuntimeConfig
from harness_evolution.evaluation.gate_common import _git, _log, _paths, _report_root, _resolve_baseline


@dataclass
class GateCheck:
    name: str
    passed: bool
    duration_sec: float
    detail: str = ""
    log_path: str = ""


@dataclass
class StaticGateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    candidate_commit: str = ""
    changed_files: list[str] = field(default_factory=list)
    binary_path: str = ""


def _build_runtime_config(config: dict[str, Any]) -> RuntimeConfig:
    """Minimal RuntimeConfig so the evolver's build helper can be reused."""
    workspace_root = Path(_paths(config)["workspace_root"]).expanduser()
    report_root = _report_root(config)
    return RuntimeConfig(
        raw={},
        initial_harness_path=str(_paths(config)["baseline_repo"]),
        initial_base_commit=str(_paths(config)["baseline_commit"]),
        iterations=0,
        rollouts_per_attempt=0,
        workspace_root=workspace_root,
        report_root=report_root,
        evolver_timeout_sec=0,
        terminal_bench={},
        build=dict(config["build"]),
        opencode_evolver={},
        task_selection={},
        evidence={},
        candidate_selection={},
        validation={},
    )


_BINARY_DIFF_NOISE_PATTERNS = (
    # opencode's script/build.ts embeds f"0.0.0-{git branch --show-current}-
    # {build wall-clock time}" as the CLI's own --version string on *every*
    # single build -- so even rebuilding the exact same commit twice yields a
    # different byte for byte binary. The channel half is the worktree's own
    # branch name (unique per candidate/rollout), the timestamp half changes
    # every minute.
    re.compile(r"0\.0\.0-\S+"),
    # bun's bundler assigns each internal chunk a fresh random filename
    # (chunk-<8 random chars>.js) baked into the binary's embedded virtual
    # filesystem on every build, independent of whether the chunk's actual
    # *content* changed.
    re.compile(r"/\$bunfs/root/chunk-[0-9a-z]+\.js"),
)


def _binary_content_fingerprint(path: str | Path) -> str:
    """A build-nondeterminism-insensitive fingerprint of a built opencode
    binary, for detecting whether two builds actually differ in substance.

    Do not just sha256 the raw file: see _BINARY_DIFF_NOISE_PATTERNS above --
    two builds of the *identical* commit already differ on every single
    build due to those two sources of noise, which would make a raw whole-
    file hash comparison always report "different" and never catch a true
    no-op candidate. Extract printable strings, drop every line matching
    either noise pattern, and hash the sorted deduplicated remainder so only
    genuine source-content differences register. (Empirically verified
    against a real promoted-vs-baseline pair from this run: raw file hashes
    differed, filtered-string fingerprints were identical -- confirming that
    pair's source diff never reached the binary.)
    """
    proc = subprocess.run(
        ["strings", "-n", "8", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    lines = {
        line
        for line in proc.stdout.splitlines()
        if not any(pattern.search(line) for pattern in _BINARY_DIFF_NOISE_PATTERNS)
    }
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line.encode("utf-8", "surrogateescape"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_static_gate(config: dict[str, Any], workspace: Path) -> StaticGateResult:
    static_cfg = config["static_gate"]
    baseline = _resolve_baseline(config)
    base_commit = baseline["commit"]
    baseline_binary_path = baseline.get("binary_path", "")
    result = StaticGateResult(passed=False)
    logs_dir = _report_root(config) / "static_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    head = _git(workspace, "rev-parse", "HEAD")
    if head.returncode != 0:
        result.checks.append(GateCheck("resolve_head", False, 0.0, detail=head.stdout.strip()))
        return result
    result.candidate_commit = head.stdout.strip()

    diff = _git(workspace, "diff", "--name-only", f"{base_commit}..HEAD")
    result.changed_files = [line for line in diff.stdout.splitlines() if line.strip()]

    # 1. no-op
    t0 = time.monotonic()
    if not result.changed_files:
        result.checks.append(
            GateCheck("no_op", False, round(time.monotonic() - t0, 3), detail="candidate changed no files")
        )
        _log("static gate: REJECT (no-op candidate, zero changed files)")
        return result
    result.checks.append(
        GateCheck("no_op", True, round(time.monotonic() - t0, 3), detail=f"{len(result.changed_files)} files changed")
    )
    _log(f"static gate: no-op check ok ({len(result.changed_files)} changed files)")

    # 2. optional diff size limit
    diff_cfg = static_cfg.get("diff_limit") or {}
    if diff_cfg.get("enabled", False):
        t0 = time.monotonic()
        numstat = _git(workspace, "diff", "--numstat", f"{base_commit}..HEAD")
        files = added = deleted = 0
        for line in numstat.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            files += 1
            added += int(parts[0]) if parts[0].isdigit() else 1000
            deleted += int(parts[1]) if parts[1].isdigit() else 1000
        max_files = int(diff_cfg.get("max_files", 20))
        max_lines = int(diff_cfg.get("max_lines", 800))
        ok = files <= max_files and (added + deleted) <= max_lines
        detail = f"files={files} (max {max_files}), lines={added + deleted} (max {max_lines})"
        result.checks.append(GateCheck("diff_limit", ok, round(time.monotonic() - t0, 3), detail=detail))
        _log(f"static gate: diff limit {'ok' if ok else 'REJECT'} ({detail})")
        if not ok:
            return result

    # 3. typecheck, scoped to the packages that ship into the harness binary
    # (opencode + its internal dependency closure, which includes core).
    # Full-workspace typecheck is deliberately NOT used: desktop/app UI
    # packages fail typecheck in worktrees with symlinked node_modules even
    # on a pristine baseline, and none of them are part of the built binary.
    if static_cfg.get("typecheck", True):
        t0 = time.monotonic()
        typecheck_filter = str(static_cfg.get("typecheck_filter", "opencode..."))
        log_path = logs_dir / f"{result.candidate_commit[:12]}_typecheck.log"
        proc = subprocess.run(
            ["bun", "run", "typecheck", f"--filter={typecheck_filter}"],
            cwd=str(workspace),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=evolver._command_env(
                _build_runtime_config(config),
                {"HUSKY": "0", "TURBO_TELEMETRY_DISABLED": "1"},
            ),
            check=False,
        )
        log_path.write_text(proc.stdout or "", encoding="utf-8")
        ok = proc.returncode == 0
        result.checks.append(
            GateCheck(
                "typecheck",
                ok,
                round(time.monotonic() - t0, 3),
                detail=f"exit {proc.returncode}",
                log_path=str(log_path),
            )
        )
        _log(f"static gate: typecheck {'ok' if ok else 'REJECT'} in {time.monotonic() - t0:.1f}s")
        if not ok:
            return result

    # 4. build (registers the binary keyed by commit + recipe)
    binary: Path | None = None
    if static_cfg.get("build", True):
        t0 = time.monotonic()
        log_path = logs_dir / f"{result.candidate_commit[:12]}_build.log"
        try:
            binary = evolver.build_opencode_binary_from_source(_build_runtime_config(config), workspace)
        except Exception as exc:  # noqa: BLE001
            log_path.write_text(str(exc), encoding="utf-8")
            result.checks.append(
                GateCheck("build", False, round(time.monotonic() - t0, 3), detail=str(exc)[:400], log_path=str(log_path))
            )
            _log("static gate: build REJECT")
            return result
        result.binary_path = str(binary)
        result.checks.append(
            GateCheck("build", True, round(time.monotonic() - t0, 3), detail=f"binary: {binary}")
        )
        _log(f"static gate: build ok in {time.monotonic() - t0:.1f}s -> {binary}")

    # 4b. binary-diff: the candidate changed source files (check 1) but if the
    # *compiled* binary is byte-identical to baseline's, none of that source
    # diff actually reached the shipped harness (e.g. a build-environment bug
    # silently discarding changes to a workspace package) -- reject rather
    # than let a no-op-in-practice candidate get promoted on the strength of
    # its git diff alone. Skipped (not rejected) when no baseline binary is on
    # record yet, e.g. the very first evolution round of a fresh lineage.
    if binary is not None and static_cfg.get("binary_diff", True):
        if baseline_binary_path and Path(baseline_binary_path).exists():
            t0 = time.monotonic()
            candidate_hash = _binary_content_fingerprint(binary)
            baseline_hash = _binary_content_fingerprint(baseline_binary_path)
            ok = candidate_hash != baseline_hash
            detail = f"candidate {candidate_hash[:12]} vs baseline {baseline_hash[:12]}"
            result.checks.append(
                GateCheck("binary_diff", ok, round(time.monotonic() - t0, 3), detail=detail)
            )
            _log(f"static gate: binary diff {'ok (differs from baseline)' if ok else 'REJECT (identical to baseline despite source diff)'}")
            if not ok:
                return result
        else:
            _log("static gate: binary diff check skipped (no baseline binary on record yet)")

    # 5. smoke test: the binary must at least start up and answer --version
    if static_cfg.get("smoke", True) and binary is not None:
        t0 = time.monotonic()
        timeout = int(static_cfg.get("smoke_timeout_sec", 60))
        with tempfile.TemporaryDirectory(prefix="gate_smoke_xdg_") as xdg:
            env = {
                **os.environ,
                "XDG_CONFIG_HOME": f"{xdg}/config",
                "XDG_DATA_HOME": f"{xdg}/data",
                "XDG_CACHE_HOME": f"{xdg}/cache",
                "XDG_STATE_HOME": f"{xdg}/state",
            }
            try:
                proc = subprocess.run(
                    [str(binary), "--version"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=timeout,
                    check=False,
                )
                ok = proc.returncode == 0 and bool(proc.stdout.strip())
                detail = f"exit {proc.returncode}, version: {proc.stdout.strip()[:80]}"
            except subprocess.TimeoutExpired:
                ok = False
                detail = f"timed out after {timeout}s"
        result.checks.append(GateCheck("smoke", ok, round(time.monotonic() - t0, 3), detail=detail))
        _log(f"static gate: smoke {'ok' if ok else 'REJECT'} ({detail})")
        if not ok:
            return result

    result.passed = True
    _log("static gate: PASSED")
    return result


def run_static_gate_batch(config: dict[str, Any], workspaces: list[Path]) -> list[StaticGateResult]:
    """Run the (local, sandbox-free) static gate for several candidate
    worktrees concurrently. Order of the returned list matches `workspaces`."""
    if len(workspaces) == 1:
        return [run_static_gate(config, workspaces[0])]
    results: list[StaticGateResult | None] = [None] * len(workspaces)
    with ThreadPoolExecutor(max_workers=len(workspaces)) as ex:
        futures = {ex.submit(run_static_gate, config, ws): i for i, ws in enumerate(workspaces)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results  # type: ignore[return-value]
