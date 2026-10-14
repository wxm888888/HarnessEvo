"""One YAML file describing a whole two-sided evolution run.

A run used to be assembled from three places: the evolve YAML, the gate YAML,
and a launch script that exported environment variables and spelled out the
loop's command-line flags. ``run.yaml`` holds all of it::

    run_dir: /abs/path/to/run            # every artifact of the run goes here
    evolve: evolve.yaml                  # path (relative to run.yaml) or inline mapping
    gate: gate.yaml                      # path or inline mapping
    data:
      held_in_catalog: tmax.jsonl        # fresh held-in batch per step
      held_in_size: 32
      held_in_seed: 0
      held_out: tb21_89.json             # fixed held-out split for the gate
      task_catalogs: [tmax.jsonl, tb21.jsonl]
    loop:
      steps: 10
      candidates: 2
      repeats: 1
      sampling_repeats: 1
      gate_repeats: 2
      validation_pool_size: 32
      sampling_concurrency: 32
      trajectory_retention_steps: 2
    gate_model:
      names: [model-a, model-b]          # replicas, distributed across trials
      gateway: http://...
    env_files: [/path/to/.env]           # KEY=VALUE lines, optional `export `
    env:                                 # knobs that modules read from the environment
      HARNESSRL_OPENCODE_SETUP: afs_bin

Only ``run_dir``, ``evolve``, ``gate`` and ``data.held_out`` are required;
omitted loop settings keep the loop's own defaults. Values in ``env`` are
exported for the run and override the caller's environment; ``env_files`` are
applied first so ``env`` can override them.
"""
from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_LOOP_FLAGS = {
    "steps": "--steps",
    "candidates": "--candidates",
    "repeats": "--repeats",
    "sampling_repeats": "--sampling-repeats",
    "gate_repeats": "--gate-repeats",
    "validation_pool_size": "--validation-pool-size",
    "sampling_concurrency": "--sampling-concurrency",
    "trajectory_retention_steps": "--trajectory-retention-steps",
}
_DATA_FLAGS = {
    "held_in": "--held-in",
    "held_in_catalog": "--held-in-catalog",
    "held_in_size": "--held-in-size",
    "held_in_seed": "--held-in-seed",
}
_PATH_KEYS = {"held_in", "held_in_catalog", "held_out"}


@dataclass
class RunConfig:
    source: Path
    run_dir: Path
    evolve: Path | dict[str, Any]
    gate: Path | dict[str, Any]
    data: dict[str, Any]
    loop: dict[str, Any] = field(default_factory=dict)
    gate_model: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    env_files: list[Path] = field(default_factory=list)

    @property
    def evolve_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def gate_path(self) -> Path:
        return self.run_dir / "gate.yaml"


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_run_config(path: str | Path) -> RunConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected a mapping")
    unknown = set(raw) - {"run_dir", "evolve", "gate", "data", "loop", "gate_model", "env", "env_files"}
    if unknown:
        raise ValueError(f"{source}: unknown keys {sorted(unknown)}")
    base = source.parent
    for key in ("run_dir", "evolve", "gate", "data"):
        if key not in raw:
            raise ValueError(f"{source}: '{key}' is required")
    data = dict(raw["data"] or {})
    if "held_out" not in data:
        raise ValueError(f"{source}: 'data.held_out' is required")
    for key in _PATH_KEYS & set(data):
        data[key] = _resolve(base, data[key])
    data["task_catalogs"] = [_resolve(base, item) for item in data.get("task_catalogs") or []]
    unknown_data = set(data) - set(_DATA_FLAGS) - {"held_out", "task_catalogs"}
    if unknown_data:
        raise ValueError(f"{source}: unknown data keys {sorted(unknown_data)}")
    loop = dict(raw.get("loop") or {})
    if set(loop) - set(_LOOP_FLAGS):
        raise ValueError(f"{source}: unknown loop keys {sorted(set(loop) - set(_LOOP_FLAGS))}")

    def section(value: Any) -> Path | dict[str, Any]:
        return dict(value) if isinstance(value, dict) else _resolve(base, value)

    return RunConfig(
        source=source,
        run_dir=_resolve(base, raw["run_dir"]),
        evolve=section(raw["evolve"]),
        gate=section(raw["gate"]),
        data=data,
        loop=loop,
        gate_model=dict(raw.get("gate_model") or {}),
        env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
        env_files=[_resolve(base, item) for item in raw.get("env_files") or []],
    )


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (optionally prefixed by `export`); quotes are shell-unquoted."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep or not key.strip().isidentifier():
            continue
        parsed = shlex.split(value, comments=True)
        values[key.strip()] = parsed[0] if parsed else ""
    return values


def run_environment(config: RunConfig) -> dict[str, str]:
    """Variables the run exports: env files first, then explicit `env`."""
    env: dict[str, str] = {}
    for path in config.env_files:
        env.update(read_env_file(path))
    env.update(config.env)
    env.setdefault("HARNESSRL_RUN_DIR", str(config.run_dir))
    if env.get("OPEN_SANDBOX_CAPACITY_MODE", "").strip().lower() == "local":
        # Run-scoped admission: the reservation journal belongs to this run,
        # so the project-wide sandbox/CPU check of "strict" mode is skipped.
        env.setdefault("SANDBOX_CAPACITY_ROOT", str(config.run_dir / "capacity"))
        env.setdefault("SANDBOX_CAPACITY_RUN_ID", config.run_dir.name)
    return env


def loop_argv(config: RunConfig, *, resume: bool = False) -> list[str]:
    """Command-line arguments for ``harness_evolution.loop.two_sided``."""
    argv: list[str] = []
    if resume:
        argv.append("--resume")
    argv += ["--config", str(config.evolve_path), "--gate-config", str(config.gate_path)]
    for key, flag in _DATA_FLAGS.items():
        if key in config.data:
            argv += [flag, str(config.data[key])]
    argv += ["--held-out", str(config.data["held_out"])]
    for catalog in config.data["task_catalogs"]:
        argv += ["--task-catalog", str(catalog)]
    names = config.gate_model.get("names")
    if names:
        argv += ["--gate-model", ",".join(names) if isinstance(names, list) else str(names)]
    if config.gate_model.get("gateway"):
        argv += ["--gate-gateway", str(config.gate_model["gateway"])]
    argv += ["--run-dir", str(config.run_dir)]
    for key, flag in _LOOP_FLAGS.items():
        if key in config.loop:
            argv += [flag, str(config.loop[key])]
    return argv


def prepare_run_dir(config: RunConfig, *, resume: bool) -> None:
    """Freeze evolve/gate configs into the run directory, as the launch scripts did.

    A fresh run refuses an existing state.json; a resume requires it and keeps
    the frozen configs already in the run directory.
    """
    state = config.run_dir / "state.json"
    if resume and not state.is_file():
        raise SystemExit(f"resume requested but state is missing: {state}")
    if not resume and state.exists():
        raise SystemExit(f"fresh start refused because state exists: {state} (use --resume)")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for value, target in ((config.evolve, config.evolve_path), (config.gate, config.gate_path)):
        if target.is_file() and target.stat().st_size > 0:
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        if isinstance(value, dict):
            temporary.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            shutil.copyfile(value, temporary)
        temporary.replace(target)


def apply_environment(env: dict[str, str]) -> None:
    os.environ.update(env)
