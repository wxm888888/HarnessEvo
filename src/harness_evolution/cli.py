"""`hevo`: the command-line entry point.

  hevo run RUN_YAML [--resume] [--dry-run]   two-sided loop from one run.yaml
  hevo evolve ...        diagnose + propose from an existing trajectory job (no sampling, no gate)
  hevo loop ...          two-sided loop with its raw flags (what `run` builds from run.yaml)
  hevo gate ...          static and held-out gate for one candidate workspace
  hevo slime-pipeline ...   one co-evolution step called from slime training
  hevo status RUN_DIR    summarize a run's state.json

Subcommands other than `run` and `status` pass their arguments unchanged to
the module they wrap; `hevo <command> --help` shows that module's flags.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable

_PASSTHROUGH: dict[str, tuple[str, str]] = {
    "evolve": ("harness_evolution.loop.from_existing_job", "diagnose and propose from an existing trajectory job"),
    "loop": ("harness_evolution.loop.two_sided", "two-sided loop with raw flags"),
    "gate": ("harness_evolution.evaluation.gate", "gate one candidate workspace"),
    "slime-pipeline": ("harness_evolution.integrations.slime.pipeline", "co-evolution step called from slime"),
}


def _call_main(module_name: str, prog: str, argv: list[str]) -> int:
    main: Callable[[], object] = importlib.import_module(module_name).main
    saved = sys.argv
    sys.argv = [prog, *argv]
    try:
        result = main()
    finally:
        sys.argv = saved
    return int(result or 0)


def _run(argv: list[str]) -> int:
    from harness_evolution.core.run_config import (
        apply_environment,
        load_run_config,
        loop_argv,
        prepare_run_dir,
        read_env_file,
        run_environment,
    )

    parser = argparse.ArgumentParser(prog="hevo run", description="Run the two-sided evolution loop from run.yaml.")
    parser.add_argument("run_yaml", type=Path)
    parser.add_argument("--resume", action="store_true", help="continue the run recorded in run_dir/state.json")
    parser.add_argument("--dry-run", action="store_true", help="print what would run and change nothing")
    args = parser.parse_args(argv)

    config = load_run_config(args.run_yaml)
    env = run_environment(config)
    loop_args = loop_argv(config, resume=args.resume)
    if args.dry_run:
        # Values from env files are usually secrets; show only their names.
        file_keys = {key for path in config.env_files for key in read_env_file(path)} - set(config.env)
        shown = {key: ("<from env file>" if key in file_keys else value) for key, value in env.items()}
        print(json.dumps({"run_dir": str(config.run_dir), "env": shown, "loop_argv": loop_args}, indent=2))
        return 0
    prepare_run_dir(config, resume=args.resume)
    apply_environment(env)
    return _call_main("harness_evolution.loop.two_sided", "hevo run", loop_args)


def _status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="hevo status", description="Summarize a run's state.json.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    state_path = args.run_dir / "state.json"
    if not state_path.is_file():
        print(f"no state.json in {args.run_dir}", file=sys.stderr)
        return 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    print(f"status: {state.get('status')}  next_step: {state.get('next_step')}")
    print(f"current commit: {state.get('current_commit')}")
    for entry in state.get("history") or []:
        score = ((entry.get("winner_score") or {}).get("held_out") or {})
        result = f"{score.get('passed_attempts')}/{score.get('total_attempts')}" if score else "-"
        verdict = "accepted" if entry.get("accepted") else "kept parent"
        parent = str(entry.get("parent_commit") or "")[:7]
        winner = str(entry.get("winner_commit") or "")[:7]
        print(f"  step {entry.get('step')}: {verdict}  {parent} -> {winner}  held-out {result}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "run":
        return _run(rest)
    if command == "status":
        return _status(rest)
    if command in _PASSTHROUGH:
        return _call_main(_PASSTHROUGH[command][0], f"hevo {command}", rest)
    print(f"unknown command {command!r}\n\n{__doc__.strip()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
