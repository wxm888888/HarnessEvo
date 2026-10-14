from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from harness_evolution import cli
from harness_evolution.core.run_config import (
    load_run_config,
    loop_argv,
    prepare_run_dir,
    read_env_file,
    run_environment,
)
from harness_evolution.loop import two_sided


def _write_run_yaml(tmp_path: Path, **overrides: object) -> Path:
    (tmp_path / "evolve.yaml").write_text("initial: {}\n", encoding="utf-8")
    raw = {
        "run_dir": "runs/demo",
        "evolve": "evolve.yaml",
        "gate": {"heldout_gate": {"backend": "opensandbox"}},
        "data": {
            "held_in_catalog": "data/held_in.jsonl",
            "held_in_size": 8,
            "held_out": "data/held_out.json",
            "task_catalogs": ["data/held_in.jsonl", "/abs/tb.jsonl"],
        },
        "loop": {"steps": 3, "candidates": 2, "gate_repeats": 2},
        "gate_model": {"names": ["a", "b"], "gateway": "http://gw"},
        "env": {"SANDBOX_TRIAL_RETRIES": 2},
    }
    raw.update(overrides)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _parse_loop(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(sys, "argv", ["two_sided", *argv])
    return vars(two_sided._parse_args())


def test_paths_resolve_relative_to_run_yaml(tmp_path: Path) -> None:
    config = load_run_config(_write_run_yaml(tmp_path))
    assert config.run_dir == (tmp_path / "runs/demo").resolve()
    assert config.data["held_out"] == (tmp_path / "data/held_out.json").resolve()
    assert config.data["task_catalogs"] == [(tmp_path / "data/held_in.jsonl").resolve(), Path("/abs/tb.jsonl")]


def test_loop_argv_parses_into_the_loop_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(_write_run_yaml(tmp_path))
    args = _parse_loop(loop_argv(config, resume=True), monkeypatch)
    assert args["resume"] is True
    assert args["config"] == config.run_dir / "config.yaml"
    assert args["gate_config"] == config.run_dir / "gate.yaml"
    assert args["held_in_size"] == 8 and args["steps"] == 3 and args["candidates"] == 2
    assert args["gate_repeats"] == 2 and args["sampling_repeats"] is None
    assert args["gate_model"] == "a,b" and args["gate_gateway"] == "http://gw"
    assert len(args["task_catalog"]) == 2


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown loop keys"):
        load_run_config(_write_run_yaml(tmp_path, loop={"stepz": 1}))
    with pytest.raises(ValueError, match="data.held_out"):
        load_run_config(_write_run_yaml(tmp_path, data={"held_in_size": 1}))


def test_env_files_then_env_override(tmp_path: Path) -> None:
    env_file = tmp_path / "secrets.env"
    env_file.write_text("# comment\nexport TOKEN='abc def'\nSANDBOX_TRIAL_RETRIES=9\nnot a line\n", encoding="utf-8")
    assert read_env_file(env_file) == {"TOKEN": "abc def", "SANDBOX_TRIAL_RETRIES": "9"}
    config = load_run_config(_write_run_yaml(tmp_path, env_files=["secrets.env"]))
    env = run_environment(config)
    assert env["TOKEN"] == "abc def"
    assert env["SANDBOX_TRIAL_RETRIES"] == "2"
    assert env["HARNESSRL_RUN_DIR"] == str(config.run_dir)
    assert "SANDBOX_CAPACITY_ROOT" not in env


def test_local_capacity_mode_gets_a_run_scoped_journal(tmp_path: Path) -> None:
    config = load_run_config(_write_run_yaml(tmp_path, env={"OPEN_SANDBOX_CAPACITY_MODE": "local"}))
    env = run_environment(config)
    assert env["SANDBOX_CAPACITY_ROOT"] == str(config.run_dir / "capacity")
    assert env["SANDBOX_CAPACITY_RUN_ID"] == "demo"
    explicit = {"OPEN_SANDBOX_CAPACITY_MODE": "local", "SANDBOX_CAPACITY_ROOT": "/shared/cap"}
    config = load_run_config(_write_run_yaml(tmp_path, env=explicit))
    assert run_environment(config)["SANDBOX_CAPACITY_ROOT"] == "/shared/cap"


def test_prepare_run_dir_freezes_configs_and_guards_state(tmp_path: Path) -> None:
    config = load_run_config(_write_run_yaml(tmp_path))
    with pytest.raises(SystemExit, match="state is missing"):
        prepare_run_dir(config, resume=True)
    prepare_run_dir(config, resume=False)
    assert config.evolve_path.read_text(encoding="utf-8") == "initial: {}\n"
    assert yaml.safe_load(config.gate_path.read_text(encoding="utf-8")) == {"heldout_gate": {"backend": "opensandbox"}}
    # Frozen configs are kept on resume even if the sources change.
    (tmp_path / "evolve.yaml").write_text("changed: true\n", encoding="utf-8")
    (config.run_dir / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="fresh start refused"):
        prepare_run_dir(config, resume=False)
    prepare_run_dir(config, resume=True)
    assert config.evolve_path.read_text(encoding="utf-8") == "initial: {}\n"


def test_dry_run_masks_env_file_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "secrets.env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    path = _write_run_yaml(tmp_path, env_files=["secrets.env"])
    assert cli.main(["run", str(path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "hunter2" not in output
    assert json.loads(output)["env"]["TOKEN"] == "<from env file>"
    assert not (tmp_path / "runs").exists()


def test_status_summarizes_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state = {
        "status": "complete",
        "next_step": 3,
        "current_commit": "b" * 40,
        "history": [
            {"step": 1, "accepted": True, "parent_commit": "a" * 40, "winner_commit": "b" * 40,
             "winner_score": {"held_out": {"passed_attempts": 30, "total_attempts": 89}}},
            {"step": 2, "accepted": False, "parent_commit": "b" * 40},
        ],
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert cli.main(["status", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "status: complete  next_step: 3"
    assert lines[2] == "  step 1: accepted  aaaaaaa -> bbbbbbb  held-out 30/89"
    assert lines[3].startswith("  step 2: kept parent")


def test_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err
