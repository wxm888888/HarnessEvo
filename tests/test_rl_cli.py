"""Tests for the one-round CLI the training loop calls."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from harness_dist import state as version_state
from harness_evolution.rl import __main__ as cli
from harness_evolution.rl.evolve import EvolveResult

_PARENT = "a" * 40
_WINNER = "b" * 40


def _result(*, accepted: bool, candidate: str = _WINNER) -> EvolveResult:
    return EvolveResult(
        accepted=accepted,
        commit=_WINNER if accepted else _PARENT,
        parent_commit=_PARENT,
        binary_path="/stage/binaries/bbbbbbb/opencode" if accepted else "",
        workspace="/stage/workspaces/cand" if accepted else "",
        report_path="/stage/reports/gate_multi.json",
        reason="" if accepted else "none cleared the gate",
        candidate_commit=candidate,
        round_index=8,
        memory_recorded=accepted,
    )


class EnvOverrideTest(unittest.TestCase):
    def test_env_wins_over_the_shared_yaml(self) -> None:
        raw = {"n_fail": 12, "work_root": "/from/yaml"}
        with mock.patch.dict(
            "os.environ",
            {"HARNESS_EVOLVE_N_FAIL": "3", "HARNESS_EVOLVE_WORK_ROOT": "/from/env"},
            clear=False,
        ):
            merged = cli._apply_env_overrides(raw)
        self.assertEqual(merged["n_fail"], 3)
        self.assertEqual(merged["work_root"], "/from/env")

    def test_an_empty_env_value_is_not_an_override(self) -> None:
        # The launcher exports these unconditionally; an unset one must not
        # blank out the config.
        with mock.patch.dict("os.environ", {"HARNESS_EVOLVE_WORK_ROOT": ""}, clear=False):
            merged = cli._apply_env_overrides({"work_root": "/from/yaml"})
        self.assertEqual(merged["work_root"], "/from/yaml")


class RoundSettingsTest(unittest.TestCase):
    def test_relative_dirs_resolve_next_to_the_config(self) -> None:
        settings = cli._round_settings(
            {"n_fail": 2, "n_success": 1, "work_root": "runs", "stage_dir": "stage"},
            base=Path("/base"),
        )
        self.assertEqual(settings.work_root, Path("/base/runs"))
        self.assertEqual(settings.stage_dir, Path("/base/stage"))
        self.assertEqual(settings.window, cli.DEFAULT_WINDOW)

    def test_a_round_needs_somewhere_to_work_and_something_to_read(self) -> None:
        with self.assertRaises(ValueError):
            cli._round_settings({"n_fail": 2}, base=Path("/base"))
        with self.assertRaises(ValueError):
            cli._round_settings({"work_root": "/runs"}, base=Path("/base"))


class ResolveParentTest(unittest.TestCase):
    def _dir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def _settings(self, root: Path) -> cli.RoundSettings:
        return cli.RoundSettings(
            n_fail=1, n_success=0, window=4, work_root=root, stage_dir=root / "stage"
        )

    def test_initialises_a_fresh_lineage_from_the_configured_baseline(self) -> None:
        root = self._dir()
        state_path = root / "state.json"
        with mock.patch.dict(
            "os.environ", {"HARNESS_BASELINE_COMMIT": _PARENT, "HARNESS_BASELINE_BINARY": ""}
        ):
            active = cli._resolve_parent(state_path, self._settings(root))
        self.assertEqual(active["commit"], _PARENT)
        # With no explicit binary, it is derived from stage_dir the same way
        # the training-side bootstrap derives it.
        self.assertEqual(active["binary_path"], str(root / "stage" / "binaries" / "aaaaaaa" / "opencode"))

    def test_an_existing_state_file_is_the_parent(self) -> None:
        root = self._dir()
        state_path = root / "state.json"
        version_state.init_state(state_path, _WINNER, "/stage/bin/opencode")
        with mock.patch.dict("os.environ", {"HARNESS_BASELINE_COMMIT": _PARENT}):
            active = cli._resolve_parent(state_path, self._settings(root))
        self.assertEqual(active["commit"], _WINNER)

    def test_without_a_baseline_there_is_no_parent_to_evolve_from(self) -> None:
        root = self._dir()
        with mock.patch.dict("os.environ", {"HARNESS_BASELINE_COMMIT": ""}):
            with self.assertRaises(RuntimeError):
                cli._resolve_parent(root / "state.json", self._settings(root))


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.state_path = self.root / "state.json"
        version_state.init_state(self.state_path, _PARENT, "/stage/binaries/aaaaaaa/opencode")
        self.config_path = self.root / "rl_evolve.yaml"
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "n_fail": 2,
                    "n_success": 1,
                    "window": 4,
                    "evolve_config": "/abs/evolve.yaml",
                    "gate_config": "/abs/gate.yaml",
                    "work_root": str(self.root / "evolution-runs"),
                    "stage_dir": str(self.root / "stage"),
                }
            ),
            encoding="utf-8",
        )
        self.manifest = {
            "converted_count": 3,
            "selection": {
                "picked": {"n_fail": 2, "n_success": 1},
                "scanned_rollout_dirs": ["rollout_000007"],
                "short": False,
            },
        }

    def _argv(self) -> list[str]:
        return [
            "--config",
            str(self.config_path),
            "--rollout-id",
            "7",
            "--rollout-root",
            str(self.root / "train_trajectories"),
            "--version-state",
            str(self.state_path),
        ]

    def _patch(self, *, result: EvolveResult | Exception):
        select = mock.patch.object(cli, "select_evidence", return_value=self.manifest)
        if isinstance(result, Exception):
            evolve = mock.patch.object(cli, "evolve_once", side_effect=result)
        else:
            evolve = mock.patch.object(cli, "evolve_once", return_value=result)
        self.addCleanup(select.stop)
        self.addCleanup(evolve.stop)
        return select.start(), evolve.start()

    def test_accept_promotes_the_returned_commit(self) -> None:
        _select, _evolve = self._patch(result=_result(accepted=True))
        self.assertEqual(cli.main(self._argv()), 0)
        active = version_state.read_active(self.state_path)
        self.assertEqual(active["commit"], _WINNER)
        self.assertEqual(active["binary_path"], "/stage/binaries/bbbbbbb/opencode")
        history = version_state.read_state(self.state_path)["history"]
        self.assertEqual(history[-1]["action"], "promoted")
        self.assertEqual(history[-1]["previous_commit"], _PARENT)

    def test_reject_keeps_the_parent_active_and_logs_the_candidate(self) -> None:
        self._patch(result=_result(accepted=False))
        self.assertEqual(cli.main(self._argv()), 0)
        self.assertEqual(version_state.read_active(self.state_path)["commit"], _PARENT)
        history = version_state.read_state(self.state_path)["history"]
        self.assertEqual(history[-1]["action"], "rejected")
        # Naming the parent here would read as "the harness we run was
        # rejected"; the candidate is what the gate turned down.
        self.assertEqual(history[-1]["commit"], _WINNER)
        self.assertEqual(version_state.consecutive_rejections(self.state_path), 1)

    def test_the_round_is_written_to_the_run_dir_and_to_stdout(self) -> None:
        self._patch(result=_result(accepted=True))
        run_dir = self.root / "round"
        with mock.patch("sys.stdout") as stdout:
            self.assertEqual(cli.main([*self._argv(), "--run-dir", str(run_dir)]), 0)
        payload = json.loads((run_dir / "round.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["commit"], _WINNER)
        self.assertEqual(payload["selection"]["picked"], {"n_fail": 2, "n_success": 1})
        self.assertEqual(payload["run_dir"], str(run_dir))
        printed = "".join(
            call.args[0] for call in stdout.write.call_args_list if call.args
        )
        self.assertIn(_WINNER, printed)

    def test_the_round_passes_the_configured_quotas_to_selection(self) -> None:
        select, evolve = self._patch(result=_result(accepted=True))
        cli.main(self._argv())
        self.assertEqual(select.call_args.kwargs["n_fail"], 2)
        self.assertEqual(select.call_args.kwargs["n_success"], 1)
        self.assertEqual(select.call_args.kwargs["window"], 4)
        self.assertEqual(select.call_args.kwargs["end_rollout_id"], 7)
        # The parent comes from the state file, not from the config.
        self.assertEqual(evolve.call_args.args[1], _PARENT)
        self.assertEqual(evolve.call_args.kwargs["parent_binary"], "/stage/binaries/aaaaaaa/opencode")
        self.assertEqual(evolve.call_args.kwargs["round_index"], 8)

    def test_a_flag_overrides_the_config(self) -> None:
        select, _evolve = self._patch(result=_result(accepted=True))
        cli.main([*self._argv(), "--n-fail", "5", "--window", "2"])
        self.assertEqual(select.call_args.kwargs["n_fail"], 5)
        self.assertEqual(select.call_args.kwargs["window"], 2)

    def test_a_broken_round_leaves_the_active_harness_alone(self) -> None:
        self._patch(result=RuntimeError("evolver crashed"))
        self.assertEqual(cli.main(self._argv()), 1)
        state = version_state.read_state(self.state_path)
        self.assertEqual(state["active_commit"], _PARENT)
        # Not even a rejection: nothing was judged.
        self.assertEqual([entry["action"] for entry in state["history"]], ["init"])

    def test_a_missing_rollout_id_is_refused(self) -> None:
        self._patch(result=_result(accepted=True))
        with mock.patch.dict("os.environ", {"HARNESS_EVOLVE_ROLLOUT_ID": ""}):
            argv = [
                "--config",
                str(self.config_path),
                "--rollout-root",
                str(self.root),
                "--version-state",
                str(self.state_path),
            ]
            self.assertEqual(cli.main(argv), 1)


if __name__ == "__main__":
    unittest.main()
