"""Tests for the RL-side evolution round."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from harness_evolution.core.types import RuntimeConfig
from harness_evolution.rl import evolve as evolve_module
from harness_evolution.rl.evolve import (
    EvolveConfig,
    _candidate_from_metadata,
    _check_memory_is_stable,
    _deep_merge,
    _heldout_sides,
    _prepare_gate_config,
    evolve_once,
    resolve_evolve_config,
)

_GATE_YAML = {
    "paths": {
        "baseline_commit": "0" * 40,
        "baseline_binary": "/stage/binaries/0000000/opencode",
        "baseline_repo": "/stage/opencode",
        "workspace_root": "/stage/workspaces",
        "report_root": "/stage/reports",
        "version_state": "/stage/harness_version_state.json",
    },
    "static_gate": {"enabled": True},
    "build": {},
    "heldout_gate": {"enabled": True, "tasks_dir": "__PLACEHOLDER__"},
}


def _runtime_config(*, memory: dict | None = None, report_root: Path | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        raw={},
        initial_harness_path="/stage/opencode",
        initial_base_commit="0" * 40,
        iterations=1,
        rollouts_per_attempt=1,
        workspace_root=Path("/stage/workspaces"),
        report_root=report_root or Path("/stage/reports"),
        evolver_timeout_sec=60,
        terminal_bench={},
        build={},
        opencode_evolver={},
        task_selection={},
        evidence={},
        candidate_selection={},
        validation={},
        memory=memory or {},
    )


def _evolver_meta(*, commit: str, workspace: str, errored: bool = False) -> dict:
    candidate = {
        "candidate_idx": 0,
        "attempt_idx": 0,
        "rollout_idx": 0,
        "workspace": workspace,
        "branch": "rl-rollout/opencode/test",
        "commit": commit,
        "commit_created": True,
        "changed_files": ["packages/opencode/src/session/prompt.ts"],
        "summary": "tighten the bash tool preamble",
        "evidence": {},
        "skill_copy": {},
        "evolver_metadata": {"error": "boom"} if errored else {},
        "validation": {"enabled": True, "passed": not errored},
        "diff_stats": {"files": 1, "added": 12, "deleted": 3, "score": 15},
    }
    return {
        "accepted_candidates": [] if errored else [{"workspace": workspace, "commit": commit}],
        "candidates": [candidate],
        "fallback_reason": "validation_exhausted" if errored else None,
    }


def _verdict(*, accepted: bool, commit: str = "", workspace: str = "", heldout: bool = True) -> dict:
    tasks = [
        {"task": "t-a", "baseline_passes": 0, "candidate_passes": 2, "n_attempts": 2, "outcome": "win"},
        {"task": "t-b", "baseline_passes": 1, "candidate_passes": 1, "n_attempts": 2, "outcome": "tie"},
    ]
    return {
        "accepted": accepted,
        "num_candidates": 1,
        "num_static_gate_survivors": 1 if accepted else 0,
        "winner_commit": commit,
        "winner_binary_path": f"/stage/binaries/{commit[:7]}/opencode" if commit else "",
        "winner_workspace": workspace,
        "report_path": "/stage/reports/gate_multi.json",
        "candidates": [
            {
                "workspace": workspace,
                "static_gate": {"candidate_commit": commit, "passed": accepted},
                "heldout_gate": {"passed": accepted, "net": 1, "tasks": tasks} if heldout else None,
            }
        ],
    }


class EvolveConfigTest(unittest.TestCase):
    def _dir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_relative_paths_resolve_next_to_the_config(self) -> None:
        root = self._dir()
        (root / "evolve.yaml").write_text("x", encoding="utf-8")
        (root / "gate.yaml").write_text("x", encoding="utf-8")
        config_path = root / "rl.yaml"
        config_path.write_text(
            yaml.safe_dump({"evolve_config": "evolve.yaml", "gate_config": "gate.yaml"}),
            encoding="utf-8",
        )
        config = EvolveConfig.from_yaml(config_path)
        self.assertEqual(config.evolve_config, root / "evolve.yaml")
        self.assertEqual(config.gate_config, root / "gate.yaml")
        self.assertEqual(config.gate_mode, "dynamic")
        self.assertIsNone(config.rollouts_per_attempt)

    def test_absolute_paths_are_kept(self) -> None:
        root = self._dir()
        config_path = root / "rl.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "evolve_config": "/abs/evolve.yaml",
                    "gate_config": "/abs/gate.yaml",
                    "gate_mode": "static_only",
                    "rollouts_per_attempt": 2,
                    "evolver_timeout_sec": 120,
                }
            ),
            encoding="utf-8",
        )
        config = EvolveConfig.from_yaml(config_path)
        self.assertEqual(config.evolve_config, Path("/abs/evolve.yaml"))
        self.assertEqual(config.gate_mode, "static_only")
        self.assertEqual(config.rollouts_per_attempt, 2)
        self.assertEqual(config.evolver_timeout_sec, 120)

    def test_missing_key_and_bad_rollouts_are_rejected(self) -> None:
        root = self._dir()
        missing = root / "missing.yaml"
        missing.write_text(yaml.safe_dump({"gate_config": "/abs/gate.yaml"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            EvolveConfig.from_yaml(missing)
        bad = root / "bad.yaml"
        bad.write_text(
            yaml.safe_dump(
                {
                    "evolve_config": "/a.yaml",
                    "gate_config": "/b.yaml",
                    "rollouts_per_attempt": 0,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            EvolveConfig.from_yaml(bad)


class EvolverPresetTest(unittest.TestCase):
    def _dir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def _config(self, root: Path, raw: dict, *, evolver: str = "") -> EvolveConfig:
        path = root / "evolve.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return EvolveConfig(
            evolve_config=path, gate_config=root / "gate.yaml", evolver=evolver
        )

    def test_deep_merge_recurses_into_mappings_only(self) -> None:
        merged = _deep_merge(
            {"a": {"keep": 1, "replace": 1}, "list": [1], "scalar": 1},
            {"a": {"replace": 2, "add": 3}, "list": [2], "scalar": 2},
        )
        self.assertEqual(
            merged,
            {"a": {"keep": 1, "replace": 2, "add": 3}, "list": [2], "scalar": 2},
        )

    def test_a_config_without_presets_is_run_as_is(self) -> None:
        root = self._dir()
        config = self._config(root, {"evidence": {"method": "agent_analyzed"}})
        self.assertEqual(resolve_evolve_config(config, root / "round"), config.evolve_config)

    def test_the_chosen_preset_is_merged_and_written_to_the_round_dir(self) -> None:
        root = self._dir()
        run_dir = root / "round"
        run_dir.mkdir()
        config = self._config(
            root,
            {
                "evidence": {"options": {"success_threshold": 1.0}},
                "opencode_evolver": {"tool_call": True},
                "evolver_presets": {
                    "sglang": {
                        "evidence": {"method": "agent_analyzed"},
                        "opencode_evolver": {"client_name": "opencode_bun_dev_client"},
                    },
                    "codex": {
                        "evidence": {"method": "per_trace_then_aggregate"},
                        "opencode_evolver": {
                            "client_name": "codex_cli_client",
                            "model": "gpt-5.6-sol",
                        },
                    },
                },
            },
            evolver="codex",
        )
        resolved_path = resolve_evolve_config(config, run_dir)
        self.assertEqual(resolved_path, run_dir / "evolve.resolved.yaml")
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        self.assertEqual(resolved["evidence"]["method"], "per_trace_then_aggregate")
        self.assertEqual(resolved["opencode_evolver"]["client_name"], "codex_cli_client")
        self.assertEqual(resolved["opencode_evolver"]["model"], "gpt-5.6-sol")
        # Shared settings survive the overlay, and the other preset is gone.
        self.assertEqual(resolved["evidence"]["options"]["success_threshold"], 1.0)
        self.assertTrue(resolved["opencode_evolver"]["tool_call"])
        self.assertNotIn("evolver_presets", resolved)

    def test_presets_without_a_choice_and_a_choice_without_presets_both_fail(self) -> None:
        root = self._dir()
        with_presets = self._config(root, {"evolver_presets": {"sglang": {}}})
        with self.assertRaises(ValueError):
            resolve_evolve_config(with_presets, root)
        unknown = self._config(root, {"evolver_presets": {"sglang": {}}}, evolver="nope")
        with self.assertRaises(ValueError):
            resolve_evolve_config(unknown, root)
        without_presets = self._config(root, {"evidence": {}}, evolver="codex")
        with self.assertRaises(ValueError):
            resolve_evolve_config(without_presets, root)


class MemoryConfigTest(unittest.TestCase):
    def test_no_memory_needs_no_storage_path(self) -> None:
        _check_memory_is_stable(_runtime_config(memory={}))
        _check_memory_is_stable(_runtime_config(memory={"method": "none"}))

    def test_a_per_round_memory_path_is_refused(self) -> None:
        # Unset storage_path means report_root/memory/..., and report_root is
        # this round's dir -- memory that silently never remembers.
        with self.assertRaises(ValueError):
            _check_memory_is_stable(_runtime_config(memory={"method": "accepted_lineage"}))
        with self.assertRaises(ValueError):
            _check_memory_is_stable(
                _runtime_config(
                    memory={
                        "method": "accepted_lineage",
                        "options": {"storage_path": "memory/lineage.jsonl"},
                    }
                )
            )

    def test_an_absolute_memory_path_is_accepted(self) -> None:
        _check_memory_is_stable(
            _runtime_config(
                memory={
                    "method": "accepted_lineage",
                    "options": {"storage_path": "/lineage/accepted.jsonl"},
                }
            )
        )


class GateConfigTest(unittest.TestCase):
    def _gate_config_path(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "gate.yaml"
        path.write_text(yaml.safe_dump(_GATE_YAML), encoding="utf-8")
        return path

    def test_version_state_is_dropped_and_the_parent_becomes_the_baseline(self) -> None:
        config = EvolveConfig(
            evolve_config=Path("/a.yaml"), gate_config=self._gate_config_path()
        )
        prepared = _prepare_gate_config(
            config, parent_commit="a" * 40, parent_binary="/stage/binaries/aaaaaaa/opencode"
        )
        # The caller owns the state file; the gate must not promote into it.
        self.assertNotIn("version_state", prepared["paths"])
        self.assertEqual(prepared["paths"]["baseline_commit"], "a" * 40)
        self.assertEqual(
            prepared["paths"]["baseline_binary"], "/stage/binaries/aaaaaaa/opencode"
        )

    def test_without_a_parent_binary_the_config_keeps_its_own(self) -> None:
        config = EvolveConfig(
            evolve_config=Path("/a.yaml"), gate_config=self._gate_config_path()
        )
        prepared = _prepare_gate_config(config, parent_commit="a" * 40, parent_binary="")
        self.assertEqual(
            prepared["paths"]["baseline_binary"], "/stage/binaries/0000000/opencode"
        )


class HeldoutSidesTest(unittest.TestCase):
    def test_totals_come_from_the_per_task_comparison(self) -> None:
        baseline, winner = _heldout_sides(_verdict(accepted=True, commit="b" * 40), "b" * 40)
        self.assertEqual(baseline, {"passed_attempts": 1, "total_attempts": 4, "rate": 0.25, "complete": True})
        self.assertEqual(winner, {"passed_attempts": 3, "total_attempts": 4, "rate": 0.75, "complete": True})

    def test_a_static_only_round_has_no_sides(self) -> None:
        self.assertEqual(
            _heldout_sides(_verdict(accepted=True, commit="b" * 40, heldout=False), "b" * 40),
            (None, None),
        )

    def test_an_unknown_winner_has_no_sides(self) -> None:
        self.assertEqual(
            _heldout_sides(_verdict(accepted=True, commit="b" * 40), "c" * 40), (None, None)
        )


class CandidateRebuildTest(unittest.TestCase):
    def test_rebuilds_what_memory_reads(self) -> None:
        meta = _evolver_meta(commit="b" * 40, workspace="/stage/workspaces/cand")
        candidate = _candidate_from_metadata(meta["candidates"][0])
        self.assertEqual(candidate.commit, "b" * 40)
        self.assertEqual(candidate.workspace, Path("/stage/workspaces/cand"))
        self.assertEqual(candidate.changed_files, ["packages/opencode/src/session/prompt.ts"])
        self.assertEqual(candidate.summary, "tighten the bash tool preamble")
        self.assertEqual(candidate.diff_stats.added, 12)
        self.assertTrue(candidate.validation.passed)

    def test_tolerates_an_unknown_diff_stat_field(self) -> None:
        record = _evolver_meta(commit="b" * 40, workspace="/w")["candidates"][0]
        record["diff_stats"] = {"files": 2, "renamed": 1}
        candidate = _candidate_from_metadata(record)
        self.assertEqual(candidate.diff_stats.files, 2)


class EvolveOnceTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.evidence_dir = self.root / "tb_job"
        self.evidence_dir.mkdir()
        (self.evidence_dir / "conversion_manifest.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "dataset_task_id": "fix-the-thing",
                            "dataset_task_dir": "/tasks/fix-the-thing",
                            "dataset_image_tag": "img:1",
                        },
                        {"dataset_task_id": "fix-the-thing"},
                        {"dataset_task_id": ""},
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.gate_config_path = self.root / "gate.yaml"
        self.gate_config_path.write_text(yaml.safe_dump(_GATE_YAML), encoding="utf-8")
        # No evolver_presets: the round runs this file as-is.
        (self.root / "evolve.yaml").write_text(
            yaml.safe_dump({"initial": {"base_commit": "a" * 40}}), encoding="utf-8"
        )
        self.config = EvolveConfig(
            evolve_config=self.root / "evolve.yaml",
            gate_config=self.gate_config_path,
            require_publish=False,
        )
        self.memory_path = self.root / "lineage" / "accepted.jsonl"

    def _runtime(self, *, memory: bool) -> RuntimeConfig:
        return _runtime_config(
            memory={
                "method": "accepted_lineage",
                "options": {"storage_path": str(self.memory_path)},
            }
            if memory
            else {"method": "none"}
        )

    def _patch(self, *, runtime: RuntimeConfig, evolver_meta: dict, verdict: dict | None):
        patches = [
            mock.patch.object(evolve_module, "load_runtime_config", return_value=runtime),
            mock.patch.object(evolve_module, "_resolve_evidence_task_ids"),
            mock.patch.object(evolve_module, "_run_evolver", return_value=evolver_meta),
        ]
        if verdict is not None:
            patches.append(mock.patch.object(evolve_module, "_full_gate", return_value=verdict))
        for patch in patches:
            self.addCleanup(patch.stop)
        return [patch.start() for patch in patches]

    def test_accept_returns_the_winner_and_records_memory(self) -> None:
        commit = "b" * 40
        workspace = str(self.root / "cand")
        self._patch(
            runtime=self._runtime(memory=True),
            evolver_meta=_evolver_meta(commit=commit, workspace=workspace),
            verdict=_verdict(accepted=True, commit=commit, workspace=workspace),
        )
        result = evolve_once(
            self.evidence_dir,
            "a" * 40,
            config=self.config,
            run_dir=self.root / "round",
            round_index=4,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.commit, commit)
        self.assertEqual(result.parent_commit, "a" * 40)
        self.assertEqual(result.binary_path, f"/stage/binaries/{commit[:7]}/opencode")
        self.assertTrue(result.memory_recorded)
        records = [
            json.loads(line)
            for line in self.memory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["winner_commit"], commit)
        self.assertEqual(records[0]["parent_commit"], "a" * 40)
        self.assertEqual(records[0]["step"], 4)
        self.assertEqual(records[0]["winner_score"]["passed_attempts"], 3)

    def test_memory_stays_off_when_the_config_says_none(self) -> None:
        commit = "b" * 40
        self._patch(
            runtime=self._runtime(memory=False),
            evolver_meta=_evolver_meta(commit=commit, workspace="/w"),
            verdict=_verdict(accepted=True, commit=commit, workspace="/w"),
        )
        result = evolve_once(
            self.evidence_dir,
            "a" * 40,
            config=self.config,
            run_dir=self.root / "round",
            round_index=1,
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.memory_recorded)
        self.assertFalse(self.memory_path.exists())

    def test_reject_returns_the_parent_commit(self) -> None:
        commit = "b" * 40
        self._patch(
            runtime=self._runtime(memory=True),
            evolver_meta=_evolver_meta(commit=commit, workspace="/w"),
            verdict=_verdict(accepted=False, commit=commit, workspace="/w"),
        )
        result = evolve_once(
            self.evidence_dir,
            "a" * 40,
            config=self.config,
            run_dir=self.root / "round",
            round_index=2,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.commit, "a" * 40)
        self.assertEqual(result.binary_path, "")
        self.assertTrue(result.reason)
        # A rejected round must not leave a promotion behind.
        self.assertFalse(self.memory_path.exists())

    def test_no_candidate_to_gate_keeps_the_parent_without_gating(self) -> None:
        meta = _evolver_meta(commit="b" * 40, workspace="/w")
        meta["accepted_candidates"] = []
        meta["fallback_reason"] = "validation_exhausted"
        gate = mock.patch.object(evolve_module, "_full_gate")
        self.addCleanup(gate.stop)
        gate_mock = gate.start()
        self._patch(runtime=self._runtime(memory=True), evolver_meta=meta, verdict=None)
        result = evolve_once(
            self.evidence_dir,
            "a" * 40,
            config=self.config,
            run_dir=self.root / "round",
            round_index=3,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.commit, "a" * 40)
        self.assertEqual(result.reason, "validation_exhausted")
        gate_mock.assert_not_called()

    def test_an_evolver_that_only_errored_is_a_failure(self) -> None:
        # An evolver that could not run is infrastructure, not a verdict: the
        # round fails loudly instead of reporting "nothing worth promoting".
        for meta in (
            _evolver_meta(commit="b" * 40, workspace="/w", errored=True),
            {"accepted_candidates": [], "candidates": [], "fallback_reason": None},
        ):
            with self.subTest(candidates=len(meta["candidates"])):
                run = mock.patch.object(evolve_module, "_run_evolver", return_value=meta)
                load = mock.patch.object(
                    evolve_module, "load_runtime_config", return_value=self._runtime(memory=True)
                )
                resolve = mock.patch.object(evolve_module, "_resolve_evidence_task_ids")
                with run, load, resolve:
                    with self.assertRaises(RuntimeError):
                        evolve_once(
                            self.evidence_dir,
                            "a" * 40,
                            config=self.config,
                            run_dir=self.root / "round",
                            round_index=1,
                        )

    def test_an_accept_without_a_winner_is_a_failure(self) -> None:
        # Promoting this would point training at a harness nobody can fetch.
        verdict = _verdict(accepted=True, commit="b" * 40, workspace="/w")
        verdict["winner_binary_path"] = ""
        self._patch(
            runtime=self._runtime(memory=True),
            evolver_meta=_evolver_meta(commit="b" * 40, workspace="/w"),
            verdict=verdict,
        )
        with self.assertRaises(RuntimeError):
            evolve_once(
                self.evidence_dir,
                "a" * 40,
                config=self.config,
                run_dir=self.root / "round",
                round_index=1,
            )

    def test_the_heldout_gate_gets_this_round_s_tasks(self) -> None:
        commit = "b" * 40
        _runtime_patch, _resolve_patch, _evolver_patch, gate_mock = self._patch(
            runtime=self._runtime(memory=False),
            evolver_meta=_evolver_meta(commit=commit, workspace="/w"),
            verdict=_verdict(accepted=True, commit=commit, workspace="/w"),
        )
        evolve_once(
            self.evidence_dir,
            "a" * 40,
            config=self.config,
            run_dir=self.root / "round",
            round_index=1,
        )
        _config, workspaces = gate_mock.call_args.args
        self.assertEqual(workspaces, [Path("/w")])
        self.assertEqual(gate_mock.call_args.kwargs["task_names"], ["fix-the-thing"])
        self.assertEqual(
            gate_mock.call_args.kwargs["task_catalog"],
            {"fix-the-thing": {"task_dir": "/tasks/fix-the-thing", "image_tag": "img:1"}},
        )

    def test_static_only_mode_skips_the_heldout_gate(self) -> None:
        commit = "b" * 40
        config = EvolveConfig(
            evolve_config=self.root / "evolve.yaml",
            gate_config=self.gate_config_path,
            gate_mode="static_only",
            require_publish=False,
        )
        self._patch(
            runtime=self._runtime(memory=False),
            evolver_meta=_evolver_meta(commit=commit, workspace="/w"),
            verdict=None,
        )
        full = mock.patch.object(evolve_module, "_full_gate")
        static = mock.patch.object(
            evolve_module,
            "_static_gate",
            return_value=_verdict(accepted=True, commit=commit, workspace="/w", heldout=False),
        )
        self.addCleanup(full.stop)
        self.addCleanup(static.stop)
        full_mock = full.start()
        static_mock = static.start()
        result = evolve_once(
            self.evidence_dir,
            "a" * 40,
            config=config,
            run_dir=self.root / "round",
            round_index=1,
        )
        self.assertTrue(result.accepted)
        full_mock.assert_not_called()
        static_mock.assert_called_once()

    def test_rejects_a_bad_gate_mode_and_a_missing_parent(self) -> None:
        with self.assertRaises(ValueError):
            evolve_once(
                self.evidence_dir,
                "",
                config=self.config,
                run_dir=self.root / "round",
                round_index=1,
            )
        bad_mode = EvolveConfig(
            evolve_config=self.root / "evolve.yaml",
            gate_config=self.gate_config_path,
            gate_mode="whatever",
            require_publish=False,
        )
        with self.assertRaises(ValueError):
            evolve_once(
                self.evidence_dir,
                "a" * 40,
                config=bad_mode,
                run_dir=self.root / "round",
                round_index=1,
            )

    def test_refuses_to_start_when_an_accept_could_not_be_published(self) -> None:
        config = EvolveConfig(
            evolve_config=self.root / "evolve.yaml",
            gate_config=self.gate_config_path,
            require_publish=True,
        )
        with mock.patch.object(
            evolve_module.harness_afs, "should_publish", return_value=False
        ):
            with self.assertRaises(RuntimeError):
                evolve_once(
                    self.evidence_dir,
                    "a" * 40,
                    config=config,
                    run_dir=self.root / "round",
                    round_index=1,
                )


if __name__ == "__main__":
    unittest.main()
