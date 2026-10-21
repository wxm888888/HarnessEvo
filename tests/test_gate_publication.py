"""The gate must make an accepted winner reachable, with or without a version state.

Publishing the binary to AFS and fetching the commit back into the baseline
repo used to sit inside the ``if version_state:`` branch, so a caller that
records promotions itself -- the RL loop, which takes the commit back from
:func:`harness_evolution.rl.evolve_once` -- would have been handed a commit no
sandbox could install.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import yaml

if "openai" not in sys.modules:
    _openai = types.ModuleType("openai")

    class OpenAI:  # noqa: D401 - stub for an unused optional dependency
        def __init__(self, *args, **kwargs) -> None:
            pass

    _openai.OpenAI = OpenAI
    sys.modules["openai"] = _openai
if "tomllib" not in sys.modules:
    try:
        import tomllib as _tomllib  # noqa: F401
    except ImportError:
        sys.modules["tomllib"] = types.ModuleType("tomllib")

import pytest

from harness_evolution.evaluation import gate
from harness_evolution.evaluation.gate import HeldoutGateResult
from harness_evolution.evaluation.static_gate import StaticGateResult

_WINNER = "b" * 40
_BINARY = "/stage/binaries/bbbbbbb/opencode"


@pytest.fixture
def gate_config(tmp_path: Path):
    def _build(*, version_state: str = "") -> dict:
        paths = {
            "baseline_commit": "a" * 40,
            "baseline_binary": "/stage/binaries/aaaaaaa/opencode",
            "baseline_repo": str(tmp_path / "opencode"),
            "workspace_root": str(tmp_path / "workspaces"),
            "report_root": str(tmp_path / "reports"),
        }
        if version_state:
            paths["version_state"] = version_state
        config = {
            "paths": paths,
            "static_gate": {"enabled": True},
            "build": {},
            "heldout_gate": {"enabled": True},
        }
        path = tmp_path / "gate.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config

    return _build


@pytest.fixture
def passing_static_gate():
    result = StaticGateResult(
        passed=True,
        checks=[],
        candidate_commit=_WINNER,
        changed_files=["packages/opencode/src/session/prompt.ts"],
        binary_path=_BINARY,
    )
    with mock.patch.object(gate, "run_static_gate_batch", return_value=[result]):
        yield result


@pytest.fixture
def recorded():
    """Capture reachability and promotion separately."""
    with mock.patch.object(gate, "_publish_for_promotion") as publish, mock.patch.object(
        gate, "_persist_candidate_commit"
    ) as persist, mock.patch.object(gate, "version_state") as state:
        yield types.SimpleNamespace(publish=publish, persist=persist, state=state)


class TestStaticOnlyGate:
    def test_publishes_the_winner_without_a_version_state(
        self, gate_config, passing_static_gate, recorded, tmp_path
    ) -> None:
        verdict = gate.run_static_gate_multi(gate_config(), [tmp_path / "cand"])

        assert verdict["accepted"] is True
        recorded.publish.assert_called_once()
        assert recorded.publish.call_args.args[:2] == (_WINNER, _BINARY)
        recorded.persist.assert_called_once()
        # Nothing to promote into: that is the caller's file.
        recorded.state.promote.assert_not_called()
        recorded.state.record_rejection.assert_not_called()

    def test_still_promotes_when_a_version_state_is_configured(
        self, gate_config, passing_static_gate, recorded, tmp_path
    ) -> None:
        config = gate_config(version_state=str(tmp_path / "state.json"))
        gate.run_static_gate_multi(config, [tmp_path / "cand"])

        recorded.publish.assert_called_once()
        recorded.state.promote.assert_called_once()

    def test_the_verdict_names_the_winner_s_binary_and_workspace(
        self, gate_config, passing_static_gate, recorded, tmp_path
    ) -> None:
        # Without these a caller holding only the verdict cannot promote.
        verdict = gate.run_static_gate_multi(gate_config(), [tmp_path / "cand"])

        assert verdict["winner_commit"] == _WINNER
        assert verdict["winner_binary_path"] == _BINARY
        assert verdict["winner_workspace"] == str(tmp_path / "cand")

    def test_a_rejected_round_publishes_nothing(
        self, gate_config, recorded, tmp_path
    ) -> None:
        failed = StaticGateResult(passed=False, candidate_commit=_WINNER, binary_path=_BINARY)
        with mock.patch.object(gate, "run_static_gate_batch", return_value=[failed]):
            verdict = gate.run_static_gate_multi(gate_config(), [tmp_path / "cand"])

        assert verdict["accepted"] is False
        assert verdict["winner_binary_path"] == ""
        recorded.publish.assert_not_called()
        recorded.persist.assert_not_called()


class TestFullGate:
    def test_publishes_the_winner_without_a_version_state(
        self, gate_config, passing_static_gate, recorded, tmp_path
    ) -> None:
        heldout = {_WINNER: HeldoutGateResult(passed=True, net=1, candidate_commit=_WINNER)}
        with mock.patch.object(gate, "run_heldout_gate_multi", return_value=heldout):
            verdict = gate.run_full_gate_multi(gate_config(), [tmp_path / "cand"])

        assert verdict["accepted"] is True
        assert verdict["winner_binary_path"] == _BINARY
        assert verdict["winner_workspace"] == str(tmp_path / "cand")
        recorded.publish.assert_called_once()
        assert recorded.publish.call_args.args[:2] == (_WINNER, _BINARY)
        recorded.persist.assert_called_once()
        recorded.state.promote.assert_not_called()

    def test_a_candidate_that_loses_the_ab_publishes_nothing(
        self, gate_config, passing_static_gate, recorded, tmp_path
    ) -> None:
        heldout = {
            _WINNER: HeldoutGateResult(
                passed=False, reason="net -1", net=-1, candidate_commit=_WINNER
            )
        }
        with mock.patch.object(gate, "run_heldout_gate_multi", return_value=heldout):
            verdict = gate.run_full_gate_multi(gate_config(), [tmp_path / "cand"])

        assert verdict["accepted"] is False
        recorded.publish.assert_not_called()
        recorded.state.promote.assert_not_called()


@pytest.mark.parametrize("mode", ["static", "multi", "single"])
@pytest.mark.parametrize("operation", ["fetch", "tag"])
def test_persistence_failure_blocks_publication_and_promotion(
    gate_config, passing_static_gate, tmp_path, mode, operation,
):
    config = gate_config(version_state=str(tmp_path / "state.json"))
    heldout = HeldoutGateResult(passed=True, net=1, candidate_commit=_WINNER)
    def git_result(_repo, *args):
        return types.SimpleNamespace(
            returncode=1 if args[0] == operation else 0,
            stdout="simulated storage error",
        )
    with mock.patch.object(gate, "_git", side_effect=git_result), \
         mock.patch.object(gate, "_publish_for_promotion") as publish, \
         mock.patch.object(gate, "version_state") as state, \
         mock.patch.object(gate, "run_static_gate", return_value=passing_static_gate), \
         mock.patch.object(gate, "run_heldout_gate", return_value=heldout), \
         mock.patch.object(gate, "run_heldout_gate_multi", return_value={_WINNER: heldout}):
        with pytest.raises(RuntimeError, match=operation + " failed"):
            if mode == "static":
                gate.run_static_gate_multi(config, [tmp_path / "cand"])
            elif mode == "multi":
                gate.run_full_gate_multi(config, [tmp_path / "cand"])
            else:
                gate.run_full_gate(config, tmp_path / "cand")
        publish.assert_not_called()
        state.promote.assert_not_called()
    import json
    reports = list((tmp_path / "reports").glob("gate*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["accepted"] is False
    assert report["gate_passed"] is True
    assert operation + " failed" in report["persistence_error"]


def test_persisted_candidate_survives_a_fresh_clone(tmp_path):
    import subprocess
    def git(repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    git(baseline, "init", "-q")
    git(baseline, "config", "user.email", "test@example.com")
    git(baseline, "config", "user.name", "Test")
    (baseline / "harness.txt").write_text("baseline")
    git(baseline, "add", ".")
    git(baseline, "commit", "-qm", "baseline")
    workspace = tmp_path / "candidate"
    subprocess.run(["git", "clone", "-q", str(baseline), str(workspace)], check=True)
    git(workspace, "config", "user.email", "test@example.com")
    git(workspace, "config", "user.name", "Test")
    (workspace / "harness.txt").write_text("candidate")
    git(workspace, "commit", "-qam", "candidate")
    commit = git(workspace, "rev-parse", "HEAD")
    gate._persist_candidate_commit({"paths": {"baseline_repo": str(baseline)}}, workspace, commit)
    future = tmp_path / "future"
    subprocess.run(["git", "clone", "-q", str(baseline), str(future)], check=True)
    assert git(future, "rev-parse", f"harness-evolved-{commit[:12]}^{{commit}}") == commit
