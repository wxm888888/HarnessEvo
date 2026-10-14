from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

if "openai" not in sys.modules:
    _openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs) -> None:
            pass

    _openai.OpenAI = OpenAI
    sys.modules["openai"] = _openai
if "tomllib" not in sys.modules:
    try:
        import tomllib as _tomllib  # noqa: F401
    except ImportError:
        sys.modules["tomllib"] = types.ModuleType("tomllib")

from polar.runtime.models import ExecResult, RuntimeSpec

from harness_evolution.evaluation.backends import opensandbox as opensandbox_gate
from harness_evolution.evaluation.backends.opensandbox import (
    _keep_raw_trace_wire,
    _promote_final_trial_artifacts,
    _run_trials,
    custom_afs_binary_path,
    expand_trials,
    heldout_backend,
    load_jsonl_catalog,
    opensandbox_settings,
    publish_side_binaries,
    resolve_task_spec,
    resolve_task_specs,
    run_trial,
    run_heldout_opensandbox,
    TaskSpec,
    Trial,
    TrialOutcome,
)


def test_raw_trace_wire_capture_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESSRL_KEEP_RAW_TRACE_WIRE", raising=False)
    assert _keep_raw_trace_wire() is False

    monkeypatch.setenv("HARNESSRL_KEEP_RAW_TRACE_WIRE", "1")
    assert _keep_raw_trace_wire() is True

    monkeypatch.setenv("HARNESSRL_KEEP_RAW_TRACE_WIRE", "off")
    assert _keep_raw_trace_wire() is False


def test_promote_final_retry_artifacts(tmp_path: Path) -> None:
    session = tmp_path / "sample-task-a0"
    retry_root = tmp_path / "sample-task-a0.retry-1"
    retry = retry_root / "agent"
    retry.mkdir(parents=True)
    (retry_root / "agent.remote.jsonl").write_text('{"type":"text"}\n', encoding="utf-8")
    (retry / "llm-context.jsonl").write_text('{"n":1}\n', encoding="utf-8")
    (session / "agent").mkdir(parents=True)
    (session / "agent" / "trajectory.json").write_text("[]\n", encoding="utf-8")
    (session / "retry_history.json").write_text(
        json.dumps([{"attempt": 1, "error": True}, {"attempt": 2, "error": False}]),
        encoding="utf-8",
    )

    _promote_final_trial_artifacts(session)

    assert (session / "agent.remote.jsonl").read_text(encoding="utf-8") == '{"type":"text"}\n'
    assert (session / "agent" / "llm-context.jsonl").read_text(encoding="utf-8") == '{"n":1}\n'


def _task_dir(root: Path, name: str, image: str = "t-data-processing-v1") -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text(f"solve {name}\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (task / "image_tag").write_text(image, encoding="utf-8")
    return task


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        self.spec = spec
        self.session_id = session_id
        self.session_dir = session_dir
        self.commit = str(spec.kwargs.get("opencode_commit") or "")
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[str] = []
        self.commands: list[str] = []
        type(self).instances.append(self)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        self.uploads.append((local_path, remote_path))

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        self.uploads.append((local_path, remote_path))

    async def download_file(self, remote_path: str, local_path: str) -> None:
        self.downloads.append(remote_path)
        Path(local_path).write_text(f"saved {remote_path}\n", encoding="utf-8")

    async def exec(self, command: str, **_: object) -> ExecResult:
        self.commands.append(command)
        if "__POLAR_VERIFIER_REWARD__" in command:
            passed = self.commit.startswith("cafe")
            return ExecResult(
                return_code=0,
                stdout=f"__POLAR_VERIFIER_REWARD__={'1' if passed else '0'}\n",
                stderr="",
            )
        if "test.sh" in command:
            # Candidate cafe* passes; baseline fails.
            passed = self.commit.startswith("cafe")
            return ExecResult(
                return_code=0 if passed else 1,
                stdout="ok" if passed else "FAILED 1",
                stderr="",
            )
        return ExecResult(return_code=0, stdout="", stderr="")


def test_heldout_backend_defaults_to_opensandbox() -> None:
    assert heldout_backend({}) == "opensandbox"
    assert heldout_backend({"backend": "opensandbox", "harbor_template": "/x"}) == "opensandbox"
    assert heldout_backend({"harbor_template": "/x"}) == "harbor"
    assert heldout_backend({"opensandbox": {}}) == "opensandbox"


def test_gate_wide_timeout_records_unfinished_trials(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, "task_timeout")
    spec = TaskSpec(
        name="task_timeout",
        task_dir=task_dir,
        image="t-data-processing-v1",
        instruction="solve",
        tests_dir=task_dir / "tests",
    )
    trials = [
        Trial("candidate-01-fast-a0", "candidate-01", "a" * 40, "/bin/a", spec, 0),
        Trial("candidate-02-slow-a0", "candidate-02", "b" * 40, "/bin/b", spec, 0),
    ]

    async def fake_runner(trial: Trial, **_: object) -> TrialOutcome:
        if trial.side == "candidate-02":
            await asyncio.sleep(1)
        return TrialOutcome(
            trial_id=trial.trial_id,
            side=trial.side,
            commit=trial.commit,
            task=trial.task.name,
            passed=True,
            error=False,
            detail="pass",
        )

    outcomes = asyncio.run(
        _run_trials(
            trials,
            settings={},
            jobs_dir=tmp_path / "jobs",
            n_concurrent=2,
            trial_runner=fake_runner,
            overall_timeout_sec=0.05,
        )
    )
    assert outcomes[0].passed is True
    assert outcomes[1].error_type == "gate_timeout"
    assert outcomes[1].imputed_zero is True
    assert (
        tmp_path / "jobs" / "candidate-02" / "candidate-02-slow-a0" / "trial.json"
    ).is_file()


def test_gate_timeout_does_not_retry_transport_error_from_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_dir = _task_dir(tmp_path, "task_cancel_retry")
    spec = TaskSpec(
        name="task_cancel_retry",
        task_dir=task_dir,
        image="t-data-processing-v1",
        instruction="solve",
        tests_dir=task_dir / "tests",
    )
    trial = Trial("candidate-01-slow-a0", "candidate-01", "a" * 40, "/bin/a", spec, 0)
    attempts = 0

    async def fake_run_once(*args, **kwargs) -> TrialOutcome:
        nonlocal attempts
        attempts += 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return TrialOutcome(
                trial_id=trial.trial_id,
                side=trial.side,
                commit=trial.commit,
                task=trial.task.name,
                passed=False,
                error=True,
                detail="_ControlPlaneError: Command termination unconfirmed; verifier must not run",
            )
        raise AssertionError("trial should have been cancelled")

    monkeypatch.setattr(opensandbox_gate, "_run_trial_once", fake_run_once)
    started = time.monotonic()
    outcomes = asyncio.run(
        _run_trials(
            [trial],
            settings={"trial_retries": 2, "gate_cleanup_grace_sec": 0.1},
            jobs_dir=tmp_path / "jobs",
            n_concurrent=1,
            trial_runner=run_trial,
            overall_timeout_sec=0.05,
        )
    )

    assert time.monotonic() - started < 0.5
    assert attempts == 1
    assert outcomes[0].error_type == "gate_timeout"


def test_gate_timeout_bounds_cancel_cleanup(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, "task_slow_cleanup")
    spec = TaskSpec(
        name="task_slow_cleanup",
        task_dir=task_dir,
        image="t-data-processing-v1",
        instruction="solve",
        tests_dir=task_dir / "tests",
    )
    trial = Trial("candidate-01-slow-a0", "candidate-01", "a" * 40, "/bin/a", spec, 0)

    async def slow_cleanup_runner(trial: Trial, **_: object) -> TrialOutcome:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(10)
        raise AssertionError("cleanup should have been force-cancelled")

    started = time.monotonic()
    outcomes = asyncio.run(
        _run_trials(
            [trial],
            settings={"gate_cleanup_grace_sec": 0.05},
            jobs_dir=tmp_path / "jobs",
            n_concurrent=1,
            trial_runner=slow_cleanup_runner,
            overall_timeout_sec=0.05,
        )
    )

    assert time.monotonic() - started < 0.5
    assert outcomes[0].error_type == "gate_timeout"


def test_placeholder_tasks_dir_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task_dir(tmp_path, "task_a")
    catalog = {"task_a": {"task_dir": str(task), "image_tag": "img-a"}}
    specs = resolve_task_specs(
        {"tasks_dir": "__TASKS_DIR__", "tasks_index": None},
        ["task_a"],
        task_catalog=catalog,
    )
    assert specs["task_a"].image == "img-a"
    assert specs["task_a"].task_dir == task


def test_static_gate_baseline_uses_configured_binary(tmp_path: Path) -> None:
    from harness_evolution.evaluation import gate

    binary = tmp_path / "opencode"
    binary.write_bytes(b"binary")
    commit = "1" * 40
    baseline = gate._resolve_baseline(
        {
            "paths": {
                "version_state": "",
                "baseline_commit": commit,
                "baseline_binary": str(binary),
            }
        }
    )

    assert baseline == {
        "commit": commit,
        "commit_short": commit[:7],
        "binary_path": str(binary),
    }


def test_jsonl_catalog_fills_image(tmp_path: Path) -> None:
    task = _task_dir(tmp_path, "task_b")
    jsonl = tmp_path / "prompt.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "prompt": [],
                "metadata": {
                    "task_id": "task_b",
                    "task_dir": str(task),
                    "image_tag": "from-jsonl",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = load_jsonl_catalog(jsonl)
    spec = resolve_task_spec("task_b", tasks_dir=tmp_path, catalog=catalog)
    assert spec.image == "from-jsonl"


def test_expand_trials_repeats_mixed_tasks(tmp_path: Path) -> None:
    mixed = _task_dir(tmp_path, "mixed")
    fail = _task_dir(tmp_path, "fail")
    specs = {
        "mixed": resolve_task_spec("mixed", tasks_dir=tmp_path),
        "fail": resolve_task_spec("fail", tasks_dir=tmp_path),
    }
    trials = expand_trials(
        sides=[("baseline", "aaa"), ("candidate-01", "bbb")],
        task_names=["mixed", "mixed", "mixed", "fail"],
        specs=specs,
        n_attempts=1,
    )
    assert len(trials) == 8  # 2 sides * 4 entries
    assert sum(1 for t in trials if t.side == "baseline" and t.task.name == "mixed") == 3
    assert mixed.is_dir() and fail.is_dir()


def test_custom_afs_path_is_carried_by_trial_and_publisher(tmp_path: Path) -> None:
    task = _task_dir(tmp_path, "custom")
    specs = {"custom": resolve_task_spec("custom", tasks_dir=tmp_path)}
    binary = tmp_path / "opencode"
    binary.write_bytes(b"binary")
    commit = "cafebabecafebabecafebabecafebabe"
    path = custom_afs_binary_path(
        "/sandbox-data/duchenyu/opencode",
        run_tag="20261005-120000-step-01",
        side_label="candidate-01",
        commit=commit,
    )
    assert path == "/sandbox-data/duchenyu/opencode/wxh/20261005-120000-step-01-candidate-01/opencode"
    trials = expand_trials(
        sides=[("candidate-01", commit, str(binary), path)],
        task_names=["custom"],
        specs=specs,
        n_attempts=1,
    )
    assert trials[0].afs_bin == path

    published: list[tuple[str, str]] = []

    def fake_publish(source: str, commit_value: str, *, afs_path: str = "", log=None) -> str:
        published.append((commit_value, afs_path))
        return afs_path

    publish_side_binaries(
        [("candidate-01", commit, str(binary), path)],
        publish=fake_publish,
    )
    assert published == [(commit, path)]


def test_opensandbox_settings_read_training_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = tmp_path / "sandbox.env"
    secret.write_text("OPEN_SANDBOX_ACCESS_KEY=ak\nOPEN_SANDBOX_SECRET_KEY=sk\n", encoding="utf-8")
    monkeypatch.setenv("OPEN_SANDBOX_SECRET_ENV", str(secret))
    monkeypatch.setenv("AGENT_GATEWAY_URL", "http://llm-proxy.test")
    monkeypatch.setenv("AGENT_MODEL_NAME", "run-model")
    monkeypatch.setenv("SANDBOX_CREATES_PER_SEC", "0.5")
    settings = opensandbox_settings({})
    assert settings["env_file"] == str(secret)
    assert settings["agent_gateway_url"] == "http://llm-proxy.test"
    assert settings["agent_model_name"] == "run-model"
    assert settings["create_rate"] == 0.5


def test_publish_skipped_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_AFS_PUBLISH", "0")
    called: list[tuple[str, str]] = []

    def fake_publish(binary: str, commit: str, log=None) -> str:
        called.append((binary, commit))
        return "/afs"

    binary = tmp_path / "opencode"
    binary.write_text("x", encoding="utf-8")
    publish_side_binaries([("baseline", "abc1234deadbeef", str(binary))], publish=fake_publish)
    assert called == []


def test_heldout_opensandbox_pins_commit_and_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime.instances = []
    task = _task_dir(tmp_path, "task_win")
    secret = tmp_path / "sandbox.env"
    secret.write_text("OPEN_SANDBOX_ACCESS_KEY=ak\nOPEN_SANDBOX_SECRET_KEY=sk\n", encoding="utf-8")
    secret.chmod(0o600)
    monkeypatch.setenv("AGENT_GATEWAY_URL", "http://llm-proxy.test")
    monkeypatch.setenv("OPEN_SANDBOX_SECRET_ENV", str(secret))
    monkeypatch.setenv("HARNESS_AFS_PUBLISH", "1")
    monkeypatch.setenv("AGENT_MODEL_NAME", "test-model")

    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "active_commit": "1878ff1dad26daa4dde838c34ff6bb54e2dc614a",
                "active_binary_path": str(tmp_path / "base.bin"),
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "base.bin").write_text("base", encoding="utf-8")
    cand_bin = tmp_path / "cand.bin"
    cand_bin.write_text("cand", encoding="utf-8")
    published: list[str] = []

    def fake_publish(binary: str, commit: str, log=None) -> str:
        published.append(commit)
        return f"/sandbox-data/duchenyu/opencode/{commit[:7]}/opencode"

    config = {
        "paths": {
            "version_state": str(state),
            "baseline_commit": "1878ff1dad26daa4dde838c34ff6bb54e2dc614a",
            "report_root": str(tmp_path / "reports"),
        },
        "heldout_gate": {
            "backend": "opensandbox",
            "enabled": True,
            "tasks_dir": str(tmp_path),
            "task_names": ["task_win"],
            "n_attempts": 1,
            "n_concurrent": 2,
            "accept_net_wins": 1,
            "opensandbox": {"env_file": str(secret), "agent_gateway_url": "http://llm-proxy.test"},
        },
    }
    raw = run_heldout_opensandbox(
        config,
        [("cafebabecafebabecafebabecafebabe", str(cand_bin))],
        task_names=["task_win"],
        task_catalog={"task_win": {"task_dir": str(task), "image_tag": "t-data-processing-v1"}},
        runtime_factory=FakeRuntime,
        publish=fake_publish,
    )
    assert "1878ff1dad26daa4dde838c34ff6bb54e2dc614a" in published
    assert "cafebabecafebabecafebabecafebabe" in published
    commits = {rt.commit for rt in FakeRuntime.instances}
    assert "1878ff1dad26daa4dde838c34ff6bb54e2dc614a" in commits
    assert "cafebabecafebabecafebabecafebabe" in commits
    assert any(rt.uploads and rt.uploads[0][1] == "/tests" for rt in FakeRuntime.instances)
    for runtime in FakeRuntime.instances:
        assert (runtime.session_dir / "agent.stdout.log").is_file()
        assert (runtime.session_dir / "agent.stderr.log").is_file()
        assert (runtime.session_dir / "verifier.stdout.log").is_file()
        assert (runtime.session_dir / "verifier.stderr.log").is_file()
        assert (runtime.session_dir / "agent.remote.jsonl").read_text(encoding="utf-8").startswith(
            "saved /logs/agent/opencode.txt"
        )
        assert not any("/logs/agent/llm-reqs/" in path for path in runtime.downloads)
    baseline = raw["baseline"]
    candidate = raw["candidates"]["cafebabecafebabecafebabecafebabe"]
    assert baseline.passes["task_win"] == 0
    assert candidate.passes["task_win"] == 1


def test_heldout_direct_upload_skips_afs_and_selects_side_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeRuntime.instances = []
    task = _task_dir(tmp_path, "task_upload")
    secret = tmp_path / "sandbox.env"
    secret.write_text("OPEN_SANDBOX_ACCESS_KEY=ak\nOPEN_SANDBOX_SECRET_KEY=sk\n", encoding="utf-8")
    secret.chmod(0o600)
    monkeypatch.setenv("AGENT_GATEWAY_URL", "http://llm-proxy.test")
    monkeypatch.setenv("OPEN_SANDBOX_SECRET_ENV", str(secret))
    monkeypatch.setenv("AGENT_MODEL_NAME", "test-model")

    base_bin = tmp_path / "base.bin"
    candidate_bin = tmp_path / "candidate.bin"
    base_bin.write_text("base", encoding="utf-8")
    candidate_bin.write_text("candidate", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "active_commit": "1878ff1dad26daa4dde838c34ff6bb54e2dc614a",
                "active_binary_path": str(base_bin),
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    published: list[str] = []

    config = {
        "paths": {"version_state": str(state), "report_root": str(tmp_path / "reports")},
        "heldout_gate": {
            "backend": "opensandbox",
            "tasks_dir": str(tmp_path),
            "task_names": ["task_upload"],
            "n_attempts": 1,
            "n_concurrent": 2,
            "opensandbox": {
                "env_file": str(secret),
                "agent_gateway_url": "http://llm-proxy.test",
                "opencode_setup": "upload",
            },
        },
    }
    run_heldout_opensandbox(
        config,
        [("cafebabecafebabecafebabecafebabe", str(candidate_bin))],
        runtime_factory=FakeRuntime,
        publish=lambda _binary, commit, **_: published.append(commit),
    )

    assert published == []
    uploads = {rt.commit: rt.spec.kwargs["opencode_upload_path"] for rt in FakeRuntime.instances}
    assert uploads["1878ff1dad26daa4dde838c34ff6bb54e2dc614a"] == str(base_bin)
    assert uploads["cafebabecafebabecafebabecafebabe"] == str(candidate_bin)


def test_gate_scores_opensandbox_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from harness_evolution.evaluation.gate import _score_heldout_sides

    results = _score_heldout_sides(
        candidates=[("cafebabecafebabe", "/bin/c")],
        distinct_task_names=["task_win", "task_tie"],
        per_task_attempts={"task_win": 1, "task_tie": 1},
        baseline_passes={"task_win": 0, "task_tie": 1},
        baseline_errors=0,
        baseline_job_dir="/tmp/base",
        candidate_results={"cafebabecafebabe": ({"task_win": 1, "task_tie": 1}, 0, "/tmp/cand")},
        accept_net_wins=1,
    )
    result = results["cafebabecafebabe"]
    assert result.wins == 1
    assert result.ties == 1
    assert result.losses == 0
    assert result.passed is True


def test_pipeline_catalog_from_manifest(tmp_path: Path) -> None:
    job = tmp_path / "tb_job"
    job.mkdir()
    (job / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset_task_id": "task_mixed",
                        "dataset_task_dir": "/data/task_mixed",
                        "dataset_image_tag": "img-mixed",
                        "selection_bucket": "mixed",
                    },
                    {
                        "dataset_task_id": "task_fail",
                        "dataset_task_dir": "/data/task_fail",
                        "dataset_image_tag": "img-fail",
                        "selection_bucket": "pure_fail",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    from harness_evolution.integrations.slime import pipeline

    names = pipeline._evidence_task_names(job, mixed_task_attempts=3)
    catalog = pipeline._evidence_task_catalog(job)
    assert names.count("task_mixed") == 3
    assert names.count("task_fail") == 1
    assert catalog["task_mixed"]["image_tag"] == "img-mixed"
    assert catalog["task_fail"]["task_dir"] == "/data/task_fail"


def test_pipeline_task_resolution_uses_env_when_config_is_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harness_evolution.integrations.slime import pipeline

    tasks_dir = tmp_path / "tasks"
    job = tmp_path / "tb_job"
    tasks_dir.mkdir()
    job.mkdir()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("HARNESS_GATE_TASKS_DIR", str(tasks_dir))
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline._resolve_evidence_task_ids(
        {"work_root": str(tmp_path / "evolution-runs")},
        {"heldout_gate": {"tasks_dir": "__TASKS_DIR__"}},
        job,
    )

    assert len(calls) == 1
    tasks_dir_arg = calls[0].index("--tasks-dir") + 1
    assert calls[0][tasks_dir_arg] == str(tasks_dir)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("upload_fails", [False, True])
def test_gate_publishes_before_changing_active(tmp_path, monkeypatch, multi, upload_fails):
    from harness_evolution.evaluation import gate
    from harness_dist import state

    state_path = tmp_path / "state.json"
    state.init_state(state_path, "baseline", "/base")
    config = {"paths": {"version_state": str(state_path), "report_root": str(tmp_path / "reports")}}
    sr = gate.StaticGateResult(passed=True, candidate_commit="candidate", binary_path="/candidate")
    hr = gate.HeldoutGateResult(passed=True)
    monkeypatch.setattr(gate, "run_static_gate", lambda *a: sr)
    monkeypatch.setattr(gate, "run_static_gate_batch", lambda *a: [sr])
    monkeypatch.setattr(gate, "run_heldout_gate", lambda *a, **k: hr)
    monkeypatch.setattr(gate, "run_heldout_gate_multi", lambda *a, **k: {"candidate": hr})
    monkeypatch.setattr(gate, "_persist_candidate_commit", lambda *a: None)
    calls = []

    def publish(*args):
        assert state.read_active(state_path)["commit"] == "baseline"
        calls.append(args)
        if upload_fails:
            raise RuntimeError("AFS unavailable")

    monkeypatch.setattr(gate, "_publish_accepted_binary", publish)
    run = gate.run_full_gate_multi if multi else gate.run_full_gate
    workspaces = [tmp_path] if multi else tmp_path
    if upload_fails:
        with pytest.raises(RuntimeError, match="AFS unavailable"):
            run(config, workspaces)
        assert state.read_active(state_path)["commit"] == "baseline"
        report = json.loads(next((tmp_path / "reports").glob("*.json")).read_text())
        assert report["accepted"] is False
        assert report["publication_error"] == "AFS unavailable"
    else:
        assert run(config, workspaces)["accepted"]
        assert state.read_active(state_path)["commit"] == "candidate"
    assert len(calls) == 1


@pytest.mark.parametrize("candidates, expected", [
    ([], 1),
    ([{"evolver_metadata": {"error": "HTTP 404"}}], 1),
    ([{"evolver_metadata": {}, "validation": {"passed": False}}], 0),
])
def test_pipeline_distinguishes_infrastructure_errors_from_rejection(tmp_path, monkeypatch, candidates, expected):
    from harness_evolution.integrations.slime import pipeline

    config = {"gate_config": "unused", "version_state": str(tmp_path / "state"), "work_root": str(tmp_path)}
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", "unused", "--rollout-id", "1"])
    monkeypatch.setattr(pipeline, "_load_config", lambda *a: config)
    monkeypatch.setattr(pipeline, "_apply_env_overrides", lambda c: c)
    monkeypatch.setattr(pipeline, "load_gate_config", lambda *a: {"paths": {}})
    monkeypatch.setattr(pipeline, "_select_trajectories", lambda *a: None)
    monkeypatch.setattr(pipeline, "_resolve_evidence_task_ids", lambda *a: None)
    monkeypatch.setattr(pipeline, "_resolve_active", lambda *a: {"commit": "baseline", "commit_short": "baselin"})
    monkeypatch.setattr(pipeline, "_run_evolver", lambda *a: {"candidates": candidates})
    assert pipeline.main() == expected
