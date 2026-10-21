from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from polar.runtime.models import ExecResult, RuntimeSpec

from harness_evolution.evaluation.backends import mimo
from harness_evolution.evaluation.backends import opensandbox as opensandbox_gate
from harness_evolution.evaluation.trials import Trial
from harness_evolution.tasks import mimo_catalog
from harness_evolution.tasks.catalog import load_jsonl_catalog, resolve_task_spec
from harness_evolution.tasks.splits import _load_task_catalog

BASE = "a" * 40
LATER = "b" * 40
TEST_PATCH = "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"


def _write_task(root: Path, task_id: str, **overrides: object) -> tuple[Path, dict]:
    task = {
        "instance_id": task_id,
        "dataset_type": "opensource-code",
        "problem_statement": f"fix {task_id}",
        "public_image": f"docker.io/xiaomimimo/mimo-v2.6-rl-oss:{task_id}",
        "cwd": "/testbed",
        "cpus": 2,
        "memory_mb": 2048,
        "agent_timeout_sec": 900.0,
        "verifier_timeout_sec": 1800,
        "test_patch": TEST_PATCH,
        "verified_patch_paths": ["tests/test_x.py"],
        "test_command": "pytest -q tests/test_x.py",
    }
    task.update(overrides)
    path = root / "mimo-tasks" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task), encoding="utf-8")
    return path, task


def _catalog_row(task_id: str, path: Path, family: str = "opensource-code") -> dict:
    return {
        "prompt": "unused",
        "metadata": {
            "task_id": task_id,
            "task_file": str(path),
            "task_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "image_tag": f"docker.io/xiaomimimo/mimo-v2.6-rl-oss:{task_id}",
            "mimo_dataset_type": family,
            "verifier_timeout_sec": 1800,
        },
    }


def test_mimo_catalog_rows_resolve_to_task_specs(tmp_path: Path) -> None:
    path, _ = _write_task(tmp_path, "format-code-task-000001")
    catalog_path = tmp_path / "mimo.jsonl"
    catalog_path.write_text(json.dumps(_catalog_row("format-code-task-000001", path)) + "\n", encoding="utf-8")

    for catalog in (load_jsonl_catalog(catalog_path), _load_task_catalog(catalog_path)):
        spec = resolve_task_spec("format-code-task-000001", tasks_dir="", catalog=catalog)
        assert spec.is_mimo and spec.mimo_task_file == path
        assert spec.workdir == "/testbed"
        assert spec.instruction == "fix format-code-task-000001"
        assert spec.image.endswith(":format-code-task-000001")
        assert (spec.agent_timeout_sec, spec.verifier_timeout_sec) == (900, 1800)
        # Same floor as the screening runner: at least 8 GiB.
        assert (spec.cpu, spec.memory) == ("2", "8192Mi")


def test_mimo_code_task_without_agent_budget_uses_screening_budget(tmp_path: Path) -> None:
    path, _ = _write_task(tmp_path, "format-code-task-000009", agent_timeout_sec=None)
    catalog = {"format-code-task-000009": _catalog_row("format-code-task-000009", path)["metadata"]}
    spec = resolve_task_spec("format-code-task-000009", tasks_dir="", catalog=catalog)
    assert spec.agent_timeout_sec == 900


def test_mimo_task_file_change_is_rejected(tmp_path: Path) -> None:
    path, _ = _write_task(tmp_path, "format-code-task-000002")
    row = _catalog_row("format-code-task-000002", path)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    catalog = {"format-code-task-000002": row["metadata"]}
    with pytest.raises(ValueError, match="task_sha256"):
        resolve_task_spec("format-code-task-000002", tasks_dir="", catalog=catalog)


def test_patch_paths_prefers_recorded_list_and_rejects_escapes() -> None:
    assert mimo.patch_paths({"verified_patch_paths": ["a/b.py"], "test_patch": ""}) == ["a/b.py"]
    assert mimo.patch_paths({"test_patch": TEST_PATCH}) == ["tests/test_x.py"]
    with pytest.raises(ValueError):
        mimo.patch_paths({"verified_patch_paths": ["../etc/passwd"]})
    with pytest.raises(ValueError):
        mimo.patch_paths({"test_patch": "no headers"})


def test_verifier_receipts() -> None:
    assert mimo.parse_code_receipt(0, "...\n__MIMO_VERIFIER_RC__=0\n") == 0
    assert mimo.parse_code_receipt(0, "__MIMO_VERIFIER_RC__=1") == 1
    for rc, out in ((0, "no receipt"), (1, "__MIMO_VERIFIER_RC__=0"), (0, "__MIMO_VERIFIER_RC__=127")):
        with pytest.raises(mimo.MimoInfrastructureError):
            mimo.parse_code_receipt(rc, out)
    ok = "1 passed\n__POLAR_VERIFIER_REWARD__=1\n__MIMO_VERIFIER_RC__=0\n"
    assert mimo.parse_terminal_receipt(0, ok, "") == (0, True)
    assert mimo.parse_terminal_receipt(0, "__POLAR_VERIFIER_REWARD__=0\n__MIMO_VERIFIER_RC__=1", "") == (1, False)
    with pytest.raises(mimo.MimoInfrastructureError):
        mimo.parse_terminal_receipt(0, "2 failed\n__POLAR_VERIFIER_REWARD__=1\n__MIMO_VERIFIER_RC__=0", "")


class MimoFakeRuntime:
    instances: list["MimoFakeRuntime"] = []
    verifier_rc = 0
    later_commits = ""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        self.spec = spec
        self.session_dir = session_dir
        self.commands: list[str] = []
        self.uploaded: dict[str, bytes] = {}
        self.homes: dict[str, str | None] = {}
        self.capped: dict[str, bool] = {}
        type(self).instances.append(self)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        raise AssertionError(f"MiMo trials must not upload a tests dir ({remote_path})")

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        self.uploaded[remote_path] = Path(local_path).read_bytes()

    async def download_file(self, remote_path: str, local_path: str) -> None:
        Path(local_path).write_text("", encoding="utf-8")

    async def exec(self, command: str, env: dict | None = None, **_: object) -> ExecResult:
        prefix = mimo.cpu_limit_prefix()
        capped = command.startswith(prefix)
        command = command[len(prefix):] if capped else command
        self.capped[command] = capped
        self.commands.append(command)
        self.homes[command] = (env or {}).get("HOME", self.spec.env.get("HOME"))
        if "HIDDEN_VERIFIER_ABSENT" in command:
            return ExecResult(return_code=0, stdout="HIDDEN_VERIFIER_ABSENT\n", stderr="")
        if "git rev-parse HEAD" in command:
            return ExecResult(return_code=0, stdout=BASE + "\n", stderr="")
        if "git rev-list --all --not" in command:
            return ExecResult(return_code=0, stdout=type(self).later_commits, stderr="")
        if "git diff --binary" in command:
            return ExecResult(return_code=0, stdout="diff --git a/src.py b/src.py\n", stderr="")
        if command == "bash /tmp/mimo-verify.sh":
            rc = type(self).verifier_rc
            reward = "" if self.uploaded.get("/tests/test.sh") is None else f"__POLAR_VERIFIER_REWARD__={1 - rc}\n"
            return ExecResult(return_code=0, stdout=f"{reward}__MIMO_VERIFIER_RC__={rc}\n", stderr="")
        return ExecResult(return_code=0, stdout="", stderr="")


def _settings(tmp_path: Path) -> dict:
    secret = tmp_path / "sandbox.env"
    secret.write_text("OPEN_SANDBOX_ACCESS_KEY=ak\nOPEN_SANDBOX_SECRET_KEY=sk\n", encoding="utf-8")
    secret.chmod(0o600)
    return opensandbox_gate.opensandbox_settings(
        {"opensandbox": {"env_file": str(secret), "agent_gateway_url": "http://llm-proxy.test", "workdir": "/home/user"}}
    )


def _trial(tmp_path: Path, task_id: str, **overrides: object) -> Trial:
    path, _ = _write_task(tmp_path, task_id, **overrides)
    spec = resolve_task_spec(task_id, tasks_dir="", catalog={task_id: _catalog_row(task_id, path)["metadata"]})
    return Trial(trial_id=f"sample-{task_id}-a0", side="sample", commit="c" * 40, binary="", task=spec, attempt=0)


@pytest.mark.parametrize("verifier_rc, later, expected", [(0, "", True), (1, LATER + "\n", False)])
def test_mimo_code_trial_runs_hidden_verifier(tmp_path: Path, verifier_rc: int, later: str, expected: bool) -> None:
    MimoFakeRuntime.instances = []
    MimoFakeRuntime.verifier_rc = verifier_rc
    MimoFakeRuntime.later_commits = later
    trial = _trial(tmp_path, "format-code-task-000003")
    session = tmp_path / "session"
    session.mkdir()

    outcome = asyncio.run(
        opensandbox_gate._run_trial_once(
            trial, settings=_settings(tmp_path), session_dir=session, runtime_factory=MimoFakeRuntime
        )
    )

    assert outcome.error is False, outcome.detail
    assert outcome.passed is expected
    runtime = MimoFakeRuntime.instances[0]
    assert runtime.spec.workdir == "/testbed"
    assert runtime.spec.env["HOME"] == mimo.MIMO_HOME
    assert (runtime.spec.kwargs["cpu"], runtime.spec.kwargs["memory"]) == ("2", "8192Mi")
    joined = "\n".join(runtime.commands)
    # History after the baseline is removed only when it exists.
    assert ("git gc" in joined or "gc --prune=now" in joined) is bool(later)
    # The hidden patch arrives only after the agent ran.
    agent_index = next(i for i, c in enumerate(runtime.commands) if "opencode" in c and " run " in c)
    apply_index = next(i for i, c in enumerate(runtime.commands) if "git apply --check" in c)
    assert agent_index < apply_index
    # The agent keeps HOME=/home/user; setup and the verifier use root's HOME,
    # whose ~/.bashrc sets up pyenv for the tasks' test scripts.
    assert runtime.homes[runtime.commands[agent_index]] == mimo.MIMO_HOME
    assert runtime.homes["bash /tmp/mimo-verify.sh"] == "/root"
    assert runtime.homes[runtime.commands[apply_index]] == "/root"
    # Thread caps (the sandbox reports the host's CPU count) apply to the
    # agent and the verifier, and the CPU-count hooks are uploaded first.
    assert runtime.capped[runtime.commands[agent_index]] and runtime.capped["bash /tmp/mimo-verify.sh"]
    assert "OMP_NUM_THREADS=2" in mimo.cpu_limit_prefix()
    assert "os.cpu_count" in runtime.uploaded["/tmp/mimo-cpu/sitecustomize.py"].decode()
    assert "/tmp/mimo-cpu-limit.cjs" in runtime.uploaded
    assert runtime.uploaded["/tmp/mimo-hidden.patch"].decode() == TEST_PATCH
    assert "pytest -q tests/test_x.py" in runtime.uploaded["/tmp/mimo-verify.sh"].decode()
    assert (session / "agent" / "model.diff").read_text(encoding="utf-8").startswith("diff --git")
    assert (session / "verifier.stdout.log").read_text(encoding="utf-8").endswith(f"__MIMO_VERIFIER_RC__={verifier_rc}\n")


def test_mimo_terminal_trial_uploads_embedded_tests(tmp_path: Path) -> None:
    MimoFakeRuntime.instances = []
    MimoFakeRuntime.verifier_rc = 0
    files = {"test.sh": base64.b64encode(b"#!/bin/bash\nexit 0\n").decode(), "sub/check.py": base64.b64encode(b"x").decode()}
    trial = _trial(
        tmp_path,
        "candidate-0001-software",
        dataset_type="terminal_bench",
        cwd="/app",
        tests_files=json.dumps(files),
    )
    session = tmp_path / "session"
    session.mkdir()
    outcome = asyncio.run(
        opensandbox_gate._run_trial_once(
            trial, settings=_settings(tmp_path), session_dir=session, runtime_factory=MimoFakeRuntime
        )
    )
    assert outcome.error is False and outcome.passed is True
    runtime = MimoFakeRuntime.instances[0]
    assert runtime.uploaded["/tests/test.sh"] == b"#!/bin/bash\nexit 0\n"
    assert runtime.uploaded["/tests/sub/check.py"] == b"x"
    assert not any("git rev-parse HEAD" in c for c in runtime.commands)


def test_build_mimo_catalog_keeps_informative_tasks(tmp_path: Path) -> None:
    screening = tmp_path / "screening"
    screening.mkdir()
    rows = []
    for successes, task_id in [(0, "t0"), (3, "t3"), (8, "t8"), (5, "tq")]:
        path, _ = _write_task(tmp_path, task_id)
        rows.append(_catalog_row(task_id, path))
        (screening / f"success_{successes}_of_8.txt").write_text(task_id + "\n", encoding="utf-8")
    for successes in (1, 2, 4, 6, 7):
        (screening / f"success_{successes}_of_8.txt").write_text("", encoding="utf-8")
    path, _ = _write_task(tmp_path, "t_incomplete")
    rows.append(_catalog_row("t_incomplete", path))
    registry = tmp_path / "quarantine.json"
    registry.write_text(json.dumps({"mimo/tq": {"status": "quarantined"}, "mimo/t3": {"status": "recovered"}}))
    source = tmp_path / "mimo.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    output = tmp_path / "out" / "heldin.jsonl"

    assert mimo_catalog.main([
        "--mimo-jsonl", str(source), "--screening", str(screening),
        "--quarantine", str(registry), "--output", str(output),
    ]) == 0

    kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["metadata"]["task_id"] for r in kept] == ["t3"]
    assert kept[0]["metadata"]["screen_success_count"] == 3
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["dropped"] == {
        "success_0_of_8": 1, "success_8_of_8": 1, "quarantined": 1, "not_screened_or_incomplete": 1,
    }


def _gate_row(task_id: str, family: str, successes: int) -> dict:
    return {"metadata": {"task_id": task_id, "mimo_dataset_type": family, "screen_success_count": successes}}


def test_gate_split_is_stratified_and_deterministic() -> None:
    rows = [_gate_row(f"c{i}", "opensource-code", 2 + i % 3) for i in range(60)]
    rows += [_gate_row(f"t{i}", "terminal_bench", 4) for i in range(20)]
    rows += [_gate_row(f"x{i}", "opensource-code", 7) for i in range(20)]  # outside 2..6
    gate = mimo_catalog.select_gate(rows, 20, seed=0, min_success=2, max_success=6)
    assert len(gate) == len({r["metadata"]["task_id"] for r in gate}) == 20
    mix = Counter((r["metadata"]["mimo_dataset_type"], r["metadata"]["screen_success_count"]) for r in gate)
    assert mix == {("opensource-code", 2): 5, ("opensource-code", 3): 5, ("opensource-code", 4): 5,
                   ("terminal_bench", 4): 5}
    again = mimo_catalog.select_gate(list(reversed(rows)), 20, seed=0, min_success=2, max_success=6)
    assert gate == again
    with pytest.raises(ValueError, match="exceeds"):
        mimo_catalog.select_gate(rows, 81, seed=0, min_success=2, max_success=6)


def test_gate_split_is_removed_from_held_in(tmp_path: Path) -> None:
    screening = tmp_path / "screening"
    screening.mkdir()
    rows = []
    for index in range(6):
        path, _ = _write_task(tmp_path, f"t{index}")
        rows.append(_catalog_row(f"t{index}", path))
    (screening / "success_3_of_8.txt").write_text("".join(f"t{i}\n" for i in range(6)), encoding="utf-8")
    source = tmp_path / "mimo.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    output, gate = tmp_path / "heldin.jsonl", tmp_path / "gate.json"

    assert mimo_catalog.main([
        "--mimo-jsonl", str(source), "--screening", str(screening), "--output", str(output),
        "--gate-size", "2", "--gate-output", str(gate),
    ]) == 0

    gate_ids = json.loads(gate.read_text(encoding="utf-8"))
    held_in = [json.loads(line)["metadata"]["task_id"] for line in output.read_text(encoding="utf-8").splitlines()]
    gate_rows = [json.loads(line)["metadata"]["task_id"] for line in gate.with_suffix(".jsonl").read_text().splitlines()]
    assert len(gate_ids) == 2 and gate_rows == gate_ids
    assert sorted(held_in + gate_ids) == [f"t{i}" for i in range(6)]
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["n_kept"] == 4 and summary["gate"]["size"] == 2
