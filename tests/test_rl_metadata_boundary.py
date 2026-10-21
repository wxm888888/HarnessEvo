"""Task identity must survive the real adapter/dump/evidence boundaries."""
import json
from enum import Enum
from types import SimpleNamespace

import pytest
from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import adapter, _dump, _samples
from slime_bridge.config import resolve_polar_slime_config
from harness_evolution.rl.select import select_evidence
from harness_evolution.rl.evolve import _evidence_task_catalog, _evidence_task_names


class Sample(SimpleNamespace):
    class Status(str, Enum):
        COMPLETED = "completed"
        FAILED = "failed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"


@pytest.mark.parametrize("submitted", [False, True])
@pytest.mark.parametrize("file_key", ["task_file", "mimo_task_file"])
def test_session_result_to_gate_catalog_preserves_mimo_identity(tmp_path, monkeypatch, file_key, submitted):
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: Sample)
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({"cwd": "/testbed"}))
    task = {
        "task_id": "format-code-task-000588",
        file_key: str(task_file),
        "task_sha256": "sha256-task",
        "mimo_dataset_type": "opensource-code",
        "image_tag": "mimo:test",
    }
    if submitted:
        args = SimpleNamespace(
            polar_rollout_url="http://localhost:8080",
            polar_task_template={"runtime": {}, "agent": {"settings": {}},
                                 "evaluator": {"config": {}}},
        )
        config = resolve_polar_slime_config(args)
        source = Sample(group_index=0, prompt=[{"role": "user", "content": "Fix the code."}],
                        metadata=task)
        payload = _samples._build_task_payload(args=args, config=config, group=[source],
                                               rollout_id=0, task_position=0)
        _samples._attach_scheduler_metadata(payload, group_id=0, policy_version=0, rollout_step=0)
        task = {**payload["metadata"], "task_id": payload["task_id"]}
        assert task["dataset_task_id"] == "format-code-task-000588"
    dataset_task_id = "format-code-task-000588"
    result = SessionResult(
        session_id="session-mimo", task_id="polar-position-traj-0",
        status=SessionStatus.COMPLETED, node_id="test",
        timing=SessionTiming(init_ms=1, run_ms=1, postrun_ms=1),
        trajectory=Trajectory(status="COMPLETED", metadata={"task_metadata": task}, traces=[
            Trace(prompt_ids=[1], response_ids=[2], loss_mask=[1],
                  prompt_messages=[{"role": "user", "content": "Fix the code."}],
                  response_messages=[{"role": "assistant", "content": "Done."}],
                  response_logprobs=[{"token_id": 2, "logprob": -0.1}], reward=0.0)
        ]),
    )
    samples = adapter.session_result_to_samples(result, group_index=0, trajectory_index=0)
    assert "task_file" not in samples[0].metadata
    monkeypatch.setattr(_dump, "_TRAJ_DUMP_DIR", str(tmp_path / "trajectories"))
    _dump._dump_all_trajectories_to_disk(0, [samples])
    dump_path = tmp_path / "trajectories/rollout_000000/session-mimo.json"
    payload = json.loads(dump_path.read_text())
    assert payload["dataset_task_id"] == dataset_task_id
    assert payload["dataset_task_file"] == str(task_file)
    evidence = tmp_path / "evidence"
    manifest = select_evidence(tmp_path / "trajectories", n_fail=1, n_success=0,
                               output_dir=evidence, end_rollout_id=0)
    record = manifest["records"][0]
    assert record["dataset_task_file"] == str(task_file)
    assert _evidence_task_names(evidence) == [dataset_task_id]
    assert _evidence_task_catalog(evidence) == {dataset_task_id: {
        "task_file": str(task_file), "task_sha256": "sha256-task",
        "mimo_dataset_type": "opensource-code", "image_tag": "mimo:test",
    }}


def test_metadata_prefers_nested_identity_and_retains_legacy_fallback():
    sample = Sample(metadata={
        "task_id": "old", "task_file": "/old.json", "task_dir": "/legacy",
        "image_tag": "legacy-image",
        "polar": {"trajectory_metadata": {"task_metadata": {
            "task_id": "current", "mimo_task_file": "/current.json", "image_tag": "",
        }}},
    })
    metadata = _dump._dataset_metadata(sample)
    assert metadata["task_id"] == "current"
    assert metadata["task_file"] == "/current.json"
    assert metadata["task_dir"] == "/legacy"
    assert metadata["image_tag"] == "legacy-image"
    assert sample.metadata["task_file"] == "/old.json"


def test_dataset_namespace_wins_over_runtime_identity_and_alias():
    sample = Sample(metadata={
        "task_id": "legacy", "task_file": "/legacy.json",
        "polar": {"trajectory_metadata": {"task_metadata": {
            "dataset_task_id": "dataset", "task_id": "runtime",
            "dataset_task_file": "/dataset.json", "mimo_task_file": "/runtime.json",
        }}},
    })
    metadata = _dump._dataset_metadata(sample)
    assert metadata["task_id"] == "dataset"
    assert metadata["task_file"] == "/dataset.json"
