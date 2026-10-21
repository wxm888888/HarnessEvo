"""Tests for the RL-side evidence selection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness_evolution.rl.select import select_evidence, select_sessions, task_key


def _payload(*, session_id: str, task_id: str, reward: float) -> dict:
    prompt = [
        {"role": "system", "content": "You are opencode."},
        {"role": "user", "content": f"Fix the failing test in {task_id}."},
    ]
    response = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should inspect the repository.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "README.md"},
        {"role": "assistant", "content": "Done.", "reasoning_content": None, "tool_calls": []},
    ]
    return {
        "session_id": session_id,
        "task_id": task_id,
        "session_status": "COMPLETED",
        "result_error": None,
        "result_error_type": None,
        "result_error_stage": None,
        "result_error_info": None,
        "trajectory_status": "COMPLETED",
        "trajectory_error": None,
        "timing": {"run_ms": 1234},
        "session_reward": reward,
        "num_traces": 1,
        "total_assistant_tokens": 42,
        "traces": [
            {
                "session_id": session_id,
                "task_id": task_id,
                "trace_index": 0,
                "finish_reason": "stop",
                "response_length": 42,
                "status": "completed",
                "prompt_messages": prompt,
                "response_messages": response,
            }
        ],
    }


def _multi_trace(payload: dict) -> dict:
    """A resumed session: one logical attempt spread over several requests."""
    second = json.loads(json.dumps(payload["traces"][0]))
    second["trace_index"] = 1
    payload["traces"].append(second)
    payload["num_traces"] = 2
    return payload


def _write_rollout(
    root: Path, rollout_id: int, sessions: list[tuple[str, float]], *, multi_trace: bool = False
) -> None:
    """``sessions`` is a list of (task_id, reward); reward 1.0 is a success."""
    rollout_dir = root / f"rollout_{rollout_id:06d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for index, (task_id, reward) in enumerate(sessions):
        session_id = f"sk-{rollout_id:06d}-{index:03d}"
        payload = _payload(session_id=session_id, task_id=task_id, reward=reward)
        if multi_trace:
            payload = _multi_trace(payload)
        (rollout_dir / f"{session_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


class TaskKeyTest(unittest.TestCase):
    def test_strips_only_the_sample_suffix(self) -> None:
        self.assertEqual(task_key("polar-tb-harbor-3-7-traj-5"), "polar-tb-harbor-3-7")
        self.assertEqual(task_key("polar-tb-harbor-3-7"), "polar-tb-harbor-3-7")
        self.assertEqual(task_key(None), "")


class SelectSessionsTest(unittest.TestCase):
    def _root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_picks_one_trace_per_task(self) -> None:
        root = self._root()
        # Four samples of task-a all failed, one sample of task-b failed.
        _write_rollout(
            root,
            9,
            [("t-a-traj-0", 0.0), ("t-a-traj-1", 0.0), ("t-a-traj-2", 0.0), ("t-b-traj-0", 0.0)],
        )
        chosen, stats = select_sessions(root, end_rollout_id=9, n_fail=2, n_success=0)
        self.assertEqual(sorted(trace.task for trace in chosen), ["t-a", "t-b"])
        self.assertEqual(stats["picked"], {"n_fail": 2, "n_success": 0})
        self.assertEqual(stats["available_tasks"]["fail"], 2)

    def test_dataset_task_id_deduplicates_the_same_task_across_steps(self) -> None:
        root = self._root()
        _write_rollout(root, 1, [("polar-mimo-1-4", 0.0)])
        _write_rollout(root, 2, [("polar-mimo-2-4", 0.0)])
        for rollout_id in (1, 2):
            path = next((root / f"rollout_{rollout_id:06d}").glob("*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["dataset_task_id"] = "format-code-task-000588"
            path.write_text(json.dumps(payload), encoding="utf-8")

        chosen, stats = select_sessions(root, end_rollout_id=2, n_fail=2, n_success=0)

        self.assertEqual([trace.task for trace in chosen], ["format-code-task-000588"])
        self.assertEqual(stats["available_tasks"]["fail"], 1)

    def test_a_task_cannot_fill_both_quotas(self) -> None:
        root = self._root()
        # task-a both failed and succeeded; task-b only succeeded.
        _write_rollout(
            root,
            4,
            [("t-a-traj-0", 0.0), ("t-a-traj-1", 1.0), ("t-b-traj-0", 1.0)],
        )
        chosen, _stats = select_sessions(root, end_rollout_id=4, n_fail=1, n_success=1)
        by_task = {trace.task: trace.is_failure for trace in chosen}
        # The shared task spends its single slot on the failure, so the
        # success quota has to come from a different task.
        self.assertEqual(by_task, {"t-a": True, "t-b": False})

    def test_prefers_the_newest_step(self) -> None:
        root = self._root()
        _write_rollout(root, 5, [("t-old-traj-0", 0.0)])
        _write_rollout(root, 6, [("t-new-traj-0", 0.0)])
        chosen, stats = select_sessions(root, end_rollout_id=6, n_fail=1, n_success=0)
        self.assertEqual([trace.task for trace in chosen], ["t-new"])
        # Stops as soon as the quota is full instead of scanning the window.
        self.assertEqual(stats["scanned_rollout_dirs"], ["rollout_000006"])

    def test_window_bounds_the_scan(self) -> None:
        root = self._root()
        _write_rollout(root, 1, [("t-far-traj-0", 0.0)])
        _write_rollout(root, 9, [("t-near-traj-0", 0.0)])
        chosen, stats = select_sessions(root, end_rollout_id=9, n_fail=5, n_success=0, window=2)
        self.assertEqual([trace.task for trace in chosen], ["t-near"])
        self.assertEqual(stats["scanned_rollout_dirs"], ["rollout_000009"])

    def test_short_quota_is_reported_not_raised(self) -> None:
        root = self._root()
        _write_rollout(root, 2, [("t-a-traj-0", 0.0), ("t-a-traj-1", 0.0)])
        chosen, stats = select_sessions(root, end_rollout_id=2, n_fail=4, n_success=2)
        self.assertEqual(len(chosen), 1)
        self.assertTrue(stats["short"])
        self.assertEqual(stats["picked"], {"n_fail": 1, "n_success": 0})
        self.assertEqual(stats["available_tasks"], {"fail": 1, "success": 0})

    def test_missing_rollout_dirs_are_skipped(self) -> None:
        root = self._root()
        _write_rollout(root, 7, [("t-a-traj-0", 0.0)])
        chosen, stats = select_sessions(root, end_rollout_id=8, n_fail=1, n_success=0)
        self.assertEqual([trace.task for trace in chosen], ["t-a"])
        self.assertEqual(stats["scanned_rollout_dirs"], ["rollout_000007"])

    def test_rejects_empty_and_negative_quotas(self) -> None:
        root = self._root()
        with self.assertRaises(ValueError):
            select_sessions(root, end_rollout_id=1, n_fail=0, n_success=0)
        with self.assertRaises(ValueError):
            select_sessions(root, end_rollout_id=1, n_fail=-1, n_success=1)
        with self.assertRaises(ValueError):
            select_sessions(root, end_rollout_id=1, n_fail=1, n_success=0, window=0)


class SelectEvidenceTest(unittest.TestCase):
    def _root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_converts_the_selection_into_a_job_dir(self) -> None:
        root = self._root()
        _write_rollout(
            root,
            3,
            [("t-a-traj-0", 0.0), ("t-b-traj-0", 0.0), ("t-c-traj-0", 1.0)],
        )
        output_dir = self._root() / "tb_job"
        manifest = select_evidence(
            root, n_fail=2, n_success=1, output_dir=output_dir, end_rollout_id=3
        )
        self.assertEqual(manifest["converted_count"], 3)
        categories = sorted(record["selection_category"] for record in manifest["records"])
        self.assertEqual(categories, ["fail", "fail", "success"])
        self.assertEqual(
            sorted(record["selection_task"] for record in manifest["records"]),
            ["t-a", "t-b", "t-c"],
        )
        on_disk = json.loads((output_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["selection"]["picked"], {"n_fail": 2, "n_success": 1})
        self.assertFalse(on_disk["selection"]["short"])

    def test_pulls_one_round_from_several_steps(self) -> None:
        root = self._root()
        _write_rollout(root, 1, [("t-old-traj-0", 0.0)])
        _write_rollout(root, 2, [("t-new-traj-0", 0.0)])
        output_dir = self._root() / "tb_job"
        manifest = select_evidence(
            root, n_fail=2, n_success=0, output_dir=output_dir, end_rollout_id=2
        )
        self.assertEqual(manifest["converted_count"], 2)
        self.assertEqual(
            sorted(record["selection_task"] for record in manifest["records"]),
            ["t-new", "t-old"],
        )

    def test_keeps_resumed_multi_trace_sessions(self) -> None:
        # A resumed session is one attempt split over several requests. It is
        # still evidence, and discover_polar_sessions drops it unless the
        # conversion is told to include it.
        root = self._root()
        _write_rollout(root, 1, [("t-a-traj-0", 0.0)], multi_trace=True)
        output_dir = self._root() / "tb_job"
        manifest = select_evidence(
            root, n_fail=1, n_success=0, output_dir=output_dir, end_rollout_id=1
        )
        self.assertEqual(manifest["converted_count"], 1)
        self.assertTrue(manifest["include_multi_trace"])

    def test_joins_durable_agent_artifacts_into_converted_trial(self) -> None:
        root = self._root()
        _write_rollout(root, 1, [("t-a-traj-0", 0.0)])
        session_id = "sk-000001-000"
        durable = root / "agent-artifacts" / session_id
        durable.mkdir(parents=True)
        (durable / "openai-messages.jsonl").write_text("{}\n", encoding="utf-8")
        (durable / "openai-tools.json").write_text("[]\n", encoding="utf-8")
        (durable / "verifier-summary.json").write_text(
            json.dumps({"status": "ok", "reward": "0.0", "failure_class": "verifier_failure"}),
            encoding="utf-8",
        )
        (durable / "capture-status.json").write_text(
            json.dumps({"captured": True, "normalized": True}), encoding="utf-8"
        )
        output_dir = self._root() / "tb_job"

        manifest = select_evidence(
            root, n_fail=1, n_success=0, output_dir=output_dir, end_rollout_id=1
        )

        agent_dir = output_dir / session_id / "agent"
        self.assertTrue((agent_dir / "openai-messages.jsonl").is_file())
        self.assertTrue((agent_dir / "openai-tools.json").is_file())
        status = json.loads((agent_dir / "artifact-status.json").read_text(encoding="utf-8"))
        self.assertTrue(status["complete"])
        self.assertIn("artifact-status.json", manifest["records"][0]["durable_artifacts"])

    def test_raises_when_nothing_is_usable(self) -> None:
        root = self._root()
        with self.assertRaises(RuntimeError):
            select_evidence(
                root, n_fail=1, n_success=0, output_dir=self._root() / "job", end_rollout_id=3
            )


if __name__ == "__main__":
    unittest.main()
