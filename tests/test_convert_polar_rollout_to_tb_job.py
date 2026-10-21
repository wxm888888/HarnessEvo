from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path



from harness_evolution.traces.converters.polar_rollout import (  # noqa: E402
    convert_rollout,
    extract_instruction,
    merge_polar_trace_messages,
    polar_payload_to_atif,
)
from harness_evolution.traces.sanitize import copy_instruction_if_available  # noqa: E402


def _payload(
    *,
    session_id: str = "sk-polar-test",
    status: str = "COMPLETED",
    reward: float = 0.0,
) -> dict:
    prompt = [
        {"role": "system", "content": "You are opencode."},
        {"role": "user", "content": "Fix the failing test."},
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
        "task_id": "polar-task-1",
        "session_status": status,
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
                "task_id": "polar-task-1",
                "trace_index": 0,
                "finish_reason": "stop",
                "response_length": 42,
                "status": "completed",
                "prompt_messages": prompt,
                "response_messages": response,
            }
        ],
    }


class PolarConversionTest(unittest.TestCase):
    def test_atif_conversion_groups_tool_result_with_call(self) -> None:
        trajectory = polar_payload_to_atif(
            _payload(),
            source_file="source.json",
            rollout_name="rollout_000001",
            agent_name="opencode",
            agent_version="test",
            model_name="qwen-test",
        )

        self.assertEqual(trajectory["schema_version"], "ATIF-v1.6")
        self.assertEqual([step["source"] for step in trajectory["steps"]], ["system", "user", "agent", "agent"])
        tool_step = trajectory["steps"][2]
        self.assertEqual(tool_step["tool_calls"][0]["function_name"], "bash")
        self.assertEqual(tool_step["tool_calls"][0]["arguments"], {"command": "ls"})
        self.assertEqual(tool_step["observation"]["results"][0]["content"], "README.md")
        self.assertEqual(trajectory["final_metrics"]["total_completion_tokens"], 42)
        self.assertEqual(extract_instruction(_payload()), "Fix the failing test.")

    def test_continuation_prompt_is_not_duplicated(self) -> None:
        payload = _payload()
        first_trace = payload["traces"][0]
        first_trace["response_messages"] = [first_trace["response_messages"][0]]
        payload["traces"].append(
            {
                "session_id": payload["session_id"],
                "task_id": payload["task_id"],
                "trace_index": 1,
                "finish_reason": "stop",
                "response_length": 3,
                "status": "completed",
                "prompt_messages": first_trace["prompt_messages"] + first_trace["response_messages"],
                "response_messages": [{"role": "assistant", "content": "Final answer."}],
            }
        )
        payload["num_traces"] = 2

        merged = merge_polar_trace_messages(payload)

        self.assertEqual(len(merged), 4)
        self.assertEqual([entry["message"]["role"] for entry in merged], ["system", "user", "assistant", "assistant"])
        self.assertEqual(merged[-1]["trace_index"], 1)

    def test_rollout_conversion_filters_masked_and_writes_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "rollout_000001"
            output = root / "converted"
            source.mkdir()
            failure = _payload(session_id="failure", reward=0.0)
            failure.update(
                dataset_task_id="format-code-task-000588",
                dataset_task_file="/tasks/format-code-task-000588.json",
                dataset_task_sha256="abc123",
                dataset_mimo_dataset_type="opensource-code",
                dataset_image_tag="mimo:000588",
            )
            success = _payload(session_id="success", reward=1.0)
            error = _payload(session_id="error", status="ERROR", reward=0.0)
            error["result_error_type"] = "ContextLengthExceeded"
            error["result_error"] = "context length exceeded"
            truncated = _payload(session_id="truncated", reward=0.0)
            truncated["traces"][0]["status"] = "truncated"
            multi = _payload(session_id="multi", reward=1.0)
            multi["traces"].append(
                {
                    "session_id": multi["session_id"],
                    "task_id": multi["task_id"],
                    "trace_index": 1,
                    "finish_reason": "stop",
                    "response_length": 3,
                    "status": "completed",
                    "prompt_messages": multi["traces"][0]["prompt_messages"],
                    "response_messages": [{"role": "assistant", "content": "Alternative."}],
                }
            )
            multi["num_traces"] = 2
            masked = _payload(session_id="masked", status="MASKED", reward=0.0)
            masked["traces"][0]["prompt_messages"] = []
            masked["traces"][0]["response_messages"] = []
            for payload in (failure, success, error, truncated, multi, masked):
                (source / f"{payload['session_id']}.json").write_text(json.dumps(payload), encoding="utf-8")
            (source / "health.json").write_text("{}", encoding="utf-8")

            manifest = convert_rollout(source_dir=source, output_dir=output)

            self.assertEqual(manifest["converted_count"], 2)
            self.assertEqual(
                manifest["skipped_counts"],
                {
                    "health_summary": 1,
                    "masked": 1,
                    "multi_trace": 1,
                    "session_status_error": 1,
                    "trace_status_truncated": 1,
                },
            )
            for session_id in ("failure", "success"):
                self.assertTrue((output / session_id / "agent" / "trajectory.json").exists())
                self.assertTrue((output / session_id / "result.json").exists())
                self.assertTrue((output / session_id / "instruction.md").exists())
            self.assertFalse((output / "error").exists())
            self.assertFalse((output / "truncated").exists())
            self.assertFalse((output / "multi").exists())
            failure_record = next(
                record for record in manifest["records"] if record["session_id"] == "failure"
            )
            self.assertEqual(
                failure_record["dataset_task_file"], "/tasks/format-code-task-000588.json"
            )
            self.assertEqual(failure_record["dataset_task_sha256"], "abc123")
            self.assertEqual(failure_record["dataset_mimo_dataset_type"], "opensource-code")

            diagnostic_manifest = convert_rollout(
                source_dir=source,
                output_dir=root / "converted-with-errors",
                include_error_trajectories=True,
            )
            self.assertEqual(diagnostic_manifest["converted_count"], 4)
            self.assertEqual(diagnostic_manifest["converted_counts"]["execution_error"], 2)

            multi_manifest = convert_rollout(
                source_dir=source,
                output_dir=root / "converted-with-multi",
                include_multi_trace=True,
            )
            self.assertEqual(multi_manifest["converted_count"], 3)
            self.assertTrue((root / "converted-with-multi" / "multi").exists())

    def test_local_instruction_is_available_to_evidence_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "trial"
            target = root / "target"
            trial.mkdir()
            target.mkdir()
            result_path = trial / "result.json"
            result_path.write_text("{}", encoding="utf-8")
            (trial / "instruction.md").write_text("Local instruction\n", encoding="utf-8")

            copied = copy_instruction_if_available(
                result_path=result_path,
                task_cache_root=root / "unused-cache",
                target_dir=target,
            )

            self.assertEqual(copied, target / "instruction.md")
            self.assertEqual(copied.read_text(encoding="utf-8"), "Local instruction\n")


if __name__ == "__main__":
    unittest.main()
