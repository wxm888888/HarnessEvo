from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATIF_SCHEMA_VERSION = "ATIF-v1.6"
DEFAULT_SUCCESS_THRESHOLD = 1.0


@dataclass(frozen=True)
class PolarSession:
    source_path: Path
    payload: dict[str, Any]
    category: str


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _text(value: object, *, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_name(value: object, *, fallback: str) -> str:
    text = str(value or fallback).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return safe or fallback


def _trace_sort_key(trace: dict[str, Any], ordinal: int) -> tuple[int, int]:
    trace_index = _integer(trace.get("trace_index"))
    return (trace_index if trace_index is not None else ordinal, ordinal)


def merge_polar_trace_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge Polar trace segments without repeating cumulative prompt prefixes."""
    raw_traces = payload.get("traces")
    traces = [trace for trace in raw_traces if isinstance(trace, dict)] if isinstance(raw_traces, list) else []
    indexed = sorted(enumerate(traces), key=lambda item: _trace_sort_key(item[1], item[0]))
    merged: list[dict[str, Any]] = []
    consumed_messages = 0

    for ordinal, trace in indexed:
        trace_index = _integer(trace.get("trace_index"))
        if trace_index is None:
            trace_index = ordinal
        prompt_messages = trace.get("prompt_messages")
        response_messages = trace.get("response_messages")
        prompt = [message for message in prompt_messages if isinstance(message, dict)] if isinstance(prompt_messages, list) else []
        response = [message for message in response_messages if isinstance(message, dict)] if isinstance(response_messages, list) else []

        prompt_start = min(consumed_messages, len(prompt))
        for message_index, message in enumerate(prompt[prompt_start:], start=prompt_start):
            merged.append(
                {
                    "message": message,
                    "trace_index": trace_index,
                    "message_index": message_index,
                    "phase": "prompt" if consumed_messages == 0 else "prompt_delta",
                    "trace_status": trace.get("status"),
                    "finish_reason": trace.get("finish_reason"),
                }
            )
        consumed_messages = max(consumed_messages, len(prompt))

        for response_index, message in enumerate(response):
            merged.append(
                {
                    "message": message,
                    "trace_index": trace_index,
                    "message_index": response_index,
                    "phase": "response",
                    "trace_status": trace.get("status"),
                    "finish_reason": trace.get("finish_reason"),
                }
            )
        consumed_messages = max(consumed_messages, len(prompt)) + len(response)

    return merged


def _tool_call(call: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    call_id = str(call.get("id") or call.get("tool_call_id") or fallback_id)
    function_name = str(function.get("name") or call.get("function_name") or "unknown")
    raw_arguments = function.get("arguments", call.get("arguments", {}))
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"raw": raw_arguments}
    else:
        arguments = raw_arguments
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {
        "tool_call_id": call_id,
        "function_name": function_name,
        "arguments": arguments,
    }


def polar_payload_to_atif(
    payload: dict[str, Any],
    *,
    source_file: str,
    rollout_name: str,
    agent_name: str,
    agent_version: str,
    model_name: str | None,
) -> dict[str, Any]:
    merged_messages = merge_polar_trace_messages(payload)
    steps: list[dict[str, Any]] = []
    call_to_step: dict[str, int] = {}
    last_agent_step: int | None = None

    for entry in merged_messages:
        message = entry["message"]
        role = str(message.get("role") or "system").lower()
        step_extra = {
            "polar_trace_index": entry["trace_index"],
            "polar_message_index": entry["message_index"],
            "polar_phase": entry["phase"],
            "polar_trace_status": entry["trace_status"],
            "polar_finish_reason": entry["finish_reason"],
        }

        if role == "tool":
            source_call_id = message.get("tool_call_id")
            target_index = call_to_step.get(str(source_call_id)) if source_call_id is not None else None
            if target_index is None:
                target_index = last_agent_step
            observation = {
                "source_call_id": str(source_call_id) if source_call_id is not None else None,
                "content": _text(message.get("content")),
            }
            if target_index is not None:
                steps[target_index].setdefault("observation", {"results": []})["results"].append(observation)
                steps[target_index].setdefault("extra", {}).setdefault("polar_tool_messages", []).append(step_extra)
                continue

            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "source": "system",
                    "message": f"Unmatched tool result: {_text(message.get('content'))}",
                    "observation": {"results": [observation]},
                    "extra": {**step_extra, "polar_original_role": role},
                }
            )
            continue

        if role == "assistant":
            raw_calls = message.get("tool_calls")
            calls = [call for call in raw_calls if isinstance(call, dict)] if isinstance(raw_calls, list) else []
            converted_calls = [
                _tool_call(
                    call,
                    fallback_id=f"polar-call-{entry['trace_index']}-{entry['message_index']}-{call_index}",
                )
                for call_index, call in enumerate(calls)
            ]
            step: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": _text(
                    message.get("content"),
                    fallback="(tool use)" if converted_calls else "(no textual response)",
                ),
                "extra": step_extra,
            }
            reasoning = _text(message.get("reasoning_content"))
            if reasoning:
                step["reasoning_content"] = reasoning
            if converted_calls:
                step["tool_calls"] = converted_calls
            if model_name:
                step["model_name"] = model_name
            steps.append(step)
            last_agent_step = len(steps) - 1
            for call in converted_calls:
                call_to_step[call["tool_call_id"]] = last_agent_step
            continue

        source = role if role in {"system", "user"} else "system"
        if role not in {"system", "user"}:
            step_extra["polar_original_role"] = role
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": source,
                "message": _text(message.get("content"), fallback="(empty message)"),
                "extra": step_extra,
            }
        )

    if not steps:
        steps.append(
            {
                "step_id": 1,
                "source": "system",
                "message": "Polar session contained no readable messages.",
                "extra": {"polar_empty_session": True},
            }
        )

    result_error_info = payload.get("result_error_info")
    if not isinstance(result_error_info, dict):
        result_error_info = {}
    total_assistant_tokens = _integer(payload.get("total_assistant_tokens"))
    agent: dict[str, Any] = {"name": agent_name, "version": agent_version}
    if model_name:
        agent["model_name"] = model_name
    final_metrics: dict[str, Any] = {
        "total_steps": len(steps),
        "extra": {
            "polar_num_traces": _integer(payload.get("num_traces")),
            "polar_session_reward": _number(payload.get("session_reward")),
        },
    }
    if total_assistant_tokens is not None:
        final_metrics["total_completion_tokens"] = total_assistant_tokens

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": str(payload.get("session_id") or Path(source_file).stem),
        "agent": agent,
        "steps": steps,
        "notes": (
            "Converted from a Polar RL session. Cumulative prompts from continuation traces "
            "were de-duplicated; trace provenance is retained in step.extra."
        ),
        "final_metrics": final_metrics,
        "extra": {
            "source_schema": "polar_rl_session_v1",
            "source_file": source_file,
            "source_rollout": rollout_name,
            "task_id": payload.get("task_id"),
            "session_status": payload.get("session_status"),
            "trajectory_status": payload.get("trajectory_status"),
            "trajectory_error": payload.get("trajectory_error"),
            "result_error": payload.get("result_error"),
            "result_error_type": payload.get("result_error_type") or result_error_info.get("type"),
            "result_error_stage": payload.get("result_error_stage") or result_error_info.get("stage"),
            "timing": payload.get("timing"),
            "num_traces": _integer(payload.get("num_traces")),
        },
    }


def polar_payload_to_result(
    payload: dict[str, Any],
    *,
    source_file: str,
    rollout_name: str,
    source_label: str,
    agent_name: str,
    agent_version: str,
    model_name: str | None,
) -> dict[str, Any]:
    session_id = str(payload.get("session_id") or Path(source_file).stem)
    task_id = str(payload.get("task_id") or session_id)
    error_info = payload.get("result_error_info")
    if not isinstance(error_info, dict):
        error_info = {}
    error_type = payload.get("result_error_type") or error_info.get("type")
    error_message = payload.get("result_error") or error_info.get("message") or payload.get("trajectory_error")
    exception_info = None
    if error_type or error_message:
        exception_info = {
            "exception_type": error_type,
            "exception_message": error_message,
        }
    output_tokens = _integer(payload.get("total_assistant_tokens"))
    agent_info: dict[str, Any] = {"name": agent_name, "version": agent_version}
    if model_name:
        agent_info["model_name"] = model_name

    return {
        "id": session_id,
        "task_name": task_id,
        "trial_name": session_id,
        "trial_uri": None,
        "task_id": {"name": task_id, "ref": None},
        "source": source_label,
        "task_checksum": None,
        "config": {},
        "agent_info": agent_info,
        "agent_result": {
            "n_input_tokens": None,
            "n_cache_tokens": None,
            "n_output_tokens": output_tokens,
            "n_total_tokens": None,
            "cost_usd": None,
        },
        "verifier_result": {"rewards": {"reward": _number(payload.get("session_reward"))}},
        "exception_info": exception_info,
        "started_at": None,
        "finished_at": None,
        "polar_metadata": {
            "source_file": source_file,
            "source_rollout": rollout_name,
            "session_status": payload.get("session_status"),
            "trajectory_status": payload.get("trajectory_status"),
            "result_error_stage": payload.get("result_error_stage") or error_info.get("stage"),
            "timing": payload.get("timing"),
            "num_traces": _integer(payload.get("num_traces")),
            # Which concrete harbor task/image this session's rollout actually
            # used (from the dataset row's metadata, distinct from the
            # position-only "polar-tb-harbor-{rollout}-{group}" task_id
            # above). Lets the evolution pipeline point the held-out gate at
            # the exact tasks that produced this evidence instead of a fixed
            # canary set. None for trajectories dumped before this field
            # existed.
            "dataset_task_id": payload.get("dataset_task_id"),
            "dataset_task_dir": payload.get("dataset_task_dir"),
            "dataset_image_tag": payload.get("dataset_image_tag"),
        },
    }


def extract_instruction(payload: dict[str, Any]) -> str:
    for entry in merge_polar_trace_messages(payload):
        message = entry["message"]
        if str(message.get("role") or "").lower() == "user":
            return _text(message.get("content")).strip()
    return ""


def _has_messages(payload: dict[str, Any]) -> bool:
    traces = payload.get("traces")
    if not isinstance(traces, list):
        return False
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        prompt = trace.get("prompt_messages")
        response = trace.get("response_messages")
        if (isinstance(prompt, list) and prompt) or (isinstance(response, list) and response):
            return True
    return False


def _clean_completion_issue(payload: dict[str, Any]) -> str | None:
    status = str(payload.get("session_status") or "UNKNOWN").upper()
    if status != "COMPLETED":
        return f"session_status_{status.lower()}"
    error_info = payload.get("result_error_info")
    error_info_values = (
        (
            error_info.get("type"),
            error_info.get("stage"),
            error_info.get("message"),
        )
        if isinstance(error_info, dict)
        else ()
    )
    if any(
        (
            payload.get("result_error"),
            payload.get("result_error_type"),
            payload.get("result_error_stage"),
            payload.get("trajectory_error"),
            *error_info_values,
        )
    ):
        return "error_metadata"
    trajectory_status = str(payload.get("trajectory_status") or "COMPLETED").upper()
    if trajectory_status != "COMPLETED":
        return f"trajectory_status_{trajectory_status.lower()}"
    traces = payload.get("traces")
    if not isinstance(traces, list) or not traces:
        return "missing_traces"
    for trace in traces:
        if not isinstance(trace, dict):
            return "invalid_trace"
        trace_status = str(trace.get("status") or "UNKNOWN").lower()
        if trace_status != "completed":
            return f"trace_status_{trace_status}"
    return None


def _category(payload: dict[str, Any], success_threshold: float) -> str:
    if _clean_completion_issue(payload) is not None:
        return "execution_error"
    reward = _number(payload.get("session_reward"))
    if reward is not None and reward >= success_threshold:
        return "success"
    return "completed_failure"


def discover_polar_sessions(
    source_dir: Path,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    include_error_trajectories: bool = False,
    include_multi_trace: bool = False,
) -> tuple[list[PolarSession], Counter[str]]:
    if not source_dir.is_dir():
        raise ValueError(f"Polar rollout directory does not exist: {source_dir}")
    sessions: list[PolarSession] = []
    skipped: Counter[str] = Counter()
    for source_path in sorted(source_dir.glob("*.json")):
        if source_path.name == "health.json":
            skipped["health_summary"] += 1
            continue
        payload = _load_json_object(source_path)
        if "session_id" not in payload or not isinstance(payload.get("traces"), list):
            skipped["not_polar_session"] += 1
            continue
        if str(payload.get("session_status") or "").upper() == "MASKED":
            skipped["masked"] += 1
            continue
        if len(payload["traces"]) > 1 and not include_multi_trace:
            skipped["multi_trace"] += 1
            continue
        if not _has_messages(payload):
            skipped["empty_messages"] += 1
            continue
        completion_issue = _clean_completion_issue(payload)
        if completion_issue is not None and not include_error_trajectories:
            skipped[completion_issue] += 1
            continue
        sessions.append(
            PolarSession(
                source_path=source_path,
                payload=payload,
                category=_category(payload, success_threshold),
            )
        )
    return sessions, skipped


def _limit_sessions(
    sessions: list[PolarSession],
    limits: dict[str, int | None],
) -> tuple[list[PolarSession], Counter[str]]:
    grouped: dict[str, list[PolarSession]] = defaultdict(list)
    for session in sessions:
        grouped[session.category].append(session)
    selected: list[PolarSession] = []
    skipped: Counter[str] = Counter()
    for category in ("completed_failure", "execution_error", "success"):
        category_sessions = sorted(grouped.get(category, []), key=lambda item: item.source_path.name)
        limit = limits.get(category)
        selected.extend(category_sessions if limit is None else category_sessions[:limit])
        if limit is not None and len(category_sessions) > limit:
            skipped[f"limited_{category}"] += len(category_sessions) - limit
    return selected, skipped


def convert_rollout(
    *,
    source_dir: Path,
    output_dir: Path,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    max_successes: int | None = None,
    max_completed_failures: int | None = None,
    max_errors: int | None = None,
    include_error_trajectories: bool = False,
    include_multi_trace: bool = False,
    source_label: str = "polar_rl",
    agent_name: str = "opencode",
    agent_version: str = "unknown",
    model_name: str | None = None,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty or absent: {output_dir}")

    sessions, skipped = discover_polar_sessions(
        source_dir,
        success_threshold=success_threshold,
        include_error_trajectories=include_error_trajectories,
        include_multi_trace=include_multi_trace,
    )
    selected, limited = _limit_sessions(
        sessions,
        {
            "success": max_successes,
            "completed_failure": max_completed_failures,
            "execution_error": max_errors,
        },
    )
    skipped.update(limited)
    if not selected:
        raise ValueError(f"no usable Polar sessions found under {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    used_trial_names: set[str] = set()
    for session in selected:
        payload = session.payload
        session_id = str(payload.get("session_id") or session.source_path.stem)
        trial_name = _safe_name(session_id, fallback=session.source_path.stem)
        if trial_name in used_trial_names:
            raise ValueError(f"duplicate converted trial name: {trial_name}")
        used_trial_names.add(trial_name)
        trial_dir = output_dir / trial_name
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        result_path = trial_dir / "result.json"
        instruction_path = trial_dir / "instruction.md"

        trajectory = polar_payload_to_atif(
            payload,
            source_file=session.source_path.name,
            rollout_name=source_dir.name,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=model_name,
        )
        result = polar_payload_to_result(
            payload,
            source_file=session.source_path.name,
            rollout_name=source_dir.name,
            source_label=source_label,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=model_name,
        )
        instruction = extract_instruction(payload)
        _write_json(trajectory_path, trajectory)
        _write_json(result_path, result)
        if instruction:
            instruction_path.write_text(instruction.rstrip() + "\n", encoding="utf-8")

        records.append(
            {
                "session_id": session_id,
                "dataset_task_id": payload.get("dataset_task_id"),
                "task_id": payload.get("task_id"),
                "category": session.category,
                "reward": _number(payload.get("session_reward")),
                "source_file": session.source_path.name,
                "trial_dir": trial_name,
                "trajectory_steps": len(trajectory["steps"]),
                "instruction_written": bool(instruction),
            }
        )

    converted_counts = Counter(record["category"] for record in records)
    manifest = {
        "schema_version": "polar-to-terminal-bench-conversion-v1",
        "source_rollout_dir": str(source_dir),
        "output_job_dir": str(output_dir),
        "success_threshold": success_threshold,
        "include_error_trajectories": include_error_trajectories,
        "include_multi_trace": include_multi_trace,
        "converted_count": len(records),
        "converted_counts": dict(sorted(converted_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "records": records,
    }
    _write_json(output_dir / "conversion_manifest.json", manifest)
    return manifest


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert readable Polar RL session dumps into a Terminal-Bench/Harbor-style "
            "job directory with ATIF-v1.6 trajectories."
        )
    )
    parser.add_argument("source_dir", help="Polar rollout directory containing session JSON files.")
    parser.add_argument("--output-dir", required=True, help="Empty/new destination job directory.")
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--max-successes", type=_non_negative_int)
    parser.add_argument("--max-completed-failures", type=_non_negative_int)
    parser.add_argument("--max-errors", type=_non_negative_int)
    parser.add_argument(
        "--include-error-trajectories",
        action="store_true",
        help=(
            "Include non-completed, truncated, or error-bearing sessions. "
            "By default only clean completed trajectories are converted."
        ),
    )
    parser.add_argument(
        "--include-multi-trace",
        action="store_true",
        help=(
            "Include sessions containing multiple traces. By default they are skipped "
            "because Polar traces may represent branches rather than linear continuation."
        ),
    )
    parser.add_argument("--source-label", default="polar_rl")
    parser.add_argument("--agent-name", default="opencode")
    parser.add_argument("--agent-version", default="unknown")
    parser.add_argument("--model-name", help="Optional model name recorded in ATIF metadata.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = convert_rollout(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        success_threshold=float(args.success_threshold),
        max_successes=args.max_successes,
        max_completed_failures=args.max_completed_failures,
        max_errors=args.max_errors,
        include_error_trajectories=args.include_error_trajectories,
        include_multi_trace=args.include_multi_trace,
        source_label=args.source_label,
        agent_name=args.agent_name,
        agent_version=args.agent_version,
        model_name=args.model_name,
    )
    print(f"converted_job_dir: {manifest['output_job_dir']}")
    print(f"converted_count: {manifest['converted_count']}")
    print(f"converted_counts: {json.dumps(manifest['converted_counts'], sort_keys=True)}")
    print(f"skipped_counts: {json.dumps(manifest['skipped_counts'], sort_keys=True)}")
    print(f"manifest_path: {Path(manifest['output_job_dir']) / 'conversion_manifest.json'}")


if __name__ == "__main__":
    main()
