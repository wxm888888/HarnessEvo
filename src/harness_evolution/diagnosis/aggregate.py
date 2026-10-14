"""Helpers for loading per-trace diagnoses, indexing them for aggregation and rendering the aggregate report."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


_TRACE_DIAGNOSIS_FILENAME = "diagnosis.json"


_TRACE_DIAGNOSIS_PROTOCOL_VERSION = "file-v8-aggregate-relevance-filter"


_TRACE_DIAGNOSIS_REQUIRED_KEYS = {
    "terminal_cause",
    "first_divergence_step",
    "agent_mechanism",
    "harness_relevance",
    "success_patterns",
    "failure_patterns",
    "causal_chain",
    "evidence_refs",
    "confidence",
}


def _load_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_json_object_or_empty(path: Path) -> dict[str, Any]:
    value = _load_json_any(path)
    return value if isinstance(value, dict) else {}


def _load_trace_diagnosis(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"diagnosis model did not write {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"diagnosis model wrote invalid JSON to {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"diagnosis model output in {path.name} is not a JSON object")
    missing = sorted(_TRACE_DIAGNOSIS_REQUIRED_KEYS - value.keys())
    if missing:
        raise ValueError(
            f"diagnosis model output in {path.name} is missing required fields: "
            + ", ".join(missing)
        )
    return value


def _build_aggregate_index(
    diagnoses: list[dict[str, Any]], coverage: dict[str, Any]
) -> dict[str, Any]:
    """Build a lightweight index containing only successful diagnoses.

    The full diagnosis remains in its per-trace file. Keeping it out of the
    index gives the aggregator one representation of each diagnosis and keeps
    unavailable model outputs out of the aggregation input.
    """

    records: list[dict[str, Any]] = []
    metadata_fields = (
        "trace_id",
        "task_name",
        "trial_name",
        "outcome",
        "reward",
        "diagnosis_status",
        "diagnosis_path",
    )
    for diagnosis in diagnoses:
        if diagnosis.get("diagnosis_status") != "ok":
            continue
        if _diagnosis_harness_relevance_level(diagnosis) not in {"low", "medium", "high"}:
            continue
        records.append(
            {field: diagnosis.get(field) for field in metadata_fields}
        )
    return {
        "schema_version": "tb21-trace-diagnosis-index-v2",
        "records": records,
        "counts": coverage,
    }


def _copy_candidate_diagnosis_view(
    *,
    result_dir: Path,
    index: dict[str, Any],
    destination: Path,
) -> None:
    """Materialize only aggregate-eligible diagnoses for the candidate.

    The persistent result directory remains the complete audit record.  The
    candidate view deliberately omits raw analyzer logs, original trajectories,
    diagnoses excluded from the aggregate index, and the duplicate Markdown
    rendering of aggregate.json.
    """

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("index.json", "coverage.json", "aggregate.json"):
        source = result_dir / name
        if not source.is_file():
            raise ValueError(f"missing candidate diagnosis artifact: {source}")
        shutil.copy2(source, destination / name)

    per_trace_destination = destination / "per_trace"
    per_trace_destination.mkdir(parents=True, exist_ok=True)
    records = index.get("records")
    if not isinstance(records, list):
        raise TypeError("diagnosis index records must be an array")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("diagnosis index record must be an object")
        raw_path = record.get("diagnosis_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("diagnosis index record is missing diagnosis_path")
        relative_path = Path(raw_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 2
            or relative_path.parts[0] != "per_trace"
        ):
            raise ValueError(f"invalid diagnosis_path in index: {raw_path!r}")
        source = result_dir / relative_path
        if not source.is_file():
            raise ValueError(f"allowlisted diagnosis file is missing: {source}")
        target = destination / relative_path
        shutil.copy2(source, target)


def _diagnosis_harness_relevance_level(diagnosis_record: dict[str, Any]) -> str:
    diagnosis = diagnosis_record.get("diagnosis")
    if not isinstance(diagnosis, dict):
        return ""
    relevance = diagnosis.get("harness_relevance")
    if not isinstance(relevance, dict):
        return ""
    return str(relevance.get("level") or "").strip().lower()


def _apply_outcome_pattern_limits(
    diagnosis: dict[str, Any], *, outcome: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    limits = (
        {"success_patterns": 1, "failure_patterns": 3}
        if outcome == "failed"
        else {"success_patterns": 3, "failure_patterns": 1}
    )
    normalized = dict(diagnosis)
    counts: dict[str, dict[str, int]] = {}
    truncated = False
    for field, limit in limits.items():
        patterns = diagnosis.get(field)
        if not isinstance(patterns, list):
            raise ValueError(f"diagnosis field {field} must be an array")
        limited = patterns[:limit]
        normalized[field] = limited
        counts[field] = {"input": len(patterns), "kept": len(limited), "limit": limit}
        truncated = truncated or len(limited) != len(patterns)
    return normalized, {"outcome": outcome, "truncated": truncated, "counts": counts}


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8", errors="replace").splitlines())


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:180] or "trace"


def _relative_ref(path: str, evidence_root: Path) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(evidence_root.resolve()))
    except ValueError:
        return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_aggregate_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Aggregated terminal-task diagnosis",
        "",
        f"- aggregation status: `{aggregate.get('aggregation_status', 'unknown')}`",
        f"- confidence: `{aggregate.get('confidence', 0)}`",
        "",
    ]
    for title, key in (
        ("Shared harness patterns", "shared_patterns"),
        ("Success patterns", "success_patterns"),
        ("Task-specific patterns", "task_specific_patterns"),
        ("Harness change hypotheses", "harness_change_hypotheses"),
    ):
        lines.extend([f"## {title}", ""])
        values = aggregate.get(key) or []
        if not isinstance(values, list) or not values:
            lines.extend(["No items reported.", ""])
            continue
        for item in values:
            if isinstance(item, dict):
                summary = item.get("summary") or item.get("hypothesis") or item.get("description") or json.dumps(item, ensure_ascii=False)
                traces = item.get("supporting_traces") or item.get("traces") or []
                confidence = item.get("confidence")
                suffix = f" (confidence={confidence})" if confidence is not None else ""
                lines.append(f"- {summary}{suffix}")
                if traces:
                    lines.append(f"  - traces: {', '.join(map(str, traces))}")
            else:
                lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)
