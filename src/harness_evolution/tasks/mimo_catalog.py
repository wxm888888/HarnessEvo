"""Build a held-in catalog from MiMo tasks and their k-attempt screening.

The screening run (Qwen3.6 base, k=8 per task) writes one task id per line to
``success_<c>_of_<k>.txt``.  Tasks that are always solved or never solved carry
little signal for harness evolution, so the default keeps 1..k-1 successes.

    python -m harness_evolution.tasks.mimo_catalog \\
        --mimo-jsonl   <screen>/inputs/mimo.jsonl \\
        --screening    <screen>/screening/mimo \\
        --quarantine   <screen>/screening/env_quality/quarantine_registry.json \\
        --output       mimo_heldin.jsonl

The output keeps the original rows (prompt + metadata with ``task_file``) and
adds ``screen_success_count`` / ``screen_k``; a ``.summary.json`` sits next to it.

``--gate-size N --gate-output gate.json`` also draws a fixed gate split for the
loop's ``held_out``: N tasks stratified by (family, success count) within
``--gate-min-success..--gate-max-success``.  They are removed from the held-in
output, and ``gate.jsonl`` holds their catalog rows for ``task_catalogs``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SUCCESS_FILE = re.compile(r"success_(\d+)_of_(\d+)\.txt")


def read_screening(directory: Path) -> tuple[dict[str, int], int]:
    """task_id -> number of successes, and k."""
    counts: dict[str, int] = {}
    ks: set[int] = set()
    for path in sorted(directory.glob("success_*_of_*.txt")):
        match = _SUCCESS_FILE.fullmatch(path.name)
        if not match:
            continue
        successes, k = int(match.group(1)), int(match.group(2))
        ks.add(k)
        for line in path.read_text(encoding="utf-8").splitlines():
            task_id = line.strip()
            if not task_id:
                continue
            if task_id in counts:
                raise ValueError(f"{task_id} appears in more than one screening list")
            counts[task_id] = successes
    if len(ks) != 1:
        raise ValueError(f"expected one k across screening lists, found {sorted(ks)}")
    return counts, ks.pop()


def quarantined_ids(registry: Path | None, family: str = "mimo") -> set[str]:
    if registry is None:
        return set()
    data = json.loads(registry.read_text(encoding="utf-8"))
    prefix = f"{family}/"
    return {
        key[len(prefix):]
        for key, value in data.items()
        if key.startswith(prefix) and isinstance(value, dict) and value.get("status") == "quarantined"
    }


def build_catalog(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    k: int,
    *,
    min_success: int,
    max_success: int,
    families: set[str] | None,
    exclude: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for row in rows:
        meta = row.get("metadata") or {}
        task_id = str(meta.get("task_id") or "")
        if task_id not in counts:
            dropped["not_screened_or_incomplete"] += 1
            continue
        if task_id in exclude:
            dropped["quarantined"] += 1
            continue
        if families and meta.get("mimo_dataset_type") not in families:
            dropped["family"] += 1
            continue
        successes = counts[task_id]
        if not min_success <= successes <= max_success:
            dropped[f"success_{successes}_of_{k}"] += 1
            continue
        task_file = Path(str(meta.get("task_file") or ""))
        if not task_file.is_file():
            raise FileNotFoundError(f"{task_id}: task_file missing: {task_file}")
        expected = str(meta.get("task_sha256") or "")
        if expected and hashlib.sha256(task_file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{task_id}: task_file does not match task_sha256")
        out = json.loads(json.dumps(row))
        out["metadata"]["screen_success_count"] = successes
        out["metadata"]["screen_k"] = k
        kept.append(out)
    summary = {
        "n_input": len(rows),
        "n_kept": len(kept),
        "k": k,
        "success_range": [min_success, max_success],
        "families": sorted(families) if families else "all",
        "kept_by_family": dict(Counter(r["metadata"].get("mimo_dataset_type") for r in kept)),
        "kept_by_success": dict(sorted(Counter(r["metadata"]["screen_success_count"] for r in kept).items())),
        "dropped": dict(dropped),
    }
    return kept, summary


def select_gate(
    rows: list[dict[str, Any]],
    size: int,
    *,
    seed: int,
    min_success: int,
    max_success: int,
) -> list[dict[str, Any]]:
    """Draw ``size`` rows, allocated to (family, success) strata in proportion."""
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row["metadata"]
        if min_success <= meta["screen_success_count"] <= max_success:
            strata[(str(meta.get("mimo_dataset_type")), meta["screen_success_count"])].append(row)
    eligible = sum(len(items) for items in strata.values())
    if size > eligible:
        raise ValueError(f"gate size {size} exceeds {eligible} eligible tasks")
    keys = sorted(strata)
    # Largest-remainder allocation keeps the gate's mix equal to the pool's.
    quotas = {key: size * len(strata[key]) / eligible for key in keys}
    counts = {key: int(quotas[key]) for key in keys}
    by_remainder = sorted(keys, key=lambda key: (counts[key] - quotas[key], key))
    for key in by_remainder[: size - sum(counts.values())]:
        counts[key] += 1
    rng = random.Random(seed)
    gate: list[dict[str, Any]] = []
    for key in keys:
        items = sorted(strata[key], key=lambda row: row["metadata"]["task_id"])
        rng.shuffle(items)
        gate.extend(items[: counts[key]])
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mimo-jsonl", type=Path, required=True)
    parser.add_argument("--screening", type=Path, required=True, help="directory with success_<c>_of_<k>.txt")
    parser.add_argument("--quarantine", type=Path, default=None, help="env_quality quarantine_registry.json")
    parser.add_argument("--min-success", type=int, default=1)
    parser.add_argument("--max-success", type=int, default=None, help="default: k-1")
    parser.add_argument("--families", default="", help="comma list, e.g. opensource-code,terminal_bench")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-size", type=int, default=0, help="tasks in a fixed gate split (0: none)")
    parser.add_argument("--gate-output", type=Path, default=None, help="gate split JSON; rows go to <stem>.jsonl")
    parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument("--gate-min-success", type=int, default=2)
    parser.add_argument("--gate-max-success", type=int, default=None, help="default: k-2")
    args = parser.parse_args(argv)
    if args.gate_size and args.gate_output is None:
        parser.error("--gate-size needs --gate-output")

    rows = [json.loads(line) for line in args.mimo_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts, k = read_screening(args.screening)
    max_success = k - 1 if args.max_success is None else args.max_success
    families = {item.strip() for item in args.families.split(",") if item.strip()} or None
    kept, summary = build_catalog(
        rows,
        counts,
        k,
        min_success=args.min_success,
        max_success=max_success,
        families=families,
        exclude=quarantined_ids(args.quarantine),
    )
    summary["sources"] = {
        "mimo_jsonl": str(args.mimo_jsonl),
        "screening": str(args.screening),
        "quarantine": str(args.quarantine) if args.quarantine else None,
    }
    if args.gate_size:
        gate_max = k - 2 if args.gate_max_success is None else args.gate_max_success
        gate = select_gate(
            kept, args.gate_size, seed=args.gate_seed, min_success=args.gate_min_success, max_success=gate_max
        )
        gate_ids = {row["metadata"]["task_id"] for row in gate}
        kept = [row for row in kept if row["metadata"]["task_id"] not in gate_ids]
        gate_rows = args.gate_output.with_suffix(".jsonl")
        args.gate_output.parent.mkdir(parents=True, exist_ok=True)
        args.gate_output.write_text(
            json.dumps([row["metadata"]["task_id"] for row in gate], indent=2) + "\n", encoding="utf-8"
        )
        gate_rows.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gate), encoding="utf-8")
        summary["n_kept"] = len(kept)
        summary["kept_by_family"] = dict(Counter(r["metadata"].get("mimo_dataset_type") for r in kept))
        summary["kept_by_success"] = dict(sorted(Counter(r["metadata"]["screen_success_count"] for r in kept).items()))
        summary["gate"] = {
            "split": str(args.gate_output),
            "catalog": str(gate_rows),
            "size": len(gate),
            "seed": args.gate_seed,
            "success_range": [args.gate_min_success, gate_max],
            "by_family": dict(Counter(r["metadata"].get("mimo_dataset_type") for r in gate)),
            "by_success": dict(sorted(Counter(r["metadata"]["screen_success_count"] for r in gate).items())),
            "split_sha256": hashlib.sha256(args.gate_output.read_bytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    summary["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
