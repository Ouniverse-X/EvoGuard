#!/usr/bin/env python3
"""Build EvoGuard evaluation rollouts from TraceSafe-Bench JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.envs.tracesafe_env import TRACE_SAFE_CATEGORIES, TraceSafeEnv
from evoguard.evaluation.metrics import compute_metrics
from evoguard.types import TrajectoryRecord


DEFAULT_TRACESAFE_ROOT = Path("data/raw/tracesafe_bench")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TraceSafe-Bench EvoGuard rollout files.")
    parser.add_argument("--tracesafe-root", default=str(DEFAULT_TRACESAFE_ROOT))
    parser.add_argument("--output-dir", default="data/eval")
    parser.add_argument("--round-id", type=int, default=400)
    args = parser.parse_args()

    tracesafe_root = Path(args.tracesafe_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(tracesafe_root.glob("golden_*.jsonl"), key=category_sort_key)
    if not paths:
        raise FileNotFoundError(
            f"No TraceSafe JSONL files found in {tracesafe_root}. "
            "Download the gated Hugging Face dataset CyCraftAI/TraceSafe first."
        )

    records_by_category: dict[str, list[TrajectoryRecord]] = {}
    all_records: list[TrajectoryRecord] = []
    for path in paths:
        env = TraceSafeEnv.from_paths([path])
        category = category_from_path(path)
        records = env.to_records(round_id=args.round_id, split="tracesafe_bench")
        records_by_category[category] = records
        all_records.extend(records)

    rollout_path = output_dir / "tracesafe_bench_rollouts.jsonl"
    write_rollouts(rollout_path, all_records)

    summary = {
        "source": "CyCraftAI/TraceSafe",
        "paper": "arXiv:2604.07223",
        "input_root": str(tracesafe_root),
        "rollouts": str(rollout_path),
        "records": len(all_records),
        "distribution": distribution(all_records),
        "metrics": compute_metrics(all_records),
        "categories": {
            category: {
                "records": len(records),
                "distribution": distribution(records),
                "metrics": compute_metrics(records),
                "source_file": str(next(path for path in paths if category_from_path(path) == category)),
            }
            for category, records in records_by_category.items()
        },
    }
    summary_path = output_dir / "tracesafe_bench_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def write_rollouts(path: Path, records: list[TrajectoryRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def distribution(records: list[TrajectoryRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.attack_type or 'unknown'}:{record.trajectory_type.value}:{record.gold_safety_action.value}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def category_from_path(path: Path) -> str:
    return TRACE_SAFE_CATEGORIES.get(path.stem, ("UNKNOWN", path.stem))[1]


def category_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        index = int(stem.split("_", 2)[1])
    except (IndexError, ValueError):
        index = 999
    return index, stem


if __name__ == "__main__":
    main()
