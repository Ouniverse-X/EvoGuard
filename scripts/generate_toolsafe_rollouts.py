#!/usr/bin/env python3
"""Convert local ToolSafe/TS-Bench JSON or JSONL files into EvoGuard rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.envs.toolsafe_env import ToolSafeEnv
from evoguard.evaluation.metrics import compute_metrics
from evoguard.types import TrajectoryRecord, TrajectoryType


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EvoGuard train/held-out rollouts from ToolSafe files.")
    parser.add_argument("inputs", nargs="+", help="ToolSafe/TS-Bench JSON or JSONL files.")
    parser.add_argument("--train-output", default="data/rollouts/toolsafe_train_tri_rollouts.jsonl")
    parser.add_argument("--heldout-output", default="data/eval/toolsafe_heldout_tri_rollouts.jsonl")
    parser.add_argument("--summary-output", default="data/eval/toolsafe_rollout_summary.json")
    parser.add_argument("--heldout-ratio", type=float, default=0.2)
    parser.add_argument("--round-id", type=int, default=300)
    args = parser.parse_args()

    env = ToolSafeEnv.from_paths(args.inputs)
    train_records, heldout_records = env.split_records(heldout_ratio=args.heldout_ratio, round_id=args.round_id)

    write_records(Path(args.train_output), train_records)
    write_records(Path(args.heldout_output), heldout_records)

    summary = {
        "environment": env.name,
        "inputs": args.inputs,
        "round_id": args.round_id,
        "total_records": len(train_records) + len(heldout_records),
        "train_records": len(train_records),
        "heldout_records": len(heldout_records),
        "train_distribution": trajectory_distribution(train_records),
        "heldout_distribution": trajectory_distribution(heldout_records),
        "train_metrics": compute_metrics(train_records),
        "heldout_metrics": compute_metrics(heldout_records),
        "outputs": {
            "train": args.train_output,
            "heldout": args.heldout_output,
        },
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def write_records(path: Path, records: list[TrajectoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def trajectory_distribution(records: list[TrajectoryRecord]) -> dict[str, int]:
    return {trajectory_type.value: sum(record.trajectory_type == trajectory_type for record in records) for trajectory_type in TrajectoryType}


if __name__ == "__main__":
    main()
