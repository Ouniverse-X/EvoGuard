#!/usr/bin/env python
"""Prepare train/eval/history JSONL files for EvoGuard self-evolution rounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.training.llm_rl import read_rollout_jsonl
from evoguard.types import SafetyAction, TrajectoryRecord, TrajectoryType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--attacks", required=True)
    train.add_argument("--benign", default="data/rollouts/toolsafe_train_tri_rollouts.jsonl")
    train.add_argument("--output", required=True)
    train.add_argument("--max-benign", type=int, default=165)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--attacks", required=True)
    eval_parser.add_argument("--benign", default="data/eval/toolsafe_heldout_tri_rollouts.jsonl")
    eval_parser.add_argument("--output", required=True)

    history = subparsers.add_parser("history")
    history.add_argument("--evaluated", required=True)
    history.add_argument("--success-output", required=True)
    history.add_argument("--blocked-output", required=True)
    history.add_argument("--low-confidence-threshold", type=float, default=0.75)
    history.add_argument("--max-cases", type=int, default=24)

    args = parser.parse_args()
    if args.command == "train":
        attacks = read_rollout_jsonl(Path(args.attacks))
        benign = _benign_records(read_rollout_jsonl(Path(args.benign)))[: args.max_benign]
        _write_jsonl(Path(args.output), attacks + benign)
        print(json.dumps({"attacks": len(attacks), "benign": len(benign), "output": args.output}, indent=2))
    elif args.command == "eval":
        attacks = read_rollout_jsonl(Path(args.attacks))
        benign = _benign_records(read_rollout_jsonl(Path(args.benign)))
        _write_jsonl(Path(args.output), attacks + benign)
        print(json.dumps({"attacks": len(attacks), "benign": len(benign), "output": args.output}, indent=2))
    elif args.command == "history":
        records = read_rollout_jsonl(Path(args.evaluated))
        successful = [
            record
            for record in records
            if record.trajectory_type == TrajectoryType.ATTACKED_TOOL and record.attack_success
        ][: args.max_cases]
        blocked = [
            record
            for record in records
            if record.trajectory_type == TrajectoryType.ATTACKED_TOOL
            and not record.attack_success
            and record.confidence <= args.low_confidence_threshold
        ][: args.max_cases]
        if not blocked:
            blocked = [
                record
                for record in records
                if record.trajectory_type == TrajectoryType.ATTACKED_TOOL and not record.attack_success
            ][: args.max_cases]
        _write_jsonl(Path(args.success_output), successful)
        _write_jsonl(Path(args.blocked_output), blocked)
        print(
            json.dumps(
                {
                    "success_cases": len(successful),
                    "blocked_or_low_confidence_cases": len(blocked),
                    "success_output": args.success_output,
                    "blocked_output": args.blocked_output,
                },
                indent=2,
            )
        )


def _benign_records(records: list[TrajectoryRecord]) -> list[TrajectoryRecord]:
    return [
        record
        for record in records
        if record.trajectory_type == TrajectoryType.CLEAN_TOOL and record.gold_safety_action == SafetyAction.EXECUTE
    ]


def _write_jsonl(path: Path, records: list[TrajectoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
