#!/usr/bin/env python
"""Create a balanced EvoGuard safety SFT set from correct rollout records."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.training.llm_rl import read_rollout_jsonl
from evoguard.training.sft_dataset import write_safety_sft_jsonl
from evoguard.types import SafetyAction, TrajectoryRecord


def main() -> None:
    args = parse_args()
    records = read_rollout_jsonl(args.input_rollouts)
    by_action = group_by_action(records)
    targets = compute_targets(by_action, args.execute_min, args.ask_min, args.refuse_max)
    selected = stratified_sample(by_action, targets, seed=args.seed)
    random.Random(args.seed).shuffle(selected)
    write_safety_sft_jsonl(selected, args.output)

    summary = {
        "input_rollouts": str(args.input_rollouts),
        "output": str(args.output),
        "available_action_distribution": {action.value: len(by_action[action]) for action in SafetyAction},
        "target_action_distribution": {action.value: targets[action] for action in SafetyAction},
        "sampled_action_distribution": action_distribution(selected),
        "sampled_trajectory_distribution": trajectory_distribution(selected),
        "insufficient_classes": insufficient_classes(by_action, targets),
        "num_examples": len(selected),
        "seed": args.seed,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance EvoGuard safety SFT data.")
    parser.add_argument(
        "--input-rollouts",
        type=Path,
        default=Path("data/rollouts/safety_sft_correct_rollouts.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/safety_sft_balanced.jsonl"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/safety_sft_balanced_summary.json"),
    )
    parser.add_argument("--execute-min", type=int, default=120)
    parser.add_argument("--ask-min", type=int, default=40)
    parser.add_argument("--refuse-max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def group_by_action(records: list[TrajectoryRecord]) -> dict[SafetyAction, list[TrajectoryRecord]]:
    grouped: dict[SafetyAction, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.gold_safety_action].append(record)
    return {action: grouped[action] for action in SafetyAction}


def compute_targets(
    by_action: dict[SafetyAction, list[TrajectoryRecord]],
    execute_min: int,
    ask_min: int,
    refuse_max: int,
) -> dict[SafetyAction, int]:
    execute_target = min(len(by_action[SafetyAction.EXECUTE]), max(execute_min, len(by_action[SafetyAction.EXECUTE])))
    ask_target = min(len(by_action[SafetyAction.ASK_CONFIRMATION]), ask_min)
    refuse_target = min(len(by_action[SafetyAction.REFUSE]), refuse_max, execute_target)
    return {
        SafetyAction.EXECUTE: execute_target,
        SafetyAction.ASK_CONFIRMATION: ask_target,
        SafetyAction.REFUSE: refuse_target,
    }


def stratified_sample(
    by_action: dict[SafetyAction, list[TrajectoryRecord]],
    targets: dict[SafetyAction, int],
    seed: int,
) -> list[TrajectoryRecord]:
    rng = random.Random(seed)
    selected: list[TrajectoryRecord] = []
    for action in SafetyAction:
        candidates = list(by_action[action])
        rng.shuffle(candidates)
        selected.extend(candidates[: targets[action]])
    return selected


def action_distribution(records: list[TrajectoryRecord]) -> dict[str, int]:
    counts = Counter(record.gold_safety_action.value for record in records)
    return {action.value: counts.get(action.value, 0) for action in SafetyAction}


def trajectory_distribution(records: list[TrajectoryRecord]) -> dict[str, int]:
    counts = Counter(record.trajectory_type.value for record in records)
    return dict(sorted(counts.items()))


def insufficient_classes(
    by_action: dict[SafetyAction, list[TrajectoryRecord]],
    targets: dict[SafetyAction, int],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for action in SafetyAction:
        available = len(by_action[action])
        target = targets[action]
        if available < target:
            result[action.value] = {"available": available, "target": target}
    return result


if __name__ == "__main__":
    main()
