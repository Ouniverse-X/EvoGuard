#!/usr/bin/env python
"""Prepare supervised safety-judge training data from EvoGuard rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import PromptDrivenAttackGenerator
from evoguard.attacks.prompt_templates import TRAIN_ATTACK_TEMPLATES
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.rollouts.tri_rollout import collect_tri_rollouts
from evoguard.training.sft_dataset import write_safety_sft_jsonl
from evoguard.types import SafetyAction, TrajectoryRecord


def main() -> None:
    args = parse_args()
    env = TextToolEnv()
    agent = DefenseAgent()

    records: list[TrajectoryRecord] = []
    round_id = 0
    while len(records) < args.min_examples or round_id < args.rounds:
        generator = PromptDrivenAttackGenerator(
            templates=round_attack_templates(round_id),
            split=f"train_r{round_id}",
        )
        round_records = collect_tri_rollouts(agent, env, generator, round_id=round_id)
        if args.correct_only:
            round_records = [record for record in round_records if is_correct_safety_decision(record)]
        records.extend(round_records)
        round_id += 1

    records = records[: args.max_examples] if args.max_examples else records
    output_path = write_safety_sft_jsonl(records, args.output)
    write_rollout_jsonl(records, args.rollout_output)

    summary = {
        "output": str(output_path),
        "rollout_output": str(args.rollout_output),
        "rounds_generated": round_id,
        "num_examples": len(records),
        "correct_only": args.correct_only,
        "action_distribution": action_distribution(records),
        "trajectory_distribution": trajectory_distribution(records),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EvoGuard safety-judge SFT data.")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--min-examples", type=int, default=500)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--correct-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/safety_sft_train.jsonl"))
    parser.add_argument(
        "--rollout-output",
        type=Path,
        default=Path("data/rollouts/safety_sft_correct_rollouts.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/safety_sft_train_summary.json"),
    )
    return parser.parse_args()


def round_attack_templates(round_id: int) -> dict[str, str]:
    wrappers = (
        "{template}",
        "Round-specific memo: {template}",
        "Retrieved webpage fragment: {template}",
        "Shared document footnote: {template}",
        "Tool instruction appendix: {template}",
        "External ticket comment: {template}",
        "Archived user note: {template}",
    )
    wrapper = wrappers[round_id % len(wrappers)]
    templates: dict[str, str] = {}
    for attack_type, template in TRAIN_ATTACK_TEMPLATES.items():
        templates[f"{attack_type}_round_{round_id}"] = wrapper.format(template=template)
    return templates


def is_correct_safety_decision(record: TrajectoryRecord) -> bool:
    return record.agent_safety_action == record.gold_safety_action


def action_distribution(records: list[TrajectoryRecord]) -> dict[str, int]:
    return {action.value: sum(record.gold_safety_action == action for record in records) for action in SafetyAction}


def trajectory_distribution(records: list[TrajectoryRecord]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for record in records:
        distribution[record.trajectory_type.value] = distribution.get(record.trajectory_type.value, 0) + 1
    return distribution


def write_rollout_jsonl(records: list[TrajectoryRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
