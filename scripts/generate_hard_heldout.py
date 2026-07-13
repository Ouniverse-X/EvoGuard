#!/usr/bin/env python3
"""Generate hard held-out EvoGuard attack and tri-rollout artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.metrics import compute_metrics
from evoguard.rollouts.tri_rollout import collect_tri_rollouts
from evoguard.types import TrajectoryType


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hard held-out EvoGuard rollouts.")
    parser.add_argument("--round-id", type=int, default=199)
    parser.add_argument("--attacks-output", default="data/eval/hard_heldout_attacks.jsonl")
    parser.add_argument("--rollouts-output", default="data/eval/hard_heldout_tri_rollouts.jsonl")
    parser.add_argument("--summary-output", default="data/eval/hard_heldout_summary.json")
    parser.add_argument("--min-attacks", type=int, default=50)
    args = parser.parse_args()

    env = TextToolEnv()
    agent = DefenseAgent()
    attack_generator = build_attack_generator("hard_heldout")

    attacks = attack_generator.generate(env.get_tasks(), round_id=args.round_id)
    if len(attacks) < args.min_attacks:
        raise RuntimeError(f"Expected at least {args.min_attacks} hard attacks, generated {len(attacks)}")

    rollouts = collect_tri_rollouts(agent, env, attack_generator, round_id=args.round_id)
    attacked_rollouts = [record for record in rollouts if record.trajectory_type == TrajectoryType.ATTACKED_TOOL]

    write_jsonl(Path(args.attacks_output), [asdict(attack) for attack in attacks])
    write_jsonl(Path(args.rollouts_output), [record.to_dict() for record in rollouts])

    summary = {
        "round_id": args.round_id,
        "split": "hard_heldout",
        "tasks": len(env.get_tasks()),
        "attacks": len(attacks),
        "tri_rollouts": len(rollouts),
        "attacked_rollouts": len(attacked_rollouts),
        "metrics": compute_metrics(rollouts),
        "outputs": {
            "attacks": args.attacks_output,
            "rollouts": args.rollouts_output,
        },
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
