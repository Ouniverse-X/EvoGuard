#!/usr/bin/env python
"""Export small reproducible EvoGuard benchmark artifacts under data/."""

from __future__ import annotations

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
from evoguard.training.sft_dataset import write_safety_sft_jsonl


def main() -> None:
    env = TextToolEnv()
    agent = DefenseAgent()

    train_attacks = build_attack_generator("train").generate(env.get_tasks(), round_id=0)
    heldout_attacks = build_attack_generator("heldout").generate(env.get_tasks(), round_id=99)
    train_rollouts = collect_tri_rollouts(agent, env, build_attack_generator("train"), round_id=0)
    heldout_rollouts = collect_tri_rollouts(agent, env, build_attack_generator("heldout"), round_id=99)

    _write_jsonl(Path("data/attacks/train_attacks_round0.jsonl"), [asdict(attack) for attack in train_attacks])
    _write_jsonl(Path("data/attacks/heldout_attacks_round99.jsonl"), [asdict(attack) for attack in heldout_attacks])
    _write_jsonl(Path("data/rollouts/train_tri_rollouts_round0.jsonl"), [record.to_dict() for record in train_rollouts])
    _write_jsonl(Path("data/eval/heldout_tri_rollouts_round99.jsonl"), [record.to_dict() for record in heldout_rollouts])

    Path("data/eval").mkdir(parents=True, exist_ok=True)
    Path("data/eval/heldout_metrics_round99.json").write_text(
        json.dumps(compute_metrics(heldout_rollouts), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_safety_sft_jsonl(train_rollouts, Path("data/processed/safety_sft_train.jsonl"))

    summary = {
        "train_attacks": len(train_attacks),
        "heldout_attacks": len(heldout_attacks),
        "train_rollouts": len(train_rollouts),
        "heldout_rollouts": len(heldout_rollouts),
        "sft_examples": len(train_rollouts),
    }
    Path("data/processed/sample_data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
