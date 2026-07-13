#!/usr/bin/env python
"""Run the EvoGuard MVP training loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.agents.safety_head import TrainableSafetyHead
from evoguard.attacks.attack_generator import PromptDrivenAttackGenerator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.training.trainer import EvoGuardTrainer
from evoguard.utils.logging import append_jsonl


def main() -> None:
    trainer = EvoGuardTrainer(
        agent=DefenseAgent(TrainableSafetyHead()),
        env=TextToolEnv(),
        attack_generator=PromptDrivenAttackGenerator(),
    )
    results = trainer.train(num_rounds=3)
    summary = [
        {
            "round_id": result.round_id,
            "num_records": len(result.records),
            "policy_updates": result.policy_updates,
            "update_stats": result.update_stats or {},
            "metrics": result.metrics,
            "mean_reward": sum(item["total"] for item in result.rewards) / len(result.rewards),
        }
        for result in results
    ]
    append_jsonl(Path("outputs/logs/train_evoguard_summary.jsonl"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
