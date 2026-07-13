#!/usr/bin/env python
"""Prepare supervised safety-judge training data from EvoGuard rollouts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.rollouts.tri_rollout import collect_tri_rollouts
from evoguard.training.sft_dataset import write_safety_sft_jsonl


def main() -> None:
    env = TextToolEnv()
    records = collect_tri_rollouts(DefenseAgent(), env, build_attack_generator("train"), round_id=0)
    output_path = write_safety_sft_jsonl(records, Path("data/processed/safety_sft_train.jsonl"))
    print(output_path)


if __name__ == "__main__":
    main()
