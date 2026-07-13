#!/usr/bin/env python
"""Run one tri-trajectory rollout batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import PromptDrivenAttackGenerator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.rollouts.tri_rollout import collect_tri_rollouts


def main() -> None:
    env = TextToolEnv()
    agent = DefenseAgent()
    attack_generator = PromptDrivenAttackGenerator()
    records = collect_tri_rollouts(agent, env, attack_generator, round_id=0)
    print(json.dumps([record.to_dict() for record in records], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
