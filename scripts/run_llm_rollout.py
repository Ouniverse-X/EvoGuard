#!/usr/bin/env python
"""Run one LLM-backed rollout when OPENAI_API_KEY and EVOGUARD_LLM_MODEL are set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.agents.llm_safety_head import LLMSafetyHead
from evoguard.agents.tool_calling_agent import LLMToolCallingAgent
from evoguard.attacks.llm_attack_generator import LLMRedTeamAttackGenerator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.llm.client import OpenAIResponsesClient
from evoguard.rollouts.agentic_tool_rollout import rollout_agentic_attacked_tool, rollout_agentic_clean_tool


def main() -> None:
    client = OpenAIResponsesClient()
    env = TextToolEnv()
    task = env.get_tasks()[0]
    tool_agent = LLMToolCallingAgent(client)
    defense_agent = DefenseAgent(LLMSafetyHead(client))
    attack_generator = LLMRedTeamAttackGenerator(client, attacks_per_task=1)

    clean = rollout_agentic_clean_tool(tool_agent, defense_agent, env, task, round_id=0)
    attack = attack_generator.generate([task], round_id=0)[0]
    attacked = rollout_agentic_attacked_tool(tool_agent, defense_agent, env, attack)
    print(json.dumps([clean.to_dict(), attacked.to_dict()], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
