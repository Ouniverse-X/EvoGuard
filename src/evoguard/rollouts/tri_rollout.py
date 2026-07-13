"""Tri-trajectory rollout orchestration."""

from __future__ import annotations

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import PromptDrivenAttackGenerator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.rollouts.attacked_tool_rollout import rollout_attacked_tool
from evoguard.rollouts.clean_tool_rollout import rollout_clean_tool
from evoguard.rollouts.no_tool_rollout import rollout_no_tool
from evoguard.types import TrajectoryRecord


def collect_tri_rollouts(
    agent: DefenseAgent,
    env: TextToolEnv,
    attack_generator: PromptDrivenAttackGenerator,
    round_id: int,
) -> list[TrajectoryRecord]:
    tasks = env.get_tasks()
    records: list[TrajectoryRecord] = []
    for task in tasks:
        records.append(rollout_no_tool(agent, task, round_id=round_id))
        records.append(rollout_clean_tool(agent, env, task, round_id=round_id))
    for attack in attack_generator.generate(tasks, round_id=round_id):
        records.append(rollout_attacked_tool(agent, env, attack))
    return records
