"""Evaluation entry points."""

from __future__ import annotations

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import PromptDrivenAttackGenerator, build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.metrics import compute_metrics
from evoguard.rollouts.tri_rollout import collect_tri_rollouts


class Evaluator:
    def __init__(self, env: TextToolEnv, attack_generator: PromptDrivenAttackGenerator | None = None) -> None:
        self.env = env
        self.attack_generator = attack_generator or build_attack_generator("heldout")

    def evaluate(self, agent: DefenseAgent, round_id: int) -> dict[str, float]:
        records = collect_tri_rollouts(agent, self.env, self.attack_generator, round_id=round_id)
        return compute_metrics(records)
