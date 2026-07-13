"""Training helpers for the local neural safety head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.agents.neural_safety_head import NeuralSafetyConfig, NeuralSafetyHead
from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.evaluator import Evaluator
from evoguard.rollouts.tri_rollout import collect_tri_rollouts
from evoguard.types import TrajectoryRecord


@dataclass(frozen=True)
class NeuralTrainingResult:
    records: list[TrajectoryRecord]
    train_stats: dict[str, float | int]
    eval_metrics: dict[str, float]
    checkpoint_path: Path


def train_neural_safety_head(
    *,
    rounds: int = 3,
    config: NeuralSafetyConfig | None = None,
    checkpoint_path: str | Path = "outputs/checkpoints/neural_safety_head.pt",
) -> NeuralTrainingResult:
    env = TextToolEnv()
    collector_agent = DefenseAgent()
    train_generator = build_attack_generator("train")
    records: list[TrajectoryRecord] = []
    for round_id in range(rounds):
        records.extend(collect_tri_rollouts(collector_agent, env, train_generator, round_id=round_id))

    head = NeuralSafetyHead(config)
    agent = DefenseAgent(head)
    head.update(records)
    output_path = head.save(checkpoint_path)
    metrics = Evaluator(env, build_attack_generator("heldout")).evaluate(agent, round_id=99)
    return NeuralTrainingResult(
        records=records,
        train_stats=agent.last_update_stats(),
        eval_metrics=metrics,
        checkpoint_path=output_path,
    )
