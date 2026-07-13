"""Combined reward computation."""

from __future__ import annotations

from dataclasses import dataclass

from evoguard.rewards.attribution_reward import attribution_reward
from evoguard.rewards.safety_reward import SafetyRewardConfig, safety_reward
from evoguard.rewards.task_reward import task_reward
from evoguard.types import TrajectoryRecord


@dataclass(frozen=True)
class RewardWeights:
    lambda_safety: float = 1.0
    lambda_attr: float = 0.25
    beta_kl: float = 0.0
    safety: SafetyRewardConfig = SafetyRewardConfig()


def combined_reward(
    record: TrajectoryRecord,
    current_round: int,
    max_round: int,
    weights: RewardWeights | None = None,
    kl_value: float = 0.0,
) -> dict[str, float]:
    weights = weights or RewardWeights()
    task = task_reward(record)
    safety = safety_reward(record, current_round=current_round, max_round=max_round, config=weights.safety)
    attr = attribution_reward(record)
    total = task + weights.lambda_safety * safety + weights.lambda_attr * attr - weights.beta_kl * kl_value
    return {
        "total": total,
        "task": task,
        "safety": safety,
        "attribution": attr,
        "kl": kl_value,
    }
