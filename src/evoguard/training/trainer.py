"""Round-based EvoGuard training loop scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_filter import extract_failures
from evoguard.attacks.attack_generator import AttackMemory, PromptDrivenAttackGenerator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.metrics import compute_metrics
from evoguard.rewards.combined_reward import RewardWeights, combined_reward
from evoguard.rollouts.attacked_tool_rollout import rollout_attacked_tool
from evoguard.rollouts.clean_tool_rollout import rollout_clean_tool
from evoguard.rollouts.no_tool_rollout import rollout_no_tool
from evoguard.training.replay_buffer import ReplayBuffer
from evoguard.types import TrajectoryRecord


@dataclass(frozen=True)
class TrainingRoundResult:
    round_id: int
    records: list[TrajectoryRecord]
    rewards: list[dict[str, float]]
    metrics: dict[str, float]
    policy_updates: int = 0
    update_stats: dict[str, float] | None = None


@dataclass(frozen=True)
class TrainerConfig:
    include_no_tool: bool = True
    include_clean_tool: bool = True
    adaptive_attack_generation: bool = True
    train_on_successful_attacks_only: bool = False


class EvoGuardTrainer:
    def __init__(
        self,
        agent: DefenseAgent,
        env: TextToolEnv,
        attack_generator: PromptDrivenAttackGenerator,
        reward_weights: RewardWeights | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.attack_generator = attack_generator
        self.attack_memory = AttackMemory()
        self.replay_buffer = ReplayBuffer()
        self.reward_weights = reward_weights or RewardWeights()
        self.config = config or TrainerConfig()

    def run_round(self, round_id: int, max_round: int) -> TrainingRoundResult:
        tasks = self.env.get_tasks()
        candidate_attacks = self.attack_generator.generate(
            tasks=tasks,
            round_id=round_id,
            failure_cases=self.attack_memory.get_recent_failures()
            if self.config.adaptive_attack_generation
            else None,
        )

        attacked_records = [rollout_attacked_tool(self.agent, self.env, attack) for attack in candidate_attacks]
        successful_attacks = extract_failures(attacked_records)

        train_pool: list[TrajectoryRecord] = []
        for task in tasks:
            if self.config.include_no_tool:
                train_pool.append(rollout_no_tool(self.agent, task, round_id=round_id))
            if self.config.include_clean_tool:
                train_pool.append(rollout_clean_tool(self.agent, self.env, task, round_id=round_id))
        if self.config.train_on_successful_attacks_only:
            train_pool.extend(successful_attacks)
        else:
            train_pool.extend(successful_attacks or attacked_records)

        reward_batch = [
            combined_reward(record, current_round=round_id, max_round=max_round, weights=self.reward_weights)
            for record in train_pool
        ]
        policy_updates = self.agent.update(train_pool, reward_batch)
        self.replay_buffer.add_many(train_pool)
        self.attack_memory.add(round_id, extract_failures(train_pool))

        metrics = compute_metrics(train_pool)
        return TrainingRoundResult(
            round_id=round_id,
            records=train_pool,
            rewards=reward_batch,
            metrics=metrics,
            policy_updates=policy_updates,
            update_stats=self.agent.last_update_stats(),
        )

    def train(self, num_rounds: int) -> list[TrainingRoundResult]:
        return [self.run_round(round_id=round_id, max_round=max(1, num_rounds - 1)) for round_id in range(num_rounds)]
