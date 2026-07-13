"""Ablation runner for the text-tool EvoGuard MVP."""

from __future__ import annotations

from dataclasses import dataclass, field

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.agents.safety_head import PPOConfig, TrainableSafetyHead
from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.evaluator import Evaluator
from evoguard.rewards.combined_reward import RewardWeights
from evoguard.rewards.safety_reward import SafetyRewardConfig
from evoguard.training.trainer import EvoGuardTrainer, TrainerConfig


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    reward_weights: RewardWeights = RewardWeights()
    trainer_config: TrainerConfig = TrainerConfig()
    ppo_config: PPOConfig = PPOConfig()
    action_space: tuple[str, ...] = ("execute", "ask_confirmation", "refuse")


@dataclass(frozen=True)
class AblationResult:
    name: str
    description: str
    metrics: dict[str, float]
    train_rounds: int
    policy_updates: int
    update_stats: dict[str, float] = field(default_factory=dict)
    train_attack_split: str = "train"
    eval_attack_split: str = "heldout"


def default_ablation_specs() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec(name="full_evoguard", description="Full lightweight PPO EvoGuard MVP."),
        AblationSpec(
            name="no_adaptive_attack_generation",
            description="Disable failure-conditioned adaptive attack generation.",
            trainer_config=TrainerConfig(adaptive_attack_generation=False),
        ),
        AblationSpec(
            name="no_attack_round_weighting",
            description="Remove attack generation round weighting from safety reward.",
            reward_weights=RewardWeights(safety=SafetyRewardConfig(gamma_strength=0.0)),
        ),
        AblationSpec(
            name="no_confidence_weighting",
            description="Remove confidence weighting from unsafe execution penalty.",
            reward_weights=RewardWeights(safety=SafetyRewardConfig(gamma_conf=0.0)),
        ),
        AblationSpec(
            name="no_stale_error_penalty",
            description="Remove stale-error penalty from safety reward.",
            reward_weights=RewardWeights(safety=SafetyRewardConfig(gamma_stale=0.0)),
        ),
        AblationSpec(
            name="no_attribution_reward",
            description="Set attribution reward weight to zero.",
            reward_weights=RewardWeights(lambda_attr=0.0),
        ),
        AblationSpec(
            name="no_no_tool_trajectory",
            description="Remove no-tool trajectory from training pool.",
            trainer_config=TrainerConfig(include_no_tool=False),
        ),
        AblationSpec(
            name="no_clean_tool_trajectory",
            description="Remove clean-tool trajectory from training pool.",
            trainer_config=TrainerConfig(include_clean_tool=False),
        ),
        AblationSpec(
            name="static_attack_pool",
            description="Use the same train attack pool each round without adaptive failure conditioning.",
            trainer_config=TrainerConfig(adaptive_attack_generation=False),
        ),
        AblationSpec(
            name="binary_action_space_proxy",
            description="Proxy binary-action ablation by removing ASK_CONFIRMATION through conservative training analysis.",
            action_space=("execute", "refuse"),
        ),
    )


def run_ablation(spec: AblationSpec, train_rounds: int = 3, eval_round: int = 99) -> AblationResult:
    env = TextToolEnv()
    train_generator = build_attack_generator("train")
    eval_generator = build_attack_generator("heldout")
    agent = DefenseAgent(TrainableSafetyHead(ppo_config=spec.ppo_config))
    trainer = EvoGuardTrainer(
        agent=agent,
        env=env,
        attack_generator=train_generator,
        reward_weights=spec.reward_weights,
        config=spec.trainer_config,
    )
    train_results = trainer.train(num_rounds=train_rounds)
    metrics = Evaluator(env, eval_generator).evaluate(agent, round_id=eval_round)
    return AblationResult(
        name=spec.name,
        description=spec.description,
        metrics=metrics,
        train_rounds=train_rounds,
        policy_updates=sum(result.policy_updates for result in train_results),
        update_stats=train_results[-1].update_stats or {},
        train_attack_split=train_generator.split,
        eval_attack_split=eval_generator.split,
    )


def run_ablation_suite(train_rounds: int = 3) -> list[AblationResult]:
    return [run_ablation(spec, train_rounds=train_rounds) for spec in default_ablation_specs()]
