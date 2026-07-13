"""Runnable baseline registry and comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.agents.safety_head import (
    AlwaysExecuteSafetyHead,
    AlwaysRefuseSafetyHead,
    RuleBasedSafetyHead,
    TrainableSafetyHead,
)
from evoguard.attacks.attack_generator import build_attack_generator
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.evaluation.evaluator import Evaluator
from evoguard.training.trainer import EvoGuardTrainer


BASELINES = (
    "no_guard",
    "rule_based_guard",
    "always_refuse_guard",
    "static_guard",
    "eapo_fixed_contrastive_rollout",
    "evoguard_adaptive",
)


@dataclass(frozen=True)
class BaselineResult:
    name: str
    metrics: dict[str, float]
    train_rounds: int
    policy_updates: int
    train_attack_split: str = "train"
    eval_attack_split: str = "heldout"
    update_stats: dict[str, float] | None = None


def build_baseline_agent(name: str) -> DefenseAgent:
    if name == "no_guard":
        return DefenseAgent(AlwaysExecuteSafetyHead())
    if name == "rule_based_guard":
        return DefenseAgent(RuleBasedSafetyHead())
    if name == "always_refuse_guard":
        return DefenseAgent(AlwaysRefuseSafetyHead())
    if name in {"static_guard", "eapo_fixed_contrastive_rollout", "evoguard_adaptive"}:
        return DefenseAgent(TrainableSafetyHead())
    raise ValueError(f"Unknown baseline: {name}")


def evaluate_baseline(name: str, train_rounds: int = 3, eval_round: int = 99) -> BaselineResult:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}")

    env = TextToolEnv()
    train_attack_generator = build_attack_generator("train")
    eval_attack_generator = build_attack_generator("heldout")
    agent = build_baseline_agent(name)
    updates = 0
    update_stats: dict[str, float] = {}

    if name in {"static_guard", "eapo_fixed_contrastive_rollout", "evoguard_adaptive"}:
        trainer = EvoGuardTrainer(agent, env, train_attack_generator)
        if name == "static_guard":
            results = [trainer.run_round(round_id=0, max_round=1) for _ in range(train_rounds)]
        elif name == "eapo_fixed_contrastive_rollout":
            results = [trainer.run_round(round_id=0, max_round=1)]
        else:
            results = trainer.train(num_rounds=train_rounds)
        updates = sum(result.policy_updates for result in results)
        update_stats = results[-1].update_stats or {}

    metrics = Evaluator(env, eval_attack_generator).evaluate(agent, round_id=eval_round)
    return BaselineResult(
        name=name,
        metrics=metrics,
        train_rounds=train_rounds,
        policy_updates=updates,
        train_attack_split=train_attack_generator.split,
        eval_attack_split=eval_attack_generator.split,
        update_stats=update_stats,
    )


def run_baseline_suite(baselines: tuple[str, ...] = BASELINES, train_rounds: int = 3) -> list[BaselineResult]:
    return [evaluate_baseline(name, train_rounds=train_rounds) for name in baselines]
