"""Attacker: LLM genome generator + genetic algorithm."""

from __future__ import annotations

from evoguard.attacks.base import AttackGenerator
from evoguard.attacks.genetic import EvaluatedAttack, GeneticAttacker
from evoguard.attacks.llm_attacker import LLMAttackGenerator
from evoguard.config import AttackerConfig
from evoguard.llm.base import LLMClient


def build_attack_generator(
    config: AttackerConfig,
    *,
    client: LLMClient | None = None,
    seed: int = 0,
) -> AttackGenerator:
    """Construct the default LLM-backed attack genome generator."""

    return LLMAttackGenerator(config, client=client, seed=seed)


__all__ = [
    "AttackGenerator",
    "LLMAttackGenerator",
    "GeneticAttacker",
    "EvaluatedAttack",
    "build_attack_generator",
]
