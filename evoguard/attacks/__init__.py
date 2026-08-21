"""Attacker: LLM genome generator + search backends."""

from __future__ import annotations

from typing import Optional, Sequence

from evoguard.attacks.base import AttackGenerator
from evoguard.attacks.genetic import EvaluatedAttack, GeneticAttacker
from evoguard.attacks.llm_attacker import LLMAttackGenerator
from evoguard.attacks.mct_searcher import DeltaGuidedMCTSAttacker
from evoguard.config import AttackerConfig
from evoguard.core.types import Task, ToolSpec
from evoguard.llm.base import LLMClient


def build_attack_generator(
    config: AttackerConfig,
    *,
    client: LLMClient | None = None,
    seed: int = 0,
) -> AttackGenerator:
    """Construct the default LLM-backed attack genome generator."""

    return LLMAttackGenerator(config, client=client, seed=seed)


def build_attacker(
    task: Task,
    tools: Sequence[ToolSpec],
    generator: AttackGenerator,
    config: AttackerConfig,
    *,
    rng=None,
    defense_max_turns: Optional[int] = None,
):
    """Factory selecting attacker backend by ``config.search_method``.

    Keeps call sites backend-agnostic so swapping GA <-> MCTS <-> future
    variants happens purely through yaml::

        attackers[t.task_id] = build_attacker(t, tools, gen, cfg.attacker,
                                               defense_max_turns=K)

    Recognised values of :attr:`AttackerConfig.search_method`:
      * ``"ga"``         -- legacy :class:`GeneticAttacker` (default).
      * ``"mcts_delta"`` -- Δ-guided MCTS (:class:`DeltaGuidedMCTSAttacker`).
    """
    method = getattr(config, "search_method", "ga") or "ga"
    if method == "ga":
        return GeneticAttacker(
            task=task, tools=tools, generator=generator, config=config,
            rng=rng, defense_max_turns=defense_max_turns,
        )
    if method == "mcts_delta":
        return DeltaGuidedMCTSAttacker(
            task=task, tools=tools, generator=generator, config=config,
            rng=rng, defense_max_turns=defense_max_turns,
        )
    raise ValueError(
        f"Unknown AttackerConfig.search_method={method!r}; "
        f"expected one of {{'ga','mcts_delta'}}."
    )


__all__ = [
    "AttackGenerator",
    "LLMAttackGenerator",
    "GeneticAttacker",
    "DeltaGuidedMCTSAttacker",
    "EvaluatedAttack",
    "build_attack_generator",
    "build_attacker",
]
