"""
The attacker is an LLM that proposes injection strategies (:class:`AttackSpec`
genomes) and applies genetic variation operators to them. It is split into an
:class:`AttackGenerator` (the LLM primitives: seed generation, crossover,
mutation) and the genetic loop in :mod:`evoguard.attacks.genetic`, which owns
selection / elitism / diversity accounting.
"""

from __future__ import annotations

import abc
from typing import Sequence

from evoguard.core.types import AttackSpec, Task, ToolSpec


class AttackGenerator(abc.ABC):
    """LLM primitives that produce and recombine attack genomes."""

    @abc.abstractmethod
    def seed(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        n: int,
        *,
        max_turns: int,
        generation: int,
    ) -> list[AttackSpec]:
        """Generate ``n`` fresh seed attacks for ``task``."""

    @abc.abstractmethod
    def crossover(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        parents: Sequence[AttackSpec],
        *,
        generation: int,
    ) -> AttackSpec:
        """Recombine ``parents`` into a single child attack."""

    @abc.abstractmethod
    def mutate(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        individual: AttackSpec,
        *,
        generation: int,
    ) -> AttackSpec:
        """Return a mutated variant of ``individual``."""
