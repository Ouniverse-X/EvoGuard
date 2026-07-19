"""LLM-backed :class:`AttackGenerator`.

Wraps a chat LLM with the attacker prompt templates and parses the returned
genome JSON into :class:`AttackSpec` objects. Parsing is defensive: malformed or
partial genomes are dropped (seed) or fall back to a copy of a parent
(crossover/mutation) so the genetic loop always receives a valid individual.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from evoguard.attacks.base import AttackGenerator
from evoguard.attacks.prompts import crossover_prompt, mutation_prompt, seed_prompt
from evoguard.config import AttackerConfig
from evoguard.core.types import AttackSpec, Message, Role, Task, ToolSpec
from evoguard.llm import build_client
from evoguard.llm.base import LLMClient
from evoguard.llm.schemas import ATTACK_GENOMES_SCHEMA
from evoguard.utils.logging import get_logger

logger = get_logger("attacks.llm")


class LLMAttackGenerator(AttackGenerator):
    """Attack genome generator driven by a chat LLM."""

    def __init__(self, config: AttackerConfig, *, client: Optional[LLMClient] = None, seed: int = 0):
        self.config = config
        self._client = client or build_client(config.llm, seed=seed)

    # ---- primitives ------------------------------------------------------- #
    def seed(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        n: int,
        *,
        max_turns: int,
        generation: int,
    ) -> list[AttackSpec]:
        system = seed_prompt(task, tools, n, max_turns)
        genomes = self._query(system)
        specs = [
            self._to_spec(task, g, generation=generation, origin="seed")
            for g in genomes
        ]
        specs = [s for s in specs if s is not None]
        # If the model under-produced, pad by cloning with perturbed turns.
        while len(specs) < n and specs:
            base = specs[len(specs) % len(specs)]
            clone = _clone(base, generation=generation, origin="seed")
            clone.target_turn = max(1, (base.target_turn % max(1, max_turns)) + 1)
            specs.append(clone)
        return specs[:n]

    def crossover(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        parents: Sequence[AttackSpec],
        *,
        generation: int,
    ) -> AttackSpec:
        system = crossover_prompt(task, tools, parents)
        genomes = self._query(system)
        if genomes:
            child = self._to_spec(task, genomes[0], generation=generation, origin="crossover")
            if child is not None:
                child.parents = [p.attack_id for p in parents]
                return child
        # Fallback: clone the fitter-looking parent (first).
        child = _clone(parents[0], generation=generation, origin="crossover")
        child.parents = [p.attack_id for p in parents]
        return child

    def mutate(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        individual: AttackSpec,
        *,
        generation: int,
    ) -> AttackSpec:
        system = mutation_prompt(task, tools, individual)
        genomes = self._query(system)
        if genomes:
            mutant = self._to_spec(task, genomes[0], generation=generation, origin="mutation")
            if mutant is not None:
                mutant.parents = [individual.attack_id]
                return mutant
        mutant = _clone(individual, generation=generation, origin="mutation")
        mutant.parents = [individual.attack_id]
        return mutant

    # ---- helpers ---------------------------------------------------------- #
    def _query(self, system: str) -> list[dict[str, Any]]:
        resp = self._client.chat(
            [
                Message(role=Role.SYSTEM, content=system),
                Message(role=Role.USER, content="Produce the attack(s) now as JSON."),
            ],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            response_format=ATTACK_GENOMES_SCHEMA,
        )
        data = _extract_json_object(resp.text)
        if not data:
            return []
        attacks = data.get("attacks")
        if isinstance(attacks, list):
            return [a for a in attacks if isinstance(a, dict)]
        return []

    def _to_spec(
        self,
        task: Task,
        genome: dict[str, Any],
        *,
        generation: int,
        origin: str,
    ) -> Optional[AttackSpec]:
        payload = str(genome.get("payload", "")).strip()
        if not payload:
            return None
        try:
            target_turn = int(genome.get("target_turn", 1))
        except (TypeError, ValueError):
            target_turn = 1
        return AttackSpec(
            attack_id=AttackSpec.new_id(),
            task_id=task.task_id,
            target_turn=max(0, target_turn),
            injection_channel=str(genome.get("injection_channel", "tool_result")),
            method=str(genome.get("method", "unknown")),
            payload=payload,
            goal=str(genome.get("goal", "")),
            generation=generation,
            origin=origin,
        )


def _clone(spec: AttackSpec, *, generation: int, origin: str) -> AttackSpec:
    return AttackSpec(
        attack_id=AttackSpec.new_id(),
        task_id=spec.task_id,
        target_turn=spec.target_turn,
        injection_channel=spec.injection_channel,
        method=spec.method,
        payload=spec.payload,
        goal=spec.goal,
        generation=generation,
        origin=origin,
        metadata=dict(spec.metadata),
    )


def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
