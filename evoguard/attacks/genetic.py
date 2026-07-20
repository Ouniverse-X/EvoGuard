"""
Implements the exact evolutionary loop described in the plan:

* **Population**: ``N`` attack genomes evaluated per round (one rollout each,
  yielding a trajectory that is A/B/C and a fitness).
* **Fitness**: ``0`` if the attack failed (C), else the normalized delta of the
  successful attack (B). Only attacks that fool the defense are eligible.
* **Selection**: tournament of size ``k=3`` on *diversity-adjusted* fitness. The
  adjustment discounts an individual whose injection position and method closely
  duplicate individuals already drawn into the parent pool, maintaining coverage
  over both the "position" and "method" axes.
* **Crossover + mutation**: parents are recombined pairwise and LLM-mutated to
  produce ``M`` offspring.
* **Elitism**: the top ``E`` individuals of the current population are copied
  unchanged into the next generation.

``M + E = N`` reconstitutes the next generation.

The GA is stateful across rounds: it holds the current population and, after the
pipeline evaluates it (attaching fitness via :class:`EvaluatedAttack`), produces
the next generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from evoguard.attacks.base import AttackGenerator
from evoguard.config import AttackerConfig
from evoguard.core.types import AttackSpec, Task, ToolSpec
from evoguard.utils.logging import get_logger

logger = get_logger("attacks.genetic")


@dataclass
class EvaluatedAttack:
    spec: AttackSpec
    fitness: float
    success: bool = False
    metadata: dict = field(default_factory=dict)


class GeneticAttacker:
    def __init__(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        generator: AttackGenerator,
        config: AttackerConfig,
        *,
        rng: Optional[random.Random] = None,
        defense_max_turns: Optional[int] = None,
    ):
        self.task = task
        self.tools = list(tools)
        self.generator = generator
        self.config = config
        self.rng = rng or random.Random(config.random_seed)
        self.generation = 0
        self._population: list[AttackSpec] = []
        
        ctrl_cap = int(defense_max_turns) if defense_max_turns else None
        if ctrl_cap is not None:
            self._inject_turn_ceiling = max(1, ctrl_cap)
        else:
            self._inject_turn_ceiling = max(1, len(self.tools))

    @property
    def injectable_turn_ceiling(self) -> int:
        """Exclusive upper bound for any :attr:`AttackSpec.target_turn`."""
        return self._inject_turn_ceiling

    def sanitize_spec(self, spec: AttackSpec) -> AttackSpec:
        """Enforce ``target_turn ∈ [0, ceiling-1]``; log loudly when clamping.

        This is the single choke point through which every genome produced or
        imported into this GA must pass. It guarantees that downstream rollouts
        will actually observe the injection at the recorded turn -- without it,
        out-of-range genomes silently degrade to zero-Δ failures polluting both
        fitness statistics and termination decisions.
        """

        ub = self._inject_turn_ceiling - 1  # inclusive maximum reachable turn
        original = spec.target_turn
        clamped = max(0, min(int(original), ub))
        if clamped != original:
            logger.warning(
                "Task %s: clamped out-of-bound target_turn %d -> %d "
                "(gen=%d origin=%s)",
                self.task.task_id, original, clamped,
                spec.generation, spec.origin,
            )
            return replace(spec, target_turn=clamped)
        return spec

    # ---- population access ------------------------------------------------ #
    def current_population(self) -> list[AttackSpec]:
        """Return the population to roll out this round (seeding if empty)."""

        if not self._population:
            seeded = self.generator.seed(
                self.task,
                self.tools,
                self.config.population_size,
                max_turns=self._inject_turn_ceiling,
                generation=self.generation,
            )
            self._population = [self.sanitize_spec(s) for s in seeded]
            logger.info(
                "Task %s: seeded generation %d with %d attacks "
                "(injectable turns [0,%d])",
                self.task.task_id,
                self.generation,
                len(self._population),
                self._inject_turn_ceiling - 1,
            )
        return list(self._population)

    # ---- evolution -------------------------------------------------------- #
    def evolve(self, evaluated: list[EvaluatedAttack]) -> list[AttackSpec]:
        """Produce the next generation from this round's evaluated population.

        Returns the new population and installs it as the current one.
        """

        if not evaluated:
            self._population = []
            return self.current_population()

        # Elitism: keep the top-E by raw fitness.
        ranked = sorted(evaluated, key=lambda e: e.fitness, reverse=True)
        elites = [
            _clone_as(e.spec, generation=self.generation + 1, origin="elite")
            for e in ranked[: self.config.elite_size]
        ]
        elites = [self.sanitize_spec(s) for s in elites]

        # Detect premature convergence: when mean best fitness drops sharply
        # vs the previous generation we inject random immigrants to escape the
        # local optimum (prevents r5-style collapse seen in evoguard_agentdojo_full).
        current_best = float(ranked[0].fitness) if ranked else 0.0
        prev_best = getattr(self, "_prev_gen_best_fitness", None)
        trigger_immigrants = (
            prev_best is not None
            and current_best < prev_best * self.config.fitness_drop_threshold
            and current_best <= 0.0  # only kick in on actual collapse to zero
        )
        self._prev_gen_best_fitness = current_best
        if trigger_immigrants:
            logger.warning(
                "Task %s gen %d->%d: best_fit %.4f -> %.4f below threshold; "
                "injecting random immigrants at rate=%.2f",
                self.task.task_id,
                self.generation,
                self.generation + 1,
                prev_best or 0.0,
                current_best,
                self.config.immigrant_injection_rate,
            )

        # Build offspring via crowding-aware tournament selection + crossover/mutation.
        offspring: list[AttackSpec] = []
        selected_history: list[AttackSpec] = []
        target_offspring = self.config.offspring_size

        # If immigrants are triggered AND the generator can re-seed, replace a
        # fraction of the worst-ranked individuals with fresh random genomes.
        n_immigrants = 0
        if trigger_immigrants and self.config.immigrant_injection_rate > 0.0:
            n_immigrants = int(round(target_offspring * self.config.immigrant_injection_rate))
            try:
                fresh_seeds = self.generator.seed(
                    self.task, self.tools, max(1, n_immigrants),
                    max_turns=self._inject_turn_ceiling,
                    generation=self.generation + 1,
                )
                for fs in fresh_seeds[:n_immigrants]:
                    tagged = _clone_as(
                        self.sanitize_spec(fs),
                        generation=self.generation + 1,
                        origin="immigrant",
                    )
                    offspring.append(tagged)
                    # Track in history so crowding penalty treats them like any other.
                    selected_history.append(tagged)
            except Exception as exc:                                            # noqa: BLE001
                logger.warning(
                    "Task %s: immigrant seeding failed (%s); skipping injection.",
                    self.task.task_id, str(exc)[:200],
                )

        guard = 0
        while len(offspring) < target_offspring and guard < target_offspring * 10:
            guard += 1
            p1 = self._tournament(evaluated, selected_history)
            p2 = self._tournament(evaluated, selected_history)
            selected_history.extend([p1.spec, p2.spec])

            if self.rng.random() < self.config.crossover_rate:
                child = self.generator.crossover(
                    self.task, self.tools, [p1.spec, p2.spec],
                    generation=self.generation + 1,
                )
            else:
                child = _clone_as(p1.spec, generation=self.generation + 1, origin="crossover")

            if self.rng.random() < self.config.mutation_rate:
                child = self.generator.mutate(
                    self.task, self.tools, child, generation=self.generation + 1
                )
            offspring.append(self.sanitize_spec(child))

        self.generation += 1
        self._population = offspring[:target_offspring] + elites
        logger.info(
            "Task %s: generation %d -> %d offspring (+%d immigrants) + %d elites "
            "(best fitness=%.3f)",
            self.task.task_id,
            self.generation,
            len(offspring[:target_offspring]),
            n_immigrants,
            len(elites),
            ranked[0].fitness,
        )
        return list(self._population)

    # ---- selection with niching / crowding --------------------------------- #
    def _tournament(
        self,
        evaluated: Sequence[EvaluatedAttack],
        selected_history: Sequence[AttackSpec],
    ) -> EvaluatedAttack:
        """Tournament selection on diversity-adjusted fitness.

        When ``crowding_factor`` >= 2 this enforces method-niche limits per bracket:
        no more than ``crowding_factor`` contenders sharing the same ``method``
        label may enter the same tournament sample. This forces exploration across
        distinct attack methods instead of letting one dominant niche take over --
        directly addressing the r5 collapse pattern observed when pure tournament
        drove every contender into the same immediate-trigger corner of the search space.
        """

        k = min(self.config.tournament_k, len(evaluated))
        cf = max(1, int(getattr(self.config, "crowding_factor", 1)))
        pool = list(evaluated)

        if cf <= 1 or k <= 1:
            # Pure-tournament fast path identical to legacy behavior.
            contenders = self.rng.sample(pool, k)
        else:
            # Sample k candidates while enforcing that no more than `cf`
            # share the same `method` label. Falls back to plain sampling if
            # diversity is exhausted before reaching k.
            shuffled = self.rng.sample(pool, min(len(pool), k * 4))
            chosen: list[EvaluatedAttack] = []
            method_counts: dict[str, int] = {}
            for cand in shuffled:
                m = cand.spec.method or ""
                if method_counts.get(m, 0) >= cf:
                    continue
                chosen.append(cand)
                method_counts[m] = method_counts.get(m, 0) + 1
                if len(chosen) >= k:
                    break
            if len(chosen) < k:
                # Top up ignoring crowding so we always return something valid.
                leftover = [c for c in shuffled if c not in chosen]
                while leftover and len(chosen) < k:
                    chosen.append(leftover.pop(0))
            contenders = chosen

        best = None
        best_score = float("-inf")
        for c in contenders:
            score = self._adjusted_fitness(c, selected_history)
            if score > best_score:
                best_score = score
                best = c
        assert best is not None
        return best

    def _adjusted_fitness(
        self,
        candidate: EvaluatedAttack,
        selected_history: Sequence[AttackSpec],
    ) -> float:
        """Discount fitness by similarity to already-selected individuals.

        Similarity counts an already-selected genome as a duplicate when its
        injection position is within ``diversity_position_window`` turns AND it
        shares the same method label. Each duplicate multiplies the fitness by
        ``(1 - diversity_penalty)``, so coverage over position and method is
        rewarded.
        """

        base = candidate.fitness
        if base <= 0.0 or not selected_history:
            return base
        window = self.config.diversity_position_window
        duplicates = 0
        for prior in selected_history:
            close_pos = abs(prior.target_turn - candidate.spec.target_turn) <= window
            same_method = prior.method == candidate.spec.method
            if close_pos and same_method:
                duplicates += 1
        factor = (1.0 - self.config.diversity_penalty) ** duplicates
        return base * factor


def _clone_as(spec: AttackSpec, *, generation: int, origin: str) -> AttackSpec:
    return AttackSpec(
        attack_id=AttackSpec.new_id(),
        task_id=spec.task_id,
        target_turn=spec.target_turn,
        injection_channel=spec.injection_channel,
        method=spec.method,
        payload=spec.payload,
        goal=spec.goal,
        generation=generation,
        parents=[spec.attack_id],
        origin=origin,
        metadata=dict(spec.metadata),
    )
