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
import statistics
import time
from collections import deque
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
        # Stagnation tracking for immigrant-injection alternative trigger.
        # ``_last_improvement_gen`` records the generation index at which the
        # population's best fitness strictly increased; reset to 0 on first run.
        self._last_improvement_gen: int = -1
        self._prev_best_fitness_for_stagnation: Optional[float] = None
        # Behavioral-novelty archive (rolling deque of canonical signatures
        # drawn from successful B-trajectory elites). Used to give small
        # additive bonus to offspring whose behavior signature has rarely been
        # seen before, encouraging exploration of distinct action sequences
        # rather than re-discovering the same immediate-trigger attack.
        archive_cap = max(0, int(getattr(config, "behavioral_archive_size", 20)))
        self._behavior_archive: deque[str] = deque(maxlen=archive_cap if archive_cap > 0 else 1)

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

        # Detect premature convergence via TWO complementary triggers:
        #
        # (a) Drop-based: best_fit collapsed by > fitness_drop_threshold vs prev gen.
        #     Catches the original r5-style sudden collapse pattern.
        #
        # (b) Stagnation-based: ``immigrant_stagnation_gens`` consecutive gens
        #     passed without ANY strict improvement in elite_fitness.  This
        #     catches the flat-zero plateau observed in evoguard_banking_local_improved
        #     where drop-rule never fired because there was nothing left to drop.
        current_best = float(ranked[0].fitness) if ranked else 0.0
        prev_best = getattr(self, "_prev_gen_best_fitness", None)
        self._prev_gen_best_fitness = current_best

        if (
            self._prev_best_fitness_for_stagnation is None
            or current_best > self._prev_best_fitness_for_stagnation + 1e-9
        ):
            self._last_improvement_gen = self.generation
            self._prev_best_fitness_for_stagnation = current_best

        stagnation_window = max(0, int(getattr(self.config, "immigrant_stagnation_gens", 0)))
        stagnation_elapsed = (
            (self.generation - self._last_improvement_gen)
            if self._last_improvement_gen >= 0 else 0
        )
        trigger_by_drop = (
            prev_best is not None
            and current_best < prev_best * self.config.fitness_drop_threshold
            and current_best <= 0.0
        )
        trigger_by_stagnation = (
            stagnation_window > 0
            and stagnation_elapsed >= stagnation_window
            and current_best <= 0.0   # only kick in when we're truly stuck at zero, not when holding steady on real signal
        )
        trigger_immigrants = bool(trigger_by_drop or trigger_by_stagnation)

        if trigger_immigrants:
            why = "drop" if trigger_by_drop else f"stagnation({stagnation_elapsed}g)"
            logger.warning(
                "Task %s gen %d->%d: best_fit %.4f -> %.4f (%s); "
                "injecting random immigrants at rate=%.2f",
                self.task.task_id,
                self.generation,
                self.generation + 1,
                prev_best or 0.0,
                current_best,
                why,
                self.config.immigrant_injection_rate,
            )

        # Adaptive mutation-rate scaling: when phenotypic variance of the just-
        # evaluated population collapses near zero, scale mutation probability UP
        # for THIS generation only so offspring get perturbed harder -- this is a
        # cheap-but-effective anti-premature-convergence lever that doesn't need
        # extra LLM calls beyond what crossover/mutation already cost us.
        eff_mutation_rate: float = float(self.config.mutation_rate)
        adaptive_enabled = bool(getattr(self.config, "adaptive_mutation_enabled", False))
        mut_min = float(getattr(self.config, "mutation_rate_min", 0.05))
        mut_max = float(getattr(self.config, "mutation_rate_max", 0.85))
        if adaptive_enabled:
            fit_vals = [float(e.fitness) for e in evaluated]
            try:
                if len(fit_vals) >= 3:
                    stdev_pop = statistics.pstdev(fit_vals) or 0.0
                    mean_pop = statistics.fmean(fit_vals) or 0.0
                    cv_pop = (stdev_pop / abs(mean_pop)) if abs(mean_pop) > 1e-12 \
                             else stdev_pop
                    # Coefficient-of-variation below ~0.25 => high homogeneity =>
                    # boost mutation multiplicatively toward ceiling; otherwise keep nominal.
                    if cv_pop < 0.25:
                        scaled = min(
                            mut_max,
                            eff_mutation_rate * (1.0 + (0.25 - cv_pop) / 0.30),
                        )
                        scaled = max(scaled, eff_mutation_rate)  # only ever raise it.
                        if scaled > eff_mutation_rate:
                            logger.info(
                                "Task %s gen %d->%d: low pop-variance(cv=%.4f) -> "
                                "raising mutation_rate %.2f -> %.4f for this generation.",
                                self.task.task_id,
                                self.generation,
                                self.generation + 1,
                                cv_pop,
                                eff_mutation_rate,
                                scaled,
                            )
                            eff_mutation_rate = float(min(mut_max, max(mut_min, scaled)))
            except Exception as exc:                                            # noqa: BLE001
                logger.debug("adaptive-mutation scaling skipped: %s", exc)

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

            if self.rng.random() < eff_mutation_rate:
                child = self.generator.mutate(
                    self.task, self.tools, child, generation=self.generation + 1
                )
            offspring.append(self.sanitize_spec(child))

        # Update behavioral-novelty archive using this round's successful elites'
        # canonical (method, target_turn) signatures. Future generations get a
        # small additive fitness boost when their signature is rare in the
        # archive -- implemented inside ``_adjusted_fitness`` via the bonus weight.
        try:
            for e in ranked[: max(0, int(getattr(self.config, "elite_size", 0)))]:
                if not getattr(e, "success", False):
                    continue
                sig = f"{e.spec.method or ''}|t{int(e.spec.target_turn)}"
                if sig in set(self._behavior_archive):
                    continue   # dedup so the deque stays diverse.
                self._behavior_archive.append(sig)
        except Exception:
            pass

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

        Additionally, when ``behavioral_archive_size`` > 0 a small additive
        bonus is granted for individuals whose (method, target_turn) signature
        is rare in the rolling archive of past successful elites. This nudges
        tournament selection toward genuinely novel attack behaviors rather
        than re-discovering the same immediate-trigger pattern.
        """

        base = candidate.fitness
        if base <= 0.0:
            # Failed attacks have zero raw fitness; we still allow novelty
            # bonus to lift them slightly so they survive selection pressure
            # during early exploration rounds where everything fails.
            bonus_only_base: float = 0.0
        else:
            bonus_only_base = base

        if not selected_history:
            score_no_bonus = base
        else:
            window = self.config.diversity_position_window
            duplicates = 0
            for prior in selected_history:
                close_pos = abs(prior.target_turn - candidate.spec.target_turn) <= window
                same_method = prior.method == candidate.spec.method
                if close_pos and same_method:
                    duplicates += 1
            factor = (1.0 - self.config.diversity_penalty) ** duplicates
            score_no_bonus = base * factor

        # Behavioral-novelty additive bonus.
        archive_size_cap = max(0, int(getattr(self.config, "behavioral_archive_size", 0)))
        if (
            archive_size_cap > 0
            and len(self._behavior_archive) > 0
            and getattr(self.config, "novelty_bonus_weight", 0.0) > 0.0
        ):
            sig_cand = f"{candidate.spec.method or ''}|t{int(candidate.spec.target_turn)}"
            unique_total = float(len(set(self._behavior_archive)) or 1)
            matches = sum(
                1 for s in self._behavior_archive if s == sig_cand
            )
            rarity_ratio = 1.0 - min(1.0, matches / max(1.0, unique_total))
            weight = float(getattr(self.config, "novelty_bonus_weight", 0.03))
            bonus = weight * rarity_ratio * (1.0 if bonus_only_base >= 0 else 0.0)
            return score_no_bonus + bonus

        return score_no_bonus


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
