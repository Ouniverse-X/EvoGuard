"""Frozen attacker: replay the dataset's OWN injections instead of searching.

This is the backend for the **attacker ablation** -- the arm in which the attack
generator does not participate in co-evolution at all. Selected by
``AttackerConfig.search_method == "vendored"``.

What it removes
---------------
Everything on the attacker side, and nothing else. No attacker LLM call is ever
issued (no seeding, no crossover, no mutation, no MCTS expansion or backprop),
``evolve()`` is a no-op, and therefore the population handed to round *N* is
byte-identical to the one handed to round 0. Defender-side machinery is
untouched: the tri-trajectory controller, the Delta/turning-point signals, the
SFT corpus builder, GDPO, advantage shaping and the reward all see specs of
exactly the same shape they see under GA/MCTS.

Where the attacks come from
---------------------------
``process/vendored_attack_loaders.load_vendored_scenarios``, i.e. the same loader
``eval/vendored_replay.py`` uses -- for AgentDojo that is the ``<INFORMATION>``
block scraped out of each transcript by ``vendored_attack_parser``. The split dir
is ``AttackerConfig.vendored_dataset_dir`` and MUST be the train leaf, since
these attacks now enter training.

The one thing that is NOT dataset-shipped
-----------------------------------------
``target_turn``. Training injects by TURN INDEX (``Controller._run`` splices the
payload when ``turn == attack.target_turn``) whereas replay substitutes on TOOL
IDENTITY (``Controller.run_replay``). So a turn has to be chosen, and the only
dataset-derived answer is the carrier action's position in the original
transcript -- ``VendoredAttack.carrier_index``, clamped into the injectable
range. Consequence, and it is the same one the searching attackers live with:
the defender's own plan may put a different tool at that index, and if its clean
trajectory is shorter the payload is never delivered at all. That is visible as
``injection_visible_turn: null`` in the records, exactly as in the base-model
replay measurements.
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

from evoguard.config import AttackerConfig
from evoguard.core.types import AttackSpec, Task, ToolSpec
from evoguard.utils.logging import get_logger

logger = get_logger("attacks.vendored")

# Loading is per-PROCESS, not per-task: the AgentDojo loader re-reads and
# re-regexes all four suite files (710 records on the train split) on every call,
# and build_attacker() is called once per task. Keyed by the full argument tuple
# so a mistyped dir cannot silently reuse another split's cache.
_CACHE: dict[tuple, dict[str, list]] = {}
_CACHE_LOCK = threading.Lock()


def _load_by_task(dataset_dir: str, dataset: str,
                  suites: Optional[Sequence[str]]) -> dict[str, list]:
    """Return ``task_id -> [VendoredAttack, ...]`` for one split, cached."""

    key = (dataset_dir, dataset, tuple(suites) if suites else None)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit

    from evoguard.process.vendored_attack_loaders import load_vendored_scenarios

    scenarios = load_vendored_scenarios(
        dataset_dir=dataset_dir,
        dataset=dataset,
        suites=list(suites) if suites else None,
    )
    by_task: dict[str, list] = {}
    for va in scenarios:
        by_task.setdefault(va.task_id, []).append(va)

    logger.info(
        "vendored attacker corpus: dir=%s dataset=%s scenarios=%d tasks=%d "
        "(per-task min=%d max=%d)",
        dataset_dir, dataset, len(scenarios), len(by_task),
        min((len(v) for v in by_task.values()), default=0),
        max((len(v) for v in by_task.values()), default=0),
    )
    with _CACHE_LOCK:
        _CACHE[key] = by_task
    return by_task


class VendoredAttacker:
    """Attacker duck-type over a FROZEN, dataset-shipped population.

    Implements exactly the surface the pipeline touches -- ``current_population``
    (``rollouts/rollout.py`` phase 0 and ``pipeline/driver.py``'s population
    persistence) and ``evolve`` (``pipeline/driver.py::evolve_after_round``) --
    plus ``injectable_turn_ceiling`` / ``sanitize_spec`` for parity with
    :class:`~evoguard.attacks.genetic.GeneticAttacker`.
    """

    def __init__(
        self,
        *,
        task: Task,
        tools: Sequence[ToolSpec],
        config: AttackerConfig,
        defense_max_turns: Optional[int] = None,
    ) -> None:
        self.task = task
        self.tools = list(tools)
        self.config = config
        self.generation = 0

        # Same ceiling rule as GeneticAttacker, so target_turn ranges are
        # comparable across arms.
        ctrl_cap = int(defense_max_turns) if defense_max_turns else None
        self._inject_turn_ceiling = max(
            1, ctrl_cap if ctrl_cap is not None else len(self.tools)
        )

        dataset_dir = (getattr(config, "vendored_dataset_dir", "") or "").strip()
        if not dataset_dir:
            raise ValueError(
                "search_method='vendored' requires attacker.vendored_dataset_dir "
                "to name the TRAIN split leaf dir (e.g. "
                "'data/toolsafe/agentdojo-tragjnew/train'). Refusing to fall back "
                "to the unsplit default, which would mix in val/test attacks."
            )
        # The env name is the task_id's first field (ToolSafeEnv builds
        # "<dataset>:<suite>:<sha1[:10]>"), and the loader must be handed the SAME
        # name or it emits task_ids that match nothing.
        dataset = str(task.task_id).split(":", 1)[0] or "agentdojo"
        by_task = _load_by_task(dataset_dir, dataset, None)

        self._population: list[AttackSpec] = [
            self.sanitize_spec(self._to_spec(va))
            for va in by_task.get(task.task_id, [])
        ]
        if not self._population:
            # Not fatal: rollout.py treats an empty population as "this task
            # contributes no attacked trajectories" and still collects its clean
            # arm, which is what we want -- dropping the task would also shrink
            # the clean corpus and make the arm non-comparable.
            logger.warning(
                "Task %s: dataset %s ships NO parseable injection; this task "
                "contributes CLEAN rollouts only in the vendored-attacker arm.",
                task.task_id, dataset_dir,
            )
        else:
            logger.info(
                "Task %s: frozen vendored population of %d attack(s) "
                "(target_turns=%s, injectable [0,%d])",
                task.task_id, len(self._population),
                sorted({s.target_turn for s in self._population}),
                self._inject_turn_ceiling - 1,
            )

    # ---- spec construction ------------------------------------------------ #
    def _to_spec(self, va) -> AttackSpec:
        return AttackSpec(
            attack_id=AttackSpec.new_id(),
            task_id=self.task.task_id,
            target_turn=int(getattr(va, "carrier_index", 0) or 0),
            # The payload is a poisoned TOOL RESULT in the source transcript, so
            # the plain in-band append is the faithful channel; anything else
            # would wrap it in delimiters the dataset never had.
            injection_channel="tool_result",
            method="vendored_replay",
            payload=va.payload,
            goal=va.goal,
            generation=0,
            parents=[],
            origin="vendored",
            metadata={
                "vendored": True,
                "target_tool": va.target_tool,
                "mal_tool": va.mal_tool,
                "mal_args": va.mal_args,
                "suite": va.suite,
                "ground_truth_score": va.ground_truth_score,
                "carrier_index": int(getattr(va, "carrier_index", 0) or 0),
            },
        )

    # ---- attacker surface ------------------------------------------------- #
    @property
    def injectable_turn_ceiling(self) -> int:
        return self._inject_turn_ceiling

    def sanitize_spec(self, spec: AttackSpec) -> AttackSpec:
        """Clamp ``target_turn`` into ``[0, ceiling-1]``, same as the GA."""

        from dataclasses import replace

        ub = self._inject_turn_ceiling - 1
        clamped = max(0, min(int(spec.target_turn), ub))
        if clamped != spec.target_turn:
            logger.warning(
                "Task %s: clamped vendored target_turn %d -> %d",
                self.task.task_id, spec.target_turn, clamped,
            )
            return replace(spec, target_turn=clamped)
        return spec

    def current_population(self) -> list[AttackSpec]:
        return list(self._population)

    def evolve(self, evaluated: list) -> list[AttackSpec]:
        """No-op. The whole point of this backend is a frozen population."""

        self.generation += 1
        logger.info(
            "Task %s: vendored attacker does NOT evolve (gen bookkeeping %d, "
            "%d evals ignored, population still %d)",
            self.task.task_id, self.generation,
            len(evaluated or []), len(self._population),
        )
        return list(self._population)


__all__ = ["VendoredAttacker"]
