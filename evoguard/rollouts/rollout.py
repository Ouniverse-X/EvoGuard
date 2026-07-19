"""Dual-trajectory collection driver (``docs/plan.md``, ``rollouts/intro.md``).

This is the glue that, for every task in a round, produces the pair described in
the plan:

* trajectory A (clean tool use), and
* one attacked trajectory (B or C) per attack individual in the task's current
  genetic population.

Because the behavior turning point of an attacked trajectory is measured against
its *clean twin*, A is always rolled out first and cached, then handed to every
attacked rollout for the same task. The result is a flat list of
:class:`TrajectoryRecord` plus, per attack, the fitness needed by the genetic
algorithm (attached via :class:`~evoguard.attacks.EvaluatedAttack`).

Two-layer concurrency
---------------------
Both the outer task loop AND the inner per-task attack loop can fan out into a
``ThreadPoolExecutor`` so that many remote-LLM HTTP round-trips overlap in time.
This matters because each GLM thinking-model call costs ~25 s of wall-clock,
most spent waiting on network IO -- with concurrency enabled, throughput scales
roughly linearly until saturating your paid-endpoint RPM quota.

The two knobs live on :class:`~evoguard.config.PipelineConfig`
(``task_concurrency``, ``attack_concurrency``). Setting either to ``<=1``
disables parallelism at that layer cleanly. Default product 4 × 4 = 16 peak
in-flight requests stays comfortably below Baidu Qianfan's default tier ceiling
(RPM=60 / TPM=250K) while still saturating available bandwidth.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from evoguard.attacks.genetic import EvaluatedAttack, GeneticAttacker
from evoguard.config import ProcessConfig
from evoguard.controller import Controller
from evoguard.core.types import AttackOutcome, Task, TrajectoryKind, TrajectoryRecord
from evoguard.judge import AttackJudge
from evoguard.rollouts.base import AttackedRollout, CleanRollout
from evoguard.utils.logging import get_logger

logger = get_logger("rollouts.driver")


@dataclass
class RoundRollouts:
    """All records collected in a round plus per-task attack evaluations."""

    records: list[TrajectoryRecord] = field(default_factory=list)
    # task_id -> list of evaluated attacks (for GeneticAttacker.evolve).
    evaluations: dict[str, list[EvaluatedAttack]] = field(default_factory=dict)

    def attacked_records(self) -> list[TrajectoryRecord]:
        return [r for r in self.records if r.kind is TrajectoryKind.ATTACKED]

    def success_count(self) -> int:
        return sum(
            1
            for r in self.records
            if r.kind is TrajectoryKind.ATTACKED and r.outcome is AttackOutcome.SUCCESS
        )

    def attack_total(self) -> int:
        return len(self.attacked_records())


def collect_tri_rollouts(
    controller: Controller,
    tasks: list[Task],
    attackers: dict[str, GeneticAttacker],
    judge: AttackJudge,
    process_config: ProcessConfig,
    round_id: int,
    *,
    task_concurrency: int = 1,
    attack_concurrency: int = 1,
) -> RoundRollouts:
    """Collect the (A, B/C...) records for every task in a round.

    Parameters mirror plan.md terminology:

      * ``tasks``       -- subset to roll out this round; usually train split only.
      * ``attackers``   -- one GA instance per task_id holding its current population.
                           Populations are pre-computed SYNCHRONOUSLY here before any
                           thread dispatch begins, sidestepping lazy-seeding race
                           conditions inside :meth:`GeneticAttacker.current_population`.
      * ``judge``        -- Attack-success judge invoked once per attacked trajectory.
      * ``process_config``-- Signal-computation settings passed through to AttackedRollout.
      * ``round_id``     -- Current co-evolution round index used for record tagging.
      * ``task_concurrency`` / ``attack_concurrency`` --
                            Two independent fan-out layers. Effective peak outbound
                            request count ≈ their product; bounded externally by paid-
                            endpoint rate quota. Either <=1 disables that layer's pool.
    """

    clean_runner = CleanRollout(controller, round_id)
    attacked_runner = AttackedRollout(controller, round_id, judge, process_config)

    # ------------------------------------------------------------------ #
    # Phase 0 — pre-compute populations synchronously                    #
    # ------------------------------------------------------------------ #
    # Avoids races where multiple threads simultaneously trigger lazy seed()
    # generation when attacker._population happens to be empty for some task.
    # Each task gets exactly ONE seeding call now regardless of how much we
    # later parallelize downstream rollout work.
    precomputed_populations: dict[str, list] = {}
    for t in tasks:
        ga = attackers.get(t.task_id)
        if ga is None:
            continue
        try:
            precomputed_populations[t.task_id] = list(ga.current_population())
        except Exception as exc:                                          # noqa: BLE001
            logger.warning(
                "Round %d task %s: population pre-compute failed (%s); skipping.",
                round_id, t.task_id, exc,
            )
            precomputed_populations[t.task_id] = []

    n_total_attacks_planned = sum(len(v) for v in precomputed_populations.values())
    logger.info(
        "Round %d dispatching tasks=%d attacks=%d "
        "(concurrency task=%d attack=%d)",
        round_id, len(tasks), n_total_attacks_planned,
        max(1, int(task_concurrency)), max(1, int(attack_concurrency)),
    )

    result = RoundRollouts()

    # ------------------------------------------------------------------ #
    # Per-task worker body                                               #
    # ------------------------------------------------------------------ #
    def _run_one_task(task: Task) -> tuple[
        "TrajectoryRecord | None",
        dict[int, tuple["TrajectoryRecord", EvaluatedAttack]],
    ]:
        """Roll out trajectory A then all N attacks for this single task.

        Returns ``(clean_record_or_None, {attack_index_in_population -> (record, eval)})``
        keyed by index-in-population rather than spec identity so the caller can
        reassemble evaluations back into original order deterministically even if
        threads complete out-of-order.
        """

        try:
            clean_record = clean_runner.rollout(task)
        except Exception as exc:                                          # noqa: BLE001
            logger.error("Round %d task %s CLEAN rollout failed: %s",
                         round_id, task.task_id, exc)
            return None, {}

        pop_for_this_task = precomputed_populations.get(task.task_id) or []
        local_results: dict[int, tuple[TrajectoryRecord, EvaluatedAttack]] = {}
        if not pop_for_this_task or attack_concurrency <= 1:
            # Sequential inner path keeps ordering trivially correct & avoids
            # spawning a worker-per-item overhead for tiny populations.
            for idx, atk_spec in enumerate(pop_for_this_task):
                rec = _safe_attack(attacked_runner, task, atk_spec,
                                   clean_record.trajectory, round_id)
                local_results[idx] = (
                    rec,
                    EvaluatedAttack(
                        spec=atk_spec,
                        fitness=rec.fitness(),
                        success=(rec.outcome is AttackOutcome.SUCCESS),
                        metadata={"record_id": rec.record_id},
                    ),
                )
            return clean_record, local_results

        def _one_attack(idx_and_spec):
            idx, spec = idx_and_spec
            rec = _safe_attack(attacked_runner, task, spec,
                               clean_record.trajectory, round_id)
            return idx, rec, EvaluatedAttack(
                spec=spec,
                fitness=rec.fitness(),
                success=(rec.outcome is AttackOutcome.SUCCESS),
                metadata={"record_id": rec.record_id},
            )

        with ThreadPoolExecutor(max_workers=int(attack_concurrency)) as pool:
            futures = [pool.submit(_one_attack, (i, s))
                       for i, s in enumerate(pop_for_this_task)]
            for fut in as_completed(futures):
                try:
                    idx, rec, ev = fut.result()
                    local_results[idx] = (rec, ev)
                except Exception as exc:                                  # noqa: BLE001
                    logger.error(
                        "Round %d task %s: attack future raised: %s",
                        round_id, task.task_id, exc,
                    )
        return clean_record, local_results

    # ------------------------------------------------------------------ #
    # Outer-level dispatcher                                              #
    # ------------------------------------------------------------------ #
    if not tasks or task_concurrency <= 1:
        # Fully sequential fallback preserves exact prior behavior when both
        # knobs are off (--smoke test relies on deterministic ordering).
        ordered_outcomes = []
        for t in tasks:
            ordered_outcomes.append((t, _run_one_task(t)))
    else:
        indexed_futures: dict = {}
        with ThreadPoolExecutor(max_workers=max(1, int(task_concurrency))) as pool:
            for t in tasks:
                f = pool.submit(_run_one_task, t)
                indexed_futures[f] = t
            collected_pairs: list[tuple[Task, object]] = []
            for f in as_completed(indexed_futures):
                t_done = indexed_futures[f]
                try:
                    outcome = f.result()
                except Exception as exc:                                  # noqa: BLE001
                    logger.error(
                        "Round %d task %s OUTER failed: %s",
                        round_id, t_done.task_id, exc,
                    )
                    outcome = (None, {})
                collected_pairs.append((t_done, outcome))

        # Re-sort by input-task order so persisted records stay stable across runs.
        order_index = {t.task_id: i for i, t in enumerate(tasks)}
        ordered_outcomes = sorted(collected_pairs,
                                   key=lambda p: order_index[p[0].task_id])

    # ------------------------------------------------------------------ #
    # Assemble final results in canonical order                          #
    # ------------------------------------------------------------------ #
    for t, (clean_record, attk_map_by_idx) in ordered_outcomes:
        if clean_record is not None:
            result.records.append(clean_record)

        pop_list = precomputed_populations.get(t.task_id) or []
        evals_ordered: list[EvaluatedAttack] = []
        for idx in range(len(pop_list)):
            pair = attk_map_by_idx.get(idx)
            if pair is None:
                continue
            rec_obj, ev_obj = pair
            result.records.append(rec_obj)
            evals_ordered.append(ev_obj)
        result.evaluations[t.task_id] = evals_ordered
        logger.info(
            "Round %d task %s: %d attacks, %d successful",
            round_id,
            t.task_id,
            len(evals_ordered),
            sum(1 for e in evals_ordered if e.success),
        )

    return result


# ---------------------------------------------------------------------------- #
# Helpers                                                                       #
# ---------------------------------------------------------------------------- #
def _safe_attack(runner: AttackedRollout, task: Task, attack, clean_traj, rid: int):
    """Run one attacked-trajectory rollout capturing exceptions instead of letting them kill workers."""
    try:
        return runner.rollout(task, attack=attack, clean=clean_traj)
    except Exception as exc:                                              # noqa: BLE001 - keep pool alive
        # Synthesize a minimal FAIL-record placeholder so bookkeeping stays consistent;
        # judge never got called but downstream metrics treat it as C-class fail.
        empty_traj = type(clean_traj)(task_id=task.task_id,
                                       actions=[],
                                       kind=TrajectoryKind.ATTACKED)
        fake_record = TrajectoryRecord(
            record_id=TrajectoryRecord.new_id(),
            round_id=rid,
            task_id=task.task_id,
            kind=TrajectoryKind.ATTACKED,
            trajectory=empty_traj,
            outcome=AttackOutcome.FAIL,
            attack=attack,                       # reuse incoming spec verbatim -- no rebuild needed
            signals=None,
            utility=None,
            metadata={"error": str(exc)[:500]},
        )
        logger.warning(
            "[round %d task %s] attack error captured→FAIL placeholder: %s",
            rid, task.task_id, str(exc)[:200],
        )
        return fake_record
