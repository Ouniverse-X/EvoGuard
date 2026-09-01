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
import time
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


_VLLM_HEALTH_MAX_WAIT = 600  # seconds to wait for vLLM to come back
_VLLM_HEALTH_POLL_INTERVAL = 10  # poll interval


def _ensure_vllm_healthy(controller: Controller) -> None:
    """Block until the defender vLLM endpoint responds, restarting if needed.

    The MCTS attacker population pre-compute phase can take hours calling remote
    QianFan API; meanwhile the local vLLM server may crash / get killed / run OOM.
    This check runs AFTER population pre-compute and BEFORE any clean/attacked
    rollout dispatch, giving us the chance to detect+restart vLLM so rollouts
    don't all fail with Connection errors.

    No-ops for the ``mock`` backend: it serves no HTTP endpoint, so the probe
    would fall back to the default ``127.0.0.1:8000`` URL, fail, shell out to
    ``scripts/start_vllm.sh`` and then poll for ``_VLLM_HEALTH_MAX_WAIT``
    seconds -- turning the offline smoke run (documented as needing no GPU and
    no network) into a ten-minute stall per round.
    """
    import subprocess

    # Extract base_url from the defense agent's LLM client config.
    agent = controller.agent
    base_url = getattr(getattr(agent, "config", None), "llm", None)
    if base_url is None:
        return
    if str(getattr(base_url, "backend", "")).lower() == "mock":
        logger.info("[vllm_health] mock backend: no endpoint to probe, skipping")
        return
    base_url_str = getattr(base_url, "base_url", None) or "http://127.0.0.1:8000/v1"
    # Strip /v1 suffix to get health URL.
    health_url = base_url_str.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[:-3]
    models_url = health_url.rstrip("/") + "/v1/models"

    def _is_healthy() -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    if _is_healthy():
        logger.info("[vllm_health] defender endpoint responsive at %s", models_url)
        return

    logger.warning(
        "[vllm_health] defender endpoint NOT responding at %s; "
        "attempting restart via scripts/start_vllm.sh",
        models_url,
    )

    # Try restarting vLLM via the project's start script.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    start_script = os.path.join(repo_root, "scripts", "start_vllm.sh")

    if os.path.isfile(start_script):
        # Remove stale PID file so start_vllm.sh doesn't bail early.
        pid_file = os.path.join(repo_root, "rounds", "vllm.pid")
        if os.path.isfile(pid_file):
            try:
                os.remove(pid_file)
            except OSError:
                pass

        try:
            subprocess.run(
                ["bash", start_script],
                cwd=repo_root,
                timeout=_VLLM_HEALTH_MAX_WAIT,
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error("[vllm_health] start_vllm.sh failed: %s", exc)

    # Poll until healthy or timeout.
    deadline = time.time() + _VLLM_HEALTH_MAX_WAIT
    while time.time() < deadline:
        if _is_healthy():
            logger.info("[vllm_health] defender endpoint recovered at %s", models_url)
            return
        time.sleep(_VLLM_HEALTH_POLL_INTERVAL)

    logger.error(
        "[vllm_health] defender endpoint STILL not responding after %ds; "
        "rollouts will likely fail.",
        _VLLM_HEALTH_MAX_WAIT,
    )


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
    eligible_pairs = [
        (t.task_id, attackers[t.task_id])
        for t in tasks
        if attackers.get(t.task_id) is not None
    ]

    def _precompute_one(tid, ga):
        return tid, list(ga.current_population())

    # Tasks whose seeding raised. A task left with an empty population
    # contributes zero attacked trajectories, which shrinks the round's
    # training set without any error surfacing -- historically caused by the
    # attacker gateway's RPM quota exhausting the client's retries. Retried
    # serially below (the QianFan client paces process-wide, so a serial retry
    # sees an uncontended window) before being reported as data loss.
    failed_precompute: dict[str, str] = {}

    n_precompute_workers = max(
        1, min(int(task_concurrency), len(eligible_pairs) or 1)
    )
    if len(eligible_pairs) > 1 and n_precompute_workers > 1:
        # Attacker trees are per-task instances with no shared mutable state
        # (prewarm cache is read-only by now; QianFan/openai clients issue
        # stateless per-call HTTP requests), so cross-task fan-out is safe.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            "Round %d pre-computing populations for %d tasks with %d workers",
            round_id, len(eligible_pairs), n_precompute_workers,
        )
        with ThreadPoolExecutor(max_workers=n_precompute_workers) as pool:
            futures = {
                pool.submit(_precompute_one, tid, ga): tid
                for tid, ga in eligible_pairs
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    _, pop = fut.result()
                    precomputed_populations[tid] = pop
                except Exception as exc:                                  # noqa: BLE001
                    failed_precompute[tid] = str(exc)
    else:
        for tid, ga in eligible_pairs:
            try:
                precomputed_populations[tid] = list(ga.current_population())
            except Exception as exc:                                      # noqa: BLE001
                failed_precompute[tid] = str(exc)

    if failed_precompute:
        attacker_by_tid = dict(eligible_pairs)
        logger.warning(
            "Round %d: %d task(s) failed population pre-compute; retrying serially: %s",
            round_id, len(failed_precompute), ", ".join(sorted(failed_precompute)),
        )
        for tid in sorted(failed_precompute):
            try:
                precomputed_populations[tid] = list(
                    attacker_by_tid[tid].current_population()
                )
                del failed_precompute[tid]
            except Exception as exc:                                      # noqa: BLE001
                failed_precompute[tid] = str(exc)

    for tid, err in sorted(failed_precompute.items()):
        precomputed_populations[tid] = []
        logger.error(
            "Round %d task %s: population pre-compute FAILED after serial retry "
            "(%s). This task contributes NO attacked trajectories -- the round's "
            "training set is incomplete.",
            round_id, tid, err,
        )

    n_total_attacks_planned = sum(len(v) for v in precomputed_populations.values())
    logger.info(
        "Round %d dispatching tasks=%d attacks=%d "
        "(concurrency task=%d attack=%d)",
        round_id, len(tasks), n_total_attacks_planned,
        max(1, int(task_concurrency)), max(1, int(attack_concurrency)),
    )

    # ------------------------------------------------------------------ #
    # Ensure vLLM defender server is reachable before starting rollouts   #
    # ------------------------------------------------------------------ #
    _ensure_vllm_healthy(controller)

    # Warm-up: send a single trivial request to wake up vLLM's engine
    # and allocate GPU cache before blasting it with concurrent requests.
    try:
        from evoguard.core.types import Message, Role
        from evoguard.llm import build_client
        warmup_client = build_client(controller.agent.config.llm)
        warmup_resp = warmup_client.chat([Message(role=Role.USER, content="hello")])
        logger.info("[warmup] vLLM warmed up successfully (got %d tokens)", warmup_resp.completion_tokens)
    except Exception as warmup_exc:                                      # noqa: BLE001
        logger.warning("[warmup] failed (non-fatal): %s", warmup_exc)
        # If warmup fails, wait and retry once more to give vLLM time to ready
        time.sleep(15)
        try:
            warmup_resp = warmup_client.chat([Message(role=Role.USER, content="hello")])
            logger.info("[warmup] vLLM warmed up on retry (got %d tokens)", warmup_resp.completion_tokens)
        except Exception:                                                # noqa: BLE001
            logger.error("[warmup] vLLM still unresponsive after retry; rollouts will likely fail")

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
            # Retry once after a short wait — vLLM may have been transiently
            # unreachable (e.g. just finishing model reload after LoRA hot-load).
            err_str = str(exc).lower()
            if "connection" in err_str or "refused" in err_str or "timeout" in err_str:
                logger.warning(
                    "Round %d task %s CLEAN rollout failed (transient: %s); "
                    "waiting 30s then retrying once.",
                    round_id, task.task_id, exc,
                )
                time.sleep(30)
                try:
                    clean_record = clean_runner.rollout(task)
                except Exception as exc2:                                  # noqa: BLE001
                    logger.error("Round %d task %s CLEAN rollout retry also failed: %s",
                                 round_id, task.task_id, exc2)
                    return None, {}
            else:
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
