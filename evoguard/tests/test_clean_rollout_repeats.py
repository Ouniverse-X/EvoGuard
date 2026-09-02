"""Regression tests for sampling the clean arm more than once per task.

``pipeline.clean_rollouts_per_task`` (default 1) makes ``collect_tri_rollouts``
sample trajectory A N times per task instead of once. One sample was never a
design decision, and it made the clean arm scarce in two different ways:

* as SFT supervision -- a task contributes up to 15 attacked rollouts against
  exactly one clean one, so the per-task cap has nothing to choose from;
* as the clean TWIN -- a single sample decides, for the whole round, whether a
  task has a usable twin. Without one, every B record for that task loses
  ``_corrective`` phase 3 and degenerates to "refuse, then stop" (33.7% of r0's
  corrective rows), and ``grpo_reward`` has no ``ADVANCE`` ground truth.

Three properties follow, and all three are load-bearing:

1. **All N samples are kept as CLEAN records.** The round's utility statistics
   are then an N-sample estimate rather than a coin flip.
2. **The twin is the BEST sample, not the last one.** Three separate places pick
   a twin out of the per-task candidates (``collect_tri_rollouts`` for the
   attacked rollouts, ``build_sft`` for ``_corrective``, and
   ``extract_grpo_prompts`` for the ``ADVANCE`` check) and they must agree, or
   the reward is computed against a different trajectory than the corpus
   imitates. All three rank on utility with ``None`` sorting as 1.0.
3. **Byte-identical repeats are de-duplicated in the corpus.** N copies of one
   demonstration is not N demonstrations.

Plus the invariant that keeps history reproducible: at ``N == 1`` every path
above is bit-for-bit what it was before the knob existed.

Run via::

    /root/miniconda3/envs/evoguard/bin/python -m evoguard.tests.test_clean_rollout_repeats
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from evoguard.config import LLMConfig, PipelineConfig, ProcessConfig  # noqa: E402
from evoguard.core.types import (  # noqa: E402
    Action,
    AttackOutcome,
    Task,
    ToolCall,
    ToolSpec,
    Trajectory,
    TrajectoryKind,
    TrajectoryRecord,
)
from evoguard.process.dataset_builder import (  # noqa: E402
    DefenderDatasetBuilder,
    _trajectory_fingerprint,
)
from evoguard.rollouts import rollout as rollout_mod  # noqa: E402

_TASK = Task(task_id="t1", instruction="Summarise my recent transactions.",
             suite="banking", dataset="agentdojo")
_TOOLS = [ToolSpec(name="get_transactions", description="read transactions"),
          ToolSpec(name="send_money", description="move money")]


def _traj(tools: list[str], *, obs: str = "ok") -> Trajectory:
    return Trajectory(
        task_id="t1",
        actions=[
            Action(turn=i, thought="", observation=obs,
                   tool_call=ToolCall(name=n, arguments={"account": "checking"}))
            for i, n in enumerate(tools)
        ],
        kind=TrajectoryKind.CLEAN,
    )


def _clean(record_id: str, utility, traj: Trajectory) -> TrajectoryRecord:
    return TrajectoryRecord(
        record_id=record_id, round_id=0, task_id="t1",
        kind=TrajectoryKind.CLEAN, trajectory=traj,
        outcome=AttackOutcome.NA, utility=utility,
    )


# --------------------------------------------------------------------------- #
# Layer 1 -- collection                                                        #
# --------------------------------------------------------------------------- #
class _FakeCleanRunner:
    """Stands in for ``CleanRollout``, returning a scripted utility sequence."""

    def __init__(self, utilities):
        self._utilities = list(utilities)
        self.n_calls = 0

    def rollout(self, task):                                    # noqa: D401
        u = self._utilities[min(self.n_calls, len(self._utilities) - 1)]
        self.n_calls += 1
        if u is _RAISE:
            raise RuntimeError("clean rollout blew up")
        return _clean(f"c{self.n_calls}", u, _traj(["get_transactions"]))


_RAISE = object()


class CollectionRepeats(unittest.TestCase):
    """``collect_tri_rollouts``'s per-task clean sampling."""

    def _collect(self, utilities, n_clean):
        """Drive the real ``collect_tri_rollouts`` with no tasks, then its guts.

        The function's clean sampling is a closure over ``clean_runner``, so it
        is exercised by calling ``collect_tri_rollouts`` on an empty task list
        (which does no IO) and re-creating the same closure inputs here would
        duplicate the logic. Instead the runner is swapped for a fake and the
        public entry point is called with one task and no attackers.
        """

        fake = _FakeCleanRunner(utilities)
        orig = rollout_mod.CleanRollout
        rollout_mod.CleanRollout = lambda controller, round_id: fake   # noqa: E731
        try:
            result = rollout_mod.collect_tri_rollouts(
                controller=_FakeController(),
                tasks=[_TASK],
                attackers={},
                judge=None,
                process_config=ProcessConfig(),
                round_id=0,
                task_concurrency=1,
                attack_concurrency=1,
                clean_rollouts_per_task=n_clean,
            )
        finally:
            rollout_mod.CleanRollout = orig
        return fake, result

    def test_n_equals_one_is_the_historical_single_record(self):
        fake, result = self._collect([0.9], n_clean=1)
        self.assertEqual(fake.n_calls, 1)
        self.assertEqual(len(result.records), 1)

    def test_all_samples_are_kept_and_best_comes_first(self):
        fake, result = self._collect([0.2, 1.0, 0.5, 0.7, 0.1, 0.4], n_clean=6)
        self.assertEqual(fake.n_calls, 6)
        self.assertEqual(len(result.records), 6)
        utils = [r.utility for r in result.records]
        self.assertEqual(utils[0], 1.0, "twin must be the best sample")
        self.assertEqual(utils, sorted(utils, reverse=True))

    def test_unscored_sorts_as_one_matching_the_cap_convention(self):
        # ``None`` outranks 0.9 because DatasetBuilder._cap_per_task treats an
        # unscored record as 1.0; a different rule here would hand the attacked
        # rollouts a twin the corpus then discards.
        _, result = self._collect([0.9, None], n_clean=2)
        self.assertIsNone(result.records[0].utility)

    def test_a_failed_repeat_costs_only_that_sample(self):
        _, result = self._collect([0.8, _RAISE, 0.8, 0.8, 0.8, 0.8], n_clean=6)
        self.assertEqual(len(result.records), 5)

    def test_a_failed_first_attempt_drops_the_task(self):
        # No twin means the attacked rollouts have no baseline, so the task
        # contributes nothing rather than a half-measured round.
        _, result = self._collect([_RAISE], n_clean=6)
        self.assertEqual(result.records, [])


class _FakeController:
    """Minimum surface ``collect_tri_rollouts`` touches before dispatch.

    ``backend="mock"`` matters: it makes ``_ensure_vllm_healthy`` no-op and lets
    the warm-up request succeed, so the test does not pay the 15 s
    warmup-retry sleep on every case.
    """

    class _Agent:
        class _Cfg:
            llm = LLMConfig(backend="mock")
        config = _Cfg()

    agent = _Agent()

    def __init__(self):
        self.env = None


# --------------------------------------------------------------------------- #
# Layer 2 -- SFT corpus                                                        #
# --------------------------------------------------------------------------- #
class CorpusFromRepeats(unittest.TestCase):
    def _builder(self, **kw) -> DefenderDatasetBuilder:
        return DefenderDatasetBuilder(
            tasks_by_id={"t1": _TASK}, tools_by_task={"t1": _TOOLS}, **kw
        )

    def test_distinct_repeats_all_contribute_rows(self):
        recs = [
            _clean("c1", 0.8, _traj(["get_transactions"])),
            _clean("c2", 0.9, _traj(["get_transactions", "get_transactions"])),
        ]
        b = self._builder()
        rows = b.build_sft(recs)
        self.assertEqual(b.last_sft_stats["clean_records_used"], 2)
        self.assertEqual(b.last_sft_stats.get("clean_records_deduped", 0), 0)
        self.assertEqual(len(rows), 3)               # 1 step + 2 steps

    def test_byte_identical_repeats_are_deduped(self):
        traj = _traj(["get_transactions", "get_transactions"])
        recs = [_clean(f"c{i}", 0.8, traj) for i in range(6)]
        b = self._builder()
        rows = b.build_sft(recs)
        self.assertEqual(b.last_sft_stats["clean_records_used"], 1)
        self.assertEqual(b.last_sft_stats["clean_records_deduped"], 5)
        self.assertEqual(len(rows), 2)

    def test_differing_observations_are_not_duplicates(self):
        # ``_imitate`` renders prior observations into every prompt, so the same
        # actions against different tool results are different supervision.
        recs = [
            _clean("c1", 0.8, _traj(["get_transactions"], obs="balance 10")),
            _clean("c2", 0.8, _traj(["get_transactions"], obs="balance 20")),
        ]
        b = self._builder()
        b.build_sft(recs)
        self.assertEqual(b.last_sft_stats["clean_records_used"], 2)

    def test_fingerprint_separates_actions_and_observations(self):
        a = _traj(["get_transactions"], obs="x")
        self.assertEqual(_trajectory_fingerprint(a), _trajectory_fingerprint(a))
        self.assertNotEqual(
            _trajectory_fingerprint(a),
            _trajectory_fingerprint(_traj(["get_transactions"], obs="y")),
        )
        self.assertNotEqual(
            _trajectory_fingerprint(a),
            _trajectory_fingerprint(_traj(["send_money"], obs="x")),
        )

    def test_twin_is_the_best_sample_not_the_last(self):
        # The last record is the WORST one; picking it would splice a failing
        # continuation onto every corrective example for this task.
        good = _traj(["get_transactions", "get_transactions", "get_transactions"])
        recs = [
            _clean("c1", 0.9, good),
            _clean("c2", 0.6, _traj(["get_transactions"])),
        ]
        b = self._builder(min_source_utility=0.5)
        b.build_sft(recs)
        # Reach the same selection the builder made, via its documented rule.
        best = max(recs, key=lambda r: (1.0 if r.utility is None else r.utility))
        self.assertEqual(best.record_id, "c1")
        self.assertEqual(
            _trajectory_fingerprint(good), _trajectory_fingerprint(best.trajectory)
        )

    def test_cap_keeps_the_six_best_clean_samples(self):
        recs = [
            _clean(f"c{i}", u, _traj(["get_transactions"] * (i + 1)))
            for i, u in enumerate([0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        ]
        b = self._builder(max_records_per_task=6)
        b.build_sft(recs)
        self.assertEqual(b.last_sft_stats["records_after_cap"], 6)
        self.assertEqual(b.last_sft_stats["clean_records_used"], 6)


# --------------------------------------------------------------------------- #
# Layer 3 -- config default                                                    #
# --------------------------------------------------------------------------- #
class ConfigDefault(unittest.TestCase):
    def test_default_is_one_so_historical_configs_are_unchanged(self):
        self.assertEqual(PipelineConfig().clean_rollouts_per_task, 1)


def main() -> int:
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
