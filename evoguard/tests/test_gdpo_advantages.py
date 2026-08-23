"""Tests for GDPO advantage normalisation (`_gdpo_advantages`).

GDPO (NVlabs, ICML 2026, arXiv:2601.05242) replaces GRPO's
"sum the reward terms, then standardise the scalar total inside the group" with
"standardise each reward term inside the group, sum the per-term advantages, then
rescale batch-wise". The paper's exact equations could not be fetched from this
host, so the implementation is a derivation from its published description and
these tests lock the properties that derivation is meant to have:

* it is OFF by default and a no-op whenever it cannot act (G<2, empty, ragged);
* a term the group agrees on contributes exactly zero -- and, crucially, does not
  flatten the terms the group disagrees on (the whole point for EvoGuard, whose
  ``r_safety`` term spans 10 points against ``p_drift``'s 0.5);
* the advantage magnitude does not grow with the number of reward terms K;
* it resolves groups that GRPO collapses to a single advantage value.
"""

from __future__ import annotations

import math
import unittest

from evoguard.training.native_grpo_runner import _gdpo_advantages


def _grpo_advantages(components, *, num_generations):
    """Reference GRPO advantage: sum first, then standardise inside the group."""
    g = num_generations
    out = []
    for lo in range(0, len(components) - g + 1, g):
        totals = [float(sum(components[lo + j])) for j in range(g)]
        mu = sum(totals) / g
        sd = math.sqrt(sum((x - mu) ** 2 for x in totals) / g)
        out.extend([(x - mu) / (sd + 1e-4) for x in totals])
    return out


def _std(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


class TestDisabledAndDegenerateInputs(unittest.TestCase):
    """Every unusable input must return [] so the caller keeps TRL's advantages."""

    def test_group_of_one_is_a_no_op(self):
        # G=1 has no sibling to normalise against; GRPO itself is undefined here.
        self.assertEqual(_gdpo_advantages([(1.0, 2.0, 0.0)], num_generations=1), [])

    def test_zero_generations_is_a_no_op(self):
        self.assertEqual(_gdpo_advantages([(1.0,)], num_generations=0), [])

    def test_empty_batch_is_a_no_op(self):
        self.assertEqual(_gdpo_advantages([], num_generations=4), [])

    def test_batch_smaller_than_one_group_is_a_no_op(self):
        self.assertEqual(
            _gdpo_advantages([(1.0,), (2.0,)], num_generations=4), [])

    def test_ragged_component_tuples_are_rejected_wholesale(self):
        # A single malformed row must not produce a partially-normalised vector:
        # writing half a batch of advantages is worse than writing none.
        rows = [(1.0, 2.0, 0.0), (3.0, 1.0), (1.0, 2.0, 0.0), (0.0, 0.0, 0.0)]
        self.assertEqual(_gdpo_advantages(rows, num_generations=2), [])

    def test_none_row_is_rejected_wholesale(self):
        rows = [(1.0, 2.0, 0.0), None]
        self.assertEqual(_gdpo_advantages(rows, num_generations=2), [])

    def test_empty_component_tuples_are_rejected(self):
        self.assertEqual(
            _gdpo_advantages([(), ()], num_generations=2), [])

    def test_trailing_partial_group_is_dropped_not_normalised(self):
        # 5 rows at G=2 -> one full group plus a leftover. The leftover has no
        # complete group so it is truncated; length must be a multiple of G.
        rows = [(1.0,), (3.0,), (0.0,), (4.0,), (2.0,)]
        out = _gdpo_advantages(rows, num_generations=2)
        self.assertEqual(len(out), 4)


class TestPerTermDecoupling(unittest.TestCase):
    """The property that motivated the change on this project."""

    def test_unanimous_term_contributes_exactly_zero(self):
        # Two terms; term 0 is unanimous, term 1 disagrees. The unanimous term
        # must add nothing, so the result equals the single-term case.
        two = _gdpo_advantages(
            [(5.0, 1.0), (5.0, -1.0)], num_generations=2, batch_normalize=False)
        one = _gdpo_advantages(
            [(1.0,), (-1.0,)], num_generations=2, batch_normalize=False)
        self.assertEqual(len(two), 2)
        for a, b in zip(two, one):
            self.assertAlmostEqual(a, b, places=9)

    def test_all_terms_unanimous_yields_zero_advantages(self):
        out = _gdpo_advantages(
            [(2.0, 1.2, 0.0), (2.0, 1.2, 0.0)], num_generations=2)
        self.assertEqual(len(out), 2)
        for v in out:
            self.assertAlmostEqual(v, 0.0, places=9)

    def test_a_wide_agreeing_term_does_not_flatten_a_narrow_disagreeing_one(self):
        # This is the EvoGuard case verbatim: both siblings were BAITED
        # (r_safety = -8.00, unanimous and huge), but one advanced the task and
        # the other wasted the step (r_progress +1.20 vs -2.50). Under GRPO the
        # spread survives because the unanimous term cancels in the mean -- but
        # the SCALE is set by the summed total, so a third, tiny term can no
        # longer influence anything. Under GDPO each term is unit-scaled.
        rows = [(-8.0, 1.20, 0.0), (-8.0, -2.50, -0.50)]
        out = _gdpo_advantages(rows, num_generations=2, batch_normalize=False)
        # progress contributes +/-1 (its own std), drift contributes +/-1 too,
        # despite spanning 0.5 against safety's 10.
        self.assertGreater(out[0], 1.5)
        self.assertLess(out[1], -1.5)

    def test_tiny_term_gets_the_same_weight_as_a_huge_one(self):
        # p_drift's whole range is 0.5; r_safety's is 10. Per-term
        # standardisation must make a disagreement in either count the same.
        #
        # Only to ~1e-4: the denominator is ``sd + std_eps`` (additive, matching
        # TRL's own convention), so a narrow term keeps a slightly smaller
        # coefficient than a wide one -- 0.25/(0.25+1e-4)=0.99960 against
        # 5.0/(5.0+1e-4)=0.99998. That residual is 4e-4 of the signal, four
        # orders of magnitude smaller than the 20x scale gap it replaces, so it
        # is documented rather than removed (a multiplicative epsilon would be
        # scale-free but would diverge from TRL's advantage normalisation).
        drift_only = _gdpo_advantages(
            [(2.0, 1.2, 0.0), (2.0, 1.2, -0.50)],
            num_generations=2, batch_normalize=False)
        safety_only = _gdpo_advantages(
            [(2.0, 1.2, 0.0), (-8.0, 1.2, 0.0)],
            num_generations=2, batch_normalize=False)
        self.assertAlmostEqual(drift_only[0], safety_only[0], places=3)
        self.assertAlmostEqual(drift_only[1], safety_only[1], places=3)

    def test_per_term_advantages_sum_to_zero_within_each_group(self):
        rows = [(2.0, 1.2, 0.0), (-8.0, -2.5, -0.5), (-1.0, -0.15, -0.25),
                (2.0, 1.2, 0.0), (2.0, -0.15, 0.0), (-8.0, 1.2, -0.5)]
        out = _gdpo_advantages(rows, num_generations=3, batch_normalize=False)
        self.assertEqual(len(out), 6)
        self.assertAlmostEqual(sum(out[0:3]), 0.0, places=9)
        self.assertAlmostEqual(sum(out[3:6]), 0.0, places=9)


class TestGrpoCollapseIsResolved(unittest.TestCase):
    """GDPO must produce signal where GRPO produces none, and keep the ordering."""

    def test_a_two_way_tie_at_g2_is_NOT_rescued_and_that_is_expected(self):
        # Honest negative, and the reason trajectory pooling is still wired in
        # behind GDPO. At G=2 a disagreeing term always contributes exactly +/-1,
        # so if the two siblings' TOTALS tie, the term deviations sum to zero,
        # which at G=2 forces the surviving contributions to cancel pairwise.
        # Real example: the same total 0.20 reached as (HELD_BUT_FIRED, advance,
        # no drift) vs (UNCLEAR, advance, heavy drift) -- safety prefers the
        # second, drift prefers the first, exactly one point each.
        rows = [(-1.00, 1.20, 0.0), (-0.50, 1.20, -0.50)]
        self.assertAlmostEqual(sum(rows[0]), sum(rows[1]), places=9)
        out = _gdpo_advantages(rows, num_generations=2, batch_normalize=False)
        for v in out:
            self.assertAlmostEqual(v, 0.0, places=9)

    def test_group_that_grpo_collapses_gets_a_nonzero_gdpo_signal(self):
        # Abstract three-term values, not the EvoGuard tables: no exact
        # three-way-disagreeing tie exists among the real constants (the
        # achievable term gaps are +/-{0.50,2.50,3.00,7.00,7.50,10.00} on safety
        # and +/-{1.35,2.35,3.70} on progress, and none of their sums lands on a
        # +/-{0.25,0.50} drift gap). The property under test is a property of the
        # normalisation, so it is exercised on values that isolate it.
        #
        # Siblings 0 and 1 total identically by different routes, so GRPO sees
        # std=0 across the whole group and emits zero gradient for both.
        rows = [(1.0, -1.0, 0.0), (-1.0, 0.5, 0.5), (1.0, 1.0, 0.0)]
        self.assertAlmostEqual(sum(rows[0]), sum(rows[1]), places=9)

        grpo = _grpo_advantages(rows, num_generations=3)
        self.assertAlmostEqual(grpo[0], grpo[1], places=9)

        gdpo = _gdpo_advantages(rows, num_generations=3, batch_normalize=False)
        self.assertGreater(abs(gdpo[0] - gdpo[1]), 0.5)
        self.assertAlmostEqual(sum(gdpo), 0.0, places=9)

    def test_three_distinct_reward_combinations_stay_three_distinct_advantages(self):
        # The paper's Figure-1 argument: GRPO maps several distinct reward
        # combinations onto FEWER distinct advantage values, GDPO retains them.
        # Same rows as above -- GRPO collapses siblings 0 and 1 into one value,
        # GDPO keeps three.
        rows = [(1.0, -1.0, 0.0), (-1.0, 0.5, 0.5), (1.0, 1.0, 0.0)]
        grpo = _grpo_advantages(rows, num_generations=3)
        gdpo = _gdpo_advantages(rows, num_generations=3, batch_normalize=False)
        self.assertEqual(len(set(round(v, 6) for v in grpo)), 2)
        self.assertEqual(len(set(round(v, 6) for v in gdpo)), 3)

    def test_sign_ordering_follows_the_total_when_only_one_term_disagrees(self):
        # Sanity: GDPO must not invert preferences. With a single disagreeing
        # term, the better completion keeps the larger advantage.
        rows = [(2.0, 1.20, 0.0), (2.0, -2.50, 0.0)]
        out = _gdpo_advantages(rows, num_generations=2)
        self.assertGreater(out[0], out[1])


class TestScaleIndependence(unittest.TestCase):
    """Batch normalisation is what keeps the magnitude independent of K."""

    def test_batch_normalised_output_has_unit_scale_regardless_of_k(self):
        # Same disagreement replicated across 1, 3 and 6 terms. Without the
        # batch step the magnitude would grow like sqrt(K).
        for k in (1, 3, 6):
            rows = [tuple([1.0] * k), tuple([-1.0] * k)]
            out = _gdpo_advantages(rows, num_generations=2)
            self.assertAlmostEqual(_std(out), 1.0, places=3,
                                   msg=f"K={k} left the scale at {_std(out)}")

    def test_without_batch_normalisation_the_scale_does_grow_with_k(self):
        # Documents WHY the batch step exists: this is the behaviour it removes.
        rows1 = [(1.0,), (-1.0,)]
        rows6 = [tuple([1.0] * 6), tuple([-1.0] * 6)]
        s1 = _std(_gdpo_advantages(rows1, num_generations=2, batch_normalize=False))
        s6 = _std(_gdpo_advantages(rows6, num_generations=2, batch_normalize=False))
        self.assertGreater(s6, 4.0 * s1)

    def test_batch_step_is_a_rescale_not_a_recentring(self):
        # Each per-term advantage already has zero mean inside its group, so the
        # batch mean is ~0 and the batch step is effectively 1/std.
        rows = [(2.0, 1.2, 0.0), (-8.0, -2.5, -0.5),
                (2.0, -0.15, 0.0), (-1.0, 1.2, -0.25)]
        raw = _gdpo_advantages(rows, num_generations=2, batch_normalize=False)
        self.assertAlmostEqual(sum(raw) / len(raw), 0.0, places=9)
        norm = _gdpo_advantages(rows, num_generations=2)
        self.assertAlmostEqual(_std(norm), 1.0, places=3)
        # Direction preserved position by position.
        for a, b in zip(raw, norm):
            self.assertEqual(a > 0, b > 0)

    def test_degenerate_batch_is_left_unscaled_rather_than_divided_by_zero(self):
        rows = [(2.0, 1.2, 0.0), (2.0, 1.2, 0.0)]
        out = _gdpo_advantages(rows, num_generations=2)
        self.assertEqual(len(out), 2)
        for v in out:
            self.assertTrue(math.isfinite(v))


class TestGroupIsolation(unittest.TestCase):
    """Groups are independent: one group's spread must not leak into another's."""

    def test_a_second_groups_disagreement_does_not_change_the_first_groups_sign(self):
        rows = [(1.0,), (-1.0,),          # group 0
                (5.0,), (-5.0,)]          # group 1, 5x the spread
        out = _gdpo_advantages(rows, num_generations=2, batch_normalize=False)
        # Per-term standardisation is per group, so both groups land on +/-1
        # regardless of how wide the other group's disagreement is. Equality only
        # to ~1e-4 because of the additive ``std_eps`` (see the note in
        # test_tiny_term_gets_the_same_weight_as_a_huge_one).
        self.assertAlmostEqual(out[0], out[2], places=3)
        self.assertAlmostEqual(out[1], out[3], places=3)

    def test_a_fully_unanimous_group_stays_zero_next_to_an_active_one(self):
        rows = [(2.0, 1.2), (2.0, 1.2),   # group 0: unanimous
                (2.0, 1.2), (-8.0, -2.5)]  # group 1: disagrees
        out = _gdpo_advantages(rows, num_generations=2, batch_normalize=False)
        self.assertAlmostEqual(out[0], 0.0, places=9)
        self.assertAlmostEqual(out[1], 0.0, places=9)
        self.assertNotAlmostEqual(out[2], 0.0, places=3)


class TestRewardDecompositionInvariant(unittest.TestCase):
    """The components handed to GDPO must sum to the scalar reward TRL sees."""

    def test_signed_components_sum_to_the_breakdown_total(self):
        from evoguard.training.grpo_reward import RewardBreakdown

        bd = RewardBreakdown()
        bd.r_safety = -8.0
        bd.r_progress = -2.50
        bd.p_drift = 0.50
        bd.total = bd.r_safety + bd.r_progress - bd.p_drift
        # This is exactly the tuple `_score_one` builds: p_drift is stored as a
        # positive magnitude and SUBTRACTED, so it must be negated to become a
        # signed contribution. Getting this wrong would let GDPO push the policy
        # TOWARDS format drift.
        comps = (bd.r_safety, bd.r_progress, -bd.p_drift)
        self.assertAlmostEqual(sum(comps), bd.total, places=9)

    def test_real_reward_grid_keeps_the_sum_invariant(self):
        from evoguard.training.grpo_reward import (
            P_PROGRESS_WASTE, R_PROGRESS_ADVANCE, R_PROGRESS_NEUTRAL,
            R_SAFETY_BAITED, R_SAFETY_CLEAN_SERVED, R_SAFETY_HELD,
            R_SAFETY_HELD_BUT_FIRED, R_SAFETY_UNCLEAR,
        )

        safeties = (R_SAFETY_HELD, R_SAFETY_HELD_BUT_FIRED, R_SAFETY_BAITED,
                    R_SAFETY_UNCLEAR, R_SAFETY_CLEAN_SERVED)
        progresses = (R_PROGRESS_ADVANCE, R_PROGRESS_NEUTRAL, P_PROGRESS_WASTE)
        drifts = (0.0, 0.25, 0.50)
        for s in safeties:
            for p in progresses:
                for d in drifts:
                    comps = (s, p, -d)
                    self.assertAlmostEqual(sum(comps), s + p - d, places=9)

    def test_unknown_row_fallback_tuple_matches_its_scalar(self):
        # `_score_one` returns (-0.5, (-0.5, 0.0, 0.0)) for a row with no meta.
        self.assertAlmostEqual(sum((-0.5, 0.0, 0.0)), -0.5, places=9)


class TestConfigDefault(unittest.TestCase):
    """Default OFF: an existing config must reproduce legacy GRPO exactly."""

    def test_grpo_gdpo_defaults_to_false(self):
        from evoguard.config import TrainingConfig

        self.assertFalse(TrainingConfig().grpo_gdpo)


class TestRewardTracePlumbing(unittest.TestCase):
    """The trace is the only channel carrying the pre-sum decomposition.

    TRL's ``_generate_and_score_completions`` surfaces advantages but not
    rewards, and never the per-term breakdown, so the reward closure stashes both
    in a single-slot list the trainer hook reads back. A mistake here would
    silently disable GDPO *and* trajectory pooling with no error anywhere, which
    is why the widening from a 2-tuple to a 3-tuple is tested directly.
    """

    def _closure(self, sink):
        from evoguard.training.native_grpo_runner import (
            build_evoguard_reward_callable,
        )
        # No metas registered -> every row takes the unknown-row default, which
        # is enough to exercise the trace shape without a judge or a model.
        return build_evoguard_reward_callable({}, reward_trace_sink=sink)

    def test_trace_carries_totals_row_indices_and_components(self):
        sink: list = []
        fn = self._closure(sink)
        out = fn(["p0", "p1"], ["c0", "c1"], row_idx=[7, 9])
        self.assertEqual(out, [-0.5, -0.5])
        self.assertEqual(len(sink), 1)
        totals, idxs, comps = sink[0][0], sink[0][1], sink[0][2]
        self.assertEqual(totals, [-0.5, -0.5])
        self.assertEqual(idxs, [7, 9])
        self.assertEqual(len(comps), 2)
        for total, comp in zip(totals, comps):
            self.assertAlmostEqual(sum(comp), total, places=9)

    def test_trl_still_receives_only_the_scalar_totals(self):
        # The widening must not change the reward function's return value --
        # anything else would alter what TRL optimises.
        sink: list = []
        out = self._closure(sink)(["p"], ["c"], row_idx=[3])
        self.assertEqual(out, [-0.5])
        self.assertTrue(all(isinstance(v, float) for v in out))

    def test_slot_is_overwritten_not_appended(self):
        sink: list = []
        fn = self._closure(sink)
        fn(["p"], ["c"], row_idx=[1])
        fn(["p"], ["c"], row_idx=[2])
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0][1], [2])


class TestAlignedTraceView(unittest.TestCase):
    """Alignment guard: refuse the batch rather than mix up prompts."""

    def _view(self, trace, row_idxs):
        from evoguard.training.native_grpo_runner import _aligned_trace_view

        return _aligned_trace_view(trace, row_idxs)

    def test_aligned_three_tuple_returns_both_views(self):
        trace = [([1.0, 2.0], [4, 5], [(1.0, 0.0), (2.0, 0.0)])]
        rewards, comps = self._view(trace, [4, 5])
        self.assertEqual(rewards, [1.0, 2.0])
        self.assertEqual(len(comps), 2)

    def test_legacy_two_tuple_yields_rewards_but_no_components(self):
        # Pooling must keep working; GDPO must decline rather than guess.
        trace = [([1.0, 2.0], [4, 5])]
        rewards, comps = self._view(trace, [4, 5])
        self.assertEqual(rewards, [1.0, 2.0])
        self.assertIsNone(comps)

    def test_length_mismatch_rejects_the_batch(self):
        trace = [([1.0, 2.0], [4, 5], [(1.0,), (2.0,)])]
        self.assertEqual(self._view(trace, [4, 5, 6]), (None, None))

    def test_row_index_mismatch_rejects_the_batch(self):
        trace = [([1.0, 2.0], [4, 5], [(1.0,), (2.0,)])]
        self.assertEqual(self._view(trace, [4, 6]), (None, None))

    def test_empty_trace_rejects_the_batch(self):
        self.assertEqual(self._view([], [1, 2]), (None, None))
        self.assertEqual(self._view(None, [1, 2]), (None, None))

    def test_non_integer_row_index_rejects_the_batch(self):
        trace = [([1.0], [None], [(1.0,)])]
        self.assertEqual(self._view(trace, [None]), (None, None))

    def test_short_component_list_is_dropped_but_rewards_survive(self):
        # A truncated components list must not be zipped against a longer reward
        # vector -- that would normalise the wrong groups.
        trace = [([1.0, 2.0], [4, 5], [(1.0,)])]
        rewards, comps = self._view(trace, [4, 5])
        self.assertEqual(rewards, [1.0, 2.0])
        self.assertIsNone(comps)


def main() -> None:
    unittest.main(module=__name__, argv=["test_gdpo_advantages"], exit=False)


if __name__ == "__main__":
    unittest.main()
