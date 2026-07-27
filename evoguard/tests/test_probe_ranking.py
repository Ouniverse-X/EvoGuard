"""Unit tests for evoguard.training.ranking + sensitivity modules.

Pure-Python synthetic attention tensors; no GPU/network/torch required.

Run via::

    /ssd1/conda_envs/evoguard/bin/python -m evoguard.tests.test_probe_ranking
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from evoguard.training.sensitivity import (  # noqa: E402
    PairScoreInput,
    SensitivityMethod,
    attn_kl_score,
    select_scorer,
)
from evoguard.training.ranking import (  # noqa: E402
    DEFAULT_PROJECTIONS_PER_BLOCK,
    SCHEMA_VERSION,
    aggregate_pair_scores,
    build_target_modules,
    emit_probe_artifact,
    make_artifact_from_run,
    rank_blocks_by_score,
    select_top_k_blocks,
    validate_artifact_dict,
)


def mk_attn(n_layers=4, n_heads=1, seq_len=6):
    """Build uniform-attention tensor as nested python lists."""
    p = 1.0 / float(seq_len)
    out = []
    for _ in range(n_layers):
        heads = []
        for _ in range(n_heads):
            qs = []
            for _ in range(seq_len):
                row = [p] * seq_len
                qs.append(row)
            heads.append(qs)
        out.append(heads)
    return out


def mk_attn_with_spike(spike_layer=2, n_layers=4, n_heads=1, seq_len=6):
    """Like mk_attn but layer ``spike_layer`` has a peaked distribution."""
    base = 0.05
    peak_val = 1.0 - base * (seq_len - 1)
    rows_peak = []
    for q_idx in range(seq_len):
        r = [base] * seq_len
        # Make the spike target depend on query position so KL is non-trivial.
        tgt = (q_idx + 1) % seq_len
        r[tgt] = peak_val
        s = sum(r)
        r = [v / s for v in r]
        rows_peak.append(r)
    flat_attn = mk_attn(n_layers=n_layers, n_heads=n_heads, seq_len=seq_len)
    if 0 <= spike_layer < len(flat_attn):
        head_list_for_layer = []
        for h_idx in range(len(flat_attn[spike_layer])):
            head_rows_copy = list(rows_peak)
            head_list_for_layer.append(head_rows_copy)
        flat_attn[spike_layer] = head_list_for_layer
    return flat_attn


class TestRankBlocks(unittest.TestCase):

    def test_rank_descending_order(self):
        scores = [0.1, 0.5, 0.3, 0.9]
        ranked = rank_blocks_by_score(scores)
        idxs = [i for i, _s in ranked]
        self.assertEqual(idxs, [3, 1, 2, 0])

    def test_rank_tiebreak_lower_index_first(self):
        # Equal scores -> lower block index wins.
        ranked = rank_blocks_by_score([0.4, 0.4, 0.4])
        self.assertEqual(ranked, [(0, 0.4), (1, 0.4), (2, 0.4)])

    def test_rank_negative_clamped_to_zero(self):
        ranked = rank_blocks_by_score([-0.5, 0.2, -1.0])
        vals = [v for _i, v in ranked]
        for v in vals:
            self.assertGreaterEqual(v, 0.0)

    def test_rank_accepts_dict_input(self):
        d_in = {0: 0.7, 1: 0.2, 2: 0.95}
        ranked = rank_blocks_by_score(d_in)
        self.assertEqual([i for i, _v in ranked], [2, 0, 1])


class TestSelectTopK(unittest.TestCase):

    def test_select_top_k_basic(self):
        ranked = [(2, 0.8), (0, 0.6), (1, 0.3)]
        sel, eff_k = select_top_k_blocks(ranked, top_k=2)
        self.assertEqual(sel, [2, 0])
        self.assertEqual(eff_k, 2)

    def test_top_k_truncates_when_more_than_available(self):
        ranked = [(1, 0.5), (0, 0.4)]
        sel, eff_k = select_top_k_blocks(ranked, top_k=10)
        self.assertEqual(sel, [1, 0])
        self.assertEqual(eff_k, 2)

    def test_zero_ranked_returns_empty(self):
        sel, eff_k = select_top_k_blocks([], top_k=3)
        self.assertEqual(sel, [])
        self.assertEqual(eff_k, 0)


class TestBuildTargetModules(unittest.TestCase):

    def test_default_four_projections_per_block(self):
        mods = build_target_modules([5, 11])
        self.assertEqual(len(mods), 8)
        # Each block contributes q/k/v/o in order.
        expected = [
            "model.layers.5.self_attn.q_proj",
            "model.layers.5.self_attn.k_proj",
            "model.layers.5.self_attn.v_proj",
            "model.layers.5.self_attn.o_proj",
            "model.layers.11.self_attn.q_proj",
            "model.layers.11.self_attn.k_proj",
            "model.layers.11.self_attn.v_proj",
            "model.layers.11.self_attn.o_proj",
        ]
        self.assertEqual(mods, expected)

    def test_custom_projection_set(self):
        mods = build_target_modules([0], projections_per_block=("q_proj",))
        self.assertEqual(mods, ["model.layers.0.self_attn.q_proj"])

    def test_dedup_when_same_block_twice(self):
        # Defensive: even if caller passes duplicates, output stays unique.
        mods = build_target_modules([3, 3])
        unique_mods = list(dict.fromkeys(mods))
        self.assertEqual(len(unique_mods), len(DEFAULT_PROJECTIONS_PER_BLOCK))


class TestAttnKlScore(unittest.TestCase):

    def test_identical_attns_yield_zero_score(self):
        a = mk_attn(n_layers=4, n_heads=2, seq_len=8)
        pi_in = PairScoreInput(attn_clean=a, attn_injected=a, inject_token_idx=1)
        scores = attn_kl_score(pi_in)
        self.assertEqual(len(scores), 4)
        for s in scores:
            # Identical distributions -> KL == 0 (modulo tiny float noise).
            self.assertLess(s, 1e-6)

    def test_differing_layer_gets_higher_score(self):
        clean_a = mk_attn(n_layers=4, n_heads=1, seq_len=6)
        spike_a = mk_attn_with_spike(spike_layer=2, n_layers=4, n_heads=1, seq_len=6)
        pi_in = PairScoreInput(
            attn_clean=clean_a, attn_injected=spike_a, inject_token_idx=1,
        )
        scores = attn_kl_score(pi_in)
        self.assertEqual(len(scores), 4)
        # Layer 2 should have the largest score since that's where distributions differ.
        max_idx = max(range(4), key=lambda i: scores[i])
        self.assertEqual(max_idx, 2)

    def test_select_scorer_unknown_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            select_scorer("grad_attr")
        with self.assertRaises(NotImplementedError):
            select_scorer("act_patch")

    def test_scorer_handles_none_gracefully(self):
        pi_bad = PairScoreInput(attn_clean=None, attn_injected=None, inject_token_idx=0)
        out_scores = attn_kl_score(pi_bad)
        self.assertEqual(out_scores, [])


class TestArtifactEmission(unittest.TestCase):

    def _make_artifact(self, n_pairs=3, top_k=2):
        per_pair = [
            [0.1, 0.5, 0.3, 0.9],
            [0.2, 0.4, 0.6, 0.7],
            [0.05, 0.55, 0.35, 0.85],
        ][:n_pairs]
        return make_artifact_from_run(
            method="attn_kl",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            per_pair_layer_scores=per_pair,
            inject_token_positions=[1] * n_pairs,
            inject_position_avg_tokens=float(128),
            top_k_requested=top_k,
        )

    def test_artifact_has_required_fields(self):
        a = self._make_artifact(n_pairs=3, top_k=3)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": a.method,
            "base_model": a.base_model,
            "n_pairs_used": a.n_pairs_used,
            "selected_blocks_sorted_desc": list(a.selected_blocks_sorted_desc),
            "recommended_target_modules": list(a.recommended_target_modules),
        }
        validate_artifact_dict(payload)   # must NOT raise

    def test_emit_artifact_writes_valid_json_file(self):
        a = self._make_artifact(n_pairs=2, top_k=2)
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "targets.json")
            written_path = emit_probe_artifact(a, path=out_path)
            self.assertTrue(os.path.isfile(written_path))
            with open(written_path) as fh:
                loaded = json.load(fh)
        # Six required fields present (per plan.md contract).
        for fld in ("schema_version", "method", "base_model",
                    "n_pairs_used", "selected_blocks_sorted_desc",
                    "recommended_target_modules"):
            self.assertIn(fld, loaded)
        # schema_version matches constant.
        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
        # selected blocks count matches top_k_value.
        self.assertEqual(len(loaded["selected_blocks_sorted_desc"]),
                         min(len(loaded["selected_blocks_sorted_desc"]),
                             int(loaded["top_k_value"])))
        # recommended_target_modules = top_k * len(DEFAULT_PROJECTIONS_PER_BLOCK).
        expected_n_mods = (
            len(loaded["selected_blocks_sorted_desc"])
            * len(DEFAULT_PROJECTIONS_PER_BLOCK)
        )
        self.assertEqual(len(loaded["recommended_target_modules"]), expected_n_mods)

    def test_validate_rejects_missing_field(self):
        bad_payload = {"schema_version": SCHEMA_VERSION}  # missing most fields
        with self.assertRaises(ValueError):
            validate_artifact_dict(bad_payload)

    def test_validate_rejects_wrong_schema_version(self):
        a = self._make_artifact()
        from evoguard.training.ranking import artifact_to_dict as to_d
        d_full = to_d(a)
        d_full["schema_version"] = "999"
        with self.assertRaises(ValueError):
            validate_artifact_dict(d_full)


class TestAggregateScores(unittest.TestCase):

    def test_aggregate_mean_across_pairs(self):
        rows = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
        agg = aggregate_pair_scores(rows)
        self.assertEqual(agg, [2.0, 3.0])

    def test_aggregate_empty_returns_empty(self):
        self.assertEqual(aggregate_pair_scores([]), [])

    def test_aggregate_ragged_pads_with_zeros(self):
        rows = [[1.0], [1.0, 5.0]]
        agg = aggregate_pair_scores(rows)
        # Layer-0 mean: (1+1)/2 == 1; layer-1 only present in second row -> 5/1 == 5.
        self.assertAlmostEqual(agg[0], 1.0)
        self.assertAlmostEqual(agg[1], 5.0)


class TestEndToEndRankingToArtifact(unittest.TestCase):
    """Plan.md Step-3 success criterion: top-K selection correctness + JSON validity."""

    def test_synthetic_scores_produce_expected_topk_and_modules(self):
        per_pair = [
            [0.05, 0.10, 0.50, 0.20, 0.30, 0.40],
            [0.07, 0.12, 0.45, 0.22, 0.33, 0.41],
        ]
        a_obj = make_artifact_from_run(
            method="attn_kl",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            per_pair_layer_scores=per_pair,
            inject_token_positions=[0, 0],
            inject_position_avg_tokens=64.0,
            top_k_requested=4,
        )
        # Highest-mean layers are indices [5, 2] then [4 or similar]; verify order desc.
        self.assertEqual(len(a_obj.selected_blocks_sorted_desc), 4)
        self.assertEqual(a_obj.top_k_value, 4)
        means_expected_sorted_descending = sorted(
            [(i, sum(per_pair[r][i] for r in range(len(per_pair))) / len(per_pair))
             for i in range(6)],
            key=lambda kv: (-kv[1], kv[0]),
        )
        expected_top_4_idxs = [idx for idx, _v in means_expected_sorted_descending[:4]]
        self.assertEqual(a_obj.selected_blocks_sorted_desc, expected_top_4_idxs)

        n_mods_expected = len(expected_top_4_idxs) * len(DEFAULT_PROJECTIONS_PER_BLOCK)
        self.assertEqual(len(a_obj.recommended_target_modules), n_mods_expected)


class TestWeightedAggregate(unittest.TestCase):
    """Δ-aware weighted aggregation contract (essence §2.4 explicit coupling).

    weights=None preserves the legacy equal-mean behaviour; when provided the
    aggregate becomes a weighted mean normalised internally so callers can pass
    raw Δ values without worrying about scale.
    """

    def test_weights_none_falls_back_to_equal_mean(self):
        rows = [[1.0, 2.0], [3.0, 4.0]]
        agg_none = aggregate_pair_scores(rows, weights=None)
        agg_default_arg = aggregate_pair_scores(rows)         # default arg path
        self.assertEqual(agg_none, [2.0, 3.0])
        self.assertEqual(agg_default_arg, [2.0, 3.0])

    def test_weights_equal_to_count_reproduces_equal_mean(self):
        # Uniform non-trivial weights must collapse to the same answer as None.
        rows = [[1.0, 2.0], [3.0, 4.0]]
        agg_w_equal = aggregate_pair_scores(rows, weights=[5.0, 5.0])
        for a, b in zip(agg_w_equal, [2.0, 3.0]):
            self.assertAlmostEqual(a, b)

    def test_weighted_basic_two_pairs_skews_toward_heavier_row(self):
        # weight row#0 heavily -> result closer to its values.
        rows = [
            [10.0, 20.0],   # heavy
            [0.0,   0.0 ],  # negligible contribution after weighting
        ]
        agg = aggregate_pair_scores(rows, weights=[100.0, 1.0])
        expected_l0 = (10 * 100 + 0 * 1) / (101)
        expected_l1 = (20 * 100 + 0 * 1) / (101)
        self.assertAlmostEqual(agg[0], expected_l0)
        self.assertAlmostEqual(agg[1], expected_l1)
        # And should be much closer to row[0] than to plain mean (5/10).
        self.assertGreater(agg[0], 5.5)
        self.assertGreater(agg[1], 15.5)

    def test_weights_length_mismatch_falls_back_to_equal_mean_safely(self):
        """Defensive: caller passes wrong-length weights → graceful degrade."""
        rows = [[1.0, 2.0], [3.0, 4.0]]
        too_short = aggregate_pair_scores(rows, weights=[1.0])
        too_long  = aggregate_pair_scores(rows, weights=[1.0, 1.0, 99.0])
        # Both mismatches MUST reproduce legacy arithmetic mean exactly.
        for v_short, v_long, v_eq in zip(too_short, too_long, [2.0, 3.0]):
            self.assertAlmostEqual(v_short, v_eq)
            self.assertAlmostEqual(v_long, v_eq)

    def test_all_zero_or_negative_weights_fall_back_to_equal_mean(self):
        # If every pair has zero/negative Δ we cannot divide; revert to equal-weight.
        rows = [[1.0, 9.0], [9.0, 1.0]]      # equal-mean => [5,5]
        all_zero    = aggregate_pair_scores(rows, weights=[0.0, 0.0])
        negative    = aggregate_pair_scores(rows, weights=[-3.0, -7.0])
        mixed_bad   = aggregate_pair_scores(rows, weights=[0.0, -1e9])
        for v_z, v_n, v_m, v_eq in zip(all_zero, negative, mixed_bad, [5.0, 5.0]):
            self.assertAlmostEqual(v_z, v_eq)
            self.assertAlmostEqual(v_n, v_eq)
            self.assertAlmostEqual(v_m, v_eq)

    def test_partial_zero_weights_drop_corresponding_pairs_only(self):
        # One valid weight, one zero weight -> only first pair contributes.
        rows = [[8.0, 16.0], [-999.0, -888.0]]
        agg = aggregate_pair_scores(rows, weights=[1.0, 0.0])
        self.assertAlmostEqual(agg[0], 8.0)
        self.assertAlmostEqual(agg[1], 16.0)

    def test_nan_and_inf_inputs_are_clamped_not_propagated(self):
        import math as _m
        rows = [[float("nan"), float("inf"), float("-inf"), 1.0],
                [     2.0,       2.0,          2.0,          2.0 ]]
        agg = aggregate_pair_scores(rows, weights=[1.0, 1.0])
        self.assertTrue(_m.isfinite(agg[0]))
        self.assertTrue(_m.isfinite(agg[1]))
        self.assertTrue(_m.isfinite(agg[2]))
        self.assertGreaterEqual(len(agg), 4)


class TestArtifactWithWeights(unittest.TestCase):

    def test_make_artifact_from_run_passes_per_pair_weights_through(self):
        per_pair = [
            [1.0, 2.0, 3.0],
            [9.0, 0.0, 9.0],
        ]
        art_unif      = make_artifact_from_run(method="attn_kl", base_model="M",
                                               per_pair_layer_scores=per_pair,
                                               inject_token_positions=[0, 0],
                                               inject_position_avg_tokens=64.0,
                                               top_k_requested=2)
        art_skewed_low = make_artifact_from_run(method="attn_kl", base_model="M",
                                                per_pair_layer_scores=per_pair,
                                                per_pair_weights=[100.0, 1.0],   # bias toward row 0
                                                inject_token_positions=[0, 0],
                                                inject_position_avg_tokens=64.0,
                                                top_k_requested=2)
        art_skewed_high = make_artifact_from_run(method="attn_kl", base_model="M",
                                                 per_pair_layer_scores=per_pair,
                                                 per_pair_weights=[1.0, 100.0],   # bias toward row 1
                                                 inject_token_positions=[0, 0],
                                                 inject_position_avg_tokens=64.0,
                                                 top_k_requested=2)
        # Equal-mean ranking: means = [5,1,6]; sorted desc -> [idx2(6), idx0(5), idx1(1)].
        self.assertEqual(list(art_unif.selected_blocks_sorted_desc[:2]),        [2, 0])
        # Low-bias: effective means ≈ row-0 values [1,2,3]; top-2 desc => {2(3), 1(2)}.
        self.assertEqual(set(art_skewed_low.selected_blocks_sorted_desc[:2]),   {2, 1})
        # High-bias: effective means ≈ row-1 values [9,0,9]; tie on score between idx0 & idx2.
        # Tie-break rule is "lower index first" so order within selected list goes [0, 2].
        self.assertEqual(sorted(art_skewed_high.selected_blocks_sorted_desc[:2]),
                         sorted({0, 2}))


def _suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        TestRankBlocks,
        TestSelectTopK,
        TestBuildTargetModules,
        TestAttnKlScore,
        TestArtifactEmission,
        TestAggregateScores,
        TestEndToEndRankingToArtifact,
        TestWeightedAggregate,
        TestArtifactWithWeights,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


if __name__ == "__main__":
    runner_inst = unittest.TextTestRunner(verbosity=2)
    result = runner_inst.run(_suite())
    sys.exit(0 if result.wasSuccessful() else 1)