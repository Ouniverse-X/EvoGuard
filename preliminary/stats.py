"""Statistical analysis suite for the preliminary entropy experiment.

Consumes ``entropy_per_scenario.csv`` (produced by ``preliminary.entropy.run``)
and emits ``summary_stats.json`` capturing:

  * Primary hypothesis test: Spearman rank correlation between bucket-ordinal
    {imm=0,d1=1,d2=2,d3=3,d4=4} and per-scenario mean Shannon entropy, with both
    two-sided and one-sided-negative-direction p-values.
  * Bootstrap percentile CI on bucket-mean entropy (N_boot iterations seeded).
  * Adjacent pairwise Welch unequal-variance t-tests with Bonferroni correction k=#planned_comparisons.
    Each contrast reports raw_p alongside corrected_alpha_applied and significant_after_correction flag,
    plus Cohen's d effect size using pooled Welch SD denominator.
  * Domain-stratified partial correlation controlling categorical domain variable via OLS residualization
    (sklearn lazy-imported; falls back gracefully if sklearn unavailable).
  * Extreme-contrast Cohen's d between imm and d4 endpoints.

All tests report BOTH point estimates AND p-values regardless of significance threshold being met —
null-results transparency mandatory. Alpha declared upfront in config BEFORE analyses run preventing post-hoc relaxation.

Implementation notes:
   * scipy.stats is required and imported lazily inside function bodies to keep module import-time cheap.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any

if __package__ in (None, ""):                                                   # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from preliminary.config import (
        ExperimentConfig,
        StatsConfig,
        BUCKET_ORDINALS,
        ordinal_level,
    )
else:
    from .config import (
        ExperimentConfig,
        StatsConfig,
        BUCKET_ORDINALS,
        ordinal_level,
    )

logger = logging.getLogger("preliminary.stats")


# --------------------------------------------------------------------------- #
# CSV loading & NaN-aware filtering helpers
# --------------------------------------------------------------------------- #
def _load_entropy_csv(csv_path: str) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows_out.append(dict(row))
    return rows_out


def _parse_float_safe(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:                                                            # noqa: BLE001 - tolerate malformed cells silently
        try:
            s = str(v).strip()
            if not s or s.lower() in ("nan", "none", "null"):
                return None
            return None
        except Exception:
            return None


def _filter_valid_rows(rows_raw: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return list of typed dicts containing only valid numeric columns needed downstream."""
    out_clean: list[dict[str, Any]] = []
    drop_reasons: Counter_like = defaultdict(int)

    H_col = "H_token_mean_full_response"
    BUCKET_col = "bucket"
    DOMAIN_col = "domain"
    DELTA_col = "delta_value_orig"

    for row in rows_raw:
        bk = (row.get(BUCKET_col) or "").strip().lower() if isinstance(row.get(BUCKET_col), str) else ""
        if bk not in BUCKET_ORDINALS:
            # Could be '?' placeholder from error-row marker -- skip but count separately only when meaningful
            continue

        h_val = _parse_float_safe(row.get(H_col))
        if h_val is None or math.isnan(h_val):
            drop_reasons["missing_or_nan_H_token_mean"] += 1
            continue

        delta_orig = _parse_float_safe(row.get(DELTA_col))
        domain_str = (row.get(DOMAIN_col) or "?").strip()

        out_clean.append({
            "scenario_id": row.get("scenario_id", ""),
            "bucket": bk,
            "delta_value_orig": int(delta_orig) if delta_orig is not None else ordinal_level(bk),
            "domain": domain_str,
            "method": row.get("method", ""),
            "truncated": bool(str(row.get("truncated", "")).lower() in ("true", "1")),
            "n_tokens_generated": int(_parse_float_safe(row.get("n_tokens_generated")) or 0),
            "H_full_response": h_val,
            "H_first_k_window": _parse_float_safe(row.get("H_first_k_mean_at_K_eq_64")) or math.nan,
            "H_normalised": _parse_float_safe(row.get("H_normalized_token_mean")) or math.nan,
            "p_argmax_avg": _parse_float_safe(row.get("p_argmax_mean")) or math.nan,
            "whitespace_excluded_h": _parse_float_safe(
                row.get("whitespace_excluded_H_token_mean")
            ) or math.nan,
        })

    return out_clean, dict(drop_reasons)


class Counter_like(defaultdict):                                                 # noqa: D101 - trivial subclass just for type clarity
    pass


# --------------------------------------------------------------------------- #
# Statistical primitives
# --------------------------------------------------------------------------- #
def _spearman_rho(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    try:
        from scipy.stats import spearmanr                                       # type: ignore[import]
    except ImportError as exc:                                                   # pragma: no cover
        logger.warning("scipy.stats unavailable (%s); Spearman rho cannot be computed.", exc)
        return None, None
    rho_obj = spearmanr(xs, ys)
    rho_v = getattr(rho_obj, "correlation", None)
    pv_two_sided = getattr(rho_obj, "pvalue", None)
    if rho_v is None or pv_two_sided is None or math.isnan(rho_v):
        return None, None
    return float(rho_v), float(pv_two_sided)


def _welch_t_test(x1: list[float], x2: list[float]) -> tuple[float | None, int | None, float | None]:
    """Returns (t_statistic_float_or_None, df_estimated_int_or_None, raw_p_two_tailed_float_or_None)."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        logger.warning("Welch t-test needs ≥2 samples/group ; got %d vs %d → skipping",
                       n1, n2)
        return None, None, None
    m1 = sum(x1) / n1
    m2 = sum(x2) / n2
    v1 = sum((xi - m1) ** 2 for xi in x1) / (n1 - 1)
    v2 = sum((yi - m2) ** 2 for yi in x2) / (n2 - 1)
    se_pooled_sq_numerator = v1 / n1 + v2 / n2
    if se_pooled_sq_numerator <= 0:
        return None, None, None
    t_stat_observed = (m1 - m2) / math.sqrt(se_pooled_sq_numerator)
    df_num = ((v1 / n1 + v2 / n2)) ** 2
    df_denom = (((v1 / n1) ** 2 / max(n1 - 1, 1))) + (((v2 / n2) ** 2 / max(n2 - 1, 1)))
    df_estimated = df_num / df_denom if df_denom > 0 else 0
    try:
        from scipy.stats import t as student_t                                  # type: ignore[import]
        p_two_tail = 2 * (1.0 - student_t.cdf(abs(t_stat_observed), df=df_estimated))
    except Exception as exc:                                                     # pragma: no cover
        logger.warning("scipy student-t cdf failed (%s)", exc)
        return float(t_stat_observed), int(df_estimated), None
    return float(t_stat_observed), int(df_estimated), float(p_two_tail)


def _cohens_d_welch(x1: list[float], x2: list[float]) -> float | None:
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return None
    m1 = sum(x1) / n1
    m2 = sum(x2) / n2
    v1 = sum((x - m1) ** 2 for x in x1) / (n1 - 1)
    v2 = sum((y - m2) ** 2 for y in x2) / (n2 - 1)
    pooled_sd = math.sqrt((v1 * (n1 - 1) + v2 * (n2 - 1)) / max(n1 + n2 - 2, 1))
    if pooled_sd <= 0:
        return None
    return (m1 - m2) / pooled_sd


def _bootstrap_ci(values_in_bucket: list[float], *, iters: int, ci_level: float,
                  rng_seed_for_resampling: int) -> tuple[float, float]:
    rng = random.Random(rng_seed_for_resampling ^ hash(tuple(values_in_bucket[:5])) & 0xFFFFF)
    n_vals = len(values_in_bucket)
    means_sampled: list[float] = []
    if n_vals == 0:
        return (float("nan"), float("nan"))
    for _i in range(iters):
        sample_xs = [rng.choice(values_in_bucket) for _j in range(n_vals)]
        means_sampled.append(sum(sample_xs) / n_vals)
    means_sampled.sort()
    alpha_lo = (1.0 - ci_level) / 2.0
    lo_idx = min(max(int(alpha_lo * len(means_sampled)), 0),
                 len(means_sampled) - 1)
    hi_idx = min(int((ci_level + alpha_lo) * len(means_sampled)),
                 len(means_sampled) - 1)
    lo_q = means_sampled[lo_idx]
    hi_q = means_sampled[hi_idx]
    return (float(lo_q), float(hi_q))


def _domain_partial_correlation(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Residualize X=bucket_ordinal, Y=H against one-hot-encoded domain indicator matrix then recompute Pearson/Spearman.

    Returns dict with adjusted_rho/adjusted_one-sided-p/direction_preserved/significance_preserved OR None when skipped due to missing deps/too few domains present.
    """
    domains_present = sorted({row["domain"] for row in rows})
    if len(domains_present) < 2:
        logger.info("domain partial corr skipped: fewer than 2 distinct domains observed")
        return {
            "skipped_reason": "fewer_than_2_distinct_domains_present",
            "adjusted_rho": None,
            "adjusted_p_one_sided_negative": None,
            "direction_preserved": False,
            "significance_preserved_at_overall_alpha": False,
            "interpretation_note": "Insufficient cross-domain variation to perform stratified adjustment.",
        }
    try:
        import numpy as np                                                        # type: ignore[import]
        from scipy.stats import pearsonr                                          # type: ignore[import]
    except ImportError as exc:                                                    # pragma: no cover
        logger.warning("numpy/scipy missing (%s); skipping domain-stratified partial correlation analysis.",
                       exc)
        return {"skipped_reason": "missing_numpy_or_scipy_dependency"}
    X_ord = np.array([ordinal_level(r["bucket"]) for r in rows], dtype=float)
    Y_ent = np.array([float(r["H_full_response"]) for r in rows], dtype=float)
    dom_to_colidx = {dm: i for i, dm in enumerate(domains_present)}
    Dom_matrix_rows: list[list[int]] = []
    for r in rows:
        one_hot_row = [0] * len(domains_present)
        cidx_dm = dom_to_colidx[r["domain"]]
        one_hot_row[cidx_dm] = 1
        Dom_matrix_rows.append(one_hot_row)
    Dom_mat_np = np.array(Dom_matrix_rows, dtype=float)

    def ols_residualize(target_vec: np.ndarray) -> np.ndarray:
        ones_const = np.ones((Dom_mat_np.shape[0], 1))
        design_dom_only = np.hstack([ones_const, Dom_mat_np])
        coeffs, *_lstsq_residual_info = np.linalg.lstsq(design_dom_only, target_vec, rcond=None)
        pred_target = design_dom_only @ coeffs
        resid_arr = target_vec - pred_target
        return resid_arr.ravel()

    res_X = ols_residualize(X_ord.copy())
    res_Y = ols_residualize(Y_ent.copy())
    pr_rval = pearsonr(res_X, res_Y)
    adj_rho = float(pr_rval[0]) if not math.isnan(pr_rval[0]) else None
    adj_p_twosided = float(pr_rval[1])
    direction_pred_neg = (adj_rho is not None) and (adj_rho < 0)
    adj_p_onetail_negative = (
        (adj_p_twosided / 2.0) if direction_pred_neg else (1.0 - adj_p_twosided / 2.0)
    )
    sig_threshold_overall = 0.05                                                  # default fallback used during internal call site below
    return {
        "adjusted_method_label": "Pearson correlation of OLS-residualized variables after regressing out one-hot encoded domain indicator matrix",
        "adjusted_rho": round(adj_rho, 6) if adj_rho is not None else None,
        "adjusted_p_two_sided": round(adj_p_twosided, 6),
        "adjusted_p_one_sided_negative_direction":
            round(adj_p_onetail_negative, 6) if adj_p_onetail_negative is not None else None,
        "direction_preserved_after_adjustment": bool(direction_pred_neg),
        "significance_preserved_at_overall_alpha_default_0p05_placeholder": False,  # filled later by caller once overall-alpha known
        "interpretation_note": "",
        "_domains_count_used_as_indicators": int(len(domains_present)),
    }


# --------------------------------------------------------------------------- #
# Driver producing summary_stats.json
# --------------------------------------------------------------------------- #
def analyze_and_emit_summary(*,
                             input_csv_path: str,
                             output_json_path: str,
                             stats_cfg: StatsConfig,
                             primary_metric_column_name: str = "H_full_response") -> dict[str, Any]:

    rows_loaded_raw = _load_entropy_csv(input_csv_path)
    rows_valid, drops_breakdown_dict = _filter_valid_rows(rows_loaded_raw)

    by_bucket: dict[str, list[float]] = defaultdict(list)
    by_domain_by_bucket_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_sizes_per_bucket: dict[str, int] = {}

    for rw in rows_valid:
        bname = rw["bucket"]
        by_bucket[bname].append(float(rw.get(primary_metric_column_name)))
        by_domain_by_bucket_counter[bname][rw["domain"]] += 1

    for bn_kk in ["imm", "d1", "d2", "d3", "d4"]:
        sample_sizes_per_bucket[bn_kk] = len(by_bucket.get(bn_kk, []))

    xs_ordinals_all: list[float] = []
    ys_metric_all: list[float] = []

    for rw in rows_valid:
        xs_ordinals_all.append(float(ordinal_level(rw["bucket"])))
        ys_metric_all.append(float(rw.get(primary_metric_column_name)))

    rho_estimate, p_two_sided_primary = _spearman_rho(xs_ordinals_all, ys_metric_all)

    predicted_dir_is_negative = (rho_estimate is not None) and (rho_estimate < 0)
    p_one_side_negative = None
    if p_two_sided_primary is not None:
        p_one_side_negative = (p_two_sided_primary / 2.0) \
                              if predicted_dir_is_negative \
                              else (1.0 - p_two_sided_primary / 2.0)

    hyp_supported_bool = (
        (rho_estimate is not None) and (predicted_dir_is_negative) and
        (p_one_side_negative is not None) and (p_one_side_negative < stats_cfg.alpha_overall)
    )

    bootstrap_ci_summary: dict[str, list[float]] = {}
    base_rng_state = stats_cfg.random_state_for_resampling or 42
    ci_iter_offset_acc = 0
    for bk_label in ["imm", "d1", "d2", "d3", "d4"]:
        vals_bk_list = by_bucket.get(bk_label, [])
        seed_for_this_bootstrap = base_rng_state + ci_iter_offset_acc
        lo_hi_pair = _bootstrap_ci(vals_bk_list,
                                   iters=int(stats_cfg.bootstrap_iters),
                                   ci_level=float(stats_cfg.ci_level),
                                   rng_seed_for_resampling=seed_for_this_bootstrap)
        bootstrap_ci_summary[bk_label] = [round(lo_hi_pair[0], 6),
                                           round(lo_hi_pair[1], 6)]
        ci_iter_offset_acc += 7

    pairwise_results: list[dict[str, Any]] = []
    bonferroni_corrected_alpha = stats_cfg.alpha_overall / max(stats_cfg.bonferroni_correction_count, 1)
    for pair_a, pair_b in stats_cfg.planned_pairwise_comparisons:
        xa = by_bucket.get(pair_a, [])
        xb = by_bucket.get(pair_b, [])
        ts, dfs, ps = _welch_t_test(list(map(float, xa)), list(map(float, xb)))
        cd_d = _cohens_d_welch(list(map(float, xa)), list(map(float, xb)))
        pairwise_results.append({
            "pair": [str(pair_a), str(pair_b)],
            "sample_size_each_group": [len(xa), len(xb)],
            "mean_diff_signed_A_minus_B": (
                round(float(sum(xa) / len(xa) - sum(xb) / len(xb)), 6)
                if xa and xb else None
            ),
            "welch_t_statistic": round(ts, 4) if ts is not None else None,
            "df_estimated": int(dfs) if dfs is not None else None,
            "raw_p_two_tailed": round(ps, 6) if ps is not None else None,
            "bonferroni_corrected_alpha_effective_per_comparison":
                round(bonferroni_corrected_alpha, 6),
            "significant_after_correction":
                bool(ps is not None and ps < bonferroni_corrected_alpha),
            "cohen_d_with_sign": round(cd_d, 4) if cd_d is not None else None,
        })

    extreme_cohens_d_imm_vs_d4 = _cohens_d_welch(by_bucket.get("imm", []),
                                                  by_bucket.get("d4", []))

    confound_check_block = _domain_partial_correlation(rows_valid)
    if confound_check_block is not None and "significance_preserved_at_overall_alpha_default_0p05_placeholder" in confound_check_block:
        ap = confound_check_block.get("adjusted_p_one_sided_negative_direction")
        dir_prsv = confound_check_block.get("direction_preserved_after_adjustment", False)
        sig_prsv = bool(ap is not None and ap < stats_cfg.alpha_overall and dir_prsv)
        del confound_check_block["significance_preserved_at_overall_alpha_default_0p05_placeholder"]
        confound_check_block["alpha_declared_upfront"] = stats_cfg.alpha_overall
        confound_check_block["significant_after_adjustment_at_alpha"] = sig_prsv
        confound_check_block["confound_control_passed_both_direction_AND_significance"] = bool(dir_prsv and sig_prsv)

    summary_blob = {
        "primary_hypothesis": {
            "test_method_short_name": "Spearman_rank_correlation",
            "metric_analyzed": primary_metric_column_name,
            "predicted_direction_prior_declaration": "negative",
            "rho_estimate": round(rho_estimate, 6) if rho_estimate is not None else None,
            "p_two_sided": round(p_two_sided_primary, 6) if p_two_sided_primary is not None else None,
            "p_one_sided_negative_direction": (
                round(p_one_side_negative, 6) if p_one_side_negative is not None else None
            ),
            "alpha_declared_upfront_before_running_analysis": stats_cfg.alpha_overall,
            "hypothesis_supported_at_alpha": bool(hyp_supported_bool),
            "interpretation_note_if_null_result_was_obtained": "" if hyp_supported_bool else
                ("If hypothesis NOT supported at declared alpha this may indicate either "
                 "(a) insufficient statistical power given small high-Delta buckets OR "
                 "(b) genuine absence of monotone relationship between Delta-bucket level and agent alertness entropy."),
        },
        "bootstrap_ci_by_bucket": bootstrap_ci_summary,
        "pairwise_adjacent_welch_t_tests": pairwise_results,
        "extreme_contrast_cohens_d_imm_vs_d4":
            round(extreme_cohens_d_imm_vs_d4, 4) if extreme_cohens_d_imm_vs_d4 is not None else None,
        "confound_control_domain_partial_correlation": confound_check_block,
        "sample_sizes_by_bucket": sample_sizes_per_bucket,
        "domain_distribution_within_buckets_observed":
            {bk_: dict(sorted(dm_counts.items(), key=lambda kv: -kv[1]))
             for bk_, dm_counts in by_domain_by_bucket_counter.items()},
        "rows_processed_total": len(rows_valid),
        "dropped_records_breakdown": drops_breakdown_dict,
        "analysis_metadata": {
            "stats_config_snapshot_yaml_dumpable_subset": {
                "bootstrap_iters_planned": stats_cfg.bootstrap_iters,
                "ci_level_targeted": stats_cfg.ci_level,
                "alpha_overall_declared_upfront": stats_cfg.alpha_overall,
                "bonferroni_correction_count_planned": stats_cfg.bonferroni_correction_count,
                "bonferroni_corrected_alpha_effective_per_pairwise_comparison": round(bonferroni_corrected_alpha, 6),
                "random_state_for_resampling_basis": stats_cfg.random_state_for_resampling,
                "planned_pairwise_comparisons_tuple_of_tuples":
                    [[a, b] for a, b in stats_cfg.planned_pairwise_comparisons],
            },
            "transparency_disclaimer":
                "All tests reported regardless of outcome including null results transparency mandated.",
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    json_tmp_path = output_json_path + ".tmp"
    with open(json_tmp_path, "w", encoding="utf-8") as fout:
        json.dump(summary_blob, fout, ensure_ascii=False, indent=2)
    os.replace(json_tmp_path, output_json_path)

    print("\n=== STATS ANALYSIS SUMMARY ===")
    print(f"output_json={output_json_path}")
    phb = summary_blob["primary_hypothesis"]
    print(f"Primary Spearman ρ={phb['rho_estimate']} ; p_two={phb['p_two_sided']}"
          f" ; p_one_neg={phb['p_one_sided_negative_direction']}"
          f" ; supported@α{phb['alpha_declared_upfront_before_running_analysis']}="
          f"{phb['hypothesis_supported_at_alpha']}")
    print("\nBootstrap CIs:")
    for bk_lbl, lohi in summary_blob["bootstrap_ci_by_bucket"].items():
        print(f"  [{bk_lbl}] ({lohi[0]:.4f}, {lohi[1]:.4f})  n={summary_blob['sample_sizes_by_bucket'].get(bk_lbl)}")

    print("\nPairwise adjacent contrasts (Bonferroni α_eff=%.4f):" %
          bonferroni_corrected_alpha)
    for ppres in summary_blob["pairwise_adjacent_welch_t_tests"]:
        sg = "*" if ppres["significant_after_correction"] else "ns"
        cd_repr = ("%+.3f" % ppres["cohen_d_with_sign"]
                   if ppres["cohen_d_with_sign"] is not None else "NA")
        print(f"  {' vs '.join(ppres['pair'])}: Δmean={ppres['mean_diff_signed_A_minus_B']:.4f} "
              f"t({ppres['df_estimated']})={ppres['welch_t_statistic']:>+.3f} "
              f"raw_p={ppres['raw_p_two_tailed']} {sg}; Cohen's d={cd_repr}")

    ccblk = summary_blob["confound_control_domain_partial_correlation"]
    if ccblk and "adjusted_rho" in ccblk and ccblk["adjusted_rho"] is not None:
        print(f"\ndomain-confound control: adjusted_ρ={ccblk['adjusted_rho']:+.4f}, "
              f"dir_preserved={ccblk['direction_preserved_after_adjustment']}, "
              f"sig_preserved={ccblk['significant_after_adjustment_at_alpha']}")
    elif ccblk and "skipped_reason" in ccblk:
        print(f"\ndomain-confound control SKIPPED: {ccblk['skipped_reason']}")

    return summary_blob


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m preliminary.stats")
    ap.add_argument("--input-csv-path", required=True,
                    help="Path to rounds/_preliminary/<ts>/entropy_per_scenario.csv.")
    ap.add_argument("--output-json-path", required=True,
                    help="Path where summary_stats.json should be written.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")

    cfg_fallback_stats = StatsConfig()                                            # use defaults aligned with configs/preliminary_entropy.yaml
    analyze_and_emit_summary(
        input_csv_path=args.input_csv_path,
        output_json_path=args.output_json_path,
        stats_cfg=cfg_fallback_stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
