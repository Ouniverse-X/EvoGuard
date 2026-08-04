"""Statistical analysis for the paired KL-divergence surprise experiment.

Reads ``paired_kl.csv``, runs:
  1. Wilcoxon signed-rank test on ΔKL (H0: median ΔKL = 0, H1: ΔKL > 0)
  2. Mann-Whitney U on ΔKL between AttackFail / AttackSuccess (H1: AF > AS)
  3. Bootstrap 95% CIs for median ΔKL per group
  4. Length-confound Spearman(ΔKL, obs_token_len_delta)
  5. Auxiliary: same MWU on raw KL_injected (for comparison with prior experiment)

Bonferroni α' = 0.05 / 2 = 0.025 for the two primary tests.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any

logger = logging.getLogger("preliminary.kld_paired_stats")

BONFERRONI_ALPHA_PRIME = 0.05 / 2  # two primary tests


def _load_csv(csv_path: str) -> list[dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("dropped", "").lower() in ("true", "1"):
                continue
            try:
                r["kl_clean"] = float(r["kl_clean"])
                r["kl_injected"] = float(r["kl_injected"])
                r["delta_kl"] = float(r["delta_kl"])
                r["obs_token_len_clean"] = int(r.get("obs_token_len_clean") or 0)
                r["obs_token_len_injected"] = int(r.get("obs_token_len_injected") or 0)
            except (ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def wilcoxon_signed_rank(deltas: list[float]) -> dict[str, Any]:
    """One-sided Wilcoxon signed-rank test H0: median=0, H1: median > 0."""
    from scipy import stats
    import numpy as np
    arr = np.asarray([d for d in deltas if d != 0], dtype="float64")
    n_nonzero = len(arr)
    n_total = len(deltas)
    n_zero = n_total - n_nonzero
    if n_nonzero < 1:
        return {"W": None, "p_one_sided_greater": None, "n_nonzero": n_nonzero,
                "n_zero": n_zero, "n_total": n_total, "median_delta": None}
    try:
        W, p_two = stats.wilcoxon(arr, alternative="two-sided")
        # one-sided greater: re-run with alternative="greater"
        W2, p_one = stats.wilcoxon(arr, alternative="greater")
    except ValueError:
        # All same sign / degenerate
        return {"W": None, "p_one_sided_greater": 1.0, "n_nonzero": n_nonzero,
                "n_zero": n_zero, "n_total": n_total,
                "median_delta": float(np.median(deltas))}
    return {"W": float(W2), "p_two_sided": float(p_two),
            "p_one_sided_greater": float(p_one),
            "n_nonzero": n_nonzero, "n_zero": n_zero, "n_total": n_total,
            "median_delta": float(np.median(deltas))}


def pairwise_mwu_delta(a: list[float], b: list[float]) -> dict[str, Any]:
    """Mann-Whitney U on ΔKL, one-sided H1: a > b (AttackFail > AttackSuccess)."""
    from scipy import stats
    import numpy as np
    a_arr = np.asarray(a, dtype="float64")
    b_arr = np.asarray(b, dtype="float64")
    if len(a_arr) == 0 or len(b_arr) == 0:
        return {"U": None, "p_two_sided": None, "p_one_sided_greater": None,
                "rank_biserial": None, "median_a": None, "median_b": None,
                "n_a": len(a_arr), "n_b": len(b_arr)}
    U, p_two = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    U2, p_one = stats.mannwhitneyu(a_arr, b_arr, alternative="greater")
    rank_biserial = 1.0 - (2.0 * float(U) / (len(a_arr) * len(b_arr))) if (len(a_arr) * len(b_arr)) else 0.0
    return {"U": float(U), "p_two_sided": float(p_two),
            "p_one_sided_greater": float(p_one),
            "rank_biserial": float(rank_biserial),
            "median_a": float(np.median(a_arr)), "median_b": float(np.median(b_arr)),
            "n_a": len(a_arr), "n_b": len(b_arr)}


def _bootstrap_ci(vals, n_boot=10000, seed=42, stat="median"):
    import numpy as np
    arr = np.asarray(vals, dtype="float64")
    if len(arr) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(np.median(s)) if stat == "median" else float(np.mean(s)))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return (lo, hi)


def _spearman(a, b):
    from scipy import stats
    if len(a) < 3:
        return (float("nan"), float("nan"))
    rho, p = stats.spearmanr(a, b)
    return (float(rho), float(p))


def analyze(csv_path: str, output_dir: str) -> dict[str, Any]:
    """Run the full paired statistical analysis. Writes JSON + plots."""
    import numpy as np
    rows = _load_csv(csv_path)
    logger.info("loaded %d non-dropped rows", len(rows))

    all_deltas = [r["delta_kl"] for r in rows]
    all_kl_injected = [r["kl_injected"] for r in rows]
    all_kl_clean = [r["kl_clean"] for r in rows]

    # --- Primary test 1: Wilcoxon signed-rank on ΔKL ---
    wilcoxon_res = wilcoxon_signed_rank(all_deltas)

    # --- Split by bucket ---
    by_bucket: dict[str, list[float]] = {}
    by_bucket_kl_inj: dict[str, list[float]] = {}
    for r in rows:
        b = r.get("bucket", "")
        if not b:
            continue
        by_bucket.setdefault(b, []).append(r["delta_kl"])
        by_bucket_kl_inj.setdefault(b, []).append(r["kl_injected"])

    # --- Primary test 2: MWU on ΔKL, AttackFail > AttackSuccess ---
    af_deltas = by_bucket.get("AttackFail", [])
    as_deltas = by_bucket.get("AttackSuccess", [])
    mwu_delta = pairwise_mwu_delta(af_deltas, as_deltas)

    # --- Auxiliary: MWU on raw KL_injected (compare with prior experiment) ---
    af_kl_inj = by_bucket_kl_inj.get("AttackFail", [])
    as_kl_inj = by_bucket_kl_inj.get("AttackSuccess", [])
    mwu_raw = pairwise_mwu_delta(af_kl_inj, as_kl_inj)

    # --- Bootstrap CIs ---
    cis_delta = {b: {"median_ci": _bootstrap_ci(v)} for b, v in by_bucket.items()}
    cis_delta["all"] = {"median_ci": _bootstrap_ci(all_deltas)}

    # --- Length confound ---
    len_deltas = [r["obs_token_len_injected"] - r["obs_token_len_clean"] for r in rows]
    confound = {
        "spearman_dkl_len_delta": _spearman(all_deltas, len_deltas),
        "spearman_kl_inj_len": _spearman(all_kl_injected, [r["obs_token_len_injected"] for r in rows]),
        "spearman_kl_clean_len": _spearman(all_kl_clean, [r["obs_token_len_clean"] for r in rows]),
    }
    # Per-bucket confound
    confound_per_bucket = {}
    for b, deltas_list in by_bucket.items():
        lens = [r["obs_token_len_injected"] - r["obs_token_len_clean"]
                for r in rows if r.get("bucket") == b]
        if len(deltas_list) >= 3:
            confound_per_bucket[b] = {"spearman_dkl_len_delta": _spearman(deltas_list, lens)}

    # --- Summary ---
    n_per_bucket = {b: len(v) for b, v in by_bucket.items()}
    summary = {
        "n_total_rows": len(rows),
        "n_per_bucket": n_per_bucket,
        "primary_test_1_wilcoxon": wilcoxon_res,
        "primary_test_2_mwu_delta_kl": mwu_delta,
        "auxiliary_mwu_raw_kl_injected": mwu_raw,
        "bootstrap_ci_delta_kl": cis_delta,
        "length_confound": confound,
        "length_confound_per_bucket": confound_per_bucket,
        "bonferroni_alpha_prime": BONFERRONI_ALPHA_PRIME,
        "overall_median_delta_kl": float(np.median(all_deltas)) if all_deltas else None,
        "overall_mean_delta_kl": float(np.mean(all_deltas)) if all_deltas else None,
        "n_delta_positive": int(sum(1 for d in all_deltas if d > 0)),
        "n_delta_negative": int(sum(1 for d in all_deltas if d < 0)),
        "n_delta_zero": int(sum(1 for d in all_deltas if d == 0)),
    }

    # Significance flags
    summary["primary_1_significant"] = (
        wilcoxon_res.get("p_one_sided_greater") is not None
        and wilcoxon_res["p_one_sided_greater"] < BONFERRONI_ALPHA_PRIME
    )
    summary["primary_2_significant"] = (
        mwu_delta.get("p_one_sided_greater") is not None
        and mwu_delta["p_one_sided_greater"] < BONFERRONI_ALPHA_PRIME
    )

    json.dump(summary, open(os.path.join(output_dir, "paired_kl_stats.json"), "w"), indent=2)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Histogram of ΔKL
        plt.figure(figsize=(7, 4))
        plt.hist(all_deltas, bins=40, edgecolor="black", alpha=0.7)
        plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
        plt.axvline(x=float(np.median(all_deltas)), color="orange", linestyle="-",
                    linewidth=1, label=f"median={np.median(all_deltas):.4f}")
        plt.xlabel("ΔKL (nats) = KL_injected − KL_clean")
        plt.ylabel("Count")
        plt.title("Paired ΔKL distribution (injection surprise increment)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "plot_delta_kl_histogram.png"), dpi=120, bbox_inches="tight")
        plt.close()

        # Plot 2: Boxplot of ΔKL by bucket
        boxes = [by_bucket.get(b, []) for b in ("AttackFail", "AttackSuccess") if b in by_bucket]
        labels = [b for b in ("AttackFail", "AttackSuccess") if b in by_bucket]
        if boxes:
            plt.figure(figsize=(6, 4))
            try:
                plt.boxplot(boxes, tick_labels=labels)
            except TypeError:
                plt.boxplot(boxes, labels=labels)
            plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
            plt.ylabel("ΔKL (nats)")
            plt.title("ΔKL by bucket (injection surprise increment)")
            plt.savefig(os.path.join(output_dir, "plot_delta_kl_by_bucket.png"), dpi=120, bbox_inches="tight")
            plt.close()

        # Plot 3: Paired scatter KL_clean vs KL_injected
        plt.figure(figsize=(6, 6))
        plt.scatter(all_kl_clean, all_kl_injected, alpha=0.4, s=20)
        max_val = max(max(all_kl_clean), max(all_kl_injected)) * 1.1
        plt.plot([0, max_val], [0, max_val], "r--", linewidth=1)
        plt.xlabel("KL_clean (nats)")
        plt.ylabel("KL_injected (nats)")
        plt.title("Paired: KL_clean vs KL_injected")
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.savefig(os.path.join(output_dir, "plot_paired_scatter.png"), dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot failed: %s", exc)

    logger.info("analysis done. primary_1_significant=%s primary_2_significant=%s",
                summary["primary_1_significant"], summary["primary_2_significant"])
    return summary
