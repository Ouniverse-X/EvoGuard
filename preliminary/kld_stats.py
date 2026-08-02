"""Statistical analysis for the KL-divergence surprise experiment.

Reads ``kl_per_scenario.csv``, runs Kruskal-Wallis + pairwise Mann-Whitney U
(Bonferroni-corrected) + bootstrap CIs + length-confound checks, writes
``kl_summary_stats.json`` and plots.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any

logger = logging.getLogger("preliminary.kld_stats")

PRIMARY_BUCKETS = ("NormalClean", "AttackSuccess", "AttackFail")


def _load_csv(csv_path: str) -> list[dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("dropped", "").lower() in ("true", "1"):
                continue
            try:
                r["kl_nats"] = float(r["kl_nats"])
                r["obs_token_len"] = int(r.get("obs_token_len") or 0)
            except (ValueError, TypeError):
                continue
            rows.append(r)
    return rows


def kruskal_wallallis(groups: dict[str, list[float]]) -> tuple[float, float]:
    """Kruskal-Wallis H test across >=2 groups. Returns (H, p_two_sided)."""
    from scipy import stats
    vals = [list(v) for v in groups.values() if len(v) >= 1]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    try:
        H, p = stats.kruskal(*vals, nan_policy="omit")
    except ValueError:
        # scipy raises when all values are identical across groups (degenerate).
        return (0.0, 1.0)
    return (float(H), float(p))


def pairwise_mwu(a: list[float], b: list[float]) -> dict[str, Any]:
    """Mann-Whitney U between two groups. Returns dict with U, p_two_sided,
    p_one_sided_greater (a > b), rank_biserial, median_a, median_b."""
    from scipy import stats
    import numpy as np
    a_arr = np.asarray(a, dtype="float64"); b_arr = np.asarray(b, dtype="float64")
    if len(a_arr) == 0 or len(b_arr) == 0:
        return {"U": None, "p_two_sided": None, "p_one_sided_greater": None,
                "rank_biserial": None, "median_a": None, "median_b": None,
                "n_a": len(a_arr), "n_b": len(b_arr)}
    U, p_two = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    U2, p_one = stats.mannwhitneyu(a_arr, b_arr, alternative="greater")
    rank_biserial = 1.0 - (2.0 * float(U) / (len(a_arr) * len(b_arr))) if (len(a_arr)*len(b_arr)) else 0.0
    return {"U": float(U), "p_two_sided": float(p_two), "p_one_sided_greater": float(p_one),
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
    """Run the full §4 statistical analysis. Writes kl_summary_stats.json + plots."""
    import numpy as np
    rows = _load_csv(csv_path)
    by_bucket: dict[str, list[float]] = {}
    for r in rows:
        b = r.get("bucket", "")
        if not b:
            continue
        by_bucket.setdefault(b, []).append(r["kl_nats"])

    primary = {b: by_bucket.get(b, []) for b in PRIMARY_BUCKETS if b in by_bucket}
    H, p_kw = kruskal_wallallis(primary) if len(primary) >= 2 else (float("nan"), float("nan"))

    pairwise: dict[str, Any] = {}
    pairs = [("AttackFail", "AttackSuccess"), ("AttackFail", "NormalClean"),
             ("AttackSuccess", "NormalClean")]
    for a, b in pairs:
        if a in by_bucket and b in by_bucket:
            pairwise[f"{a}_vs_{b}"] = pairwise_mwu(by_bucket[a], by_bucket[b])

    # Bootstrap CIs per bucket.
    cis = {b: {"median_ci": _bootstrap_ci(v)} for b, v in by_bucket.items()}

    # Length confound.
    confound = {}
    for b, vals_list in by_bucket.items():
        lens = [r["obs_token_len"] for r in rows if r.get("bucket") == b]
        if len(vals_list) >= 3:
            confound[b] = {"spearman_len_kl": _spearman(lens, vals_list)}

    n_per_bucket = {b: len(v) for b, v in by_bucket.items()}
    underpowered = n_per_bucket.get("AttackFail", 0) < 10

    summary = {
        "n_total_rows": len(rows),
        "n_per_bucket": n_per_bucket,
        "kruskal_wallis": {"H": H, "p": p_kw, "significant_005": (p_kw < 0.05 if p_kw == p_kw else False)},
        "pairwise": pairwise,
        "bootstrap_ci": cis,
        "length_confound": confound,
        "underpowered_attackfail": underpowered,
        "bonferroni_alpha_prime": 0.05 / 3,
    }
    json.dump(summary, open(os.path.join(output_dir, "kl_summary_stats.json"), "w"), indent=2)

    # Plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        boxes = [by_bucket.get(b, []) for b in PRIMARY_BUCKETS if b in by_bucket]
        labels = [b for b in PRIMARY_BUCKETS if b in by_bucket]
        if boxes:
            plt.figure(figsize=(6, 4))
            try:
                plt.boxplot(boxes, tick_labels=labels)  # matplotlib >=3.9
            except TypeError:
                plt.boxplot(boxes, labels=labels)       # older matplotlib
            plt.ylabel("KL (nats)"); plt.title("Tool-return surprise by bucket")
            plt.savefig(os.path.join(output_dir, "plot_boxplot_by_bucket.png"), dpi=120, bbox_inches="tight")
            plt.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot failed: %s", exc)

    logger.info("analysis done. summary=%s", {k: v for k, v in summary.items() if k != "pairwise"})
    return summary
