#!/usr/bin/env python
"""Render Tier-1 Plan-A bar chart: seq_entropy + seq_nll across bands.

Clean, no error bars, value labels above each bar. Saves PNG + PDF.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BANDS = ["early", "mid", "late"]
METRICS = ["seq_entropy", "seq_nll"]
BUCKETS = ["clean", "AS_avg_within_bucket_then_over_scenarios", "AF_avg_within_bucket_then_over_scenarios"]
BUCKET_LABELS = ["Clean", "AS", "AF"]
COLORS = ["#4c72b0", "#dd8452", "#55a467"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="path to as_vs_af_sequence_results.json")
    ap.add_argument("--output-dir", default=None,
                    help="where to save png/pdf (default: same dir as results)")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        d = json.load(f)
    planA = d["signal_plan_A_segment_aggregated_Wilcoxon_tests"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(BANDS))
    width = 0.26

    for ax, metric in zip(axes, METRICS):
        bands_d = planA[metric]["bands"]
        # Collect means and n_valid per band
        means = {b: [] for b in BUCKETS}  # bucket -> list of (mean or np.nan)
        n_valid = []
        for b in BANDS:
            if b not in bands_d:
                for bk in BUCKETS:
                    means[bk].append(np.nan)
                n_valid.append(0)
                continue
            bd = bands_d[b]
            n_valid.append(bd["n_valid_scenarios"])
            s = bd["summary_means_across_scenarios"]
            for bk in BUCKETS:
                means[bk].append(s[bk]["mean"])

        for i, (bk, lbl) in enumerate(zip(BUCKETS, BUCKET_LABELS)):
            offsets = x + (i - 1) * width
            vals = np.array(means[bk], dtype=float)
            bars = ax.bar(offsets, vals, width, label=lbl, color=COLORS[i],
                          edgecolor="white", linewidth=0.5)
            # Value label above each bar
            for v, off in zip(vals, offsets):
                if np.isnan(v):
                    continue
                ax.text(off, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BANDS, n_valid)],
                           fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.set_ylabel("mean across scenarios")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    png = out_dir / "plan_A_bars.png"
    pdf = out_dir / "plan_A_bars.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"saved: {png}")
    print(f"saved: {pdf}")


if __name__ == "__main__":
    main()
