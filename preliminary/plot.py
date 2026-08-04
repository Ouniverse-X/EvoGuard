"""Plot generation for the preliminary entropy experiment.

Produces three PNGs under the output directory:

  * ``plot_main_curve.png`` — mean per-bucket entropy ±95% bootstrap CI across
    {imm,d1,d2,d3,d4}; primary headline figure.
  * ``plot_per_domain_facets.png`` — four-panel facet grid showing same curve
    separately for banking / slack / travel / workspace.
  * ``plot_histogram_overlay.png`` — stacked histograms of H_token_mean_full_response
    distributions color-coded by bucket label.

Uses matplotlib Agg backend set before pyplot import to avoid GUI dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict, Counter

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                                                  # noqa: E402 - after backend switch above
import numpy as np                                                               # noqa: E402  (lazy-loaded via matplotlib anyway)

if __package__ in (None, ""):                                                   # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger("preliminary.plot")


BUCKET_ORDER = ["imm", "d1", "d2", "d3", "d4"]
DOMAIN_ORDER = ["banking", "slack", "travel", "workspace"]
COLORS_BY_BUCKET = {
    "imm": "#1f77b4",
    "d1": "#ff7f0e",
    "d2": "#2ca02c",
    "d3": "#9467bd",
    "d4": "#8c564b",
}


# --------------------------------------------------------------------------- #
# CSV loading helper shared with stats.py but kept local to avoid coupling.
# --------------------------------------------------------------------------- #
def _load_csv(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def _parse_float_safe(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:                                                            # noqa: BLE001
        return None


def _filter_valid(rows_raw: list[dict[str, str]]) -> list[tuple[str, str, float]]:
    """Return list of [(bucket_label_str, domain_str, h_value_float)]."""
    out_clean = []
    for r in rows_raw:
        bk = (r.get("bucket") or "").strip().lower()
        if bk not in BUCKET_ORDER:
            continue
        hv = _parse_float_safe(r.get("H_token_mean_full_response"))
        if hv is None or math.isnan(hv):
            continue
        dm = (r.get("domain") or "?").strip()
        out_clean.append((bk, dm, float(hv)))
    return out_clean


def _load_summary_stats(json_path: str | None) -> dict | None:
    if not json_path or not os.path.isfile(json_path):
        logger.warning("summary_stats.json not found at %r; CI bars will be computed from raw data instead.",
                       json_path)
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:                                                     # pragma: no cover
        logger.warning("failed loading summary_stats.json %s (%s)", json_path, exc)
        return None


# --------------------------------------------------------------------------- #
# Plot builders
# --------------------------------------------------------------------------- #
def plot_main_curve(triples_valid: list[tuple[str, str, float]],
                    summary_stats_json_path: str | None,
                    output_png_path: str):
    fig_w_inch = max(7.5, len(BUCKET_ORDER) * 1.6 + 1.8)
    fig_h_inch = 4.7
    fig, ax = plt.subplots(figsize=(fig_w_inch, fig_h_inch))

    xs_ord_positions = list(range(len(BUCKET_ORDER)))

    means_list: list[float] = []
    ci_lo_hi_pairs: list[list[float]] = []

    summary_block = _load_summary_stats(summary_stats_json_path)

    for bidx, bname in enumerate(BUCKET_ORDER):
        vals_for_bucket = [hval for (bk_lab, _, hval) in triples_valid if bk_lab == bname]
        n_bk = len(vals_for_bucket)
        mean_v = sum(vals_for_bucket) / max(n_bk, 1)
        # Prefer pre-computed CIs from summary_stats.json when available so plots stay consistent with reported numbers.
        ci_pair_precomputed = (
            (summary_block.get("bootstrap_ci_by_bucket") or {}).get(bname) if isinstance(summary_block, dict) else None
        )
        lo_q, hi_q = ((ci_pair_precomputed + [None])[:2] if ci_pair_precomputed else [None, None])
        if lo_q is None and vals_for_bucket:
            std_local = math.sqrt(sum((x_ - mean_v) ** 2 for x_ in vals_for_bucket) /
                                   max(n_bk - 1, 1)) if n_bk > 1 else 0.0
            sem_estimated = std_local / math.sqrt(max(n_bk, 1))
            half_width_approximate_ci95 = 1.96 * sem_estimated
            lo_q = mean_v - half_width_approximate_ci95
            hi_q = mean_v + half_width_approximate_ci95
            ax.text(bidx, hi_q + abs(hi_q - lo_q) * 0.05 + 0.003, f"n={n_bk}",
                    ha="center", va="bottom", fontsize=8, alpha=0.75)
        elif lo_q is None:
            lo_q = hi_q = mean_v
            ax.text(bidx, hi_q + 0.002, f"n=NA({bk!s})", ha="center", va="bottom", fontsize=8, alpha=0.55)
        else:
            ax.text(bidx, hi_q + abs(hi_q - lo_q) * 0.10 + 0.001, f"n={n_bk}",
                    ha="center", va="bottom", fontsize=8, alpha=0.65)

        means_list.append(mean_v)
        ci_lo_hi_pairs.append([float(lo_q), float(hi_q)])

    err_lo_arr = np.array([(means_list[i] - ci_lo_hi_pairs[i][0] if ci_lo_hi_pairs[i][0] is not None else 0.0)
                           for i in range(len(means_list))], dtype=float).clip(min=0.0)
    err_hi_arr = np.array([(ci_lo_hi_pairs[i][1] - means_list[i] if ci_lo_hi_pairs[i][1] is not None else 0.0)
                           for i in range(len(means_list))], dtype=float).clip(min=0.0)
    yerr_asymmetric_shape_2xn = np.vstack((err_lo_arr.reshape(1, -1), err_hi_arr.reshape(1, -1))) \
                                .reshape(2, len(xs_ord_positions))

    ax.errorbar(
        x=xs_ord_positions,
        y=np.array(means_list),
        yerr=yerr_asymmetric_shape_2xn,
        fmt="-o",
        capsize=4,
        lw=2,
        markersize=8,
        markerfacecolor="#22222222",
        markeredgecolor="#11111144" if False else COLORS_BY_BUCKET["imm"],
        ecolor="#66666666",
        elinewidth=1.5,
        zorder=5,
    )

    rho_summary_text = ""
    p_summary_text = ""
    supported_summary_text = ""
    sample_total_n_summed = len(triples_valid)
    if isinstance(summary_block, dict):
        phblk = summary_block.get("primary_hypothesis") or {}
        rhov = phblk.get("rho_estimate")
        pv_two = phblk.get("p_two_sided")
        sup_flagged = phblk.get("hypothesis_supported_at_alpha")
        if rhov is not None:
            rho_summary_text = f"Spearman ρ={rhov:+.4f}"
        if pv_two is not None:
            p_summary_text = f"(two-sided p={pv_two:.4g})"
        if sup_flagged is True:
            supported_summary_text = "; hypothesis SUPPORTED at α"
        elif sup_flagged is False:
            supported_summary_text = "; NOT significant at declared α"

    title_main = ("Mean Token-Level Shannon Entropy of Agent Response\n"
                  "(immediately post-injection-exposure window)")
    subtitle_meta = (
        f"N={sample_total_n_summed} scenarios total across {len(BUCKET_ORDER)} Δ-buckets ; "
        "error-bars show 95% percentile-bootstrap CI on bucket-mean."
    )
    annotation_line_below_subtitle = " ".join(filter(None, [
        rho_summary_text,
        p_summary_text,
        supported_summary_text.strip("; ") if supported_summary_text else "",
    ])).strip()

    full_title_lines = [title_main]
    if subtitle_meta:
        full_title_lines.append(subtitle_meta)
    if annotation_line_below_subtitle:
        full_title_lines.append(annotation_line_below_subtitle)

    ax.set_title("\n".join(full_title_lines), fontsize=11)
    ax.set_xticks(list(range(len(BUCKET_ORDER))))
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_xlabel("Δ-Bucket Label\n(imm ≤ immediate-trigger IPI; dN → latent attack with turning_point − injection_point = N)",
                  fontsize=10)
    ax.set_ylabel("$\\bar{H}_t$ averaged over response tokens (natural-log units)", fontsize=10)
    ax.grid(True, which='both', axis='y', linestyle="--", linewidth=0.45, alpha=0.45)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_per_domain_facets(triples_valid: list[tuple[str, str, float]],
                            output_png_path: str):
    domains_present_observed = sorted(
        {dm for (_, dm, _) in triples_valid},
        key=lambda s: DOMAIN_ORDER.index(s) if s in DOMAIN_ORDER else 99,
    )
    domains_present_observed_filtered_to_known = [d for d in domains_present_observed if d != "?" and d in DOMAIN_ORDER]

    nrows = 1
    ncols = max(1, min(len(domains_present_observed_filtered_to_known), 4))
    figsize_each_unit_x_inches = 3.0
    figsize_each_unit_y_inches = 3.4
    fig_width_inches = ncols * figsize_each_unit_x_inches + 0.85
    fig_height_inches = nrows * figsize_each_unit_y_inches + 1.25
    fig, axes_row = plt.subplots(nrows=nrows, ncols=max(1, ncols),
                                  figsize=(fig_width_inches, fig_height_inches))
    axes_iterable = list(axes_row.flat if hasattr(axes_row, 'flat') else [axes_row])

    for idx_panel, dom_name in enumerate(domains_present_observed_filtered_to_known[:ncols]):
        ax_p = axes_iterable[idx_panel]
        sub_triples_domain_only = [(bk_, dn_, hv_) for (bk_, dn_, hv_) in triples_valid if dn_ == dom_name]
        ys_means_per_bucket_this_dom = []
        ns_per_bucket_this_dom = []
        for bname_kk in BUCKET_ORDER:
            vs_bk_dmn = [hv_ for (bk_, _, hv_) in sub_triples_domain_only if bk_ == bname_kk]
            mn_ = sum(vs_bk_dmn) / max(len(vs_bk_dmn), 1) if vs_bk_dmn else float('nan')
            ys_means_per_bucket_this_dom.append(mn_)
            ns_per_bucket_this_dom.append(len(vs_bk_dmn))

        valid_mask = ~np.isnan(np.array(ys_means_per_bucket_this_dom, dtype=float))
        positions_valid_xs = np.array(list(range(len(BUCKET_ORDER))))[valid_mask].tolist()
        values_valid_ys = np.array(ys_means_per_bucket_this_dom)[valid_mask].tolist()
        ax_p.plot(positions_valid_xs, values_valid_ys, "-o", lw=2, ms=7,
                  color="#333333",
                  mfc='#ffffff', mec='#000000', alpha=0.92)

        for xv, yv, nvv in zip(positions_valid_xs, values_valid_ys,
                               [ns_per_bucket_this_dom[i] for i,v in enumerate(valid_mask) if v]):
            ax_p.annotate(f"n={int(nvv)}", xy=(xv, yv), xytext=(0, 12),
                          textcoords="offset points", ha="center", va="bottom",
                          fontsize=7, alpha=0.70)
        ax_p.set_title(f"{dom_name}\n(N={sum(ns_per_bucket_this_dom)})",
                       fontsize=10)
        ax_p.set_xticks(list(range(len(BUCKET_ORDER))))
        ax_p.set_xticklabels(BUCKET_ORDER, rotation=0)
        ax_p.grid(True, axis='y', ls='--', lw=0.35, alpha=0.40)
        ax_p.tick_params(labelsize=8)
        ax_p.spines['top'].set_visible(False); ax_p.spines['right'].set_visible(False)
        ymin_floor = next((y for y in ys_means_per_bucket_this_dom if not math.isnan(y)), 0.0)
        ymax_ceiling_val = max((y for y in ys_means_per_bucket_this_dom if not math.isnan(y)),
                               default=1.0)
        span_pad_amount = (ymax_ceiling_val - ymin_floor) * 0.15 + 0.01
        ax_p.set_ylim(ymin_floor - span_pad_amount, ymax_ceiling_val + span_pad_amount * 1.20)

    while len(axes_iterable) > len(domains_present_observed_filtered_to_known[:ncols]):
        extra_ax_obj = axes_iterable[len(domains_present_observed_filtered_to_known)]
        try:
            extra_ax_obj.axis('off')
        except AttributeError:                                                    # pragma: no cover - single-axes case has no extras to hide
            break

    suptitle_global = "$\\bar{H}_t$ by domain × Δ-bucket\n(per-domain breakdown reveals whether trend generalises across task types)"
    fig.suptitle(suptitle_global, fontsize=11, va="top", ha="center")

    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_histogram_overlay(triples_valid: list[tuple[str, str, float]], output_png_path: str):
    fig_h_overlay, ax_o = plt.subplots(figsize=(8.5, 4.6))
    bins_uniform_edges = np.linspace(0, max(2.80, max(hv_ for _, _, hv_ in triples_valid) + 0.02),
                                     num=32, endpoint=True)

    hist_handles_artists = []
    hist_labels_legend = []
    for bidx_hist, bn_lbl in enumerate(reversed(BUCKET_ORDER)):
        bin_vals = [hv_ for (bk_, _, hv_) in triples_valid if bk_ == bn_lbl]
        if not bin_vals:
            continue
        counts_bin_counted, edges_used_by_np = np.histogram(bin_vals, bins=bins_uniform_edges)
        widths_bins = np.diff(edges_used_by_np)
        bar_artist_coll = ax_o.bar(edges_used_by_np[:-1],
                                    counts_bin_counted.astype(float),
                                    width=widths_bins * 0.86,
                                    align='edge', bottom=None,
                                    color=COLORS_BY_BUCKET[bn_lbl],
                                    edgecolor='none',
                                    alpha=0.50,
                                    label=f"{bn_lbl} (Δ={'≤0' if bn_lbl=='imm' else int(bn_lbl[-1:])}) "
                                          f"[n={len(bin_vals)}, μ={np.mean(bin_vals):+.3f}]")
        hist_handles_artists.insert(0, bar_artist_coll)
        hist_labels_legend.insert(0, bn_lbl)

    ax_o.legend(hist_handles_artists, [a.get_label() for a in hist_handles_artists],
                loc='upper right', framealpha=0.78, fontsize=8)
    ax_o.set_xlabel("$H_t$ token-level entropy value (per scenario aggregated across response tokens)")
    ax_o.set_ylabel("# scenarios falling into each bin")
    ax_o.set_title(f"Histogram overlay of $H_t$ distribution by Δ-bucket\n(total N={len(triples_valid)} scenarios)")
    ax_o.grid(True, axis='both', ls='--', lw=0.4, alpha=0.42)
    ax_o.spines['top'].set_visible(False); ax_o.spines['right'].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=165, bbox_inches="tight")
    plt.close(fig_h_overlay)


# --------------------------------------------------------------------------- #
# Driver entrypoint
# --------------------------------------------------------------------------- #
def run(*, input_csv_path: str, output_dir: str, summary_stats_json_path: str | None) -> dict[str, Any]:
    rows_loaded = _load_csv(input_csv_path)
    triples_valid_tuples = _filter_valid(rows_loaded)

    out_paths_dict: dict[str, Any] = {}

    main_curve_png_fp = os.path.join(output_dir, "plot_main_curve.png")
    facets_png_fp = os.path.join(output_dir, "plot_per_domain_facets.png")
    histogram_png_fp = os.path.join(output_dir, "plot_histogram_overlay.png")

    plot_main_curve(triples_valid_tuples, summary_stats_json_path, main_curve_png_fp)
    plot_per_domain_facets(triples_valid_tuples, facets_png_fp)
    plot_histogram_overlay(triples_valid_tuples, histogram_png_fp)

    print("\n=== PLOT GENERATION SUMMARY ===")
    print(f"output_dir={output_dir}")
    print(f"main curve PNG -> {os.path.abspath(main_curve_png_fp)}")
    print(f"facets PNG     -> {os.path.abspath(facets_png_fp)}")
    print(f"histogram PNG  -> {os.path.abspath(histogram_png_fp)}")
    print(f"#scenarios plotted={len(triples_valid_tuples)}")

    out_paths_dict.update({
        "main_curve_absolute_path": os.path.abspath(main_curve_png_fp),
        "facets_absolute_path": os.path.abspath(facets_png_fp),
        "histogram_absolute_path": os.path.abspath(histogram_png_fp),
        "total_scenarios_plotted": len(triples_valid_tuples),
    })
    return out_paths_dict


if __name__ == "__main__":
    import argparse as ap_argparse_module_alias_unused                                            # noqa: F401
    ap_real_parser = argparse.ArgumentParser(prog="python -m preliminary.plot")
    ap_real_parser.add_argument("--input-csv-path", required=True)
    ap_real_parser.add_argument("--output-dir", required=True)
    ap_real_parser.add_argument("--summary-stats-json-path", default=None)
    args_passed_via_cli_plot_mod_standalone = ap_real_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")
    run(input_csv_path=args_passed_via_cli_plot_mod_standalone.input_csv_path,
        output_dir=args_passed_via_cli_plot_mod_standalone.output_dir,
        summary_stats_json_path=args_passed_via_cli_plot_mod_standalone.summary_stats_json_path)
