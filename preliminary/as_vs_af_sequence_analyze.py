"""Plan A + Plan B analysis driver for sequence-level AS-vs-AF probe outputs.

PLAN A -- Segment-aggregated pairwise-Wilcoxon tests
=====================================================
Per rollout, take each per-position signal array ``s[t]`` ∈ {entropy, top1,
refuse_mass, nll_actual_yt}, truncate it to its own ``valid_len``, then
aggregate into three equal-width bands:

    Band-Early : t∈[0,16)
    Band-Mid   : t∈[16,32)
    Band-Late  : t∈[32,48)

Each rollout contributes ONE scalar per band := mean over surviving positions;
rollouts whose valid_len doesn't reach a given band are excluded from THAT
band's comparison rather than zero-padded.

Within-scenario triplets (Clean_v, AS_avg_over≥3_successes_in_scenario,
AF_avg_over≥3_fails). Pairwise contrasts tested via one-sided Wilcoxon:
- For metrics predicted to ASCEND  Clean<AS<AF use alt='greater' on diff series.
- For metrics predicted to DESCEND Clean>AS>AF we still compute differences as
  X-Y but flip alternative to 'less'.

Bonferroni-corrected alpha = 0.05/3 ≈ 0.0167 per metric across three pairwise tests.

PLAN B -- Cross-timestep paired Jensen-Shannon divergence curve
=================================================================
At every absolute decoded offset t∈[0,T_max), collect ≥3 success & ≥3 fail
rollouts sharing SAME scenario index AND having valid_len>t.
Reconstruct approximate full-vocab distributions P̄_AS(t), P̄_AF(t) via bincount
aggregation over top-K indices then sum-normalize; missing mass assumed ~zero.

JSD(t,s) = sqrt(½KL(P‖M)+½KL(Q‖M)), M=½(P+Q).
Cross-scenario aggregation yields μ_JS(t)±σ_JS(t), fraction above baseline τ₀=0.001,
peak-t identification + two-sided one-sample t-test against 0 at that position.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("preliminary.as_vs_af_seq_analyze")
EPS = 1e-12


def _safe_float(x):
    try:
        v = float(x)
        return None if np.isnan(v) or np.isinf(v) else v
    except Exception:
        return None


def _safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return default


def _wilcoxon(diffs, alt="two-sided"):
    """One-sided Wilcoxon signed-rank on non-zero diffs."""
    from scipy.stats import wilcoxon
    nz = [d for d in diffs if d is not None and abs(d) > EPS]
    if len(nz) < 5:
        return float("nan"), len(nz)
    try:
        _, p = wilcoxon(nz, alternative=alt)
        return float(p), len(nz)
    except Exception as e:
        logger.warning("wilcoxon failed (%s); n=%d", repr(e)[:80], len(nz))
        return float("nan"), len(nz)


def _summarize(vals):
    arr = np.array([v for v in vals if v is not None], dtype=np.float64)
    if not arr.size:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# --------------------------------------------------------------------------- #
# PLAN A                                                                      #
# --------------------------------------------------------------------------- #

BAND_RANGES = {
    "early":    slice(0, 16),
    "mid":      slice(16, 32),
    "late":     slice(32, 48),
    # Tier-1 extended band: covers the [48, 128) thought-token window that v0's
    # MAX_THOUGHT_TOKENS=64 hard cap clipped entirely. With the probe-side raise
    # (DEFAULT_MAX_THOUGHT_TOKENS=128) this region becomes observable; scenarios
    # whose rollouts deliberate past offset 48 finally contribute signal here.
    # Coverage will still be sparse relative to mid-band because most rollouts
    # terminate before t≈50 under the current Qwen2.5-7B-Instruct scaffold.
    "extended": slice(48, 128),
}

# Number of pairwise contrasts per metric per band = 3:
#   AS-vs-Clean / AF-vs-AS / AF-vs-Clean.
N_PAIRWISE_PER_BAND_METRIC = 3

# Direction predicted by H1 AS-vs-AF ordering hypothesis AFTER fixing design flaw.
METRIC_DIRECTION_PREDICTION = {
    "seq_entropy":     "asc",
    "seq_top1":        "desc",
    "seq_refuse_mass": "asc",
    "seq_nll":         "asc",
}


def _band_mean(seq_list, valid_len, bslice):
    start = max(bslice.start, 0)
    stop = min(bslice.stop, valid_len if valid_len > 0 else 0)
    if stop <= start or not seq_list:
        return None
    sub = list(seq_list[start:stop])
    vals = [_safe_float(v) for v in sub]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals)/len(vals))


def analyze_plan_A(features_by_scen,
                   *,
                   temperature_strata_filter: list[float] | None = None) -> dict:
    """Plan-A segment-aggregated Wilcoxon analysis.

    ``temperature_strata_filter`` (Tier-2 deconfounder): if provided as a non-
    empty list, only attacked rollouts whose ``collection_temperature`` field
    matches one of the listed values are retained for AS/AF bucket assembly.
    Clean records are never filtered (clean rollout has no T-stratum). Pass
    ``None`` or ``[]`` for legacy "all temperatures pooled" behaviour.
    """
    # Bonferroni family-wise error control across all pairwise contrasts within
    # this metric: (#bands observed at least once) * N_PAIRWISE_PER_BAND_METRIC.
    # The original v0 hard-coded 0.05/3 ≈ 0.0167 because only three pairwise
    # tests existed per metric; with Tier-1 extended band added we widen the
    # family accordingly so false-positive inflation stays controlled.
    n_bands_in_family = max(len(BAND_RANGES), 1)
    bonferroni_family_size = n_bands_in_family * N_PAIRWISE_PER_BAND_METRIC
    bonferroni_alpha_per_metric_pairwise_test = 0.05 / bonferroni_family_size

    stratum_set = (
        set(float(t) for t in temperature_strata_filter)
        if temperature_strata_filter else None
    )

    report = {}

    for key, dirn in METRIC_DIRECTION_PREDICTION.items():
        alt_kwarg_for_predicted_direction = "greater" if dirn == "asc" else "less"
        pred_label = ("Clean < AS < AF" if dirn == "asc"
                      else "Clean > AS > AF")
        per_band_outputs = {}

        for bname, bslice in BAND_RANGES.items():
            cb, ab, fb = [], [], []
            d_ca, d_aaf, d_cf = [], [], []     # AS-Clean, AF-AS, AF-Clean in raw-metric units
            sm_count = 0
            tot_eval = 0

            for sid, bd in features_by_scen.items():
                c_recs = bd.get("clean"); s_recs = bd.get("success"); f_recs = bd.get("fail")
                if not c_recs or not s_recs or not f_recs:
                    continue
                # Tier-2 deconfounder: restrict attacked-rollout buckets to
                # requested T-strata only. Clean records carry no T-tag so they
                # always pass through unchanged.
                if stratum_set is not None:
                    s_recs = [r for r in s_recs
                              if float(r.get("collection_temperature") or -1.0) in stratum_set]
                    f_recs = [r for r in f_recs
                              if float(r.get("collection_temperature") or -1.0) in stratum_set]
                cv_valid = _safe_int(c_recs[0].get('valid_len'))
                cv = _band_mean(c_recs[0].get(key) or [], cv_valid, bslice)

                sv_list = [_band_mean(r.get(key) or [],
                                      _safe_int(r.get('valid_len')),
                                      bslice) for r in s_recs]
                fv_list = [_band_mean(r.get(key) or [],
                                      _safe_int(r.get('valid_len')),
                                      bslice) for r in f_recs]
                sv_list = [x for x in sv_list if x is not None]
                fv_list = [x for x in fv_list if x is not None]

                if cv is None or not sv_list or not fv_list:
                    continue

                sm_ = float(np.mean(sv_list))
                fm_ = float(np.mean(fv_list))

                cb.append(cv); ab.append(sm_); fb.append(fm_)
                d_ca.append(sm_-cv)
                d_aaf.append(fm_-sm_)
                d_cf.append(fm_-cv)

                tot_eval += 1
                # Strictly-monotonic check uses prediction direction directly.
                if dirn == "asc":
                    if cv < sm_ < fm_: sm_count += 1
                elif dirn == "desc":
                    if cv > sm_ > fm_: sm_count += 1

            if tot_eval == 0:
                per_band_outputs[bname] = {"n_valid_scenarios": 0}
                continue

            p_wcx_AC,_  = _wilcoxon(d_ca,  alt=alt_kwarg_for_predicted_direction)
            p_wcx_AAF,_ = _wilcoxon(d_aaf, alt=alt_kwarg_for_predicted_direction)
            p_wcx_CF_,_ = _wilcoxon(d_cf,  alt=alt_kwarg_for_predicted_direction)

            per_band_outputs[bname] = {
                "sub_metric_key": key,
                "predicted_ordering": pred_label,
                "n_valid_scenarios": int(tot_eval),
                "summary_means_across_scenarios": {
                    "clean": _summarize(cb),
                    "AS_avg_within_bucket_then_over_scenarios": _summarize(ab),
                    "AF_avg_within_bucket_then_over_scenarios": _summarize(fb),
                },
                "pairwise_wilcoxon_one_sided_predicting_predicted_direction": {
                    "p_value_for_AS_vs_Clean": float(p_wcx_AC),
                    "p_value_for_AF_vs_AS": float(p_wcx_AAF),
                    "p_value_for_AF_vs_Clean": float(p_wcx_CF_),
                    "_alternative_kwarg_used": alt_kwarg_for_predicted_direction,
                },
                "bonferroni_corrected_alpha_applied_to_each_pairwise_test":
                   float(bonferroni_alpha_per_metric_pairwise_test),
                "strictly_monotonic_count_out_of_total_evaluable":
                   f"{sm_count}/{tot_eval}",
            }

        report[key] = {
            "bands": per_band_outputs,
            "_interpretation_hint":
              "If trajectory-resolved signals distinguish outcome buckets then "
              "later bands should yield small p-values below Bonferroni-alpha "
              "(unlike pos-only mechanism_probe which collapsed bit-for-bit equal)."
        }

    return report


# --------------------------------------------------------------------------- #
# PLAN B                                                                      #
# --------------------------------------------------------------------------- #

BASELINE_THRESHOLD_JSB = 0.001
VOCAB_SIZE_FALLBACK_MAX = 152064       # Qwen2.5 tokenizer size upper bound


def reconstruct_avg_dist_from_topk(rows_probs_fp16, rows_indices_int32, vocab_size):
    flat_ids = np.clip(rows_indices_int32.flatten(), 0, vocab_size - 1).astype(np.int64)
    flat_prb = rows_probs_fp16.astype(np.float64).flatten()
    summed = np.bincount(flat_ids, weights=flat_prb, minlength=vocab_size)
    norm = summed.sum()
    return summed / (norm + EPS)


def _js_distance_over_union(succ_rows_probs, succ_rows_idx,
                            fail_rows_probs, fail_rows_idx):
    """Compute Jensen-Shannon DISTANCE between two averaged top-K-truncated
    distributions restricted to the UNION of their support tokens.

    Avoids false-zero blowups inherent to naive bincount-on-full-vocab path
    where >85% entries end up exactly zero after fp16 storage rounding,
    producing ``log(0/0)=nan`` under any reasonable epsilon additive constant.

    Returns scalar float JSD ∈ [0, sqrt(ln 2)] OR None on insufficient data.
    """
    # Union of token-id supports seen in EITHER group at this position.
    union_ids = np.unique(np.concatenate([
        np.asarray(succ_rows_idx).reshape(-1),
        np.asarray(fail_rows_idx ).reshape(-1)]))
    union_ids = union_ids[union_ids >= 0]
    if union_ids.size == 0:
        return None

    def _project_mean(probs_block, idx_block):
        """Mean across rows of probabilities projected onto union-axis."""
        n_rows = probs_block.shape[0]
        out = np.zeros(union_ids.size, dtype=np.float64)
        lookup = {int(tid): j for j, tid in enumerate(union_ids)}
        for r_i in range(n_rows):
            row_p = probs_block[r_i].astype(np.float64)
            row_k = idx_block[r_i].astype(np.int64)
            for k_pos in range(row_p.size):
                tid = int(row_k[k_pos])
                pos = lookup.get(tid)
                if pos is not None:
                    out[pos] += row_p[k_pos]
        out /= max(n_rows, 1)
        return out

    ps_u = _project_mean(np.asarray(succ_rows_probs), np.asarray(succ_rows_idx ))
    pf_u = _project_mean(np.asarray(fail_rows_probs), np.asarray(fail_rows_idx ))

    # Renormalize onto union-support domain (sum may be <1 because truncated tail).
    sum_ps = ps_u.sum(); sum_pf = pf_u.sum()
    if sum_ps < EPS or sum_pf < EPS:
        return None
    ps_u /= sum_ps
    pf_u /= sum_pf

    m = 0.5*(ps_u + pf_u)
    # Standard masked KL formulation -- only count terms where source distro >0.
    mask_ps = ps_u > 0
    mask_pf = pf_u > 0
    kl_sm_terms = np.zeros_like(ps_u)
    kl_fm_terms = np.zeros_like(pf_u)
    kl_sm_terms[mask_ps] = ps_u[mask_ps]*np.log((ps_u[mask_ps])/(m[mask_ps]))
    kl_fm_terms[mask_pf] = pf_u[mask_pf]*np.log((pf_u[mask_pf])/(m[mask_pf]))

    js_sq = 0.5*float(kl_sm_terms.sum()) + 0.5*float(kl_fm_terms.sum())
    js_sq = max(js_sq, 0.0)     # numerical safety against tiny negatives
    return float(np.sqrt(js_sq))


def analyze_plan_B(topk_probs, topk_indices, index_rows, output_dir_parent,
                   *,
                   temperature_strata_filter: list[float] | None = None) -> dict:
    """Plan-B time-resolved paired-JS-divergence curve between AS and AF
    average distributions at each decoded thought-token offset ``t``.

    ``temperature_strata_filter`` (Tier-2 deconfounder): when supplied as a non-
    empty list, only rollouts whose index-row ``collection_temperature`` matches
    one of the listed values contribute to per-position bucket averages. Pass
    ``None`` or empty list for legacy "all temperatures pooled" behaviour.
    """
    N, T, K = topk_probs.shape[:3] if topk_probs.ndim >= 3 else (0, 0, 0)
    logger.info("Plan-B union-aligned JSD computation; K_per_row=%d", int(K))

    stratum_set = (
        set(float(t) for t in temperature_strata_filter)
        if temperature_strata_filter else None
    )

    si_map = {int(rw['global_idx']): i for i, rw in enumerate(index_rows)}

    grouped = defaultdict(lambda: {'success':[], 'fail':[]})
    for ri, rw in enumerate(index_rows):
        kind = rw['rollout_kind']
        gid = int(rw['global_idx'])
        assert si_map[gid] == ri, f"inconsistent mapping at gidx={gid}"
        if kind not in ('success', 'fail'):
            continue   # ignore clean rows — Plan B compares AS vs AF only
        # Tier-2 deconfounder filter on T-stratum.
        if stratum_set is not None:
            t_val_raw = rw.get('collection_temperature')
            try:
                t_float = float(t_val_raw) if t_val_raw not in (None, "") else -1.0
            except (TypeError, ValueError):
                t_float = -1.0
            if t_float not in stratum_set:
                continue
        grouped[int(rw['scenario_index'])][kind].append((ri, rw))

    coverage_counts_by_t = np.zeros(T, dtype=np.int32)
    js_curves_all_scenarios_matrix = np.full((T, max(len(grouped), 1)),
                                              np.nan, dtype=np.float64)
    col_idx_used = 0

    for jcol_pair_idx,(si,gpair) in enumerate(sorted(grouped.items())):
        sg=gpair['success'];fg=gpair['fail']
        if len(sg)<3 or len(fg)<3:
            continue

        for t in range(T):
            # Per-position filter: rollouts whose valid_len > t.
            succ_keep_local=[]
            fail_keep_local=[]
            for _,rw in sg:
                if int(_safe_int_or_zero(rw,'valid_len'))>t:
                    succ_keep_local.append(si_map[int(rw['global_idx'])])
            for _,rw in fg:
                if int(_safe_int_or_zero(rw,'valid_len'))>t:
                    fail_keep_local.append(si_map[int(rw['global_idx'])])

            if len(succ_keep_local)<3 or len(fail_keep_local)<3:
                continue
            sp=topk_probs[np.asarray(succ_keep_local)][:,t,:]
            si_=topk_indices[np.asarray(succ_keep_local)][:,t,:]
            fp=topk_probs[np.asarray(fail_keep_local )][:,t,:]
            fi_=topk_indices[np.asarray(fail_keep_local )][:,t,:]
            js_val=_js_distance_over_union(sp,si_,fp,fi_)
            if js_val is None or np.isnan(js_val):
                continue

            while col_idx_used>=js_curves_all_scenarios_matrix.shape[1]:
                new_shape=list(js_curves_all_scenarios_matrix.shape)
                new_shape[1]*=2
                expanded_mat=np.full(new_shape,np.nan,dtype=np.float64)
                expanded_mat[:,:js_curves_all_scenarios_matrix.shape[1]]=\
                     js_curves_all_scenarios_matrix
                js_curves_all_scenarios_matrix=expanded_mat
            js_curves_all_scenarios_matrix[t,col_idx_used]=float(js_val)
            coverage_counts_by_t[t]+=1

        col_idx_used+=1

    dmat_used=js_curves_all_scenarios_matrix[:, :max(col_idx_used,1)]
    curve_mu=(np.nanmean(dmat_used,axis=-1) if dmat_used.shape[-1]>0 else np.zeros(T))
    curve_std=(np.nanstd(dmat_used,axis=-1,ddof=0) if dmat_used.shape[-1]>0 else np.zeros(T))
    above_baseline_frac=(np.nanmean((dmat_used>BASELINE_THRESHOLD_JSB)&~np.isnan(dmat_used),axis=-1)
                          if dmat_used.shape[-1]>0 else np.zeros(T))
    peak_t=int(np.argmax(curve_mu)) if curve_mu.size else -1

    significance_table_full=[]
    from scipy.stats import ttest_1samp
    for ti in range(T):
        samples=dmat_used[ti,:]
        samples=samples[~np.isnan(samples)]
        pval=None
        if samples.size>=5:
            try:
                _,pval=ttest_1samp(samples,popmean=0.)
                pval=float(pval)
            except Exception:
                pass
        significance_table_full.append({
           "abs_position_index":ti,
           "coverage_scenarios_with_enough_data_at_this_t":int(coverage_counts_by_t[ti]),
           "curve_height_mean":(None if np.isnan(curve_mu[ti]) else float(curve_mu[ti])),
           "curve_height_std":(None if np.isnan(curve_std[ti]) else float(curve_std[ti])),
           "fraction_above_baseline_threshold_"+str(BASELINE_THRESHOLD_JSB):
              (None if np.isnan(above_baseline_frac[ti]) else float(above_baseline_frac[ti])),
           "pvalue_two_tailed_ttest_against_zero":pval,
        })

    Path(output_dir_parent / 'plan_B_js_curve.npy').parent.mkdir(parents=True, exist_ok=True)
    np.save(Path(output_dir_parent/'plan_B_js_curve.npy'),
            np.stack([np.arange(T),curve_mu,curve_std]))
    json.dump({"peak_abs_t":peak_t,
               "peak_height":None if peak_t<0 or np.isnan(curve_mu[peak_t]) else float(curve_mu[peak_t]),
               }, open(Path(output_dir_parent/'plan_B_peak.json'),'w'))

    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        fig,ax=plt.subplots(figsize=(10,5))
        xs=np.arange(T)
        ax.plot(xs,curve_mu,color="steelblue",lw=2,label="μ(JSD)")
        ax.fill_between(xs,curve_mu-curve_std,curve_mu+curve_std,alpha=.25,color="steelblue",
                        label="±σ across scenarios")
        ax.axhline(BASELINE_THRESHOLD_JSB,color="grey",ls="--",alpha=.6,label=f"Baseline τ₀={BASELINE_THRESHOLD_JSB}")
        ax.set_xlabel("Decoded thought-token offset t")
        ax.set_ylabel("Paired-JS distance between P̄_AS(t) & P̄_AF(t)")
        ax.set_title("Time-resolved representation divergence\n(plan B)")
        ax.legend();ax.grid(alpha=.4)
        plt.tight_layout()
        plt.savefig(Path(output_dir_parent/"plan_B_js_curve.png"),dpi=120)
        plt.close()
    except Exception as exc:
        logger.warning("plot failed:%s",exc)

    return {
      "config":{"top_K_truncated_snapshot_size":int(K),"method":"union_aligned_projection"},
      "curves_summary":{
         "decoded_offset_axis":[int(x) for x in range(T)],
         "mu_JS_distance":[None if np.isnan(c) else round(float(c),8) for c in curve_mu],
         "sigma_JS_among_scenarios":[None if np.isnan(c) else round(float(c),8) for c in curve_std],
         "above_baseline_fraction":[None if np.isnan(a) else round(float(a),4) for a in above_baseline_frac],
      },
      "peak_absolute_decoded_offset_t":int(peak_t) if peak_t>=0 else None,
      "significance_table_full_excerpt_top_eight_positions_by_height":
          sorted(significance_table_full,key=lambda x:(-(x["curve_height_mean"] if x["curve_height_mean"] is not None else -9)))[:8],
    }


def _safe_int_or_zero(rowdict,key):
    val=rowdict.get(key,"0")
    try:return int(val)
    except Exception:return 0


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--probe-dir",required=True,type=str)
    # Tier-2 deconfounder: optional comma-separated list of T-strata. When
    # supplied, ONLY those temperatures are used for the pooled analysis (i.e.
    # acts as an inclusion filter). When omitted, ALL T-strata observed in the
    # probe index.csv are auto-detected and analysed BOTH individually AND
    # pooled, producing a stratified report.
    ap.add_argument("--restrict-temperatures", type=str, default="",
                    help="Comma-separated list of sampling-T values; when "
                         "omitted all distinct collection_temperature entries "
                         "in probe/index.csv are auto-discovered.")
    args=ap.parse_args()
    pd_path=Path(args.probe_dir).resolve()

    logging.basicConfig(level=logging.INFO,format="[%(levelname)s][%(name)s] %(message)s")

    feats_records=[json.loads(line) for line in open(pd_path/'feats.jsonl')]
    index_rows=list(csv.DictReader(open(pd_path/'index.csv')))
    assert len(index_rows)==len(feats_records),(len(index_rows),len(feats_records))

    topk_npz=np.load(pd_path/'topk_dists.npz')
    topk_probs=topk_npz["topk_probs"]
    topk_indices=topk_npz["topk_indices"]

    fbyscen=defaultdict(lambda:{"clean":[],"success":[],"fail":[]})
    for r in feats_records:
        fbyscen[int(r["scenario_index"])][r["kind"]].append(r)

    print("\n=== Loaded seq_probe data ===")
    print(f"samples_total={len(feats_records)} scenarios_covered={len(fbyscen)}")

    # Tier-2: discover or apply T-stratum list.
    if args.restrict_temperatures.strip():
        requested_strata = [float(x) for x in args.restrict_temperatures.split(",")
                            if x.strip()]
        t_strata = sorted(set(requested_strata))
        print(f"[tier2] user-restricted T-strata = {t_strata}")
    else:
        seen_ts = set()
        for r in feats_records:
            tv = r.get("collection_temperature")
            if tv is None or tv == "":
                continue
            try:
                seen_ts.add(float(tv))
            except (TypeError, ValueError):
                pass
        if len(seen_ts) >= 2:
            t_strata = sorted(seen_ts)
            print(f"[tier2] discovered {len(t_strata)} T-strata in probe data "
                  f"= {t_strata}; running both pooled and per-stratum analyses")
        else:
            t_strata = []
            print("[tier2] no multi-stratum data detected; running legacy "
                  "pooled-only analysis")

    sig_planA_pooled=analyze_plan_A(fbyscen)
    print("[done] Plan A segment-aggregated Wilcoxon analyses complete (pooled)")

    sig_planB_pooled=analyze_plan_B(topk_probs,topk_indices,index_rows,pd_path.parent)
    print("[done] Plan B time-resolved JS-divergence analysis complete (pooled)")

    planA_by_stratum: dict[str, dict] = {}
    planB_by_stratum: dict[str, dict] = {}
    for ts_val in t_strata:
        s_list=[ts_val]
        sa = analyze_plan_A(fbyscen, temperature_strata_filter=s_list)
        sb = analyze_plan_B(topk_probs,topk_indices,index_rows,pd_path.parent,
                            temperature_strata_filter=s_list)
        key_str=f"T_{ts_val:.2f}"
        planA_by_stratum[key_str]=sa
        planB_by_stratum[key_str]=sb
        print(f"[done] Plan A+B at T={ts_val:.2f} complete")

    final_report={
      "experiment_metadata":{
        "model_loaded":"/ssd1/models/qwen2.5-7b-it",
        "samples_processed":len(feats_records),
        "scenarios_covered":len(fbyscen),
        "evaluator_version_underlying_features":feats_records[0]["evaluator_version"],
      },
      "temperature_sweep_metadata":{
          "strata_discovered_or_applied":[float(t) for t in t_strata],
          "pooled_includes_all_temperatures":
              not bool(args.restrict_temperatures.strip()),
          "interpretation_note":
              "If effect size monotonically scales WITH T → confound confirmed "
              "(sampling-stochasticity drives signal). If effect persists roughly "
              "constant across all T including low-T ≈0.3 → mechanism hypothesis "
              "strengthened.",
      },
      "signal_plan_A_segment_aggregated_Wilcoxon_tests":sig_planA_pooled,
      "signal_plan_B_time_resolved_paired_JS_divergence_between_AS_and_AF_average_distributions":sig_planB_pooled,
      "signal_plan_A_per_T_stratum":planA_by_stratum if t_strata else {},
      "signal_plan_B_per_T_stratum":planB_by_stratum if t_strata else {},
    }
    out_p = pd_path.parent / "as_vs_af_sequence_results.json"
    out_p.write_text(json.dumps(final_report,indent=2,default=str,ensure_ascii=False))
    print(f"\n=== Final results written to {out_p} ===")


if __name__=="__main__":
    main()


if __name__=="__main__":
    main()
