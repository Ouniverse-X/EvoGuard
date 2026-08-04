"""Statistical analysis for AS-vs-AF mechanism probe outputs — clean rewrite."""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("preliminary.as_vs_af_analysis")
EPS = 1e-12


def _safe_float(x):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v): return None
        return v
    except Exception:
        return None


def _wilcoxon(diffs, alt="two-sided"):
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
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# S1/S4 per-scenario triplet analysis for scalar metrics
# ---------------------------------------------------------------------------

SCALAR_KEYS_OF_INTEREST = ["fts_h", "fts_top1", "fts_nll", "refuse_prob_pos0"]


def analyze_scalar_triplets(features_by_scen: dict[int, dict[str, list[dict]]]) -> dict:
    """For each scalar metric compute within-scenario triplets and pairwise Wilcoxon tests."""

    bonferroni_alpha = 0.05 / 3   # three pairwise contrasts per metric
    out: dict[str, Any] = {}

    for key in SCALAR_KEYS_OF_INTEREST:
        scen_clean_v = []
        scen_as_mean = []
        scen_af_mean = []
        diff_C_minus_A_per_scen = []     # AS - Clean positive => AS > Clean
        diff_A_minus_F_per_scen = []     # AF - AS positive => AF > AS
        diff_F_minus_C_per_scen = []
        strict_monotonic_count = 0       # C < A < F strictly ascending count
        total_evaluable = 0

        for sid, bdict in features_by_scen.items():
            c_recs = bdict.get("clean"); s_recs = bdict.get("success"); f_recs = bdict.get("fail")
            if not c_recs or not s_recs or not f_recs:
                continue
            cv = _safe_float(c_recs[0].get(key))
            sv_list = [_safe_float(r.get(key)) for r in s_recs if _safe_float(r.get(key)) is not None]
            fv_list = [_safe_float(r.get(key)) for r in f_recs if _safe_float(r.get(key)) is not None]
            if cv is None or not sv_list or not fv_list:
                continue
            sm_ = float(np.mean(sv_list))
            fm_ = float(np.mean(fv_list))

            scen_clean_v.append(cv)
            scen_as_mean.append(sm_)
            scen_af_mean.append(fm_)
            diff_C_minus_A_per_scen.append(sm_ - cv)
            diff_A_minus_F_per_scen.append(fm_ - sm_)
            diff_F_minus_C_per_scen.append(fm_ - cv)

            total_evaluable += 1
            if cv < sm_ < fm_:
                strict_monotonic_count += 1

        if total_evaluable == 0:
            out[key] = {"n_valid_scenarios": 0}
            continue

        wcx_p_A_gt_C,_ = _wilcoxon(diff_C_minus_A_per_scen,"greater")    # H_alt: median(C<A)>0 i.e. AS>Clean
        wcx_p_F_gt_A,_ = _wilcoxon(diff_A_minus_F_per_scen,"greater")    # AF>AS direction prediction C<A<F
        wcx_p_F_gt_C_two_sided,n2c = _wilcoxon(diff_F_minus_C_per_scen,"greater")

        out[key] = {
            "sub_metric_key": key,
            "n_valid_scenarios": int(total_evaluable),
            "summary_means_across_scenarios": {
                "clean": _summarize(scen_clean_v),
                "AS_avg_within_bucket_then_over_scenarios": _summarize(scen_as_mean),
                "AF_avg_within_bucket_then_over_scenarios": _summarize(scen_af_mean),
            },
            "pairwise_wilcoxon_one_sided_predicting_increasing_order_C_less_than_AS_less_than_AF": {
                "p_value_for_AS_greater_than_clean": float(wcx_p_A_gt_C),
                "p_value_for_AF_greater_than_AS": float(wcx_p_F_gt_A),
                "p_value_for_AF_greater_than_clean": float(wcx_p_F_gt_C_two_sided),
            },
            "bonferroni_corrected_alpha_applied_to_each_pairwise_test": float(bonferroni_alpha),
            "strictly_monotonic_count_out_of_total_evaluable": f"{strict_monotonic_count}/{total_evaluable}",
            "_interpretation_hint":
              "If predicted order holds then all three one-sided tests should yield small p-values "
              "(below Bonferroni alpha). For metrics that may run REVERSE to the original hypothesis "
              "(e.g., NLL where lower=more-natural), inspect sign of mean differences instead of "
              "relying on directional verdicts."
        }

    return out


# ---------------------------------------------------------------------------
# S2 Top-K concentration descriptives only
# ---------------------------------------------------------------------------

def analyze_S2(topk_probs: np.ndarray, index_rows: list[dict]) -> dict:
    grp_idx = defaultdict(list)
    for ri,row in enumerate(index_rows):
        k=row["rollout_kind"]
        bucket_name={"clean":"clean","success":"AS","fail":"AF"}.get(k)
        if bucket_name:grp_idx[bucket_name].append(int(ri))
    res={}
    for bn,idxs in grp_idx.items():
        probs=topk_probs[np.asarray(idxs,dtype=int)].astype(np.float64)+EPS
        topk_sum_normalized=(probs.sum(axis=-1)/(probs.sum(axis=-1,keepdims=True)))
        eff_k_via_inv_sq_norm=((probs**-2).sum(axis=-1))**(-1)/topk_probs.shape[1]
        res[bn]={
          "N_rollouts":int(len(idxs)),
          "avg_mass_fraction_captured_by_truncated_topK_vs_self_normalization":float(probs.mean()),
          "effective_rank_inverse_squared_on_full_K_window":float(((probs**-2).sum(axis=-1)).mean()),
        }
    return {"per_group_descriptives":res,
            "note":"S2 reported descriptively; significance testing on truncated distributions unreliable."}


# ---------------------------------------------------------------------------
# S3 Paired Jensen-Shannon distance between Success-mean & Fail-mean distros
# ---------------------------------------------------------------------------

def reconstruct_dist_from_topk_stacked(rows_probs_fp16:np.ndarray,
                                        rows_indices_int32:np.ndarray,
                                        vocab_size:int)->np.ndarray:
    """Average across rows producing a single full-vocab distribution vector normalized to sum-to-one.

    Approximation: missing tail mass assumed zero since top-K captures ~99%+ of typical LLM softmax mass.
    """
    M,N=rows_probs_fp16.shape[:2]
    avg=np.zeros(vocab_size,dtype=np.float64)
    flat_ids=np.clip(rows_indices_int32.flatten(),0,vocab_size-1).astype(np.int64)
    flat_prb=rows_probs_fp16.astype(np.float64).flatten()
    summed=np.bincount(flat_ids,weights=flat_prb,minlength=vocab_size)
    avg=summed/M
    norm_factor=avg.sum()
    return avg/norm_factor if norm_factor>EPS else avg


def analyze_S3(topk_probs,topk_indices,index_rows):
    vocab_size=int(max(topk_indices.max()+1,150000))
    grouped=defaultdict(lambda:{'success':[],'fail':[]})
    for ri,rw in enumerate(index_rows):
        si=rw['scenario_index']
        kind=rw['rollout_kind']
        if kind=='success':grouped[si]['success'].append(ri)
        elif kind=='fail': grouped[si]['fail' ].append(ri)

    js_vals=[]
    succ_sizes=[]; fail_sizes=[]
    for si,g in sorted(grouped.items()):
        sg=g['success']; fg=g['fail']
        if len(sg)<3 or len(fg)<3:continue
        ps=reconstruct_dist_from_topk_stacked(
             topk_probs[np.asarray(sg)], topk_indices[np.asarray(sg)],
             vocab_size)
        pf=reconstruct_dist_from_topk_stacked(
             topk_probs[np.asarray(fg)], topk_indices[np.asarray(fg)],
             vocab_size)
        m=0.5*(ps+pf)
        kl_sm=float((ps*np.log(ps/m + EPS)).sum())
        kl_fm=float((pf*np.log(pf/m + EPS)).sum())
        js_dist_squared=kl_sm*0.5+kl_fm*0.5
        js_vals.append(js_dist_squared)
        succ_sizes.append(len(sg));fail_sizes.append(len(fg))

    arr_js=np.array(js_vals)
    baseline_threshold=0.001
    above_baseline_frac=float((arr_js>baseline_threshold).mean()) if arr_js.size else None

    return{
      'count_scenarios_evaluated':int(arr_js.size),
      'mean_JS_distance':float(arr_js.mean()) if arr_js.size else None,
      'median_JS_distance':float(np.median(arr_js)) if arr_js.size else None,
      'max_JS_distance':float(arr_js.max()) if arr_js.size else None,
      'std_JS_distance':float(arr_js.std()) if arr_js.size else None,
      'fraction_above_baseline_threshold_'+str(baseline_threshold):above_baseline_frac,
      '_note':"JS values computed via truncated-top512 reconstruction approximating full-vocab distribution."
    }


# ---------------------------------------------------------------------------
# Plan B cross-layer representation divergence profile
# ---------------------------------------------------------------------------

L_LAYERS_PLUS_EMBED=29
HIDDEN_DIM=3584


def analyze_planB(hstates_mm,index_rows,output_dir:Path):
    """Per-layer ||μ_AF−μ_AS||/σ_pool curve with peak detection."""
    L=L_LAYERS_PLUS_EMBED
    HD=HIDDEN_DIM

    grouped=defaultdict(lambda:{'success':[],'fail':[]})
    for ri,rw in enumerate(index_rows):
        si=rw['scenario_index']
        kind=rw['rollout_kind']
        if kind=='success':grouped[si]['success'].append(ri)
        elif kind=='fail': grouped[si]['fail' ].append(ri)

    sc_iter=list(sorted(grouped.keys()))
    n_total=len(sc_iter)
    distances_matrix=np.zeros(shape=(L,max(n_total,1))) * np.nan

    valid_col_mask=[]
    valid_cols=[]
    for jcol,si in enumerate(sc_iter):
        gsucc=grouped[si]['success']; gfail=grouped[si]['fail']
        if len(gsucc)<3 or len(gfail)<3:
            continue
        hs_succ=hstates_mm[np.asarray(gsucc)].astype(np.float64)
        hs_fail=hstates_mm[np.asarray(gfail )].astype(np.float64)

        mu_succ=hs_succ.mean(axis=0)         # [L,H]
        mu_fail=hs_fail .mean(axis=0)         # [L,H]
        sd_succ=hs_succ.std(axis=0)+EPS       # [L,H]
        sd_fail=hs_fail.std(axis=0)+EPS

        # Per-layer scalars: numerator=||μ_AS-μ_AF||_2 ; denominator=RMS of within-bucket σ averaged across H.
        diff_norm_per_layer=np.linalg.norm(mu_succ-mu_fail,axis=-1)             # [L]
        sd_rms_succ=np.sqrt((sd_succ**2).mean(axis=-1))                          # [L]
        sd_rms_fail=np.sqrt((sd_fail**2).mean(axis=-1))                          # [L]
        sigma_pool_per_layer=(sd_rms_succ+sd_rms_fail)*0.5                       # [L]
        delta_l_vec=diff_norm_per_layer/(sigma_pool_per_layer+EPS)              # [L]

        col=jcol
        while col>=distances_matrix.shape[1]:
            new_shape=list(distances_matrix.shape)
            new_shape[1]=new_shape[1]*2
            expanded=np.full(new_shape,np.nan)
            expanded[:,:distances_matrix.shape[1]]=distances_matrix
            distances_matrix=expanded

        distances_matrix[:,col]=delta_l_vec
        valid_cols.append(col)
        valid_col_mask.append(True)

    dmat_used=distances_matrix[:,valid_cols] if valid_cols else distances_matrix[:, :0]

    rel_depth_arr=np.arange(L,float(L))/(max(L-1,1))
    rel_depths=[i/(L-1) for i in range(L)]
    curve=dmat_used.mean(axis=-1) if dmat_used.size else np.zeros(L)
    err=dmat_used.std(axis=-1) if dmat_used.size else np.zeros(L)
    peak_layer_abs=int(curve.argmax()) if curve.size else -1
    peak_rel_depth=float(rel_depths[peak_layer_abs]) if peak_layer_abs>=0 else None

    significance_table=[]
    from scipy.stats import ttest_1samp,wilcoxon
    for li in range(L):
        samples=dmat_used[li,:]
        samples=samples[~np.isnan(samples)] if samples.size else np.array([])
        if samples.size<5 or abs(samples.mean())<EPS*10:
            pval=None
        else:
            _,pval=ttest_1samp(samples,popmean=0.)
            pval=float(pval)
        significance_table.append({
           "layer_absolute_index":li,
           "relative_depth":rel_depths[li],
           "delta_mu_sigma_ratio_curve_height_at_this_depth":float(curve[li]) if li<curve.size else None,
           "std_among_scenarios":float(err[li]) if li<err.size else None,
           "pvalue_two_tailed_ttest_against_zero":pval,
        })

    np.save(Path(output_dir.parent / "plan_B_divergence_curve.npy"),
            np.stack([np.array(rel_depths),curve,err]))

    json.dump({"peak_relative_depth":peak_rel_depth,"peak_layer_absolute_index":peak_layer_abs},
              open(Path(output_dir.parent/"plan_B_peak.json"),"w"))

    try:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        fig,ax=plt.subplots(figsize=(9,5))
        ax.plot(rel_depths,curve,color="steelblue",lw=2,label="Mean Δ‖μ‖/σ_pool")
        ax.fill_between(rel_depths,curve-err,curve+err,alpha=.25,color="steelblue",
                        label="±1 std across scenarios")
        ax.axvline(peak_rel_depth,color="red",ls="--",alpha=.6,label=f"Peak @ depth≈{peak_rel_depth:.3f}")
        ax.set_xlabel("Relative transformer depth (=layer/(L−1))")
        ax.set_ylabel("Paired ‖μ_AS − μ_AF‖₂ ÷ σ_within_bucket")
        ax.set_title("Cross-Layer Representation Divergence Profile\n(plan B)")
        ax.legend()
        ax.grid(alpha=.4)
        plt.tight_layout()
        plt.savefig(Path(output_dir.parent/"plan_B_divergence.png"),dpi=120)
        plt.close()
    except Exception as exc:
        logger.warning("plot failed:%s",exc)


    top_layers_sorted=sorted(significance_table,key=lambda x:-x["delta_mu_sigma_ratio_curve_height_at_this_depth"])[:8]
    return{
       "curves_summary":{
         "relative_depth_axis":[round(rd,4) for rd in rel_depths],
         "height_values":[round(float(c),6) for c in curve],
         "error_bars_std":[round(float(e),6) for e in err],
       },
       "peak_relative_depth":peak_rel_depth,
       "peak_layer_absolute_index":peak_layer_abs,
       "n_valid_scenarios_entered_into_aggregation":len(valid_cols),
       "significance_table_excerpt_top_eight_layers_by_height":top_layers_sorted,
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

import typing as Any_alias_unused_typing_module_placeholder
Any=dict|typing.Any if False else object

class _any_namespace_holder(object):
    pass

# proper aliasing so that `Any` works below without circular issues
del Any
try:
    from typing import Any as RealTypingAnyAlias
except ImportError:
    pass



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--probe-dir",required=True,type=str)
    args=ap.parse_args()
    pd_path=Path(args.probe_dir).resolve()

    logging.basicConfig(level=logging.INFO,format="[%(levelname)s][%(name)s] %(message)s")

    feats_records=[json.loads(line) for line in open(pd_path/'features.jsonl')]
    index_rows=list(csv.DictReader(open(pd_path/'index.csv')))
    assert len(index_rows)==len(feats_records),(len(index_rows),len(feats_records))

    hstates_mm=np.memmap(str(pd_path/'hstates.npy'),
                         dtype=np.float16,mode='r',
                         shape=(len(feats_records),29,3584))
    topk_npz=np.load(pd_path/'topk_dists.npz')
    topk_probs=topk_npz["topk_probs"]
    topk_indices=topk_npz["topk_indices"]

    fbyscen=defaultdict(lambda:{"clean":[],"success":[],"fail":[]})
    for r in feats_records:
        fbyscen[r["scenario_index"]][r["kind"]].append(r)

    print("\n=== Loaded probe data ===")
    print(f"samples_total={len(feats_records)} scenarios_covered={len(fbyscen)}")

    sig_S1_S4_combined=analyze_scalar_triplets(fbyscen)
    print("[done] analyzed S1+S4 scalar metrics")
    sig_S2=analyze_S2(topk_probs,index_rows)
    print("[done] analyzed S2 concentration metrics")
    sig_S3=analyze_S3(topk_probs,topk_indices,index_rows)
    print("[done] analyzed S3 paired-JS metric")
    planb_result=analyze_planB(hstates_mm,index_rows,pd_path)
    print("[done] plan B divergence analysis complete")

    final_report={
      "experiment_metadata":{
        "model_loaded":"/ssd1/models/qwen2.5-7b-it",
        "samples_processed":len(feats_records),
        "scenarios_covered":len(fbyscen),
      },
      "signal_scalar_metrics_with_pairwise_wilcoxon_tests_predicting_order_C_lt_AS_lt_AF":
          sig_S1_S4_combined,
      "signal_S2_TopK_concentration_descriptives_only":sig_S2,
      "signal_S3_Paired_JS_distance_success_vs_fail_distributions":sig_S3,
      "plan_B_cross_layer_representation_divergence_profile":planb_result,
    }
    Path(pd_path.parent/"mechanism_probe_results.json").write_text(json.dumps(final_report,indent=2,default=str,ensure_ascii=False))
    print(f"\n=== Final results written to {pd_path.parent}/mechanism_probe_results.json ===")


if __name__=="__main__":
    main()
