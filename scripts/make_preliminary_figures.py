"""Generate publication-quality figures for preliminary.md plan-A & plan-B sections."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

RESULTS_PATH = Path("rounds/_preliminary/20260803_alertness_v3/as_vs_af_sequence_results.json")
OUT_DIR = Path("docs/assets/preliminary_seq_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    p = float(p)
    if p < 1e-4:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def make_planA_figure(report):
    """Three-panel bar chart comparing Clean / AS / AF across early/mid/late bands."""
    pa = report["signal_plan_A_segment_aggregated_Wilcoxon_tests"]
    metrics = list(pa.keys())
    bands = ["early", "mid", "late"]
    band_labels = ["Early [0,16)", "Mid [16,32)", "Late [32,48)"]
    bonf_alpha = 0.05 / 3   # 0.0167 per pairwise test per metric

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metric_titles = {
        "seq_entropy":     "Thought-token Shannon entropy\n(higher = more uncertain)",
        "seq_top1":        "Top-1 token probability mass\n(lower = more uncertain)",
        "seq_refuse_mass": "Refusal-marker probability mass\n(Wait/Hold/Suspicious...)",
        "seq_nll":         "-log P(actual recorded token)\n(self-surprise)",
    }
    for ax_idx, key in enumerate(metrics):
        ax = axes[ax_idx // 2][ax_idx % 2]
        d = pa[key]["bands"]
        x = np.arange(len(bands))
        width = 0.27

        clean_means = []
        as_means = []
        af_means = []
        sig_labels = []      # one tuple (p_AC,p_AAF,p_FC,n_valid,alpha)
        for b in bands:
            bb = d.get(b, {})
            n = bb.get("n_valid_scenarios", 0)
            if not n:
                clean_means.append(np.nan); as_means.append(np.nan); af_means.append(np.nan)
                sig_labels.append(None)
                continue
            s = bb["summary_means_across_scenarios"]
            clean_means.append(s["clean"]["mean"])
            as_means.append(s["AS_avg_within_bucket_then_over_scenarios"]["mean"])
            af_means.append(s["AF_avg_within_bucket_then_over_scenarios"]["mean"])
            pp = bb["pairwise_wilcoxon_one_sided_predicting_predicted_direction"]
            sig_labels.append((
                float(pp["p_value_for_AS_vs_Clean"]),
                float(pp["p_value_for_AF_vs_AS"]),
                float(pp["p_value_for_AF_vs_Clean"]),
                int(bb.get("n_valid_scenarios",0)),
                float(bb.get("bonferroni_corrected_alpha_applied_to_each_pairwise_test",0.0167)),
                str(bb.get("strictly_monotonic_count_out_of_total_evaluable","?")),
            ))

        # Error bars = std among scenarios of bucket-mean values.
        def std_for(bk):
            out=[]
            for b in bands:
                bb=d.get(b,{})
                if not bb.get("n_valid_scenarios",0):out.append(0);continue
                out.append(bb['summary_means_across_scenarios'][bk]['std'])
            return np.array(out)

        c_err=std_for('clean')
        a_err=std_for('AS_avg_within_bucket_then_over_scenarios')
        f_err=std_for('AF_avg_within_bucket_then_over_scenarios')

        bars_c=ax.bar(x-width,clean_means,width,yerr=c_err,label="Clean",
                      color="#4C72B0",capsize=4,alpha=.85)
        bars_a=ax.bar(x,as_means,width,yerr=a_err,label="Attack-Success (AS)",
                      color="#DD8452",capsize=4,alpha=.85)
        bars_f=ax.bar(x+width,af_means,width,yerr=f_err,
                      label="Attack-Fail (AF)",color="#55A868",capsize=4,alpha=.85)

        # Annotate significance stars above each trio when pairwise tests pass Bonferroni.
        for i,(cm_,am_,fm_) in enumerate(zip(clean_means,as_means,af_means)):
            sl=sig_labels[i]
            if sl is None: continue
            p_ac,p_af_as,p_fc,n_v,alpha_v,mono_str = sl
            top_y=max(cm_ or -np.inf, am_ or -np.inf, fm_ or -np.inf)
            err_top=[c_err[i],a_err[i],f_err[i]][int(np.argmax([cm_ or -999,am_ or -999,fm_ or -999]))]
            y_pos=top_y + err_top + .03*abs(top_y if abs(top_y)>EPS else 1)+.005
            parts=[]
            if p_ac<alpha_v:parts.append("AC✓")
            if p_af_as<alpha_v:parts.append("SA✓")
            if p_fc<alpha_v:parts.append("FC✓")
            label_text=f"N={n_v}\n"+("/".join(parts) if parts else "n.s.")
            if mono_str!="?":
                num,total=int(mono_str.split('/')[0]),int(mono_str.split('/')[-1])
                label_text+=f"\nmono {num}/{total}"
            ax.text(x[i],y_pos,label_text,ha='center',va='bottom',fontsize=7.5,color='#333')

        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, fontsize=10)
        title_lines=metric_titles[key].split("\n")
        pred_dir=("Clean < AS < AF"
                  if pa[key]["bands"]["early"].get("predicted_ordering","").startswith("Clean < ")
                  else "Clean > AS > AF")
        ax.set_title(f"{key} — predicted: {pred_dir}",fontsize=11,fontweight='bold')
        ax.annotate(title_lines[1] if len(title_lines)>1 else "",
                    xy=(0.5,-0.18),xycoords='axes fraction',
                    ha='center',fontsize=8,color="#666")
        ax.grid(axis='y',alpha=.3)
        if ax_idx==3:
            handles,labels_=ax.get_legend_handles_labels()
            fig.legend(handles,labels_,loc='lower center',ncol=3,bbox_to_anchor=(.5,.02),frameon=False,fontsize=10)

    plt.tight_layout(rect=(0,.05,1,1))
    out_path=OUT_DIR/"plan_A_band_triplets.png"
    plt.savefig(out_path,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"[plan A] saved -> {out_path}")


def make_planB_figure(report):
    """Time-resolved JSD curve with peak annotation."""
    pb=(report
        ["signal_plan_B_time_resolved_paired_JS_divergence_between_AS_and_AF_average_distributions"])
    cs=pb["curves_summary"]
    t_axis=np.array(cs["decoded_offset_axis"],dtype=float)
    mu=np.array([v if v is not None else np.nan for v in cs["mu_JS_distance"]])
    sigma=np.array([v if v is not None else np.nan for v in cs["sigma_JS_among_scenarios"]])
    frac_above=np.array([v if v is not None else 0 for v in cs["above_baseline_fraction"]])

    fig,ax=plt.subplots(figsize=(12,6))
    line_mu,=ax.plot(t_axis,mu,lw=2.5,color="#C44E52",label=r"$\mu_{JSD}(t)$ across scenarios")
    fill_handle=ax.fill_between(t_axis,mu-sigma,mu+sigma,color="#C44E52",alpha=.22,
                                label=r"$\pm\sigma$ among scenarios")

    baseline_tau=0.001
    ax.axhline(baseline_tau,color="grey",ls="--",lw=1.,alpha=.65,
               label=f"Baseline τ₀={baseline_tau}")
    theoretical_max=np.sqrt(np.log(2))
    ax.axhline(theoretical_max,color="#34495e",ls=":",lw=1.,alpha=.75,
               label=f"Theoretical max √ln2≈{theoretical_max:.3f}")

    peak_t=pb["peak_absolute_decoded_offset_t"]
    peak_h=None
    for i,t_val in enumerate(cs["decoded_offset_axis"]):
        if int(t_val)==peak_t and cs["mu_JS_distance"][i] is not None:
            peak_h=cs["mu_JS_distance"][i];break
    if peak_h is not None:
        ax.plot([peak_t],[peak_h],'*',color="gold",markersize=20,zorder=5,
                markeredgecolor="black",markeredgewidth=.8)
        ax.annotate(f"peak @ t={peak_t}, height≈{peak_h:.3f}",
                    xy=(peak_t,peak_h),xytext=(max(int(peak_t)-15,5),peak_h+.04),
                    arrowprops=dict(arrowstyle="->",color="black"),fontsize=10)

    # Annotate coverage decay on secondary axis so reader sees where curve becomes noisy tail.
    cov_per_t=[]
    for entry in pb["significance_table_full_excerpt_top_eight_positions_by_height"]:
        pass   # we only have excerpt; instead reconstruct coverage via above_baseline_fraction being nonzero proxy won't work cleanly -- skip second axis.

    region_specs=[
        ("rapid-rise phase",(2,15),"#FDEBD0"),
        ("plateau",(15,30),"#FCF3CF"),
        ("secondary climb",(30,46),"#D5F5E3"),
    ]
    ymax_eff=np.nanmax(mu)*1.15 if not np.all(np.isnan(mu)) else 1.
    for name,(lo,hi),col in region_specs:
        ax.axvspan(lo-.5,hi-.5,color=col,alpha=.35,zorder=-100)
        mid_x=(lo+hi)/2
        ax.text(mid_x,ymax_eff*0.93,name,ha='center',va='top',fontsize=9,color='#444')

    ax.set_xlim(-1,max(t_axis)+1)
    ax.set_ylim(bottom=min(0,np.nanmin(mu)) if not np.all(np.isnan(mu)) else 0,top=ymax_eff)
    ax.set_xlabel("Decoded thought-token offset $t$",fontsize=11)
    ax.set_ylabel(r"$JSD(\bar{P}_{AS}(t), \bar{P}_{AF}(t))$ distance",fontsize=11)
    ax.set_title("Plan B — Time-resolved representation divergence between attack-success vs attack-fail buckets\n"
                 "(union-aligned projection over per-position top-K snapshots; cross-scenario aggregation)",
                 fontsize=11)
    ax.legend(loc='upper left',framealpha=.92,fontsize=9)
    ax.grid(alpha=.25)

    plt.tight_layout()
    out_path=OUT_DIR/"plan_B_jsd_curve.png"
    plt.savefig(out_path,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"[plan B] saved -> {out_path}")


def main():
    report=json.loads(open(RESULTS_PATH).read())
    EPS=1e-30
    globals()['EPS']=EPS
    make_planA_figure(report)
    make_planB_figure(report)


if __name__=="__main__":
    EPS=1e-30
    main()
