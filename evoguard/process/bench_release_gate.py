"""Three-stage release-gating evaluator emitting diagnostics bundle for bench_v2 lifecycle management."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from evoguard.process.bench_constants import *
from evoguard.process.bench_power_calc import compute_required_n_per_cell, decide_stagegate
from evoguard.process.bench_schema import iter_load_scenarios


def evaluate_release_state(*,bench_root:str,planned_contrasts_count:int=4,
                           output_diag_subdir:str="diagnostics")->dict[str,Any]:
    br=Path(bench_root); diag=br/output_diag_subdir; diag.mkdir(parents=True,exist_ok=True)
    scen_dir=br/"scenarios"; syn_dir=scen_dir/"_synthetic"

    counts_mined:dict[str,int]={}
    counts_total:dict[str,int]={}
    domains_seen:dict[str,set[str]]={}
    families_seen:dict[str,set[str]]={}

    for lbl in BUCKET_LABELS_LEGACY_ORDERING:
        mp=scen_dir/f"bucket_{lbl}.jsonl"
        cnt_mined=cnt_total=0
        ds:set[str]=set(); fs:set[str]=set()
        if mp.exists():
            for row in iter_load_scenarios(str(mp),exclude_synthetic=False):
                om=row.get("origin_mode")
                if om==ORIGIN_MODE_MINED: cnt_mined+=1
                cnt_total+=1
                ds.add(row.get("domain",""))
                fs.add(row.get("canonical_technique_id",""))
        counts_mined[lbl]=cnt_mined; counts_total[lbl]=cnt_total
        domains_seen[lbl]=ds; families_seen[lbl]=fs

    required_n=compute_required_n_per_cell(effect_size_cohen_d=DEFAULT_EFFECT_SIZE_COHEN_D,
                                           alpha_overall=DEFAULT_ALPHA_OVERALL,
                                           beta=DEFAULT_BETA_POWER_TARGET,
                                           bonferroni_correction_count=planned_contrasts_count)

    decision_pkg=decide_stagegate(counts_total,required_n)

    pool_audit={"eligible_count_by_bucket_mined":counts_mined,
                "eligible_count_by_bucket_total_inc_synth":counts_total,
                "domains_represented_per_bucket":{k:sorted(v) for k,v in domains_seen.items()},
                "distinct_canonical_families_per_bucket":{k:len(v) for k,v in families_seen.items()}}

    power_calc={"required_n_per_cell":required_n,
                "alpha_overall_declared":DEFAULT_ALPHA_OVERALL,
                "beta_targetted":DEFAULT_BETA_POWER_TARGET,
                "effect_size_assumed_cohen_d":DEFAULT_EFFECT_SIZE_COHEN_D,
                "bonferroni_correction_count_applied":planned_contrasts_count,
                "current_ratios_per_bucket":decision_pkg["ratios_per_bucket"],
                "minimum_ratio_observed":decision_pkg["ratio_minimum"]}

    confounds_report={"length_proxy_warnings":["not-implemented-initial-version"],
                      "instruction_entropy_proxy_warnings":["not-implemented-initial-version"],
                      "domain_balance_flags":[lbl for lbl,ds in domains_seen.items() if len(ds)>1 and DOMAIN_SCOPE_RESTRICTION_V2 in ds and len({DOMAIN_SCOPE_RESTRICTION_V2}-ds)!=0][:0]}

    _atomic_write(diag/"pool_audit.json",pool_audit)
    _atomic_write(diag/"power_calc.json",power_calc)
    _atomic_write(diag/"confound_report.json",confounds_report)

    return {"stagegate_decision":decision_pkg["decision"],
            "required_n_per_cell":required_n,
            "counts_mined":counts_mined,
            "counts_total_inc_synth":counts_total,
            "diag_dir":str(diag)}


def _atomic_write(target:Path,obj:Any)->None:
    tmp=tempfile.NamedTemporaryFile(mode="w",dir=target.parent,delete=False,encoding="utf-8",suffix=".tmp")
    try:
        json.dump(obj,tmp,ensure_ascii=False,indent=2,sort_keys=True); tmp.write("\n"); tmp.close()
        Path(tmp.name).rename(target)
    finally:
        try:Path(tmp.name).unlink(missing_ok=True)
        except OSError:pass
