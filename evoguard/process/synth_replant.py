"""Strategy-① Turn-Shift Replant Validation synthesizer for Δ-bucket benchmark v2.

Stages implemented progressively across multiple commits/tasks per docs/superpowers/plans/...md:
Stage β-1 ::extract_seeds        (this task)
Stage β-2 ::enumerate_candidates (Task 8)
Stage β-3 ::validate_forward_replay (Task 9)
Stage β-4 ::judge_acceptance     (Task 10)
Stage β-5 ::assemble_diversified (Task 11)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from evoguard.process.bench_constants import (
    DOMAIN_SCOPE_RESTRICTION_V2,
    MAX_CANDIDATE_POSITIONS_PER_SEED_FACTOR,
    SHIFT_STEP_UPPER_BOUND_FACTOR,
    REPLAY_LOOKAHEAD_BUFFER_STEPS,
    FAMILY_CONCENTRATION_CEILING,
    ORIGIN_MODE_MINED,
    ORIGIN_MODE_SYNTH_SHIFTED,
)
from evoguard.process.bench_schema import iter_load_scenarios


@dataclass
class SynthSeed:
    """Material harvested from existing successful low/mid-Δ attack ready for shift-replant experiments."""
    origin_scenario_id:str
    bucket_origin:str                              # e.g., "d1" where this seed came FROM originally
    delta_value_orig:int                           # measured Δ of source success
    domain:str                                     # always equals DOMAIN_SCOPE_RESTRICTION_V2 currently
    task_id:str
    toolkit_signature:str
    channel_class_hint:str                         # extracted from canonical-tech tuple (informational only initially)
    canonical_technique_id:str
    poisoned_observation_text:str                  # isolated malicious substring spliced into observation field
    context_prefix_actions_verbatim:list[dict]     # replay-ready action stream UP THROUGH injection site
    injection_target_turn_index_original:int
    clean_trajectory_actions_full:list[dict]       # best-effort populated when round-files accessible; empty-list sentinel otherwise
    metadata_notes:dict                            # freeform diagnostics bag growing across stages


@dataclass
class CandidateShiftPosition:
    seed:"SynthSeed"
    proposed_injection_turn_index:int              # zero-indexed turn slot along clean-A trajectory
    target_delta:int                               # desired outcome integer ∈ {1..4}
    estimated_max_lookahead_budget:int             # bound = target_delta + REPLAY_LOOKAHEAD_BUFFER_STEPS


def _load_rounds_safe(rounds_root:str,clean_record_id_hint:str)->list[dict]:
    """Best-effort loader returning actions stream from associated clean-record id hint.

    Returns empty list silently when either rounds_root inaccessible OR record_id not locatable,
    enabling graceful degradation documented via metadata['clean_twin_status']='missing'|'loaded'.
    """
    rp=Path(rounds_root)
    if not rp.is_dir(): return []
    needle=clean_record_id_hint.strip()
    if not needle: return []
    matches=list(rp.rglob("*.jsonl"))
    for jf in matches:
        try:
            with open(jf,"r",encoding="utf-8") as fh:
                for ln in fh:
                    ln=ln.strip()
                    if not ln:continue
                    obj=__import__("json").loads(ln)
                    rid=obj.get("record_id") or obj.get("id") or ""
                    if str(rid)==needle:
                        traj=obj.get("trajectory") or {}
                        acts=traj.get("actions") or []
                        if isinstance(acts,list):return acts
        except Exception:
            continue
    return []


def extract_seeds(*,bench_root:str,source_buckets:Sequence[str]=("imm","d1","d2"),
                  rounds_root:str=".")->list[SynthSeed]:
    """Read migrated scenarios/bucket_<lbl>.jsonl extracting reusable material.

    Best-effort loads paired clean-trajectory twins from rounds/<exp>/round_*/records.jsonl when available.
    """
    br=Path(bench_root)
    seeds:list[SynthSeed]=[]
    for bkt in source_buckets:
        scen_path=br/"scenarios"/f"bucket_{bkt}.jsonl"
        if not scen_path.exists(): continue
        for row in iter_load_scenarios(str(scen_path),exclude_synthetic=True):
            dom=row.get("domain")
            if dom!=DOMAIN_SCOPE_RESTRICTION_V2: continue

            ctx=row.get("context_prefix_actions") or []
            iti=row.get("injection_target_turn_index",0)

            # Extract poisoned substring: prefer dedicated field; fallback scan observation at index iti.
            pot=row.get("poisoned_observation_text") or ""
            if not pot.strip() and 0<=iti<len(ctx):
                obs_str=str(ctx[iti].get("observation",""))
                # Heuristic: take suffix beyond legitimate-tool-output heuristic boundary markers
                markers=["\n[System Update]\n","\n--- recovered conversation excerpt ---\n","ACTION REQUIRED:\n","</INFORMATION>\n"]
                cut_idx=-1
                for mkr in markers:
                    pos=obs_str.find(mkr)
                    if pos!=-1:
                        candidate=pos+len(mkr.splitlines()[0])+1
                        cut_idx=candidate if cut_idx==-1 else min(cut_idx,pos)
                pot=obs_str[max(cut_idx,0):].strip() if cut_idx>-1 else ""

            prov_blk=row.get("_provenance",{})
            cri_id=prov_blk.get("clean_record_record_id","")
            clean_acts=_load_rounds_safe(rounds_root,cri_id)

            seeds.append(SynthSeed(
                origin_scenario_id=row.get("scenario_id","unknown_seed"),
                bucket_origin=bkt,
                delta_value_orig=int(row.get("delta_value_orig",row.get("signals_ref",{}).get("delta",0))),
                domain=dom,
                task_id=row.get("task_id",""),
                toolkit_signature=row.get("toolkit_signature",""),
                channel_class_hint="",  # filled lazily later from registry join if needed
                canonical_technique_id=row.get("canonical_technique_id",""),
                poisoned_observation_text=pot,
                context_prefix_actions_verbatim=ctx,
                injection_target_turn_index_original=int(iti),
                clean_trajectory_actions_full=clean_acts,
                metadata_notes={"clean_twin_status":"loaded" if clean_acts else "missing",
                                "extraction_source_path":str(scen_path)},
            ))
    return seeds


__all__=["SynthSeed","CandidateShiftPosition","extract_seeds"]
