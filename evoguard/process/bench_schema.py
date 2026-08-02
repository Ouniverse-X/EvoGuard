"""Dataclasses & IO helpers for bench_v2 scenario records."""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from evoguard.process.bench_constants import (
    SCHEMA_VERSION_V2,
    ORIGIN_MODE_MINED,
)


@dataclass(frozen=True)
class SignalsRef:
    """Frozen snapshot of computed signals stored alongside each record."""
    injection_point:int
    turning_point:int
    delta:int
    delta_normalized:float
    edit_distance:int

    def to_dict(self)->dict[str,int|float]:
        return {"injection_point":self.injection_point,"turning_point":self.turning_point,
                "delta":self.delta,"delta_normalized":self.delta_normalized,
                "edit_distance":self.edit_distance}

    @classmethod
    def from_dict(cls,d:dict[str,Any])->"SignalsRef":
        return cls(int(d["injection_point"]),int(d["turning_point"]),
                   int(d["delta"]),float(d["delta_normalized"]),int(d.get("edit_distance",-1)))

    def to_json_line(self)->str:
        return json.dumps(self.to_dict(),ensure_ascii=False)

    @classmethod
    def from_json_line(cls,s:str)->"SignalsRef":
        return cls.from_dict(json.loads(s))


@dataclass
class ScenarioRecordV2:
    scenario_id:str
    bucket:str                      # ∈ {imm,d1,d2,d3,d4}
    origin_mode:str                 # ∈ {mined,synth_shifted}
    delta_value_orig:int
    canonical_technique_id:str
    method_tag_raw:str
    task_id:str
    domain:str
    toolkit_signature:str
    goal_instruction:str
    context_prefix_actions:list[dict[str,Any]]
    poisoned_observation_text:str
    injected_payload_sha256_first16:str
    injection_target_turn_index:int
    signals_ref:SignalsRef
    provenance:dict[str,Any]
    expected_response_length_tokens:int|None=None
    instruction_perplexity_proxy:float|None=None
    confound_flags:list[str]=field(default_factory=list)

    def to_dict(self)->dict[str,Any]:
        d:dict[str,Any]={
            "schema_version":SCHEMA_VERSION_V2,
            "scenario_id":self.scenario_id,
            "bucket":self.bucket,
            "origin_mode":self.origin_mode,
            "canonical_technique_id":self.canonical_technique_id,
            "method_tag_raw":self.method_tag_raw,
            "task_id":self.task_id,
            "domain":self.domain,
            "toolkit_signature":self.toolkit_signature,
            "goal_instruction":self.goal_instruction,
            "context_prefix_actions":list(self.context_prefix_actions),
            "poisoned_observation_text":self.poisoned_observation_text,
            "injected_payload_sha256_first16":self.injected_payload_sha256_first16,
            "injection_target_turn_index":self.injection_target_turn_index,
            "signals_ref":self.signals_ref.to_dict(),
            "expected_response_length_tokens":self.expected_response_length_tokens,
            "instruction_perplexity_proxy":self.instruction_perplexity_proxy,
            "confound_flags":list(self.confound_flags),
            "delta_value_orig":self.delta_value_orig,
        }
        for k,v in self.provenance.items():     # merge source/_provenance keys top-level preserving both shapes
            d[k]=v
        return d

    def to_json_line(self)->str:
        return json.dumps(self.to_dict(),ensure_ascii=False)


def iter_load_scenarios(path:str,*,exclude_synthetic:bool=True)->Iterator[dict[str,Any]]:
    """Stream-parse bench_v2 scenario JSONL honoring exclude_synthetic flag.

    Default behavior matches spec invariant S2: pure-mined subset returned unless caller opts-in explicitly.
    """
    with open(path,"r",encoding="utf-8") as fh:
        for lineno,raw in enumerate(fh,start=1):
            s=raw.strip()
            if not s: continue
            try:
                obj=json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} malformed ({exc})") from exc
            om=obj.get("origin_mode")
            if exclude_synthetic and om!=ORIGIN_MODE_MINED:
                continue
            yield obj
