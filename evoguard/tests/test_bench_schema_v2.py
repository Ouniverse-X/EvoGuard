"""Tests for bench_v2 constants & schemas. Runnable as python -m or pytest."""
from __future__ import annotations

import importlib
import json
import pathlib
import tempfile


def _bc():
    return importlib.import_module("evoguard.process.bench_constants")


def test_required_n_per_cell_default_is_100():
    assert _bc().REQUIRED_N_PER_CELL_DEFAULT == 100


def test_family_concentration_ceiling_value():
    assert _bc().FAMILY_CONCENTRATION_CEILING == 0.35


def test_shift_step_upper_bound_factor_value():
    assert _bc().SHIFT_STEP_UPPER_BOUND_FACTOR == 0.25


def test_dev_snapshot_threshold_ratio_value():
    assert _bc().DEV_SNAPSHOT_THRESHOLD_RATIO == 0.90


def test_public_release_threshold_ratio_value():
    assert _bc().PUBLIC_RELEASE_THRESHOLD_RATIO == 1.00


def test_legacy_unknown_sentinel_method_tag_is_empty_string():
    assert _bc().LEGACY_UNKNOWN_SENTINEL_METHOD_TAG == ""


# NOTE on ``tmp_path`` substitution below:
# Plan-supplied body used pytest's built-in ``tmp_path`` fixture; repo sibling-
# test conventions (test_probe_ranking.py / test_native_grpo_advantage_shaping.py)
# require each suite work identically under both bare ``python -m evoguard.tests.<x>``
# (no pytest runtime) AND pytest discovery — so we accept an optional-defaulted
# param that falls back to a fresh tempfile.mkdtemp() when invoked without the
# fixture. All assertion bodies are byte-for-byte identical to Task-2 spec.
def _opt_tmp(tmp_path=None) -> pathlib.Path:
    return pathlib.Path(tmp_path or tempfile.mkdtemp())


def _bs():
    return importlib.import_module("evoguard.process.bench_schema")


def test_signals_ref_roundtrip_json():
    bs=_bs()
    sr=bs.SignalsRef(injection_point=3,turning_point=7,delta=4,
                    delta_normalized=0.5,edit_distance=0)
    s=sr.to_json_line()
    restored=bs.SignalsRef.from_json_line(s)
    assert restored==sr


def test_scenario_record_v2_mined_row_emits_v2_schema_version_field(tmp_path=None):
    bs=_bs(); bc=_bc()
    rec=bs.ScenarioRecordV2(
        scenario_id="scn_deadbeefdeadbeef",
        bucket="d4",
        origin_mode=bc.ORIGIN_MODE_MINED,
        delta_value_orig=4,
        canonical_technique_id="tech_aaaaaaaaaaaa",
        method_tag_raw="deferred_trigger",
        task_id="agentdojo:workspace:abc12345",
        domain="workspace",
        toolkit_signature="get_day_calendar_events|create_calendar_event|send_email",
        goal_instruction="dummy instruction text",
        context_prefix_actions=[],
        poisoned_observation_text="[X]",
        injected_payload_sha256_first16="abc123def456abcd",
        injection_target_turn_index=2,
        signals_ref=bs.SignalsRef(injection_point=2,turning_point=6,delta=4,delta_normalized=.667,edit_distance=0),
        provenance={"source":{"exp_dir":"x","round_id":14,"record_id_attacked":"y","attack_id":"z"},
                   "_provenance":{"miner_git_sha":"abc","dedup_key_sha256_first16":"dd","clean_record_record_id":"ee","poisoned_observation_sha256_first16":"ff"}},
    )
    parsed=json.loads(rec.to_json_line())
    assert parsed["schema_version"]=="bench_v2"
    assert parsed["origin_mode"]==bc.ORIGIN_MODE_MINED
    assert parsed["source"]["round_id"]==14           # provenance merged top-level not nested


def test_synthetic_record_carries_validator_metadata_in_provenance_block():
    bs=_bs(); bc=_bc()
    rec=bs.ScenarioRecordV2(
        scenario_id="scn_synth00000000001",
        bucket="d3",
        origin_mode=bc.ORIGIN_MODE_SYNTH_SHIFTED,
        delta_value_orig=3,
        canonical_technique_id="tech_bbbbbbbbbbbb",
        method_tag_raw="<unknown>",
        task_id="agentdojo:workspace:syntheticseed001",
        domain="workspace",
        toolkit_signature="search_calendar_event|create_calendar_event",
        goal_instruction="synthetic seed instr",
        context_prefix_actions=[],
        poisoned_observation_text="[INJ]",
        injected_payload_sha256_first16="syntheticpayloadhash",
        injection_target_turn_index=1,
        signals_ref=bs.SignalsRef(injection_point=1,turning_point=4,delta=3,delta_normalized=.75,edit_distance=0),
        provenance={"_provenance":{
                       "synthesizer_version":"v2.0.0-alpha",
                       "validator_judge_model":"qwen2.5-7b-it-judge-port8002",
                       "shift_steps_from_origin":-1,
                       "origin_scenario_id":"scn_origin_seed_xyzw",
                       "replay_defender_model_state":"git-sha-placeholder"}},
    )
    p=json.loads(rec.to_json_line())["_provenance"]
    assert p["validator_judge_model"]=="qwen2.5-7b-it-judge-port8002"
    assert p["shift_steps_from_origin"]==-1


def test_iter_load_scenarios_excludes_synth_rows_by_default(tmp_path=None):
    bs=_bs(); bc=_bc()
    fp=_opt_tmp(tmp_path)/"mixed.jsonl"
    rows=[
      {"scenario_id":"a"*16,"bucket":"d3","schema_version":"bench_v2","origin_mode":bc.ORIGIN_MODE_MINED},
      {"scenario_id":"b"*16,"bucket":"d3","schema_version":"bench_v2","origin_mode":bc.ORIGIN_MODE_SYNTH_SHIFTED},
    ]
    fp.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    loaded=list(bs.iter_load_scenarios(str(fp),exclude_synthetic=True))
    ids={r["scenario_id"] for r in loaded}
    assert ("a"*16) in ids
    assert ("b"*16) not in ids


def test_iter_load_scenarios_include_synthetic_flag_optin_returns_all_rows(tmp_path=None):
    bs=_bs(); bc=_bc()
    fp=_opt_tmp(tmp_path)/"mixed.jsonl"
    rows=[
      {"scenario_id":"a"*16,"bucket":"d3","schema_version":"bench_v2","origin_mode":bc.ORIGIN_MODE_MINED},
      {"scenario_id":"b"*16,"bucket":"d3","schema_version":"bench_v2","origin_mode":bc.ORIGIN_MODE_SYNTH_SHIFTED},
    ]
    fp.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    n_default=len(list(bs.iter_load_scenarios(str(fp),exclude_synthetic=True)))
    n_all=len(list(bs.iter_load_scenarios(str(fp),exclude_synthetic=False)))
    assert n_default==1 and n_all==2


def _main() -> int:
    fns = [(k,v) for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for name, fn in fns:
        try:
            fn(); print(f"PASS {name}")
        except Exception as exc:
            print(f"FAIL {name}: {exc!r}"); failures.append(name)
    print(f"\n{len(fns)-len(failures)}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_main())
