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


def _bt():
    return importlib.import_module("evoguard.process.bench_taxonomy")


def test_classifier_unknown_sentinel_maps_to_canonical_tuple_not_null():
    bt=_bt(); bc=_bc()
    tid,axis_tuple=bt.classify_method_tag(bc.LEGACY_UNKNOWN_SENTINEL_METHOD_TAG)
    assert axis_tuple==bc.CANONICAL_TUPLE_DEFAULT_FOR_UNKNOWN_SENTINEL
    assert tid.startswith("tech_")


def test_classifier_pure_function_same_input_same_output_across_calls():
    bt=_bt()
    samples=["authority_directive","dlg_calendar_description_roleplay_extract_attendee_t3","",None,42,"deferred_trigger","developer_override_immediate_invoke"]
    cache={}
    for s in samples*3:
        try:
            tid,_=bt.classify_method_tag("" if s is None else str(s))
            key=str(s)
            if key in cache:
                assert cache[key]==tid,f"non-pure mapping diverged for {key!r}"
            else:
                cache[key]=tid
        except TypeError:
            continue


def test_envelope_pattern_information_tag_detected_from_compound_name():
    bt=_bt()
    _,tup=bt.classify_method_tag("authenv_information_tag_with_delayed_tool_result_spoof_xxx")
    assert tup[0]=="INFORMATION_TAG"


def test_authority_vector_developer_override_keyword_recognised_variants():
    bt=_bt()
    cases=["developer_override_immediate_invoke","DEVELOPER-OVERRIDE-banner","debug_protocol_xxx"]
    for c in cases:
        _,t=bt.classify_method_tag(c); assert t[3]=="DEVELOPER_OVERRIDE",f"miss on {c}"


def test_compute_techid_truncated_to_twelve_hex_chars_prefixed_tech_underscore():
    bt=_bt()
    tid,_=bt.classify_method_tag("any_random_string_input_here_for_test_only_zzzzz")
    prefix_len=len("tech_"); hexpart=tid[prefix_len:]
    assert len(hexpart)==12
    assert set(hexpart)<=set('0123456789abcdef')


def test_alias_log_writer_appends_one_jsonl_line_per_call(tmp_path=None):
    bt=_bt()
    log_path=_opt_tmp(tmp_path)/"aliases.jsonl"
    history=["foo","","bar_foo"]
    for h in history:
        tid,tup=bt.classify_method_tag(h)
        bt.record_alias_entry(str(log_path),h,(tid,tup),"regex-default-fallback")
    lines=log_path.read_text(encoding="utf-8").strip().split("\n")
    decoded=[json.loads(l) for l in lines]
    assert len(decoded)==len(history)
    assert {d["input_str"] for d in decoded}==set(history)


def test_axis_tuple_length_always_five_even_for_unmatched_inputs():
    bt=_bt()
    _,tup=bt.classify_method_tag("totally novel string nothing matches here")
    assert len(tup)==5


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
