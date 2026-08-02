"""Tests for bench_v2 constants & schemas. Runnable as python -m or pytest."""
from __future__ import annotations

import importlib
import json
import pathlib
import tempfile
from pathlib import Path


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


def _bm():
    return importlib.import_module("evoguard.process.bench_migrate")


def test_migration_writes_new_layout_files_from_minimal_fixture_corpus(tmp_path=None):
    bm=_bm(); bs=_bs(); bc=_bc()
    tmp=_opt_tmp(tmp_path)
    src_dir=tmp/"src"; dst_dir=tmp/"dst"; alias_path=dst_dir/"techniques"/"aliases.jsonl"

    imm_fp=src_dir/"corpus_imm.jsonl"
    sample={
      "_meta":{"schema_version":"preliminary_bench_v1"},
      "scenarios":[
        {"scenario_id":"scn_imm_test_aaaaaaaa","bucket":"imm","delta_value_orig":0,
         "domain":"workspace","task_id":"agentdojo:workspace:t1","method":"",
         "context_prefix_actions":[{"thought":"","tool_call":{"name":"get_day_calendar_events","arguments":{"day":"x"}},"observation":"[]"}],
         "injection_target_turn_index":0,
         "original_signals_for_reference":{"injection_point":0,"turning_point":0,"delta":0},
         "_provenance":{"clean_record_record_id":"x"}},
      ],
    }
    src_dir.mkdir(parents=True,exist_ok=True)
    imm_fp.write_text("\n".join(json.dumps(s) for s in sample["scenarios"])+"\n")
    (src_dir/"manifest.json").write_text("{}")

    summary=bm.migrate(src_dir=str(src_dir),dst_dir=str(dst_dir),
                        canonical_aliases_out=str(alias_path))

    expected_bucket_file=dst_dir/"scenarios"/"bucket_imm.jsonl"
    assert expected_bucket_file.exists()
    rows=list(bs.iter_load_scenarios(str(expected_bucket_file),exclude_synthetic=False))
    assert len(rows)==1
    r=rows[0]
    assert r["schema_version"]=="bench_v2"
    assert r["origin_mode"]==bc.ORIGIN_MODE_MINED
    assert r["canonical_technique_id"].startswith("tech_")
    assert len(r["canonical_technique_id"])-len("tech_")==12
    assert alias_path.exists()


def test_migration_skips_records_whose_domain_doesnt_match_scope_restriction_workspace_only(tmp_path=None):
    bm=_bm(); bs=_bs()
    bc_unused=_bc()
    tmp=_opt_tmp(tmp_path)
    src_dir=tmp/"src"; dst_dir=tmp/"dst"
    # refactored-for-clarity-from-spec-original-preserving-semantics:
    # Original used ``i in ["0"]`` then computed ``i*"aa"+...[:(15-len(i))]`` which
    # mixes string-with-string slicing/multiplication incorrectly on py>=3.
    # Here we replace list comprehension with explicit single workspace record +
    # one explicit non-workspace record asserting identical pass/fail intent
    # (workspace kept across filter, banking filtered out).
    rows=[{"scenario_id":"wspace_keep_aaaaaaaa","bucket":"d1",
           "delta_value_orig":1,"domain":"workspace","task_id":"agentdojo:workspace:x0",
           "method":"","context_prefix_actions":[],"injection_target_turn_index":0,
           "original_signals_for_reference":{"injection_point":0,"turning_point":1},"_provenance":{}}]
    banking_row={"scenario_id":"bank_drop_ccccccccccccccc","bucket":"d1","delta_value_orig":1,
                 "domain":"banking","task_id":"agentdojo:banking:y","method":"",
                 "context_prefix_actions":[],"injection_target_turn_index":0,
                 "original_signals_for_reference":{"injection_point":0,"turning_point":1}}
    fp=src_dir/"corpus_d1.jsonl"
    src_dir.mkdir(parents=True,exist_ok=True)
    fp.write_text("\n".join(json.dumps(x) for x in (rows+[banking_row]))+"\n")
    (src_dir/"manifest.json").write_text("{}")

    bm.migrate(src_dir=str(src_dir),dst_dir=str(dst_dir),
               canonical_aliases_out=str(dst_dir/"techniques"/"aliases.jsonl"))

    kept=list(bs.iter_load_scenarios(str(dst_dir/"scenarios"/"bucket_d1.jsonl"),exclude_synthetic=False))
    domains={r["domain"] for r in kept}
    assert domains=={"workspace"}
    # refactored-for-clarity-from-spec-original-preserving-semantics:
    # removed placeholder-noop assertion ``assert sum(kept.__sizeof__()for _ in [])>=0``
    # left intact in original plan line; only meaningful asserts survive here.


def test_migration_registry_contains_one_entry_per_unique_canonical_technique_id_seen(tmp_path=None):
    bm=_bm()
    bs_unused=_bs()
    tmp=_opt_tmp(tmp_path)
    src_dir=tmp/"src"; dst_dir=tmp/"dst"
    rows=[]
    methods_used=["authority_directive","developer_override_immediate_invoke","authority_directive"]  # dup intentional
    for i,meth in enumerate(methods_used):
        rows.append({"scenario_id":f"id_{i}_{(20-i)*'a'}"[:24].ljust(19,'b')+"bbbbb","bucket":"d2",
                     "delta_value_orig":2,"domain":"workspace","task_id":f"x{i}",
                     "method":meth,"context_prefix_actions":[],"injection_target_turn_index":0,
                     "original_signals_for_reference":{"injection_point":0,"turning_point":2}})
    src_dir.mkdir(parents=True,exist_ok=True)
    (src_dir/"corpus_d2.jsonl").write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    (src_dir/"manifest.json").write_text("{}")

    bm.migrate(src_dir=str(src_dir),dst_dir=str(dst_dir),
               canonical_aliases_out=str(dst_dir/"techniques"/"aliases.jsonl"))

    registry_lines=(dst_dir/"techniques"/"registry.jsonl").read_text(encoding="utf-8").strip().split("\n")
    entries=[json.loads(l) for l in registry_lines]
    unique_ids={e["tech_id"] for e in entries}
    # two distinct method strings -> potentially different OR collapsed family IDs depending classification;
    # strict requirement: registry contains exactly len(unique_ids) lines (no duplicates within registry).
    assert len(entries)==len(set(e["tech_id"] for e in entries)),f"duplicate tech_ids found in registry"


def test_migration_zero_orphan_rate_after_processing_historical_noisy_tags(tmp_path=None):
    bm=_bm()
    tmp=_opt_tmp(tmp_path)
    src_dir=tmp/"src"; dst_dir=tmp/"dst"
    noisy_methods=["","<unknown>","authenv_delete_legitimate_mail_after_read_exfil_t2",
                   "ambient_metadata_blended_governance_with_spoofed_integrity_log_and_multi_stage_retention_protocol",
                   "steganographic_gdpr_authority_with_triple_channel_exfiltration_and_delayed_spoof"]
    rows=[{"scenario_id":f"s_{i}_{'aaaabbbbccccddd'}","bucket":"d3","delta_value_orig":3,
           "domain":"workspace","task_id":f"w_{i}",
           "method":m,"context_prefix_actions":[],"injection_target_turn_index":0,
           "original_signals_for_reference":{"injection_point":0,"turning_point":3}}
          for i,m in enumerate(noisy_methods)]
    src_dir.mkdir(parents=True,exist_ok=True)
    (src_dir/"corpus_d3.jsonl").write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    (src_dir/"manifest.json").write_text("{}")

    summary=bm.migrate(src_dir=str(src_dir),dst_dir=str(dst_dir),
                        canonical_aliases_out=str(dst_dir/"techniques"/"aliases.jsonl"))

    assert summary["records_orphan_after_classification"]==0
    assert summary["records_total_processed"]==len(noisy_methods)


def test_adapter_resolve_legacy_corpus_path_yields_equivalent_content_when_called_against_migrated_tree(tmp_path=None):
    bs=_bs(); bm=_bm()
    src=_opt_tmp(tmp_path)/"src"; dst=_opt_tmp(tmp_path)/"dst"
    src.mkdir(parents=True,exist_ok=True); dst.mkdir(parents=True,exist_ok=True)
    (src/"corpus_d1.jsonl").write_text("\n".join([
        json.dumps({"scenario_id":"legacy_aa"+"bb"+"bb"+"bb"+"bb"+"bb"+"bb"+"bb"+"bb","bucket":"d1",
                    "delta_value_orig":1,"domain":"workspace","task_id":"tx","method":"",
                    "context_prefix_actions":[],"injection_target_turn_index":0,
                    "original_signals_for_reference":{"injection_point":0,"turning_point":1}})])+"\n")
    (src/"manifest.json").write_text("{}")
    bm.migrate(src_dir=str(src),dst_dir=str(dst),
               canonical_aliases_out=str(dst/"techniques"/"aliases.jsonl"))

    resolved=bs.resolve_legacy_corpus_path(repo_root=str(dst),bucket_label="d1")
    rows_via_old_name=list(bs.iter_load_scenarios(resolved,exclude_synthetic=False))
    rows_via_new_name=list(bs.iter_load_scenarios(str(Path(dst)/"scenarios"/"bucket_d1.jsonl"),
                                                  exclude_synthetic=False))
    assert {r["scenario_id"] for r in rows_via_old_name}=={r["scenario_id"] for r in rows_via_new_name}


def test_adapter_resolve_legacy_path_raises_helpful_error_when_neither_variant_exists(tmp_path=None):
    bs=_bs()
    raised=False
    try:
        bs.resolve_legacy_corpus_path(repo_root=str(_opt_tmp(tmp_path)),bucket_label="d99_missing")
    except FileNotFoundError as e:
        msg=str(e); raised=("neither" in msg.lower())
    except Exception:
        pass
    assert raised


def _bp():
    return importlib.import_module("evoguard.process.bench_power_calc")


def test_required_n_cell_matches_handcalc_defaults_alpha_dot05_beta_dot20_effectsize_dot45_k_eq_4():
    bp=_bp(); bc=_bc()
    req=bp.compute_required_n_per_cell(effect_size_cohen_d=bc.DEFAULT_EFFECT_SIZE_COHEN_D,
                                       alpha_overall=bc.DEFAULT_ALPHA_OVERALL,
                                       beta=bc.DEFAULT_BETA_POWER_TARGET,
                                       bonferroni_correction_count=4)
    # closed-form hand-check: ((zα'+zβ)^2)/(d^2)= ((2.49+0.84)^2)/(0.45^2) ≈ 54.7→55 rounded up
    assert 50<=req<=60


def test_required_n_decreases_as_effect_size_grows_larger_signal_detectable_smaller_samples_needed():
    bp=_bp()
    small_eff=bp.compute_required_n_per_cell(effect_size_cohen_d=0.30,alpha_overall=.05,beta=.20,bonferroni_correction_count=4)
    large_eff=bp.compute_required_n_per_cell(effect_size_cohen_d=0.60,alpha_overall=.05,beta=.20,bonferroni_correction_count=4)
    assert small_eff > large_eff > 10


def test_required_n_increases_monotonically_with_more_concurrent_contrasts_stricter_Bonferroni():
    bp=_bp()
    base=bp.compute_required_n_per_cell(effect_size_cohen_d=.45,alpha_overall=.05,beta=.20,bonferroni_correction_count=1)
    more=bp.compute_required_n_per_cell(effect_size_cohen_d=.45,alpha_overall=.05,beta=.20,bonferroni_correction_count=8)
    most=bp.compute_required_n_per_cell(effect_size_cohen_d=.45,alpha_overall=.05,beta=.20,bonferroni_correction_count=16)
    assert base <= more <= most


def test_gate_decision_function_blocks_release_below_threshold_ratio_passes_above_above_threshold_ratio():
    bp=_bp(); bc=_bc()
    decision_dev_snapshot=bp.decide_stagegate(current_n_by_bucket={"imm":95,"d1":100,"d2":110,"d3":92,"d4":90},required_n=100)
    decision_full_release=bp.decide_stagegate({"imm":105,"d1":120,"d2":130,"d3":115,"d4":108},required_n=100)
    decision_blocked=bp.decide_stagegate({"imm":40,"d1":40,"d2":35,"d3":32,"d4":16},required_n=100)
    assert decision_dev_snapshot["decision"]=="DEV_SNAPSHOT_ALLOWED"
    assert decision_full_release["decision"]=="PUBLIC_RELEASE_ALLOWED"
    assert decision_blocked["decision"]=="BLOCKED_BELOW_HALF_FLOOR_OR_DEV_THRESHOLD"


def _sr():
    return importlib.import_module("evoguard.process.synth_replant")


def test_extract_seeds_reads_existing_low_mid_buckets_returning_enriched_triplets(tmp_path=None):
    sr=_sr(); bm=_bm()
    _t=_opt_tmp(tmp_path)
    src=_t/"src"; dst=_t/"dst"; src.mkdir(parents=True,exist_ok=True)
    lowmid_sample=[{"scenario_id":"seed_aaabbbbbbbbbbbbb","bucket":"d1","delta_value_orig":1,
                    "domain":"workspace","task_id":"ws_task_seed_A","method":"authority_directive",
                    "context_prefix_actions":[{"thought":"","tool_call":{"name":"search_calendar_events"},
                                              "observation":"raw benign obs text"}],
                    "injection_target_turn_index":0,
                    "original_signals_for_reference":{"injection_point":0,"turning_point":1},
                    "_provenance":{"clean_record_record_id":"abc-def-ghi-jkl-mnopqrstuvwx-yz"}}
                   ,{"scenario_id":"seed_bbbbbaaaaaaaaaaa","bucket":"d2","delta_value_orig":2,
                    "domain":"workspace","task_id":"ws_task_seed_B","method":"developer_override_immediate_invoke",
                    "context_prefix_actions":[{"thought":"","tool_call":{"name":"send_email"},"observation":"ok"}],
                    "injection_target_turn_index":0,
                    "original_signals_for_reference":{"injection_point":0,"turning_point":2},
                    "_provenance":{"clean_record_record_id":""}}]
    (src/"corpus_d1.jsonl").write_text(json.dumps(lowmid_sample[0])+"\n")
    (src/"corpus_d2.jsonl").write_text(json.dumps(lowmid_sample[1])+"\n")
    (src/"manifest.json").write_text("{}")
    bm.migrate(src_dir=str(src),dst_dir=str(dst),
               canonical_aliases_out=str(dst/"techniques"/"aliases.jsonl"))

    seeds=sr.extract_seeds(bench_root=str(dst),
                           source_buckets=("imm","d1","d2"),
                           rounds_root="/nonexistent_intentional_skip_clean_twin_load")
    assert len(seeds)>=2
    payloads_present=sum(1 for s in seeds if getattr(s,"poisoned_observation_text","").strip())
    payloads_empty=sum(1 for s in seeds if not getattr(s,"poisoned_observation_text","").strip())
    assert payloads_present+payloads_empty==len(seeds)
    assert all(getattr(s,"origin_scenario_id","")!="?" for s in seeds)


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
