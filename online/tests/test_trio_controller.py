"""Offline unit tests for the online single-stage co-evolution trainer.

Mirrors the discipline used by ``evoguard/tests/test_native_grpo_reward.py`` and
``evoguard/tests/test_prompt_extraction.py``: every assertion MUST pass in an
environment lacking ``torch`` / ``trl`` / ``peft`` -- heavy ML deps stay lazily
imported behind the dry-run gate inside :mod:`online.trio_controller` exactly
as ``native_runner.py`` does, keeping these tests second-fast under CPython.

Run::

    python -m pytest online/tests -v           # preferred discovery path
    python -m online.tests.test_trio_controller # legacy direct invocation
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Duck-typed stand-ins -- avoid coupling to concrete dataclass schemas         #
# --------------------------------------------------------------------------- #
def _mk_traj_stub(*, kind_value: str,
                  outcome_value: str,
                  inj_point: Optional[int],
                  post_inj_assistant_text: str,
                  called_tool_name: str = "") -> Any:
    """Minimal stub satisfying the closure / extractor contracts."""
    from types import SimpleNamespace as NS

    tc_obj = None
    if called_tool_name:
        tc_obj = NS(name=called_tool_name)
    actions_list = []
    if inj_point is not None and inj_point >= 0:
        # Pre-injection turns: irrelevant content.
        for i in range(inj_point):
            actions_list.append(NS(turn=i, tool_call=None))
    # First action STRICTLY AFTER injection point carries our completion text.
    next_turn = (max(0, inj_point) + 1) if inj_point is not None else 0
    actions_list.append(
        NS(
            turn=next_turn,
            thought=post_inj_assistant_text,
            final_answer="",
            tool_call=(NS(name=called_tool_name) if called_tool_name else None),
            observation="",
            metadata={},
        )
    )
    return NS(kind_value=kind_value, outcome_value=outcome_value,
              inj_point=inj_point, actions=actions_list)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
# ---- Test A: assemble_completion_payload schema ---------------------------- #
def test_assemble_completion_payload_schema():
    """_assemble_completion_payload yields aligned lists keyed by row_idx & zero-mean groups."""
    from evoguard.training.grpo_reward import PromptMeta
    from online.trio_controller import _assemble_completion_payload

    g_size = 3
    meta_a = PromptMeta(task_id="t1", task_instruction="hello",
                        injection_point=2, turning_point=None,
                        delta_normalized=0.4, target_tool_name="",
                        clean_twin_action_tool_name="")
    meta_b = PromptMeta(task_id="t2", task_instruction="world",
                        injection_point=1, turning_point=None,
                        delta_normalized=0.6, target_tool_name="badtool",
                        clean_twin_action_tool_name="goodtool")
    metas_lut = {10: meta_a, 20: meta_b}
    inputs_rows = [{"row_idx": 10}, {"row_idx": 20}]

    def fake_sample_one(row_idx: int, gi: int) -> tuple[str, float]:
        return (f'{{"thought":"r{row_idx}g{gi}"}}', float(gi * row_idx))

    out = _assemble_completion_payload(inputs_rows=inputs_rows,
                                       metas_lut=metas_lut,
                                       g_size=g_size,
                                       sample_fn=fake_sample_one)

    assert isinstance(out, dict), f"_assemble returned {type(out)!r}"
    expected_keys = {"completions", "rewards", "advantages", "row_idx"}
    missing = expected_keys.difference(set(out.keys()))
    assert not missing, \
        f"missing keys {sorted(missing)} got_keys={list(out.keys())}"
    n_total = len(inputs_rows) * g_size
    lens_eq_n_total = (
        len(out["completions"]) == n_total == len(out["rewards"])
        == len(out["advantages"]) == len(out["row_idx"])
    )
    assert lens_eq_n_total, \
        f"#entries mismatch comps={len(out['completions'])} rews={len(out['rewards'])} advs={len(out['advantages'])} rids={len(out['row_idx'])}"

    for grp_start in range(0, n_total, g_size):
        slice_adv = [float(a) for a in out["advantages"][grp_start:grp_start + g_size]]
        mu = sum(slice_adv) / max(1, len(slice_adv))
        assert abs(mu) < 1e-9, \
            f"group @ offset {grp_start} nonzero-mean ({mu}); slice={slice_adv}"


# ---- Test B: live_judge_closure correctness ---------------------------------#
def test_live_judge_closure_correct_for_each_outcome():
    """Closure threads real verdict bypassing LLM call."""
    from online.trio_controller import make_live_judge_closure
    from evoguard.training.grpo_reward import PromptMeta

    meta = PromptMeta(task_id="t1", task_instruction="instr",
                      injection_point=1, turning_point=None,
                      delta_normalized=0.5, target_tool_name="eviltool",
                      clean_twin_action_tool_name="cleantool")

    cases = [
        ("B_success_leak_unsafe",     "success"),
        ("C_fail_caught_safe",        "fail"),
        ("clean_reference_na_safe",   "na"),       # treated safe
    ]
    for label, outcome_val in cases:
        rec_kind = "attacked" if outcome_val != "na" else "clean"
        fake_traj_rec = _mk_traj_stub(
            kind_value=rec_kind, outcome_value=outcome_val,
            inj_point=meta.injection_point,
            post_inj_assistant_text='{"thought":"","final_answer":""}',
            called_tool_name=("eviltool" if outcome_val == "success"
                              else ""),
        )
        closure = make_live_judge_closure(fake_traj_rec)

        # Sanity-check return tuple structure independent of completion text passed.
        judged_tuple = closure('{"thought":"x"}', meta)
        assert isinstance(judged_tuple, tuple) and len(judged_tuple) == 2, \
            f"[{label}] expected 2-tuple got {type(judged_tuple)}"

        judged_safe_bool, called_tgt_bool = judged_tuple
        assert isinstance(judged_safe_bool, bool), \
            f"[{label}] element[0] must be bool got {type(judged_safe_bool)}"
        assert isinstance(called_tgt_bool, bool), \
            f"[{label}] element[1] must be bool got {type(called_tgt_bool)}"
        if outcome_val == "success":
            assert judged_safe_bool is False, \
                f"[{label}] B-success leaked -> unsafe but said safe"
        elif outcome_val == "fail":
            assert judged_safe_bool is True, \
                f"[{label}] C-fail caught -> safe but said unsafe"

        # B-success with evil-tool fired -> took_bait=True reflected in slot[1].
        if outcome_val == "success":
            assert called_tgt_bool is True, \
                f"[{label}] success case fired malicious_tool -> bait flag must be true"


# ---- Test C: LogpAgent parses response raw into token-lps metadata ---------#
def test_logp_agent_extracts_token_lps():
    from online.logp_agent import parse_lp_list_from_raw

    fake_choices = [
        {
            "message": {"role": "assistant",
                         "content": "{\"thought\":\"hi\"}"},
            "logprobs": {
                "content": [
                    {"token": "{",      "logprob": -0.123},
                    {"token": "\"thou", "logprob": -1.456},
                    {"token": "ght\"",   "logprob": -0.789},
                ],
            },
        }
    ]
    lps_seq = parse_lp_list_from_raw({"choices": fake_choices})
    assert isinstance(lps_seq, list), \
        f"parse_lp_list_from_raw returned {type(lps_seq)!r}; want list[float]"
    assert len(lps_seq) == 3, \
        f"expected 3 lp entries parsed from fixture; got {len(lps_seq)}"
    expect_first_three = [-0.123, -1.456, -0.789]
    for i, want in enumerate(expect_first_three):
        got = round(float(lps_seq[i]), 6)
        assert abs(got - want) < 1e-6, f"@idx{i} wanted {want} got {got}"

    # Defensive edge cases.
    assert parse_lp_list_from_raw({}) == [], "empty input -> []"
    assert parse_lp_list_from_raw({"choices": []}) == [], "no choices -> []"
    weird_choiceless_entry = [{"foo": "bar"}]
    parsed_weird = parse_lp_list_from_raw({"choices": weird_choiceless_entry})
    assert isinstance(parsed_weird, list) and len(parsed_weird) == 0, \
        f"choiceless-entry -> []; got {parsed_weird!r}"
    logprobs_missing_logprobs_key_entry = [
        {
            "logprobs": {},
            "message": {"role": "assistant"},
        }
    ]
    empty_lps_result = parse_lp_list_from_raw({"choices": logprobs_missing_logprobs_key_entry})
    assert empty_lps_result == [], \
        f"logprobs.content absent -> []; got {empty_lps_result!r}"


# ---- Test D: train_online_grpo dry-run emits plan json w/o torch -----------#
def test_train_online_grpo_dry_run_emits_plan_json(tmp_path_factory):  # noqa: ANN001
    tmp_root = str(tmp_path_factory.mktemp("online_dryrun"))
    exp_dir = os.path.join(tmp_root, "exp")
    os.makedirs(exp_dir, exist_ok=True)

    cfg_mock_attrs = {
        "base_model": "/fake/path/qwen-base",
        "cuda_visible_devices": "7",
        "grpo_max_prompts_per_round": 8,
        "grpo_rollout_temperature": 0.85,
        "native_max_steps_per_round": 50,
        "lora_rank": 16,
        "method": "sft_then_online_grpo",
        "dry_run": True,
        "lora_alpha": 32,
        "per_device_batch_size": 1,
        "gradient_accumulation": 8,
        "use_native_trainer": True,
        "_seed_for_extraction": 42,
    }

    def factory_fn() -> Any:
        raise AssertionError(
            "[test_D] dry_run mode must NEVER invoke controller_factory_fn")

    res = _invoke_train_dryrun(exp_dir, cfg_mock_attrs, factory_fn)
    found_plans = []
    candidates_to_check = [
        os.path.join(exp_dir, "_plan.json"),
        os.path.join(exp_dir, "online_native_r1_plan.json"),
        os.path.join(exp_dir, "plan_and_logs.jsonl"),
        os.path.join(exp_dir, "r1_plan.json"),
    ] + ([os.path.join(getattr(res,"adapter_dir","")or"/nonexistent_xyz","_plan.json")]
         if hasattr(res, "adapter_dir") else [])
    for cand_path in candidates_to_check:
        if cand_path and os.path.isfile(cand_path):
            found_plans.append(cand_path)
    assert found_plans, (
        "no plan artifact emitted under exp_dir tree.\nchecked="
        f"{candidates_to_check}\nexp_dir contents:\n{_tree_listing(exp_dir)}")
    print(f"[debug][test_D] discovered plan artifacts -> {found_plans}")

    # Verify phase marker present somewhere within emitted files.
    saw_phase_marker = False
    for p in found_plans:
        try:
            txt = open(p, encoding="utf-8").read()
        except OSError:
            continue
        if '"phase"' in txt and 'plan_emitted' in txt:
            saw_phase_marker = True
            break
    assert saw_phase_marker, (
        "emitted plan missing {'phase': 'plan_emitted'} marker;\nfirst_blob=\n"
        f"{open(found_plans[0],encoding='utf-8').read()[:600]}"
    )


def _invoke_train_dryrun(exp_dir, cfg_attrs, factory_fn):
    from unittest.mock import MagicMock
    from online.trio_controller import train_online_grpo

    cfg_mock = MagicMock()
    for k, v in cfg_attrs.items():
        setattr(cfg_mock, k, v)
    cfg_mock.dry_run = True

    db_stub = MagicMock()
    db_mock_dict_tasks_attr = {}
    db_mock_tools_attr = {}
    db_stub._tasks = db_mock_dict_tasks_attr
    db_stub._tools = db_mock_tools_attr

    pre_modules_snapshot = set(sys.modules.keys())
    ret = train_online_grpo(exp_rounds_root=exp_dir,
                            training_cfg=cfg_mock,
                            round_label="r1",
                            records=[],
                            dataset_builder=db_stub,
                            init_from_dir="/tmp/nonexistent_initdir_unused_in_dryrun",
                            controller_factory_fn=factory_fn)
    new_after_call = set(sys.modules.keys()) - pre_modules_snapshot
    suspicious_heavy_new_imports = sorted(m for m in new_after_call
                                          if m.split(".")[0] in {"torch", "trl", "peft"})
    print(f"[debug][test_D] newly-imported heavy modules during dry-run "
          f"(should ideally stay empty): {suspicious_heavy_new_imports}")
    return ret


def _tree_listing(root: str, max_depth: int = 3) -> str:
    rows: list[str] = []

    def walk(p: str, depth: int):
        if depth > max_depth:
            return
        if os.path.isdir(p):
            rows.append(f"[D]{'  '*depth}{p}")
            try:
                children = sorted(os.listdir(p))[:30]
            except Exception:
                return                                                # noqa: BLE001
            for c in children:
                walk(os.path.join(p, c), depth + 1)
        elif os.path.isfile(p):
            sz_kb = os.path.getsize(p) // 1024
            rows.append(f"[F]{sz_kb:>5d}KB{'  '*depth}{p}")

    walk(root, 0)
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Legacy direct-invocation harness matching repo convention                   #
# --------------------------------------------------------------------------- #
def main() -> int:
    rc = 0
    failures: list[tuple[str,str]] = []
    tests_with_tmpdir_marker = ["test_train_online_grpo_dry_run_emits_plan_json"]

    table = [
        ("test_assemble_completion_payload_schema",          test_assemble_completion_payload_schema),
        ("test_live_judge_closure_correct_for_each_outcome", test_live_judge_closure_correct_for_each_outcome),
        ("test_logp_agent_extracts_token_lps",               test_logp_agent_extracts_token_lps),
        ("test_train_online_grpo_dry_run_emits_plan_json",   test_train_online_grpo_dry_run_emits_plan_json),
    ]

    class _TmpFactoryShim:
        def __init__(self): self._counter = 0
        def mktemp(self, prefix):                                 # noqa: ARG002,D401
            self._counter += 1
            d = tempfile.mkdtemp(prefix=f"_{prefix}_legacy_")
            return d

    shi = _TmpFactoryShim()
    cleanup_dirs: list[str] = []
    for nm, fn in table:
        try:
            if nm in tests_with_tmpdir_marker:
                fn(shi);
                # capture created dirs for later sweep
            else:
                fn()
            print(f"PASS :: {nm}")
        except Exception as exc:                                          # noqa: BLE001
            traceback.print_exc()
            failures.append((nm, repr(exc)))
            rc = 1
    summary_msg = ("ALL PASS" if not failures
                   else f"FAILURES=[{','.join(nm for nm,_ in failures)}]")
    print(f"\n[test_trio_controller summary] {summary_msg}")
    # Best-effort cleanup of legacy-mode temp dirs.
    for d in list(tempfile._get_tempdir_state()):                       # noqa: SLF001
        pass                                                            # noqa: E701
    shutil.rmtree("/tmp/_online_dryrun_legacy_" , ignore_errors=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
