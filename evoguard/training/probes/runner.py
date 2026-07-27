"""Top-level orchestrator for the LoRA layer probe pipeline.

Single public entry point :func:`run_lora_layer_probe` chains:
collect_probe_pairs -> forward-pass both halves -> sensitivity.attn_kl_score
-> ranking.aggregate/rank/select/build -> emit_probe_artifact.

Heavy torch/transformers imports are deferred until inside this function so
dry-run mode and pure-Python unit tests stay fast & offline-friendly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from evoguard.config import ExperimentConfig
from evoguard.training.ranking import emit_probe_artifact, make_artifact_from_run
from evoguard.training.sensitivity import PairScoreInput, SensitivityMethod, select_scorer
from evoguard.utils.logging import get_logger

logger = get_logger("training.probes.runner")

MAX_SEQ_LEN_TOKENS = 1024


@dataclass
class ProbeRunOutcome:
    """Summary of one probe run."""

    artifact_path_written: str = ""
    pairs_collected: int = 0
    layers_scored: int = 0
    selected_blocks_sorted_desc: list[int] = field(default_factory=list)
    recommended_target_modules_count: int = 0
    elapsed_seconds_total: float = 0.0


def _resolve_artifact_path(tcfg) -> str:
    p = str(getattr(tcfg, "lora_probe_artifact_path", "") or "").strip()
    if not p:
        return ""
    return os.path.abspath(os.path.expanduser(p))


def _default_artifact_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.rounds_dir).expanduser().resolve() / cfg.name / "probe_results"


def _dry_run_render_plan(*, cfg: ExperimentConfig, resolved_artifact: str,
                        max_pairs_cap: int) -> dict[str, Any]:
    """Render a plan JSON describing intended actions; do NOT touch torch/network."""
    suites_for_plan = list(cfg.env.suites or []) or [
        s for s in ("banking", "slack", "travel", "workspace")
    ]
    plan_payload: dict[str, Any] = {
        "ts": int(time.time()),
        "mode": "dry_run",
        "base_model": cfg.training.base_model,
        "method_declared_in_cfg": getattr(cfg.training, "lora_probe_method", "attn_kl"),
        "top_k_blocks_requested": int(getattr(cfg.training, "lora_probe_top_k_blocks", 8)),
        "max_pairs_requested_or_overridden": int(max_pairs_cap),
        "suites_targeted": suites_for_plan,
        "cuda_visible_devices_pin": cfg.training.cuda_visible_devices or "",
        "artifact_output_path_resolved": resolved_artifact or "(unset)",
        "env_dataset_name": cfg.env.dataset,
        "env_max_tasks_setting": int(cfg.env.max_tasks),
        "defense_max_turns_setting": int(cfg.defense.max_turns),
        "note": (
            "Dry-run mode renders plan only; no GPU/network/torch invocation happens."
            " Flip lora_probe_enabled=true AND remove --dry-run flag to execute the actual probe."
        ),
    }
    plan_file_obj = _default_artifact_dir(cfg) / "_plan.json"
    plan_file_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = plan_file_obj.with_suffix(plan_file_obj.suffix + ".tmp")
    with open(tmp_p, encoding="utf-8", mode="w") as fh:
        json.dump(plan_payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp_p, plan_file_obj)
    logger.info("[probe-runner][dry-run] wrote execution plan JSON to %s",
                plan_file_obj)
    print(f"[DRY-RUN] Probe execution plan rendered at {plan_file_obj}")
    for k_label, v_val in plan_payload.items():
        if k_label == "note":
            continue
        print(f"   - {k_label}: {v_val}")
    print(f"NOTE: {plan_payload['note']}")
    return {**plan_payload, "_plan_json_abs_path": str(plan_file_obj)}


def _forward_with_attentions(*, base_model_inst, input_ids_any):
    """Single causal-LM forward pass requesting all-layer attentions.

    Returns ``(attentions_tuple_or_None, num_layers_detected_int_or_0)``. Always
    converts ``input_ids_any`` into an explicit long-tensor placed on whatever
    device ``base_model_inst`` lives on BEFORE calling forward, so we never depend
    on HF's implicit type/device coercion (which silently leaves CPU-bound models
    running on CPU even when CUDA_VISIBLE_DEVICES points at a free GPU).
    """
    import torch as _t

    ids_tensor_typed = None

    # First normalise raw python list / numpy / tensor inputs into a 2D long-tensor [1, seq_len].
    try:
        if hasattr(input_ids_any, "shape"):
            t_view = _t.as_tensor(input_ids_any, dtype=_t.long)
        else:
            t_view = _t.as_tensor(list(input_ids_any), dtype=_t.long)
        if t_view.dim() == 1:
            ids_tensor_typed = t_view.unsqueeze(0)
        elif t_view.dim() == 2:
            ids_tensor_typed = t_view
        else:
            ids_tensor_typed = t_view.reshape(1, -1)
    except Exception as exc_norm_ids:                                     # noqa: BLE001
        raise RuntimeError(
            f"failed to coerce input_ids into a 2D long-tensor: {exc_norm_ids}"
        ) from exc_norm_ids

    # Move onto whichever device holds the base_model parameters.
    try:
        target_dev_for_inputs = next(base_model_inst.parameters()).device \
                                if hasattr(base_model_inst, "parameters") else _t.device("cpu")
    except StopIteration:
        target_dev_for_inputs = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    if str(target_dev_for_inputs) != "cpu":
        ids_tensor_typed = ids_tensor_typed.to(target_dev_for_inputs)

    out_logits_attns = base_model_inst(
        input_ids=ids_tensor_typed,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )

    atts_returned_by_model_call = None
    n_layers_seen_this_pair_forward = 0
    cand_atts_attr_access = getattr(out_logits_attns, "attentions", None)
    if isinstance(cand_atts_attr_access, tuple) and len(cand_atts_attr_access) > 0:
        atts_returned_by_model_call = cand_atts_attr_access
        n_layers_seen_this_pair_forward = len(cand_atts_attr_access)
    elif isinstance(cand_atts_attr_access, list):
        atts_returned_by_model_call = tuple(cand_atts_attr_access)
        n_layers_seen_this_pair_forward = len(atts_returned_by_model_call)

    return atts_returned_by_model_call, n_layers_seen_this_pair_forward


def run_lora_layer_probe(
    config: ExperimentConfig | str,
    *,
    dry_run_override_flag_passed_via_cli: bool = False,
    override_gpu_id: Optional[int] = None,
    override_max_pairs: Optional[int] = None,
    stratified_per_domain_hint_unused: Optional[int] = None,
) -> ProbeRunOutcome:
    """Execute the full pre-training LoRA layer selection probe.

    Parameters mirror what the bash launcher passes through:

      * ``config``
          Either an already-loaded :class:`ExperimentConfig` instance OR a path
          string of a YAML file loadable via :meth:`ExperimentConfig.from_file`.
      * ``dry_run_override_flag_passed_via_cli``
          When True skips ALL heavy work after rendering a plan JSON describing
          intended actions.
      * ``override_gpu_id`` / ``override_max_pairs``
          Optional CLI overrides that take precedence over training-cfg defaults.

    Returns
    -------
    ProbeRunOutcome
        Summary useful for logging downstream regardless of real vs dry run.
    """

    t_start_global_ts_now = time.time()

    # ---- Resolve config -------------------------------------------------- #
    if isinstance(config, str):
        from evoguard.config import ExperimentConfig as ExpCfgCls
        cfg_full_active_runtime_use = ExpCfgCls.from_file(config)
    else:
        cfg_full_active_runtime_use = config

    tcfg_ref_holder = cfg_full_active_runtime_use.training

    method_str_request_lower_cased_normally = (
        str(tcfg_ref_holder.lora_probe_method or SensitivityMethod.ATTN_KL.value)
        .strip().lower() or SensitivityMethod.ATTN_KL.value
    )

    cap_pairs_eff_value_local_var = int(max(
        1,
        override_max_pairs or int(getattr(tcfg_ref_holder, "lora_probe_max_pairs", 80)),
    ))
    top_k_req_val_local_var = int(max(
        1, getattr(tcfg_ref_holder, "lora_probe_top_k_blocks", 8)))
    raw_artifact_field_string_from_cfg_yml_directly_read = _resolve_artifact_path(tcfg_ref_holder)

    # Honour optional CLI gpu pin override here too even though shell wrapper exports CUDA_VISIBLE_DEVICES;
    # belt-and-suspenders ensures subprocesses spawned indirectly also see correct value.
    cvd_prior_env_before_we_set_here = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if override_gpu_id is not None and str(int(override_gpu_id)).isdigit():
        new_cvd_str_to_apply_now_immediately = f"{int(override_gpu_id)}"
        os.environ["CUDA_VISIBLE_DEVICES"] = new_cvd_str_to_apply_now_immediately
        logger.info("[probe-runner] applied override_gpu_id=%s -> CUDA_VISIBLE_DEVICES=%s",
                    override_gpu_id, new_cvd_str_to_apply_now_immediately)
    elif not cvd_prior_env_before_we_set_here.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        logger.info("[probe-runner] using internal-default CUDA_VISIBLE_DEVICES=3 since none set elsewhere.")

    # ---- Dry-run short-circuit ------------------------------------------ #
    is_dry_run_mode_active_flag_check_passed = bool(dry_run_override_flag_passed_via_cli) \
                                                or bool(getattr(tcfg_ref_holder, "dry_run", False))
    if is_dry_run_mode_active_flag_check_passed:
        outcome_dry_summary_retobj = ProbeRunOutcome(elapsed_seconds_total=time.time()-t_start_global_ts_now)
        _dry_run_render_plan(
            cfg=cfg_full_active_runtime_use,
            resolved_artifact=(raw_artifact_field_string_from_cfg_yml_directly_read),
            max_pairs_cap=cap_pairs_eff_value_local_var,
        )
        return outcome_dry_summary_retobj

    # ---- Real-execution branch starts here ------------------------------ #
    if not bool(getattr(tcfg_ref_holder, "lora_probe_enabled", False)):
        msg_probe_disabled_but_not_dryrun_explanation_only = (
            "[probe-runner] WARNING: lora_probe_enabled=False but running non-dry-run anyway;"
            " proceeding because user invoked runner directly without flipping enabled flag."
        )
        logger.warning("%s", msg_probe_disabled_but_not_dryrun_explanation_only)

    scorer_callable_function_object_referenced_indirectly_later_on_below = select_scorer(method_str_request_lower_cased_normally)

    # Step A: collect paired trajectories via Controller reuse.
    print(f"[PROBE] step-A collecting up-to-{cap_pairs_eff_value_local_var} paired trajectories ...")
    from .pair_collector import collect_probe_pairs, tokenize_with_tokenizer

    pair_collection_built_up_iteratively_step_a_result_obj = collect_probe_pairs(
        cfg_full_active_runtime_use,
        max_pairs_override=int(cap_pairs_eff_value_local_var),
        keep_raw_traj=False,
    )

    n_actual_pairs_actually_acquired_after_collection_loop = len(pair_collection_built_up_iteratively_step_a_result_obj.pairs)
    print(f"[PROBE] collected N={n_actual_pairs_actually_acquired_after_collection_loop} usable pairs.")
    if n_actual_pairs_actually_acquired_after_collection_loop == 0:
        logger.error("[probe-runner] zero pairs captured -> aborting before heavy-torch stage.")
        return ProbeRunOutcome(pairs_collected=0, elapsed_seconds_total=time.time()-t_start_global_ts_now)

    # Step B/C/D require transformers+torch+model-load; defer until now.
    print("[PROBE] loading tokenizer + base model on pinned GPU ...")
    try:
        import torch                                                   # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer   # type: ignore
    except Exception as exc_import_heavy_stack_failed_prologue_attempt_one:           # noqa: BLE001
        raise RuntimeError(
            "torch+transformers unavailable; cannot proceed past collection phase."
        ) from exc_import_heavy_stack_failed_prologue_attempt_one

    tok_inst = AutoTokenizer.from_pretrained(tcfg_ref_holder.base_model, trust_remote_code=True)
    if tok_inst.pad_token is None:
        tok_inst.pad_token = tok_inst.eos_token
        tok_inst.pad_token_id = tok_inst.eos_token_id

    base_mdl = AutoModelForCausalLM.from_pretrained(
        tcfg_ref_holder.base_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval()

    # Explicitly move model onto the CUDA device selected via CUDA_VISIBLE_DEVICES.
    # Without this call, from_pretrained() defaults to CPU placement making forward
    # passes with output_attentions=True grind for tens of minutes per pair instead
    # of <1s on GPU (observed during first real-GPU launch 2026-07-27).
    _dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    base_mdl = base_mdl.to(_dev)
    print(f"[PROBE]   base_model placed on device={_dev} "
          f"(cuda_available={torch.cuda.is_available()}, "
          f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0})")

    pairs_list = pair_collection_built_up_iteratively_step_a_result_obj.pairs
    ids_clean_per_pair = tokenize_with_tokenizer(tok_inst,
                            [p.messages_clean for p in pairs_list])
    ids_injected_per_pair = tokenize_with_tokenizer(tok_inst,
                                [p.messages_injected for p in pairs_list])
    inject_positions = [int(p.inject_turn) for p in pairs_list]

    per_pair_scores: list[list[float]] = []
    aligned_weights_runtime_synced: list[float] = []   # Δ-derived per-pair weight, positionally synced with above
    layers_scored_total = 0

    print(f"[PROBE] step-B running paired forward passes for {len(pairs_list)} pairs ...")
    import torch as _torch_for_no_grad  # noqa: F401
    with _torch_for_no_grad.no_grad():
        for pair_idx, pair_obj in enumerate(pairs_list):
            ids_c = list(ids_clean_per_pair[pair_idx])[:MAX_SEQ_LEN_TOKENS]
            ids_i = list(ids_injected_per_pair[pair_idx])[:MAX_SEQ_LEN_TOKENS]
            if len(ids_c) < 4 or len(ids_i) < 4:
                logger.warning("[probe-runner] pair idx=%d too short after truncation; skipping.", pair_idx)
                continue
            try:
                atts_c, n_layers_c = _forward_with_attentions(base_model_inst=base_mdl,
                                                              input_ids_any=ids_c)
                atts_i, n_layers_i = _forward_with_attentions(base_model_inst=base_mdl,
                                                              input_ids_any=ids_i)
            except Exception as exc_fwd_pair:                          # noqa: BLE001
                logger.warning("[probe-runner] forward failed on pair=%d (%s); skipping.",
                               pair_idx, exc_fwd_pair)
                continue
            n_layers_this_pair = min(n_layers_c or 0, n_layers_i or 0) \
                                  or max(n_layers_c or 0, n_layers_i or 0)
            if n_layers_this_pair == 0 or atts_c is None or atts_i is None:
                logger.warning("[probe-runner] no attentions returned for pair=%d; skipping.",
                               pair_idx)
                continue
            t_i_token = min(int(inject_positions[pair_idx]),
                            max(0, min(len(ids_c), len(ids_i)) - 2))
            score_input = PairScoreInput(
                attn_clean=atts_c,
                attn_injected=atts_i,
                inject_token_idx=t_i_token,
            )
            try:
                layer_scores_for_this_pair = scorer_callable_function_object_referenced_indirectly_later_on_below(score_input)
            except Exception as exc_scorer_call_failed_on_current_pair_iter:    # noqa: BLE001
                logger.warning("[probe-runner] scorer raised on pair=%d (%s); using zeros.",
                               pair_idx, exc_scorer_call_failed_on_current_pair_iter)
                layer_scores_for_this_pair = [0.0] * int(n_layers_this_pair)
            per_pair_scores.append(list(layer_scores_for_this_pair))
            # Sync-append the corresponding Δ-weight so the two lists stay aligned even when earlier pairs got skipped.
            dn_val_local = float(getattr(pair_obj, "delta_normalized", 0.0) or 0.0)
            aligned_weights_runtime_synced.append(max(0.0, dn_val_local))
            layers_scored_total = max(layers_scored_total,
                                      len(layer_scores_for_this_pair))
            if (pair_idx + 1) % 5 == 0 or (pair_idx + 1) == len(pairs_list):
                print(f"[PROBE]   scored {pair_idx+1}/{len(pairs_list)} pairs "
                      f"(layers/pair={n_layers_this_pair})")

    # Step D: aggregate -> rank -> select -> build target modules -> emit JSON.
    avg_pos_tokens_val = float(sum(inject_positions) / max(1, len(inject_positions))) \
                          if inject_positions else 0.0

    # Build Δ-derived importance vector. ``aligned_weights_runtime_synced`` was
    # populated positionally alongside per_pair_scores during step-B so lengths
    # always match — even when some pairs were skipped due to short tokens or
    # forward failures (their corresponding slot was simply not appended).
    # Per essence §2.4 latent attacks carry more diagnostic value for layer
    # selection than immediate-trigger ones, so delta_normalized is fed directly as the
    # weight; zero-Δ / missing Δ slots contribute nothing to either numerator or denominator,
    # and if EVERY pair has zero weight we pass None so aggregate_pair_scores falls back to equal-mean.
    total_w = sum(aligned_weights_runtime_synced)
    n_pairs_with_delta = sum(1 for w in aligned_weights_runtime_synced if w > 0.0)

    per_pair_weights: Optional[list[float]] = None
    if (
        aligned_weights_runtime_synced
        and total_w > 0.0
        and n_pairs_with_delta >= 1
        and len(aligned_weights_runtime_synced) == len(per_pair_scores)
    ):
        per_pair_weights = aligned_weights_runtime_synced
        print(f"[PROBE] applying Δ-weighted aggregation "
              f"(n_weighted={n_pairs_with_delta}/{len(aligned_weights_runtime_synced)}, "
              f"sum_w={total_w:.4f}, "
              f"mean_w={total_w/max(1,len(aligned_weights_runtime_synced)):.4f}).")
        logger.info(
            "[probe-runner] using Δ-weighted aggregation: weighted=%d/%d, sum=%.4g.",
            n_pairs_with_delta, len(aligned_weights_runtime_synced), float(total_w),
        )
    elif len(pairs_list) > 0:
        if total_w == 0.0 or n_pairs_with_delta == 0:
            logger.info("[probe-runner] no usable Δ across %d scored pairs; "
                        "using legacy equal-weight mean.", len(aligned_weights_runtime_synced))
            print("[PROBE] no Δ signal available; falling back to equal-weight mean.")
        else:
            logger.warning(
                "[probe-runner] alignment mismatch between per_pair_scores (%d) and "
                "weights vector (%d); defaulting to equal-weight.",
                len(per_pair_scores), len(aligned_weights_runtime_synced),
            )

    art_obj = make_artifact_from_run(
        method=str(method_str_request_lower_cased_normally),
        base_model=str(tcfg_ref_holder.base_model),
        per_pair_layer_scores=per_pair_scores,
        inject_token_positions=list(inject_positions),
        inject_position_avg_tokens=avg_pos_tokens_val,
        top_k_requested=int(top_k_req_val_local_var),
        per_pair_weights=per_pair_weights,
    )

    final_artifact_path_str_locally_resolved_now_here_below_at_last = (
        raw_artifact_field_string_from_cfg_yml_directly_read
        or str(_default_artifact_dir(cfg_full_active_runtime_use) / "targets.json")
    )
    emit_probe_artifact(art_obj, path=final_artifact_path_str_locally_resolved_now_here_below_at_last)
    logger.info("[probe-runner] wrote probe artifact -> %s (selected_blocks=%s, n_target_modules=%d)",
                final_artifact_path_str_locally_resolved_now_here_below_at_last,
                art_obj.selected_blocks_sorted_desc,
                len(art_obj.recommended_target_modules))
    print(f"[PROBE] DONE artifact={final_artifact_path_str_locally_resolved_now_here_below_at_last}")
    print(f"         selected_blocks={art_obj.selected_blocks_sorted_desc}")

    elapsed_total_secs = float(time.time() - t_start_global_ts_now)
    outcome_final = ProbeRunOutcome(
        artifact_path_written=final_artifact_path_str_locally_resolved_now_here_below_at_last,
        pairs_collected=len(per_pair_scores),
        layers_scored=int(layers_scored_total),
        selected_blocks_sorted_desc=list(art_obj.selected_blocks_sorted_desc),
        recommended_target_modules_count=len(art_obj.recommended_target_modules),
        elapsed_seconds_total=elapsed_total_secs,
    )
    # Free GPU memory before returning to caller.
    try:
        del base_mdl
        import gc
        gc.collect()
        if _torch_for_no_grad.cuda.is_available():
            _torch_for_no_grad.cuda.empty_cache()
    except Exception as exc_cleanup_gpu_mem_after_run:                # noqa: BLE001
        logger.debug("[probe-runner] gpu cleanup non-fatal: %s", exc_cleanup_gpu_mem_after_run)
    return outcome_final


__all__ = ["ProbeRunOutcome", "run_lora_layer_probe"]