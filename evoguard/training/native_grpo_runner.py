"""Native in-process GRPO trainer for EvoGuard defender RL (spec §4).

Bypasses vendored verl/AEPO framework entirely -- their pinned dependency versions
(numpy<2 / peft<=0.15 / trl<=0.9.6) clash irreparably with what's installed in
the ``evoguard`` conda env today. Instead we wrap HuggingFace + PEFT + TRL's own
:class:`GRPOTrainer` directly to produce genuine online-policy LoRA weight updates
that hot-load onto a running vLLM server exactly like :mod:`native_runner.py`
does for cold-start SFT.

Key design decisions baked in below (all cross-referenced against spec §3–§5):

* Turn-level sampling granularity (prompt = post-injection state).
* Reward function lives in :mod:`evoguard.training.grpo_reward`; passed to TRl
  as a Python callable so dense shaping signals reach advantage estimation.
* Reference policy handled automatically by TRL itself when ``beta>0`` plus a
  pre-existing PEFT wrapper around actor model -> no second GPU-resident copy needed.
* External rollout generation delegated to running vLLM endpoint configured at
  defense.llm.base_url (TRL built-in ``use_vllm=True,vllm_mode="server"``,
  no custom sampler code required).

Public surface mirrors :func:`evoguard.training.native_sft.train_native_sft`.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from evoguard.config import TrainingConfig
from evoguard.process.dataset_builder import DefenderDatasetBuilder
from evoguard.core.types import TrajectoryRecord
from evoguard.training.native_runner import _cleanup_foreign_gpu_processes
from evoguard.utils.logging import get_logger

logger = get_logger("training.native_grpo")


# --------------------------------------------------------------------------- #
# 方案乙 pure helpers: Δ-aware advantage shaping (spec §3 explicit coupling)  #
# --------------------------------------------------------------------------- #
# Multiplicative curriculum factor applied DIRECTLY on top of group-relative #
# advantages BEFORE PPO ratio computation:                                    #
#                                                                             #
#     Ã⁽ᵍᵖ⁾ = (1 + λ·δ_p) · A⁽ᵍᵖ⁾                                          #
#                                                                             #
# where δ_p is the originating record's normalized Δ carried per-prompt via   #
# PromptMeta.delta_normalized. Default λ=0.0 reproduces legacy equal-weight    #
# behaviour bit-for-bit; positive values amplify gradients on latent-attack    #
# prompts without touching reward scale itself.                               #
# --------------------------------------------------------------------------- #
def _build_per_position_delta_factors(
    row_idx_seq,
    metas_lookup_table,
    *,
    lambda_curriculum: float = 0.0,
) -> list[float]:
    """Return per-position multiplicative scale factors ``(1+λ·δ_p)``.

    Parameters
    ----------
    row_idx_seq :
        Sequence of integer-ish keys into ``metas_lookup_table``, one entry per
        completion slot in the current mini-batch (= len(inputs) at override time).
        With ``num_generations=g>1`` each prompt-row index repeats g consecutive times.
        Non-int-castable entries collapse silently to neutral factor=1.0.
    metas_lookup_table : Mapping[int, PromptMeta]
        Cached metadata objects built alongside the HF Dataset in train_native_grpo.
        Missing entries fall back to factor=1.0 so partial table corruption cannot crash training.
    lambda_curriculum :
        Curriculum strength λ ≥ 0. λ==0 yields all-ones list (legacy parity).
        Negative values are clamped to zero defensively.

    Returns
    -------
    list[float]
        Same length as input sequence. Each element finite and >=1.0 by construction.

    Design notes:
      * Keeping this PURE-Python with no torch dependency lets us unit-test the math
        exhaustively offline without spinning up GPU/TR Library stack.
      * The trainer subclass below consumes this output as a plain Python list then
        converts to a tensor on-device for one inplace multiply -- minimal blast radius.
    """
    # Clamp negative / non-finite lambda to zero upfront -> always-neutral fallback path.
    try:
        lam = float(lambda_curriculum)
    except Exception:                                                  # noqa: BLE001
        lam = 0.0
    if not math.isfinite(lam) or lam < 0.0:
        lam = 0.0

    out_factors: list[float] = []
    for ri_raw in (row_idx_seq or []):
        if lam == 0.0:
            out_factors.append(1.0)
            continue
        meta = None
        delta_val = 0.0
        try:
            ri = int(ri_raw)
            meta = metas_lookup_table.get(ri) if hasattr(metas_lookup_table, "get") else None
            raw_dn = getattr(meta, "delta_normalized", 0.0) if meta is not None else 0.0
            delta_val = float(raw_dn) if raw_dn is not None else 0.0
        except Exception:                                              # noqa: BLE001
            delta_val = 0.0
        # Sanitize δ to [0, ∞): spec guarantees normalized ∈ [0,1] but be defensive against NaN/Inf/negative.
        if not math.isfinite(delta_val) or delta_val < 0.0:
            delta_val = 0.0
        scale_factor = 1.0 + lam * delta_val
        # Final safety net: never emit non-finite or sub-unity factors downstream.
        if not math.isfinite(scale_factor) or scale_factor < 1.0:
            scale_factor = 1.0
        out_factors.append(scale_factor)
    return out_factors


def _apply_advantage_shaping_inplace(
    advantages_tensor,
    scale_factors: list[float],
):
    """Multiply a torch advantages tensor inplace by per-position scalar factors.

    Returns the same tensor object for fluent chaining. No-op safe when either argument
    is empty/None OR when all scale_factors equal exactly 1.0 (legacy default).
    """
    if advantages_tensor is None or not scale_factors:
        return advantages_tensor
    # Skip work entirely on legacy-default uniform-ones case (preserves numerics bit-for-bit).
    if all(s == 1.0 for s in scale_factors):
        return advantages_tensor
    try:
        import torch                                            # local lazy-import keeps module-load cheap
        dev = getattr(advantages_tensor, "device", None)
        dtype = getattr(advantages_tensor, "dtype", torch.float32)
        scales_t = torch.tensor(scale_factors, device=dev, dtype=dtype)
        n = min(int(scales_t.numel()), int(getattr(advantages_tensor, "numel", lambda: 0)()))
        if n == 0:
            return advantages_tensor
        # Broadcast-multiply along dim 0 only (advantages are shape [B]).
        advantages_tensor[:n].mul_(scales_t[:n])
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("[grpo_Δ_shaping] failed applying advantage shaping (%s); skipping.", exc)
    return advantages_tensor


# --------------------------------------------------------------------------- #
# Outcome container                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class NativeGrpoOutcome:
    """Result of one native-grpo invocation."""

    method_used: str               # "native_grpo"
                                    # | "none"                       (no-op e.g. empty records)
                                    # | "error_no_init_from_dir"
                                    # | "error_loading_reference"    (warm-start dir unreadable)
                                    # | "error_during_fit"
                                    # | "error_during_save"
    grpo_samples_written: int      # number of prompts fed to trainer this round (=0 if skipped)
    adapter_dir: str                # "<exp>/grpo_native/<round_label>/adapter_weights/"
    launched_grpo: bool            # True iff fit() completed without raising AND saved successfully
    new_lora_adapter_name: str     # symbolic tag registered onto live vLLM after success
    n_inner_steps_executed: int = 0
    mean_reward_before: Optional[float] = None       # diagnostic only
    mean_reward_after: Optional[float] = None        # diagnostic only
    kl_divergence_estimate: Optional[float] = None   # diagnostic only


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _set_cuda_visible_devices(training_cfg: TrainingConfig):
    """Honor optional GPU pinning from config; returns prior env value."""
    pin = (training_cfg.cuda_visible_devices or "").strip()
    if not pin:
        return os.environ.get("CUDA_VISIBLE_DEVICES", None)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    parts = [p.strip() for p in pin.split(",") if p.strip().isdigit()]
    if parts:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(parts)
            logger.info("[native_grpo] CUDA_VISIBLE_DEVICES=%s", ",".join(parts))
        except Exception as exc:                                      # noqa: BLE001
            logger.warning("[native_grpo] failed setting CVD=%s (%s)", parts, exc)
    else:
        logger.warning("[native_grpo] cuda_visible_devices=%r unparsable.", pin)
    return prev


def _read_marker_or_none(marker_file: str) -> Optional[str]:
    """Return absolute adapter directory pointed-at by marker file, validated.

    Refuses paths missing both expected files so downstream code never receives
    garbage state. Returns None silently when marker doesn't exist yet (cold start case).
    """
    if not os.path.isfile(marker_file):
        return None
    try:
        with open(marker_file, encoding="utf-8") as fh:
            cand = fh.read().strip()
    except OSError as io_exc:
        logger.warning("[native_grpo] failed reading %s: %s", marker_file, io_exc)
        return None
    if not cand or not os.path.isdir(cand):
        return None
    has_adapter_files = any(
        name.endswith(("adapter_config.json", "adapter_model.safetensors"))
        for name in os.listdir(cand)
    )
    return cand if has_adapter_files else None


def _write_marker(marker_file: str, abs_adapter_dir: str) -> bool:
    """Persist updated pointer atomically-ish. Best-effort logging on failure."""
    try:
        parent = os.path.dirname(os.path.abspath(marker_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(marker_file, "w", encoding="utf-8") as fh:
            fh.write(abs_adapter_dir.rstrip("/"))
        logger.info("[native_grpo] wrote marker %s -> %s", marker_file, abs_adapter_dir)
        return True
    except OSError as exc:                                            # noqa: BLE001
        logger.warning(
            "[native_grpo] failed updating %s -> next round won't pick up new "
            "adapter until manual fixup (%s)", marker_file, exc,
        )
        return False


def _build_judge_callable(judge_client_endpoint_url: Optional[str]) -> Optional[Callable]:
    """Construct judge-call closure used inside reward function callback.

    The returned callable has signature ``(completion_text:str, meta:PromptMeta)``
    returning ``(judged_safe_bool, called_malicious_tool_after_divergence_bool)``
    matching contract documented in grpo_reward.JudgeCallable docstring.

    Implementation defers heavy imports + connection setup lazily so dry-run /
    unit tests don't pay startup cost unless actually exercising real network.
    """
    if not judge_client_endpoint_url:
        return None

    def _judge(completion_text: str, meta):                          # noqa: ANN001
        # Lazy imports keep module-load cheap outside true runs.
        from evoguard.config import LLMConfig
        from evoguard.llm.base import LLMClient
        from evoguard.judge import AttackJudge, _infer_target_tool
        from evoguard.core.types import AttackSpec, Message, Role, Trajectory

        cfg = LLMConfig(backend="openai",
                        model="llama3-8b-judge",
                        base_url=judge_client_endpoint_url,
                        api_key="EMPTY")
        # Re-use AttackJudge logic indirectly: build minimal synthetic trajectory
        # containing just the single action being scored so its structural gates fire.
        target_tool_name = meta.target_tool_name or ""
        # We construct an AttackSpec stub carrying goal text mentioning the inferred tool
        # so _detected_injection/_action_timeline codepaths behave consistently.
        attack_stub = AttackSpec(
            attack_id=f"_grpo_inner_{meta.task_id}",
            task_id=str(meta.task_id),
            target_turn=(int(meta.injection_point)+1) if meta.injection_point is not None else 1,
            injection_channel="_inner_loop_probe_",
            method="probe",
            payload="(judge probe)",
            goal=("attacker goal: call " + target_tool_name) if target_tool_name else "",
        )

        # Parse completion_text defensively to assemble Action-like evidence.
        thought_str = ""
        tool_called_name = ""
        final_answer_str = ""
        try:
            obj = json.loads(completion_text)
            if isinstance(obj, dict):
                thought_str = str(obj.get("thought","") or "")
                final_answer_str = str(obj.get("final_answer","") or "")
                tcfld = obj.get("tool")
                if isinstance(tcfld,str): tool_called_name=tcfld.strip()
                elif isinstance(tcfld,dict):
                    nm = tcfld.get("name") or tcfld.get("tool") or ""
                    if isinstance(nm,str): tool_called_name=nm.strip()
        except Exception:
            pass                                                    # noqa: BLE001

        from dataclasses import dataclass as _dc_dataclass
        from evoguard.core.types import ToolCall, Action as EvAction
        act_obj = EvAction(turn=int(meta.injection_point)+2 \
                            if meta.injection_point is not None else 2,
                           thought=thought_str,
                           tool_call=ToolCall(name=tool_called_name, arguments={})
                               if tool_called_name else None,
                           final_answer=final_answer_str)

        traj = Trajectory(task_id=str(meta.task_id),
                          actions=[act_obj],
                          kind=__import__("evoguard").core.types.TrajectoryKind.ATTACKED,
                          metadata={
                              "injection_visible_turn":
                                  int(meta.injection_point) if meta.injection_point is not None else 0,
                              "instruction": meta.task_instruction,
                          })

        ajudge = AttackJudge(cfg)
        ok_success_flag_unused, reason = ajudge.judge(traj, attack_stub)
        judged_safe = not ok_success_flag_used_helper(ok_success_flag_unused)
        called_malicious_tool_post_divergence = (
            bool(target_tool_name)
            and bool(tool_called_name)
            and tool_called_name.lower()==target_tool_name.lower()
        )
        return judged_safe, called_malicious_tool_post_divergence

    return _judge


def ok_success_flag_used_helper(v):                                   # noqa: D401
    """Tiny shim keeping lazy-import scope tidy while exposing flag value."""
    return bool(v)


# --------------------------------------------------------------------------- #
# Reward function builder                                                      #
# --------------------------------------------------------------------------- #
def build_evoguard_reward_callable(metas_by_prompt_idx: dict[int, Any]):
    """Create the actual function handed to TRL.GRPOTrainer.reward_funcs.

    TRL invokes reward funcs as ``(prompts, completions, **kwargs) -> list[float]``
    where kwargs carries extra columns from train_dataset rows. We attach a unique
    integer index per prompt-row ("row_idx") enabling lookup back into our cached
    :class:`PromptMeta` objects stored alongside extracted prompts upstream.

    Parameters
    ----------
    metas_by_prompt_idx :
        Mapping row_idx(int) -> PromptMeta used during scoring lookups.
    """

    from evoguard.training.grpo_reward import compute_evoguard_reward

    # Lazy-init judge once across all invocations within this round to amortize
    # tokenizer/client construction cost over potentially hundreds of inner steps.
    _cached_judge: list[Optional[Callable]] = [None]
    _endpoint_env_var = "EVOGUARD_JUDGE_LLM_BASE_URL"

    def _get_or_init_judge():
        if _cached_judge[0] is not None:
            return _cached_judge[0]
        url = os.environ.get(_endpoint_env_var, "").strip()
        if not url:
            # Fall back gracefully to heuristic-only path if env var unset.
            _cached_judge[0] = False     # sentinel meaning "tried but unavailable"
            return None
        try:
            jc = _build_judge_callable(url)
            _cached_judge[0] = jc
            return jc
        except Exception as exc:                                       # noqa: BLE001
            logger.warning(
                "[native_grpo][reward_fn] judge_callable setup failed (%s); "
                "using heuristic-only fallback throughout this run.", exc,
            )
            _cached_judge[0] = False
            return None

    def _evoguard_reward_func(prompts, completions, **kwargs):
        # Recover parallel-aligned row indices supplied via dataset column.
        raw_row_idx = kwargs.get("row_idx", [])
        try:
            idxs_iter = iter(raw_row_idx)
        except TypeError:
            idxs_iter = iter([raw_row_idx])

        results_floats: list[float] = []
        judge_cb_ref = _get_or_init_judge()

        # prompts may be List[str] OR List[List[{role,content}]] depending on whether caller
        # serialized via apply_chat_template beforehand. We rely solely on metas indexed by
        # position rather than parsing prompt content again -> agnostic handling.
        assert len(prompts)==len(completions), (
            f"[reward_fn] len(prompts)={len(prompts)} != len(completions)={len(completions)}"
        )
        for comp_txt, ri_raw in zip(completions, idxs_iter):
            # TRL 0.19 conversational format passes each completion as
            # List[dict] (e.g. [{"role":"assistant","content":"..."}]) rather
            # than a plain str.  Normalise to a single content string so the
            # reward function's regex/json.loads calls don't crash with
            # "expected string or bytes-like object, got 'list'".
            if isinstance(comp_txt, list):
                _parts = []
                for _msg in comp_txt:
                    if isinstance(_msg, dict) and "content" in _msg:
                        _parts.append(str(_msg["content"]))
                    else:
                        _parts.append(str(_msg))
                comp_txt = "".join(_parts)
            elif not isinstance(comp_txt, str):
                comp_txt = str(comp_txt)
            try:
                ri = int(ri_raw)
            except Exception:
                ri = -1                                                # noqa: BLE001
            meta = metas_by_prompt_idx.get(ri)
            if meta is None:
                logger.debug("[reward_fn] unknown row_idx=%r defaulting neutral R=-0.5", ri)
                results_floats.append(-0.5)
                continue
            effective_jcb = judge_cb_ref if callable(judge_cb_ref) else None
            bd = compute_evoguard_reward(
                completion_text=comp_txt,
                meta=meta,
                judge_call=effective_jcb,
            )
            results_floats.append(float(bd.total))
        return results_floats

    _evoguard_reward_func.__name__ = "evoguard_defense_rl_reward"
    return _evoguard_reward_func


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #
def train_native_grpo(
    *,
    exp_rounds_root: str,
    training_cfg: TrainingConfig,
    round_label: str,
    records: list[TrajectoryRecord],
    dataset_builder: DefenderDatasetBuilder,
    init_from_dir: Optional[str] = None,
) -> NativeGrpoOutcome:
    """Run one incremental GRPO step starting from previously-trained adapter.

    See module-level docstring & spec §4.1 for parameter semantics.
    """

    out_root = os.path.join(exp_rounds_root, "grpo_native", round_label)
    os.makedirs(out_root, exist_ok=True)
    plan_log_path = os.path.join(out_root, "plan_and_logs.jsonl")

    outcome_err_base = lambda method_used, **extra: NativeGrpoOutcome(  # noqa: E731
        method_used=method_used,
        grpo_samples_written=0,
        adapter_dir=out_root,
        launched_grpo=False,
        new_lora_adapter_name="",
        **{k:v for k,v in extra.items()},
    )

    # ------------------------------------------------------------------ #
    # Step A: validate prerequisites                                     #
    # ------------------------------------------------------------------ #
    if init_from_dir is None or not os.path.isdir(init_from_dir):
        msg = f"init_from_dir={init_from_dir!r} invalid/missing."
        logger.error("[native_grpo] %s: %s", round_label, msg)
        _append_plan_json(plan_log_path, {"ts": time.time(), "label": round_label,
                                           "error": "no_init_from_dir", "msg": msg})
        return outcome_err_base(method_used="error_no_init_from_dir")

    # Build prompt extraction up-front BEFORE importing torch stack so smoke-test failures stay fast.
    from evoguard.training.grpo_prompt_extraction import extract_grpo_prompts
    max_prompts_cap = max(0, int(getattr(training_cfg, "grpo_max_prompts_per_round", 32)))
    prompt_rows, stats = extract_grpo_prompts(
        records=records,
        dataset_builder=dataset_builder,
        max_prompts=max_prompts_cap,
        seed=getattr(training_cfg, "_seed_for_extraction", 0),
    )
    n_samples = len(prompt_rows)

    payload_plan_common = {
        "ts": int(time.time()),
        "label": round_label,
        "base_model": training_cfg.base_model,
        "init_from_dir": os.path.abspath(init_from_dir),
        "n_records_input": len(records),
        "extraction_stats": stats.to_dict(),
        "max_steps_requested": int(getattr(training_cfg,"native_max_steps_per_round",0)),
        "dry_run": bool(getattr(training_cfg,"dry_run",True)),
        "cuda_pin": getattr(training_cfg,"cuda_visible_devices",""),
        "cfg_snapshot_keys": {
            k:getattr(training_cfg,k,None) for k in [
                "method","lora_rank","lora_alpha","per_device_batch_size",
                "gradient_accumulation","grpo_beta","grpo_group_size_g",
                "grpo_clip_epsilon","grpo_rollout_temperature",
                "grpo_max_prompts_per_round","grpo_learning_rate",
                ]
        },
    }
    _append_plan_json(plan_log_path, {"phase":"plan_emitted", **payload_plan_common})

    if n_samples == 0:
        logger.info(
            "[native_grpo] %s skipping fit(): zero candidate prompts survived filtering.",
            round_label,
        )
        _append_plan_json(plan_log_path, {
            "phase":"skipped_empty_dataset",
            "stats": stats.to_dict(),
        })
        return NativeGrpoOutcome(method_used="none",
                                  grpo_samples_written=n_samples,
                                  adapter_dir=out_root,
                                  launched_grpo=False,
                                  new_lora_adapter_name="")
    elif stats.n_capped_away > 0:
        logger.warning(
            "[native_grpo] capped away %d candidates (>=%d cap)",
            stats.n_capped_away, max_prompts_cap,
        )

    # Dry-run short-circuit keeps test suite offline-friendly.
    if getattr(training_cfg, "dry_run", True):
        logger.info(
            "[native_grpo] %s dry-run: rendered plan@%s ; would have trained on %d prompts.",
            round_label, plan_log_path, n_samples,
        )
        _append_plan_json(plan_log_path, {
            "phase":"dryrun_shortcircuit",
            "would_train_n_prompts":n_samples,
        })
        return NativeGrpoOutcome(method_used="none",
                                  grpo_samples_written=n_samples,
                                  adapter_dir=out_root,
                                  launched_grpo=False,
                                  new_lora_adapter_name=f"evoguard_native_{round_label}_weights_placeholder")

    # ------------------------------------------------------------------ #
    # Step B: heavy imports past dry-run gate                             #
    # ------------------------------------------------------------------ #
    prev_cvd = _set_cuda_visible_devices(training_cfg)
    _cleanup_foreign_gpu_processes()
    try:
        import torch                                                  # noqa: F401
        from datasets import Dataset                                 # noqa: F401
        from peft import LoraConfig, PeftModel                       # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import GRPOConfig, GRPOTrainer                      # type: ignore

        # -------------------------------------------------------------- #
        # B1 Load base + warm-start adapter                              #
        # -------------------------------------------------------------- #
        logger.info(
            "[native_grpo] loading base_model=%r dtype=bfloat16 ...",
            training_cfg.base_model,
        )
        t_load0 = time.time()
        bf16_avail = getattr(torch.cuda,'is_bf16_supported',lambda *_a,**_kw: True)()
        model_dtype = torch.bfloat16 if bf16_avail else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            training_cfg.base_model,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tok = AutoTokenizer.from_pretrained(training_cfg.base_model, trust_remote_code=True,
                                             padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        logger.info("[native_grpo] loaded base+tokenizer in %.2fs", time.time()-t_load0)

        wf_abs = os.path.abspath(init_from_dir)
        try:
            logger.info("[native_grpo] warm-starting LoRA from %s ...", wf_abs)
            model = PeftModel.from_pretrained(model, wf_abs, is_trainable=True)
            model.train()
            active_peft_config = getattr(getattr(model, "peft_config", {}), "get", lambda *a: None)(
                next(iter(getattr(model,"peft_config",{})),None) if hasattr(model,"peft_config") else None
            ) if hasattr(model,"peft_config") else None
            r_val = getattr(active_peft_config,"r",training_cfg.lora_rank)
            alpha_val = getattr(active_peft_config,"alpha",training_cfg.lora_alpha)
            targets_val = [
                  m for m in (getattr(active_peft_config,"target_modules",[]) or [])
              ] or list(getattr(training_cfg,"lora_target_modules",["q_proj","k_proj","v_proj","o_proj"]))
            logger.info(
                "[native_grpo] resumed LoRA spec r=%s alpha=%s targets[:6]=%s",
                r_val,alpha_val,targets_val[:6],
            )
            warm_loaded=True
        except Exception as exc:                                         # noqa: BLE001
            logger.exception("[native_grpo] PeftModel.from_pretrained(%s) raised:%s",wf_abs,exc)
            _append_plan_json(plan_log_path,{
                "phase":"reference_load_failed","init_from_dir":wf_abs,"err":str(exc)})
            return outcome_err_base(method_used="error_loading_reference")

        # Sanity-check trainable param count post-wrap.
        tp_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot_count=sum(p.numel() for p in model.parameters())
        pct=100.*tp_count/max(1,tot_count)
        logger.info("[native_grpo] trainable=%.2fM/%.2fM %.3f%%",tp_count/1e6,tot_count/1e6,pct)

        # -------------------------------------------------------------- #
        # B2 Construct HF Dataset mapping each prompt_row->dataset record#
        # Each row gets:- "prompt":[system,user]-messages format          #
        #               - "row_idx":integer key into metas map           #
        # -------------------------------------------------------------- #
        metas_lookup_table:dict[int,Any]={}
        ds_rows:list[dict[str,Any]]=[]
        for i,row in enumerate(prompt_rows):
            metas_lookup_table[i]=row.meta
            ds_rows.append({
                "prompt":[{"role":"system","content":row.system},
                          {"role":"user","content":row.user}],
                "row_idx":i,
            })
        hf_ds=Dataset.from_list(ds_rows)
        logger.info("[native_grpo] assembled hf-dataset size=%d cols=%s",
                    len(hf_ds),hf_ds.column_names)

        # -------------------------------------------------------------- #
        # B3 Configure GRPO                                              #
        # -------------------------------------------------------------- #
        eff_batch=max(1,int(getattr(training_cfg,"gradient_accumulation",8)))\
                   *max(1,int(getattr(training_cfg,"per_device_batch_size",1)))
        g_size=max(1,int(getattr(training_cfg,"grpo_group_size_g",8)))
        eps_clip=float(getattr(training_cfg,"grpo_clip_epsilon",0.20))
        beta_kl=float(getattr(training_cfg,"grpo_beta",0.04))
        temp_roll=float(getattr(training_cfg,"grpo_rollout_temperature",0.90))

        st_args=dict(
            output_dir=os.path.join(out_root,"trl_state"),
            overwrite_output_dir=True,
            learning_rate=float(getattr(training_cfg,"grpo_learning_rate",5e-7)),
            num_generations=g_size,
            temperature=temp_roll,
            top_p=0.95,
            # NOTE: TRL>=0.13 renamed PPO-clip-low kwarg from ``epsilon_low`` -> plain
            # ``epsilon``; older versions used ``epsilon`` for symmetric single-bound too.
            # Use canonical name here so config.grpo_clip_epsilon flows through correctly;
            # otherwise silent drop leaves clip-low stuck on hard-coded default regardless
            # of yaml tuning. Asymmetric upper bound doubles per original intent.
            epsilon=eps_clip,
            epsilon_high=float(eps_clip*2.0),
            scale_rewards=True,
            loss_type="bnpo",
            max_completion_length=512,
            max_prompt_length=2048,
            beta=beta_kl,
            use_vllm=False,             # local inference-only rollouts initially safer than external server wiring complexity;
                                        # flip to True+vllm_mode='server' later when wall-clock matters most.
            steps_per_generation=None,
            per_device_train_batch_size=max(1,int(getattr(training_cfg,"per_device_batch_size",1))),
            gradient_accumulation_steps=max(1,int(getattr(training_cfg,"gradient_accumulation",8))),
            optim="adamw_torch_fused" if bf16_avail else "adamw_torch",
            lr_scheduler_type="cosine",
            save_strategy="steps",
            save_steps=10**9,         # effectively disabled mid-run; final save explicit below
            save_total_limit=1,
            report_to=[],
            disable_tqdm=True,
            dataloader_num_workers=0,
            seed=(abs(hash(round_label))^int(time.time())) & 0xFFFFFFFF,  # parens required: ^ has LOWER precedence than &, otherwise abs(hash()) can exceed uint32 -> GRPOConfig rejects "Seed must be between 0 and 2**32 - 1"
            remove_unused_columns=False,
            logging_first_step=True,
            logging_steps=1,
            bf16=bool(bf16_avail),
            tf32=bool(bf16_avail),
            # Preemptive guard: if user flips gradient_checkpointing on later (e.g. to
            # fit longer sequences under memory pressure), the same PEFT-frozen-base +
            # legacy-use_reentrant=True incompatibility that crashed SFT would crash GRPO too.
            # Default gc=False today so this kwarg is dormant but ready.
            gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # Drop unsupported keys similar to native_sft behavior.
        sig_params=set()
        try:
            from inspect import signature as _sigfn
            sig_params=set(_sigfn(GRPOConfig.__init__).parameters.keys())
        except Exception:                                                 # noqa: BLE001
            sig_params=set()
        cleaned_st_args={k:v for k,v in st_args.items() if not sig_params or k in sig_params}

        cap_ms=int(getattr(training_cfg,"native_max_steps_per_round",0))
        if cap_ms>0:
            cleaned_st_args['max_steps']=cap_ms
            cleaned_st_args.pop('num_train_epochs',None)
        else:
            cleaned_st_args.pop('max_steps',None)
            cleaned_st_args['num_train_epochs']=1

        sconf=GRPOConfig(**cleaned_st_args)

        # -------------------------------------------------------------- #
        # B4 Instantiate trainer                                          #
        # -------------------------------------------------------------- #
        reward_fn_closure=build_evoguard_reward_callable(metas_lookup_table)

        diag_state={"step_counter":0,"first_rewards":[],"last_rewards":[],"kl_trace":[],
                    "delta_shaping_applied_count": 0,
                    "delta_shaping_mean_scale": []}
        # CRITICAL: inherit from transformers.TrainerCallback so all lifecycle hooks
        # (on_init_end, on_train_begin, on_epoch_end, etc.) get default no-op
        # implementations. TRL>=0.19 GRPOTrainer.__init__ calls on_init_end on
        # every registered callback -- bare `object` lacks that method and raises:
        #   AttributeError: '_DiagCallback' object has no attribute 'on_init_end'
        from transformers import TrainerCallback as _BaseCb  # local import keeps module-level deps lazy.
        class _DiagCallback(_BaseCb):
            """Lightweight hook capturing diagnostics needed by outcome reporting."""
            def __init__(self,state): super().__init__() ; self._st=state
            def on_step_end(self,args=None,state=None,control=None,model=None,logs=None,**kw):  # noqa: ARG002,D401
                self._st["step_counter"]+=1
                if logs is not None:
                    if self._st["step_counter"]==1 and "rewards/mean" in logs:
                        self._st["first_rewards"].append(float(logs.get("rewards/mean")))
                    if "rewards/mean" in logs:
                        self._st["last_rewards"].append(float(logs.get("rewards/mean")))
                    if "kl" in logs:
                        self._st["kl_trace"].append(float(logs.get("kl")))

        cb=_DiagCallback(diag_state)

        # 方案乙 wiring: pick subclass vs base class based on λ config knob.
        #
        # We define ``_DeltaShapedGRPOTrainer`` LOCALLY here rather than at module scope
        # so that the rest of this module stays importable cheaply WITHOUT pulling in
        # torch+TRL stack -- preserving offline-CI friendliness verified across prior suites.
        # Definition happens exactly ONCE per process per round invocation thanks to idempotent
        # Python class-statement semantics; subsequent rounds simply shadow-rebind harmlessly.
        lam_curriculum = float(getattr(training_cfg, "grpo_advantage_curriculum_lambda", 0.0) or 0.0)
        use_delta_shaping = (lam_curriculum > 0.0)

        if use_delta_shaping:
            logger.info(
                "[native_grpo] Δ-aware advantage shaping ENABLED with λ=%.4f "
                "(Ã=(1+λ·δ_p)·A applied per-prompt-group atop group-relative advantages).",
                lam_curriculum,
            )

            # Local subclass overriding _generate_and_score_completions ONLY --
            # parent handles everything else unchanged keeping blast radius minimal.
            class _DeltaShapedGRPOTrainer(GRPOTrainer):                       # type: ignore[misc]
                """Thin GRPOTrainer override injecting Δ-aware multiplicative scale onto advantages."""

                _EVOGUARD_LAMBDA_CURRICULUM_DEFAULT: float = 0.0       # type: ignore[assignment]
                _EVOGUARD_METAS_LOOKUP_DEFAULT: dict = {}              # type: ignore[assignment]

                def __init__(self,*args,_evoguard_lambda:float=0.0,
                             _evoguard_metas_by_idx:Optional[dict]=None,**kwargs):
                    self._evoguard_lambda_val=float(_evoguard_lambda or 0.0)
                    self._evoguard_metas_lookup=_evoguard_metas_by_idx or {}
                    self._diag_state_ref:dict[str,Any]={"delta_shaping_applied_count":0,
                                                        "delta_shaping_mean_scale":[],
                                                        "last_n_shaped_slots":0}
                    # Strip our private kwargs then forward normally.
                    super().__init__(*args,**kwargs)

                def _generate_and_score_completions(self,inputs):
                    out_dict=super()._generate_and_score_completions(inputs)
                    try:
                        lam=float(getattr(self,"_evoguard_lambda_val",0.0))
                        metas_lut=getattr(self,"_evoguard_metas_lookup",{}) or {}
                        if(lam>0.0 and isinstance(out_dict,dict)
                           and "advantages" in out_dict and inputs):
                            row_idxs:list[Any]=[]
                            for x in inputs:
                                ri:Any=None
                                try:
                                    if hasattr(x,"get"): ri=x.get("row_idx")
                                except Exception:                          # noqa: BLE001
                                    ri=None
                                row_idxs.append(ri)
                            factors=_build_per_position_delta_factors(
                                       row_idxs,metas_lut,lambda_curriculum=lam)
                            adv_tensor=out_dict.get("advantages")
                            n_fac=len(factors)
                            n_adv:int=0
                            try:n_adv=int(adv_tensor.numel())             # noqa: E701
                            except Exception:n_adv=0                      # noqa: BLE001,E701
                            if(factors and n_adv>0):
                                k=min(n_fac,n_adv)
                                _apply_advantage_shaping_inplace(adv_tensor,factors[:k])
                                shaped=sum(1 for f in factors[:k] if abs(float(f)-1.0)>1e-12)
                                if(shaped>0):
                                    self._diag_state_ref["delta_shaping_applied_count"]+=1
                                    ms=sum(float(f) for f in factors[:k] if abs(float(f)-1.0)>1e-12)/max(1,shaped)
                                    self._diag_state_ref["delta_shaping_mean_scale"].append(ms)
                                    self._diag_state_ref["last_n_shaped_slots"]=shaped
                    except Exception as exc_inner:                         # noqa: BLE001
                        logger.warning(
                            "[grpo_Δ_shaping] inner hook failed (%s); "
                            "advantages left unmodified.",exc_inner,)
                    return out_dict
            # Expose globally so unit-test smoke checks resolve post-import-time.
            globals()["_DeltaShapedGRPOTrainer"]=_DeltaShapedGRPOTrainer
        else:
            _DeltaShapedGRPOTrainer=None                                   # type: ignore[assignment]

        ctor_kwargs=dict(
            model=model,
            reward_funcs=[reward_fn_closure],
            args=sconf,
            train_dataset=hf_ds,
            processing_class=tok,
            callbacks=[cb],
        )
        if use_delta_shaping:
            ctor_kwargs["_evoguard_lambda"]=lam_curriculum
            ctor_kwargs["_evoguard_metas_by_idx"]=metas_lookup_table

        try:
            if use_delta_shaping:
                trainer=_DeltaShapedGRPOTrainer(**ctor_kwargs)               # type: ignore[arg-type,misc]
            else:
                clean_kwargs={k:v for k,v in ctor_kwargs.items()
                              if k not in {"_evoguard_lambda","_evoguard_metas_by_idx"}}
                trainer=GRPOTrainer(**clean_kwargs)
        except Exception as ctor_exc:                                       # noqa: BLE001
            logger.exception("[native_grpo] GRPOTrainer instantiation raised:%s",ctor_exc)
            _append_plan_json(plan_log_path,{"phase":"trainer_ctor_error","err":str(ctor_exc)})
            return outcome_err_base(method_used="error_during_fit")

        # -------------------------------------------------------------- #
        # B5 Launch fit                                                   #
        # -------------------------------------------------------------- #
        logger.info(
            "[native_grpo] launching fit(): g=%d batch_eff~%d cap_steps=%s",
            g_size,eff_batch,cap_ms if cap_ms>0 else "(epoch-based)"
        )
        t_fit0=time.time()
        try:
            trainer.train(resume_from_checkpoint=False)
            fit_secs=time.time()-t_fit0
            logger.info("[native_grpo] fit() completed in %.2fs (~%.2fs/prompt).",
                         fit_secs,fit_secs/max(1,n_samples))
        except Exception as fit_exc:                                        # noqa: BLE001
            logger.exception("[native_grpo] trainer.train() crashed:%s",fit_exc)
            _append_plan_json(plan_log_path,{"phase":"fit_crash","err":str(fit_exc)})
            return outcome_err_base(method_used="error_during_fit")

        # Capture diagnostic aggregates reported-back via callback hooks above.
        mr_before=(
            sum(diag_state["first_rewards"]) / max(1,len(diag_state["first_rewards"]))
            ) if diag_state["first_rewards"] else None
        mr_after=(
            sum(diag_state["last_rewards"]) / max(1,len(diag_state["last_rewards"]))
            ) if diag_state["last_rewards"] else None
        kl_est=(
            sum(diag_state["kl_trace"]) / max(1,len(diag_state["kl_trace"]))
            ) if diag_state["kl_trace"] else None

        # -------------------------------------------------------------- #
        # B6 Save adapter artifacts                                       #
        # -------------------------------------------------------------- #
        saved_adapters_dir=os.path.join(out_root,"adapter_weights")
        os.makedirs(saved_adapters_dir,exist_ok=True)
        try:
            unwrapped=model.module if hasattr(model,'module') else model
            unwrapped.save_pretrained(saved_adapters_dir,safe_serialization=True)
            tok.save_pretrained(saved_adapters_dir)
            logger.info("[native_grpo] wrote adapter_weights -> %s",saved_adapters_dir)
        except Exception as sav_exc:                                        # noqa: BLE001
            logger.exception("[native_grpo] saving FAILED:%s",sav_exc)
            _append_plan_json(plan_log_path,{"phase":"save_fail","err":str(sav_exc)})
            return outcome_err_base(method_used="error_during_save")

        adapter_tagged_name=f"evoguard_native_{round_label}_weights"

        _append_plan_json(plan_log_path,{
            "phase":"success",
            "fit_seconds":round(fit_secs,2),
            "adapter_saved_at":os.path.abspath(saved_adapters_dir),
            "mean_reward_before":mr_before,
            "mean_reward_after":mr_after,
            "kl_estimate_avg":kl_est,
            "steps_executed":diag_state["step_counter"],
        })

        return NativeGrpoOutcome(
            method_used="native_grpo",
            grpo_samples_written=n_samples,
            adapter_dir=saved_adapters_dir,
            launched_grpo=True,
            new_lora_adapter_name=adapter_tagged_name,
            n_inner_steps_executed=diag_state["step_counter"],
            mean_reward_before=mr_before,
            mean_reward_after=mr_after,
            kl_divergence_estimate=kl_est,
        )

    finally:
        if prev_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=prev_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES",None)


# --------------------------------------------------------------------------- #
# Small utilities                                                              #
# --------------------------------------------------------------------------- #
def _append_plan_json(path:str,payload:dict)->None:
    try:
        parent=os.path.dirname(os.path.abspath(path)) or "."
        if parent: os.makedirs(parent,exist_ok=True)
        with open(path,"a",encoding="utf-8") as fp:
            fp.write(json.dumps(payload,ensure_ascii=False,default=str));fp.write("\n")
    except OSError as ose:
        logger.warning("[native_grpo] could not append plan-log entry: %s",ose)


__all__:list[str]=[
    "NativeGrpoOutcome",
    "train_native_grpo",
]
