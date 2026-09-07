"""Native in-process LoRA-SFT trainer (``training/native_runner.py``).

Bypasses the vendored LLaMA-Factory / verl frameworks entirely -- their pinned
dependency versions (numpy<2, peft<=0.15) clash with what's installed in the
``evoguard`` conda env today. Instead we wrap HuggingFace + PEFT + TRL directly
to produce genuine per-round LoRA weight updates that hot-load onto a running
vLLM server via its ``/v1/load_lora_adapter`` endpoint.

Only handles cold-start SFT and incremental SFT (warm-started from previous
round's adapter). True online RL via GRPO is deferred to future iterations
because wiring vLLM-as-rollout-server during training requires substantially
more scaffolding than tonight's deadline allows.

Public surface:

* :func:`train_native_sft` -- run one round of in-process SFT given
  :class:`SFTExample` list produced by :class:`DefenderDatasetBuilder.build_sft`.
  Returns :class:`NativeTrainingOutcome` describing what was written.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from evoguard.config import TrainingConfig
from evoguard.process.dataset_builder import SFTExample
from evoguard.utils.logging import get_logger

logger = get_logger("training.native")


@dataclass
class NativeTrainingOutcome:
    """Result of one native-trainer invocation."""

    method_used: str          # "native_sft" | "none"
    sft_examples_written: int = 0
    rl_samples_written: int = 0  # always zero for now; reserved for GRPO later
    adapter_dir: str = ""
    launched_sft: bool = False
    launched_grpo: bool = False   # never true under current implementation
    new_lora_adapter_name: str = ""
    warm_started_from_prev_adapter: bool = False


def _set_cuda_visible_devices(training_cfg: TrainingConfig):
    """Honor optional GPU pinning from config; returns prior env value to restore."""
    pin = (training_cfg.cuda_visible_devices or "").strip()
    if not pin:
        return os.environ.get("CUDA_VISIBLE_DEVICES", None)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Validate format loosely so malformed entries don't crash silently.
    parts = [p.strip() for p in pin.split(",") if p.strip().isdigit()]
    if parts:
        new_val = ",".join(parts)
        logger.info("[native] CUDA_VISIBLE_DEVICES=%s for this training step.", new_val)
        return prev if _apply_env("CUDA_VISIBLE_DEVICES", new_val, prev) else prev
    else:
        logger.warning(
            "[native] cuda_visible_devices=%r unparsable -> leaving env untouched.",
            pin,
        )
        return prev


def _apply_env(key: str, val: str, prev: Optional[str]) -> bool:
    try:
        os.environ[key] = val
        return True
    except Exception as exc:                                            # noqa: BLE001
        logger.warning("[native] failed setting %s=%s (%s)", key, val, exc)
        return False


_TRAINER_MIN_FREE_MIB = int(os.environ.get("EVOGUARD_TRAINER_MIN_FREE_MIB", 72 * 1024))
_TRAINER_FREE_WAIT_S = float(os.environ.get("EVOGUARD_TRAINER_FREE_WAIT_S", 1800))
_TRAINER_FREE_POLL_S = 15.0
_TRAINER_RESERVE_BLOCK_MIB = 1024


def _gpu_free_mib(target_uuid: str) -> Optional[int]:
    """Free VRAM on the GPU with ``target_uuid``, or None if unreadable."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:                                                     # noqa: BLE001
        return None
    for ln in out.stdout.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 2 and parts[0] == target_uuid and parts[1].isdigit():
            return int(parts[1])
    return None


def _wait_for_free_gpu_memory(target_uuid: str, target_idx_str: str) -> None:
    """Claim enough VRAM on the trainer GPU for this process, or log and go on.

    This box runs an oversubscription "filler" workload that reserves 55-80 GiB
    per card purely to keep utilisation up. Its PIDs live in the host namespace,
    so :func:`_cleanup_foreign_gpu_processes` cannot signal them.

    Crucially the filler yields **only under allocation pressure**, not on a
    timer: measured on GPU 4 with 10.5 GiB free, polling nvidia-smi showed no
    change, but grabbing 10 GiB and holding it for 20 s made free memory jump to
    68.5 GiB and a second pass then obtained 60 GiB more. So a passive poll --
    the previous implementation -- deadlocks against it: we wait for memory the
    filler will only release once someone asks.

    We therefore *ask*, in ``_TRAINER_RESERVE_BLOCK_MIB`` chunks, retrying past
    OOM until the reservation reaches ``_TRAINER_MIN_FREE_MIB``. The blocks are
    then dropped WITHOUT ``empty_cache()``, which leaves the bytes parked in this
    process's caching allocator: the filler cannot reclaim them and the trainer's
    own allocations reuse them. Verified to hold across a 30 s idle gap and to
    satisfy a subsequent 60 GiB request.

    Non-fatal by design: if the reservation cannot be filled within
    ``_TRAINER_FREE_WAIT_S`` we log loudly and let the trainer try anyway, so a
    mis-tuned threshold degrades rather than hanging a multi-hour experiment.
    """
    need = _TRAINER_MIN_FREE_MIB
    free0 = _gpu_free_mib(target_uuid)
    if free0 is not None and free0 >= need:
        logger.info("[native-gpu-wait] GPU idx=%s has %d MiB free (>= %d MiB needed); proceeding.",
                    target_idx_str, free0, need)
        return
    try:
        import torch
    except ImportError:                                                   # pragma: no cover
        logger.warning("[native-gpu-wait] torch unavailable; cannot pressure the filler.")
        return
    if not torch.cuda.is_available():
        logger.warning("[native-gpu-wait] no visible CUDA device; skipping reservation.")
        return

    # Later rounds re-enter with the reservation from round 0 still parked in our
    # caching allocator. nvidia-smi counts that as "used", so without this check
    # we would try to claim a second 72 GiB and stall until the deadline.
    already_mib = int(torch.cuda.memory_reserved() // (1024 * 1024))
    if already_mib >= need:
        logger.info(
            "[native-gpu-wait] this process already holds %d MiB reserved on GPU idx=%s "
            "(>= %d MiB needed); reusing it.", already_mib, target_idx_str, need,
        )
        return

    logger.warning(
        "[native-gpu-wait] GPU idx=%s only %s MiB free, need >= %d MiB. Reserving in %d MiB "
        "blocks to force the oversubscription filler to yield (it cannot be killed from this "
        "PID namespace, and it ignores anything short of real allocation pressure).",
        target_idx_str, free0 if free0 is not None else "?", need, _TRAINER_RESERVE_BLOCK_MIB,
    )

    block_bytes = _TRAINER_RESERVE_BLOCK_MIB * 1024 * 1024
    n_target = max(1, (need - already_mib) // _TRAINER_RESERVE_BLOCK_MIB)
    blocks: list = []
    deadline = time.time() + _TRAINER_FREE_WAIT_S
    last_log = 0.0
    stalled = 0
    while len(blocks) < n_target and time.time() < deadline:
        try:
            blocks.append(torch.empty(block_bytes, dtype=torch.uint8, device="cuda"))
            stalled = 0
            continue
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                logger.warning("[native-gpu-wait] unexpected allocation error: %s", str(exc)[:200])
                break
        stalled += 1
        now = time.time()
        if now - last_log >= 60.0:
            last_log = now
            logger.info(
                "[native-gpu-wait] holding %d/%d MiB on GPU idx=%s; filler still resisting "
                "(%d stalled attempts, %.0fs left).",
                len(blocks) * _TRAINER_RESERVE_BLOCK_MIB, need, target_idx_str,
                stalled, deadline - now,
            )
        time.sleep(_TRAINER_FREE_POLL_S)

    reserved_mib = len(blocks) * _TRAINER_RESERVE_BLOCK_MIB + already_mib
    # Release into our OWN allocator cache: no empty_cache() call, deliberately.
    del blocks
    if reserved_mib >= need:
        logger.info(
            "[native-gpu-wait] reserved %d MiB on GPU idx=%s and parked it in this process's "
            "caching allocator; the trainer will reuse it.", reserved_mib, target_idx_str,
        )
        return
    logger.error(
        "[native-gpu-wait] only secured %d MiB of %d MiB on GPU idx=%s within %.0fs. Proceeding "
        "anyway -- expect CUDA OOM. Raise EVOGUARD_TRAINER_FREE_WAIT_S, lower "
        "EVOGUARD_TRAINER_MIN_FREE_MIB, or move the trainer to a quieter GPU.",
        reserved_mib, need, target_idx_str, _TRAINER_FREE_WAIT_S,
    )


def _cleanup_foreign_gpu_processes() -> None:
    """Kill foreign GPU processes occupying the pinned training device.

    Defensive hygiene executed immediately after ``CUDA_VISIBLE_DEVICES`` is
    pinned but BEFORE the heavy model-loading step. On shared multi-GPU boxes,
    long-running vLLM worker processes (notably the ``VLLM::Worker_TP``
    placeholder scripts that reserve cards) frequently bleed memory onto the
    card we want to train on, causing intermittent ``CUDA out of memory``
    crashes during the 3-hour cold-start SFT phase. This pre-arms the trainer
    by reaping any such squatter whose footprint exceeds 5 GiB on the target
    device, while leaving the current process tree (and sub-5 GiB stragglers)
    untouched. Best-effort: failures here are logged but never fatal so a
    transient nvidia-smi hiccup doesn't abort a multi-hour run.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or not cvd.split(",")[0].isdigit():
        logger.info("[native-gpu-cleanup] no single-device pin detected (cvd=%r); skipping.", cvd)
        return
    target_idx_str = cvd.split(",")[0].strip()
    try:
        import subprocess
        # Map logical CUDA index -> physical GPU UUID (nvidia-smi is the only
        # authoritative source; CUDA_VISIBLE_DEVICES post-fork is already
        # logical so we cannot rely on torch.cuda here).
        uuid_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        target_uuid = None
        for ln in uuid_out.stdout.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 2 and parts[0] == target_idx_str:
                target_uuid = parts[1]
                break
        if not target_uuid:
            logger.warning("[native-gpu-cleanup] could not resolve GPU UUID for idx=%s; skipping.", target_idx_str)
            return
        apps_out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        own_pid = os.getpid()
        # Collect descendant PIDs so we never kill our own trainer workers.
        desc_pids: set[int] = {own_pid}
        try:
            ps_out = subprocess.run(
                ["ps", "-eo", "pid,ppid", "--no-headers"],
                capture_output=True, text=True, timeout=10,
            )
            child_map: dict[int, list[int]] = {}
            for ln in ps_out.stdout.splitlines():
                parts = ln.split()
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    child_map.setdefault(int(parts[1]), []).append(int(parts[0]))
            frontier = list(child_map.get(own_pid, []))
            while frontier:
                ppid = frontier.pop()
                if ppid in desc_pids:
                    continue
                desc_pids.add(ppid)
                frontier.extend(child_map.get(ppid, []))
        except Exception as exc:                                              # noqa: BLE001
            logger.warning("[native-gpu-cleanup] descendant scan failed (%s); own_pid only.", exc)

        reaped: list[tuple[int, str]] = []
        for ln in apps_out.stdout.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 3:
                continue
            gpu_uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
            if gpu_uuid != target_uuid or not pid_s.isdigit():
                continue
            pid = int(pid_s)
            if pid in desc_pids:
                continue
            # Memory string like "35710 MiB" -> extract leading integer.
            mem_mib = 0
            digits = ""
            for ch in mem_s:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                mem_mib = int(digits)
            if mem_mib < 5120:  # 5 GiB threshold: ignore small (<5 GiB) stragglers
                continue
            reaped.append((pid, f"{mem_mib} MiB"))
        if not reaped:
            logger.info("[native-gpu-cleanup] target GPU idx=%s uuid=%s clear of foreign >5 GiB processes.", target_idx_str, target_uuid)
            _wait_for_free_gpu_memory(target_uuid, target_idx_str)
            return
        logger.warning(
            "[native-gpu-cleanup] GPU idx=%s uuid=%s occupied by %d foreign process(es) >=5 GiB; reaping: %s",
            target_idx_str, target_uuid, len(reaped), reaped,
        )
        # nvidia-smi reports HOST-namespace PIDs. Inside a container those numbers
        # either don't exist locally or -- worse -- name unrelated local processes,
        # so signalling them is both useless and unsafe. Probe with signal 0 first
        # and only escalate on PIDs we can actually see; report the rest honestly
        # instead of logging "reaped N" for kills that never landed (the failure
        # mode that aborted run 20260817_111147 at r0 SFT).
        signalled: list[int] = []
        invisible: list[tuple[int, str]] = []
        for pid, mem in reaped:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                invisible.append((pid, mem))
                continue
            except PermissionError:
                invisible.append((pid, mem))
                continue
            except Exception as exc:                                         # noqa: BLE001
                logger.warning("[native-gpu-cleanup] probe pid=%s failed: %s", pid, exc)
                invisible.append((pid, mem))
                continue
            try:
                os.kill(pid, 15)  # SIGTERM first
                signalled.append(pid)
            except Exception as exc:                                         # noqa: BLE001
                logger.warning("[native-gpu-cleanup] SIGTERM pid=%s failed: %s", pid, exc)
        if signalled:
            time.sleep(3.0)  # let driver reclaim memory after graceful termination
            for pid in signalled:                                        # SIGKILL survivors
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                except Exception:                                             # noqa: BLE001
                    pass
            time.sleep(2.0)
        logger.info(
            "[native-gpu-cleanup] signalled %d of %d foreign process(es); "
            "%d not visible in this PID namespace (host PIDs, cannot be killed): %s",
            len(signalled), len(reaped), len(invisible), invisible,
        )
        _wait_for_free_gpu_memory(target_uuid, target_idx_str)
    except FileNotFoundError:
        logger.warning("[native-gpu-cleanup] nvidia-smi not on PATH; skipping GPU pre-clean.")
    except subprocess.TimeoutExpired:
        logger.warning("[native-gpu-cleanup] nvidia-smi timed out; skipping GPU pre-clean.")
    except Exception as exc:                                                  # noqa: BLE001
        logger.warning("[native-gpu-cleanup] unexpected failure (non-fatal): %s", exc)


def train_native_sft(
    examples: list[SFTExample],
    *,
    exp_rounds_root: str,
    training_cfg: TrainingConfig,
    round_label: str,
    warm_start_from_dir: Optional[str] = None,
) -> NativeTrainingOutcome:
    """Run an incremental / cold-start LoRA-SFT pass using TRL.SFTTrainer.

    Parameters mirror :mod:`evoguard.training.sft_runner.prepare_and_run_sft`
    enough that callers can swap implementations behind a feature flag without
    touching their dispatch site.

      - ``examples``       -- output of :meth:`DefenderDatasetBuilder.build_sft`;
                              empty => no-op outcome returned immediately.
      - ``exp_rounds_root``-- experiment directory root (e.g. rounds/<name>).
      - ``training_cfg``   -- read for base_model/lora hyper-params/epochs/lr.
      - ``round_label``    -- e.g. "r3"; used both for naming the output dir AND
                              constructing the symbolic lora-adapter name that
                              gets registered on the running vLLM instance.
      - ``warm_start_from_dir``
                            -- path of previously-trained adapter dir from last
                              successful round when doing incremental updates;
                              ``None`` triggers fresh cold-start against the raw
                              base model.
    """

    out_root = os.path.join(exp_rounds_root, "sft_native", round_label)
    os.makedirs(out_root, exist_ok=True)
    adapter_name_tagged = f"evoguard_native_{round_label}"

    n_ex = len(examples)
    if n_ex == 0:
        logger.info(
            "[native] %s: skipping SFT — dataset_builder emitted 0 examples "
            "(likely all B-trajectories lacked clean twins); keeping existing weights.",
            round_label,
        )
        return NativeTrainingOutcome(method_used="none")

   
    plan_path = os.path.join(out_root, "plan.jsonl")
    with open(plan_path, "w", encoding="utf-8") as fplan:
        payload_plan = {
            "ts": int(time.time()),
            "label": round_label,
            "n_examples": n_ex,
            "base_model": training_cfg.base_model,
            "method_declared_in_cfg": training_cfg.method,
            "use_native_trainer_flag": getattr(training_cfg, "use_native_trainer", False),
            "cuda_visible_devices_pin": training_cfg.cuda_visible_devices or "",
            "sft_epochs": float(getattr(training_cfg, "sft_epochs", 1.0)),
            "lr": float(getattr(training_cfg, "sft_learning_rate", 1e-4)),
            "per_device_batch_size": int(getattr(training_cfg, "per_device_batch_size", 1)),
            "gradient_accumulation_steps": int(getattr(training_cfg, "gradient_accumulation", 4)),
            "lora_rank": int(getattr(training_cfg, "lora_rank", 16)),
            "lora_alpha": int(getattr(training_cfg, "lora_alpha", 32)),
            "lora_dropout": float(getattr(training_cfg, "lora_dropout", 0.05)),
            "lora_target_modules": list(getattr(training_cfg, "lora_target_modules",
                                                ["q_proj","k_proj","v_proj","o_proj"])),
            "dry_run_requested": bool(getattr(training_cfg, "dry_run", True)),
            "warm_start_from_dir": warm_start_from_dir or "",
        }
        fplan.write(json.dumps(payload_plan, ensure_ascii=False))
        fplan.write("\n")

    if getattr(training_cfg, "dry_run", True):
        logger.info(
            "[native] %s dry-run mode: rendered plan at %s but NOT launching fit() "
            "(would have trained on %d examples).",
            round_label, plan_path, n_ex,
        )
        return NativeTrainingOutcome(
            method_used="none",
            sft_examples_written=n_ex,
            adapter_dir=out_root,
            launched_sft=False,
            new_lora_adapter_name=adapter_name_tagged,
            warm_started_from_prev_adapter=bool(warm_start_from_dir),
        )

    prev_cvd = _set_cuda_visible_devices(training_cfg)
    _cleanup_foreign_gpu_processes()
    try:
        import torch                                               
        from datasets import Dataset                                   
        from peft import LoraConfig, PeftModel                         # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer    # type: ignore

        from evoguard.training._trl_compat import patch_trl_probes
        # SFT never generates through vLLM, but shim 3 is a PROCESS-WIDE, one-shot
        # decision (it fixes what trl.trainer.grpo_trainer imports at module scope).
        # `sft_then_native_grpo` runs r0 SFT and r>=1 GRPO in the SAME process, so
        # latching False here would silently veto colocate for every later round.
        # Pass the GRPO flag through instead; the aliases are inert for SFTTrainer.
        patch_trl_probes(
            enable_vllm=bool(getattr(training_cfg, "grpo_use_vllm_colocate", False))
        )
        from trl import SFTConfig, SFTTrainer                           # type: ignore

        target_modules_list = [str(m) for m in (
            getattr(training_cfg, "lora_target_modules") or ["q_proj","k_proj","v_proj","o_proj"]
        )]

        # Probe-artifact override takes precedence over static YAML defaults when
        # a JSON produced by evoguard.training.probes.run_lora_layer_probe is
        # pointed at by training_cfg.lora_probe_artifact_path. Only applies on
        # cold-start / fresh-LoRA-init paths; warm-started adapters keep their
        # own (already-chosen) target_modules via the PeftModel branch below.
        artifact_p = getattr(training_cfg, "lora_probe_artifact_path", "") or ""
        if artifact_p and os.path.isfile(artifact_p):
            try:
                probe_data = json.loads(Path(artifact_p).read_text(encoding="utf-8"))
                mods_from_probe = probe_data.get("recommended_target_modules")
                if isinstance(mods_from_probe, list) and mods_from_probe:
                    logger.info(
                        "[native] %s: overriding target_modules from probe artifact %s "
                        "(%d entries)", round_label, artifact_p, len(mods_from_probe),
                    )
                    target_modules_list = [str(m) for m in mods_from_probe]
            except Exception as exc_probe_load:                         # noqa: BLE001
                logger.warning(
                    "[native] %s: failed reading probe artifact %r (%s); keeping defaults.",
                    round_label, artifact_p, exc_probe_load,
                )

        tokenizer_path_or_repo = training_cfg.base_model
        model_load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        device_map_strategy = "auto"

        logger.info(
            "[native] %s loading base_model=%r device_map=%r dtype=bfloat16 ...",
            round_label, training_cfg.base_model, device_map_strategy,
        )
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            training_cfg.base_model, **model_load_kwargs,
        )
        tok = AutoTokenizer.from_pretrained(tokenizer_path_or_repo, trust_remote_code=True)
        if tok.pad_token is None:
            # Many causal LMs lack pad_token by default; reuse eos safely.
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        load_secs = max(0.01, time.time() - t0)
        logger.info("[native] loaded base+tokenizer in %.2fs", load_secs)

        warm_loaded = False
        if warm_start_from_dir:
            wf_abs = os.path.abspath(warm_start_from_dir)
            try:
                peft_state_present = any(
                    name.endswith(("adapter_config.json","adapter_model.safetensors"))
                    for name in os.listdir(wf_abs)
                ) if os.path.isdir(wf_abs) else False
                if peft_state_present:
                    logger.info(
                        "[native] %s warm-starting from PEFT dir %s",
                        round_label, wf_abs,
                    )
                    model = PeftModel.from_pretrained(model, wf_abs, is_trainable=True)
                    warm_loaded = True
                    active_peft_config = getattr(model, "peft_config", {})
                    first_key = next(iter(active_peft_config)) if active_peft_config else ""
                    cfg_obj = active_peft_config.get(first_key) if first_key else None
                    if cfg_obj is not None:
                        # Use same rank/target_modules/etc. as before for stability.
                        target_modules_list = [
                            m for m in (getattr(cfg_obj,"target_modules",[]) or [])
                        ] or target_modules_list
                        logger.info(
                            "[native] reusing previous LoRA spec r=%d alpha=%d targets=%s",
                            getattr(cfg_obj, "r", training_cfg.lora_rank),
                            getattr(cfg_obj, "alpha", training_cfg.lora_alpha),
                            target_modules_list[:6],
                        )
                else:
                    logger.warning(
                        "[native] %s warm_start_from_dir=%s exists but missing adapter files;"
                        " falling back to fresh LoRA init.",
                        round_label, wf_abs,
                    )
            except Exception as exc:                                        # noqa: BLE001
                logger.warning(
                    "[native] PeftModel.from_pretrained(%s) raised %s; starting fresh LoRA instead.",
                    warm_start_from_dir, exc,
                )

        if not warm_loaded:
            # Fresh cold start OR fallback after warm-start failure.
            # If somehow already wrapped (shouldn't normally), unwrap first.
            underlying = model.get_base_model() if hasattr(model, "get_base_model") \
                          and hasattr(model, "peft_config") else model
            lcfg = LoraConfig(
                r=int(getattr(training_cfg, "lora_rank", 16)),
                lora_alpha=int(getattr(training_cfg, "lora_alpha", 32)),
                lora_dropout=float(getattr(training_cfg, "lora_dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules_list,
            )
            # Re-import get_peft_model lazily since it lives alongside LoraConfig.
            from peft import get_peft_model                                # type: ignore
            model = get_peft_model(underlying, lcfg)

        trainable_params_count = sum(p.numel() for p in model.parameters()
                                     if p.requires_grad)
        total_params_count     = sum(p.numel() for p in model.parameters())
        pct_train = (100.0 * trainable_params_count /
                     max(1, total_params_count))
        logger.info(
            "[native] trainable params=%.2fM/%.2fM (%.3f%%)",
            trainable_params_count / 1_000_000,
            total_params_count / 1_000_000,
            pct_train,
        )

        # ---- Build HF datasets.Dataset from SFTExample.to_llamafactory ---- #
        rows = []
        skipped_empty_resp = 0
        skipped_tooshort_prompt = 0
        min_chars_for_valid_sample = 32
        for ex in examples:
            resp_text = ex.response or ""
            prompt_text = ex.prompt or ""
            sys_text = ex.system or ""
            if len(resp_text.strip()) < 5:
                skipped_empty_resp += 1
                continue
            if len(prompt_text.strip()) < min_chars_for_valid_sample // 2:
                skipped_tooshort_prompt += 1
                continue
            msgs = [
                {"role":"system", "content":sys_text} if sys_text else {"role":"system","content":""},
                {"role":"user", "content":prompt_text},
                {"role":"assistant", "content":resp_text},
            ]
            rows.append({"messages":msgs})
            if len(rows[-1]["messages"][0]["content"]) == 0:
                # Drop placeholder-empty system rather than risk template issues downstream.
                rows[-1]["messages"] = rows[-1]["messages"][1:]
        ds_dict = {"messages":[row["messages"] for row in rows]}
        hf_ds = Dataset.from_dict(ds_dict)
        logger.info(
            "[native] built dataset rows=%d (skipped empty-resp=%d too-short-prompt=%d)",
            len(rows), skipped_empty_resp, skipped_tooshort_prompt,
        )
        if len(rows) == 0:
            logger.error(
                "[native] %s aborting fit(): every example was filtered out."
                , round_label,
            )
            return NativeTrainingOutcome(
                method_used="none",
                sft_examples_written=n_ex,
                adapter_dir=out_root,
                launched_sft=False,
                new_lora_adapter_name="",
                warm_started_from_prev_adapter=warm_loaded,
            )

        eff_batch = max(1, int(getattr(training_cfg, "gradient_accumulation", 4))) \
                  * max(1, int(getattr(training_cfg, "per_device_batch_size", 1)))
        epochs_float = max(0.05, float(getattr(training_cfg, "sft_epochs", 1.0)))
        approx_total_steps = max(1, int(len(rows) * epochs_float / eff_batch))

        cap_max_steps = int(getattr(training_cfg, "native_max_steps_per_round", 0))
        final_max_steps_setting = -1   # TRL convention: <0 means use num_train_epochs
        capped_reason_msg = ""
        if cap_max_steps > 0 and cap_max_steps < approx_total_steps:
            final_max_steps_setting = cap_max_steps
            capped_reason_msg = (
                f"capped@{cap_max_steps}<est{approx_total_steps}"
            )
        elif cap_max_steps > 0:
            capped_reason_msg = f"cap={cap_max_steps}>est{approx_total_steps};using_epochs"
        else:
            capped_reason_msg = "no_cap_using_epochs"

        save_total_limit_int = 1   # keep just newest checkpoint each round; saves disk space across multi-round runs.
        # save_strategy="no": the final LoRA adapter is saved explicitly via
        # save_pretrained() to <out_root>/adapter_weights; trainer-driven
        # checkpoints (incl. the end-of-training final save, ~950M each with
        # optimizer state) are never resumed (resume_from_checkpoint=False)
        # and were pure disk waste — 12-round runs accumulated ~11GiB.
        save_strategy_str = "no"
        logging_first_step_bool = True
        bf16_avail = getattr(torch.cuda, 'is_bf16_supported', lambda *_a,**_kw:True)()

        st_args_runtime = dict(
            output_dir=os.path.join(out_root, "_ckpt"),
            overwrite_output_dir=True,
            do_eval=False,
            eval_strategy='no',  # we have no held-out eval split per round; rely on downstream pipeline metrics instead.
            logging_steps=max(1, approx_total_steps//10),
            logging_first_step=logging_first_step_bool,
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
            label_names=None,
            gradient_checkpointing=getattr(torch,'is_grad_enabled',lambda:True)(),  # off-by-default unless enabled explicitly below.
            # CRITICAL: with PEFT-wrapped frozen base + only LoRA params trainable, the legacy
            # `use_reentrant=True` default breaks the backward graph because checkpoint inputs
            # (frozen base activations) lack requires_grad. Non-reentrant path recomputes
            # forward inside autograd graph so it works regardless of input grad state.
            # Without this kwarg fit() crashes on first step:
            #   RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
            # See HF docs: https://huggingface.co/docs/transformers/main/en/training#gradient-checkpointing
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            optim="adamw_torch_fused" if bf16_avail else "adamw_torch",
            lr_scheduler_type="cosine",
            learning_rate=float(getattr(training_cfg, "sft_learning_rate", 1e-4)),
            per_device_train_batch_size=int(getattr(training_cfg, "per_device_batch_size", 1)),
            gradient_accumulation_steps=int(getattr(training_cfg, "gradient_accumulation", 4)),
            seed=(abs(hash(str(round_label))) ^ int(time.time())) & 0xFFFFFFFF,
            bf16=bf16_avail,
            tf32=bf16_avail,
            save_safetensors=True,
            save_only_model=False,
            save_strategy=save_strategy_str,
            save_total_limit=save_total_limit_int,
            completion_only_loss=True,         # mask user/system tokens, supervise assistant response only.
            packing=False,
            # Sequence-length cap prevents pathological long banking tool-call histories from OOMing.
            # Modern TRL accepts ``max_length`` directly; older versions may rename -- our arg-filter
            # below handles both cases gracefully.
            max_length=2048,
        )
        # Drop keys unsupported by current TRL version gracefully:
        cleaned_st_args = {}
        sig_params_known = set()
        try:
            from inspect import signature as _sigfn
            sig_params_known = set(_sigfn(SFTConfig.__init__).parameters.keys())
        except Exception:                                                  # noqa: BLE001
            sig_params_known = set()
        for k,v in st_args_runtime.items():
            if k.startswith('_') :
                continue
            if k == "max_length":
                if "max_seq_length" in sig_params_known:
                    cleaned_st_args["max_seq_length"]=int(v)
                else:
                    cleaned_st_args[k]=int(v)
                continue
            if not sig_params_known or k in sig_params_known:
                cleaned_st_args[k]=v
            else:
                logger.debug("[native] dropping unknown SFTConfig arg %r",k)
        st_args_final = {**cleaned_st_args}
        if final_max_steps_setting >= 0:
            st_args_final['max_steps']=final_max_steps_setting
            st_args_final.pop('num_train_epochs',None)
            # When passing max_steps, also drop epoch-style args that conflict.
        else:
            st_args_final.pop('max_steps',None)
            st_args_final['num_train_epochs']=float(epochs_float)

        sconf = SFTConfig(**st_args_final)

        class _NullCollatorWithTokenizer:
            """Minimal collator delegating chat-template work back to SFTTrainer internals."""
            def __init__(self,_tok,_pad_token_id): self._tok=_tok ; self._pid=int(_pad_token_id)
            def __call__(self,batch_features:list[dict[str,list]]):
                input_ids=[]
                labels=[]
                attn_mask=[]
                for row in batch_features:
                    ids=row.get('input_ids') or []; lbls=row.get('labels')
                    input_ids.append(list(ids)+([self._pid]*(0)))
                    labels.append(list(lbls) if lbls else [])
                    amask=[1]*len(ids)
                    attn_mask.append(amask)
                maxlen=max((len(x)for x in input_ids),default=0)
                padded_ids=[]; padded_lbls=[]; padded_amasks=[]
                pid=self._pid
                ignore=-100
                for i,(ids,lbls,amask)in enumerate(zip(input_ids,labels,attn_mask)):
                    padlen=maxlen-len(ids)
                    padded_ids.append( list(ids)+[pid]*padlen )
                    padded_lbls.append( list(lbls)+[ignore]*padlen )
                    padded_amasks.append( list(amask)+[0]*padlen )
                import torch as _t
                return {
                    'input_ids':_t.tensor(padded_ids,dtype=_t.long),
                    'attention_mask':_t.tensor(padded_amasks,dtype=_t.long),
                    'labels':_t.tensor(padded_lbls,dtype=_t.long),
                }

        trainer=SFTTrainer(
            model=model,
            args=sconf,
            train_dataset=hf_ds,
            processing_class=tok,
        )

        logger.info(
            "[native] launching fit(): est_total_steps≈%d batch_eff=%d cap_info='%s'",
            approx_total_steps,eff_batch,capped_reason_msg,
        )
        t_fit0=time.time()
        try:
            trainer.train(resume_from_checkpoint=False)
            fit_seconds=time.time()-t_fit0
            logger.info("[native] fit() completed in %.2fs (~%.2fs/example).",
                       fit_seconds,fit_seconds/max(1,len(rows)))

            saved_adapters_dir=os.path.join(out_root,"adapter_weights")
            os.makedirs(saved_adapters_dir,exist_ok=True)
            merged_save_dir=os.path.join(saved_adapters_dir,"merged_full")   # unused currently; kept for offline export needs.
            os.makedirs(merged_save_dir,exist_ok=True)
            # Save ONLY adapter weights so vLLM can register them cheaply via /load_lora_adapter.
            try:
                unwrapped=model.module if hasattr(model,'module')else model
                # `peft` exposes .save_pretrained(save_directory)` writing adapter_config.json + adapter_model.safetensors exactly matching what vLLM expects.
                unwrapped.save_pretrained(saved_adapters_dir,safe_serialization=True)
                # Tokenizer copy next door keeps things self-contained.
                tok.save_pretrained(saved_adapters_dir)
                logger.info(
                    "[native] wrote adapter weights →%s",saved_adapters_dir,
                )
                outcome_method_used_str="native_sft"
                outcome_launched_bool=True
                outcome_new_name=adapter_name_tagged+"::"+os.path.basename(saved_adapters_dir.rstrip('/'))
                # Simpler: just expose stable tag itself, registration step uses path anyway.
                outcome_new_name=adapter_name_tagged+"_weights"
            except Exception as sav_exc:                                  # noqa: BLE001
                logger.exception("[native] saving adapter FAILED:%s",sav_exc)
                saved_adapters_dir=""
                outcome_method_used_str="error_during_save"
                outcome_launched_bool=False
                outcome_new_name=""

            return NativeTrainingOutcome(
                method_used=outcome_method_used_str,
                sft_examples_written=len(rows),
                rl_samples_written=0,
                adapter_dir=saved_adapters_dir,
                launched_sft=outcome_launched_bool,
                launched_grpo=False,
                new_lora_adapter_name=outcome_new_name,
                warm_started_from_prev_adapter=warm_loaded,
            )
        except Exception as fit_exc:                                      # noqa: BLE001
            logger.exception("[native] trainer.train() crashed:%s",fit_exc)
            # Re-raise: a crashed trainer must abort the co-evolution loop
            # instead of silently freezing the defender at its previous
            # adapter for all remaining rounds (same rationale as the GRPO
            # runner's fit() crash handler).
            raise
    finally:
        # Restore CVD env var so subsequent non-training calls aren't affected.
        if prev_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=prev_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES",None)


__all__:list[str]=[
    "NativeTrainingOutcome",
    "train_native_sft",
]
