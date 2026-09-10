"""Native in-process GRPO trainer for EvoGuard defender RL
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

#  Δ-aware advantage shaping (spec §3 explicit coupling) 
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
# Trajectory-pooled advantage baseline (轨C.3)                                  #
# --------------------------------------------------------------------------- #
def _traj_pooled_advantage_overrides(
    rewards: list[float],
    traj_ids: list[str],
    *,
    num_generations: int,
    std_eps: float = 1e-4,
    zero_std_tol: float = 1e-8,
) -> dict[int, float]:
    """Advantages for the completions whose OWN prompt group carries no gradient.

    GRPO standardises rewards inside each block of ``num_generations`` siblings
    drawn from one prompt (``trl/trainer/grpo_trainer.py`` ~:2020): ``A = (r -
    mean_group) / (std_group + 1e-4)``. When all G siblings score the same the
    numerator is identically zero, so that prompt contributes **literally no
    gradient**. Measured over the 4400 logged GRPO steps of the plan_abc run:
    ``reward_std == 0`` in 62.8% of steps (2762/4400), reward median 3.2000 =
    exactly the reward ceiling. Ten rounds of training moved training-loop ASR
    0.2255 -> 0.3546.

    This function returns a REPLACEMENT advantage for exactly those positions,
    computed against a baseline pooled over every step sampled from the same
    trajectory (``traj_ids`` equal). Rationale: a saturated step still carries
    information relative to its own trajectory -- "every sibling held the line
    here, and the same policy stalled at the terminal step" is a gradient the
    per-prompt baseline cannot express.

    Deliberately a FALLBACK, not a replacement of TRL's baseline:

      * positions whose prompt group has ``std > zero_std_tol`` are left
        untouched, so wherever the local signal exists it wins unchanged;
      * a trajectory pool that is itself degenerate yields no override, so the
        function never manufactures signal out of a constant;
      * a trajectory represented by a single prompt group can never be helped and
        is skipped -- which is why ``steps_per_trajectory == 1`` makes this a
        guaranteed no-op and reproduces legacy numerics bit-for-bit;
      * empty / falsy ``traj_ids`` entries are treated as ungrouped and skipped.

    Parameters
    ----------
    rewards :
        Flat per-completion rewards, laid out as consecutive blocks of
        ``num_generations`` siblings per prompt (TRL's ``RepeatSampler`` layout).
    traj_ids :
        Same length as ``rewards``; the trajectory id of each completion's prompt.
    num_generations :
        G. Values < 2 disable the function (no group to be degenerate about).

    Returns
    -------
    dict[int, float]
        Position -> new advantage, containing ONLY positions to override. Pure
        Python and torch-free so the arithmetic is unit-testable offline.
    """
    g = int(num_generations or 0)
    n = len(rewards)
    if g < 2 or n == 0 or len(traj_ids) != n:
        return {}

    n_blocks = n // g
    if n_blocks == 0:
        return {}

    # Per-prompt-group mean/std plus the trajectory each block belongs to.
    block_mean: list[float] = []
    block_std: list[float] = []
    block_traj: list[str] = []
    for b in range(n_blocks):
        chunk = [float(x) for x in rewards[b * g:(b + 1) * g]]
        mu = sum(chunk) / float(g)
        var = sum((x - mu) ** 2 for x in chunk) / float(g)
        block_mean.append(mu)
        block_std.append(math.sqrt(max(0.0, var)))
        ids = {str(traj_ids[b * g + j] or "") for j in range(g)}
        block_traj.append(ids.pop() if len(ids) == 1 else "")

    # Pool statistics per trajectory, over all its completions.
    pool_vals: dict[str, list[float]] = {}
    for b in range(n_blocks):
        tid = block_traj[b]
        if not tid:
            continue
        pool_vals.setdefault(tid, []).extend(
            float(x) for x in rewards[b * g:(b + 1) * g]
        )

    pool_stats: dict[str, tuple[float, float, int]] = {}
    for tid, vals in pool_vals.items():
        m = sum(vals) / float(len(vals))
        v = sum((x - m) ** 2 for x in vals) / float(len(vals))
        pool_stats[tid] = (m, math.sqrt(max(0.0, v)), len(vals))

    overrides: dict[int, float] = {}
    for b in range(n_blocks):
        if block_std[b] > zero_std_tol:
            continue                       # local signal exists -> leave alone
        tid = block_traj[b]
        if not tid:
            continue
        stat = pool_stats.get(tid)
        if stat is None:
            continue
        pool_mean, pool_std, pool_n = stat
        if pool_n <= g:                    # trajectory has only this prompt group
            continue
        if pool_std <= zero_std_tol:       # pool is degenerate too -> no signal
            continue
        denom = pool_std + float(std_eps)
        for j in range(g):
            pos = b * g + j
            overrides[pos] = (float(rewards[pos]) - pool_mean) / denom
    return overrides


# --------------------------------------------------------------------------- #
# GDPO: group reward-Decoupled normalization                                    #
# --------------------------------------------------------------------------- #
def _gdpo_advantages(
    components: list[tuple[float, ...]],
    *,
    num_generations: int,
    std_eps: float = 1e-4,
    zero_std_tol: float = 1e-8,
    batch_normalize: bool = True,
) -> list[float]:
    """GDPO advantages: normalise each reward term inside its group, then sum.

    GRPO sums the reward terms *first* and standardises the scalar total inside
    each group of ``num_generations`` siblings. GDPO (NVlabs, ICML 2026,
    arXiv:2601.05242) inverts the order -- it normalises **each reward
    independently** within the group and then sums the per-reward advantages,
    finally rescaling batch-wise so the numeric range does not grow with the
    number of rewards.

    Why this matters for EvoGuard specifically: the three terms live on wildly
    different scales. ``r_safety`` spans 10 points ({+2.00, −1.00, −8.00,
    −0.50}), ``r_progress`` spans 3.7 ({+1.20, −0.15, −2.50}), and ``p_drift``
    spans 0.5 ({0, 0.25, 0.50}). Summing first lets ``r_safety`` set the group
    std almost by itself, so a group that agrees on safety but disagrees on
    progress gets a near-degenerate signal -- and when it agrees on all of the
    total, exactly zero (measured: ``reward_std == 0`` in 62.8-73.3% of steps).
    Normalising per reward keeps each term's disagreement at unit scale, so
    progress can still steer a group that already agrees about the bait.

    Note this is a *normalisation* change, not a new reward term -- the reward
    stays the same three components (``R = r_safety + r_progress − p_drift``).

    Parameters
    ----------
    components :
        One tuple per completion, laid out as consecutive blocks of
        ``num_generations`` siblings per prompt (TRL's ``RepeatSampler``
        layout). Every tuple must have the same arity K >= 1, and the tuples are
        expected to be SIGNED contributions that sum to the scalar reward (so
        ``p_drift`` enters as ``-p_drift``).
    num_generations :
        G. Values < 2 disable the function -- a group of one has no spread to
        normalise against.
    batch_normalize :
        Apply the second, batch-wise standardisation. Since each per-reward
        advantage has exactly zero mean inside its own group, the batch mean is
        already ~0, so in practice this is a rescale by ``1/std_batch``: it is
        what keeps the advantage magnitude independent of K.

    Returns
    -------
    list[float]
        Replacement advantage per completion, length ``(len(components) // G) *
        G``, or ``[]`` when the inputs are unusable (which the caller must treat
        as "leave TRL's advantages alone").
    """
    g = int(num_generations or 0)
    n = len(components)
    if g < 2 or n == 0:
        return []
    n_blocks = n // g
    if n_blocks == 0:
        return []
    k_terms = len(components[0]) if components[0] is not None else 0
    if k_terms < 1:
        return []
    used = n_blocks * g
    for row in components[:used]:
        if row is None or len(row) != k_terms:
            return []

    out = [0.0] * used
    for b in range(n_blocks):
        lo = b * g
        for k in range(k_terms):
            chunk = [float(components[lo + j][k]) for j in range(g)]
            mu = sum(chunk) / float(g)
            var = sum((x - mu) ** 2 for x in chunk) / float(g)
            sd = math.sqrt(max(0.0, var))
            if sd <= zero_std_tol:
                # This term is unanimous in this group: it carries no
                # information here, so it contributes nothing -- and crucially
                # it does not drag the other terms' scale down either.
                continue
            denom = sd + float(std_eps)
            for j in range(g):
                out[lo + j] += (chunk[j] - mu) / denom

    if batch_normalize:
        bm = sum(out) / float(used)
        bvar = sum((x - bm) ** 2 for x in out) / float(used)
        bsd = math.sqrt(max(0.0, bvar))
        if bsd > zero_std_tol:
            bdenom = bsd + float(std_eps)
            out = [(x - bm) / bdenom for x in out]
    return out


def _aligned_trace_view(trace, row_idxs) -> tuple[Optional[list], Optional[list]]:
    """Validated ``(rewards, components)`` from the reward closure's trace slot.

    Returns ``(None, None)`` unless the stashed trace lines up positionally with
    ``row_idxs``. The reward function and the trainer hook must be looking at the
    SAME batch in the SAME order -- TRL shuffles only later, in
    ``_prepare_inputs`` -- and a silent misalignment would pool or normalise
    across unrelated prompts, so verify rather than assume.

    ``components`` is ``None`` on an older two-element trace, which callers must
    treat as "GDPO cannot run on this batch". The trace is INDEXED rather than
    unpacked precisely so widening it again cannot break this.
    """
    if not trace:
        return (None, None)
    entry = trace[0]
    try:
        rewards_seq = list(entry[0])
        traced_idxs = list(entry[1])
    except Exception:                                               # noqa: BLE001
        return (None, None)
    comps: Optional[list] = None
    try:
        if len(entry) > 2 and entry[2] is not None:
            comps = list(entry[2])
    except Exception:                                               # noqa: BLE001
        comps = None
    if len(rewards_seq) != len(row_idxs):
        return (None, None)
    for a, b in zip(traced_idxs, row_idxs):
        try:
            if int(a) != int(b):
                return (None, None)
        except Exception:                                           # noqa: BLE001
            return (None, None)
    if comps is not None and len(comps) != len(rewards_seq):
        comps = None
    return (rewards_seq, comps)


def _apply_advantage_overrides_inplace(advantages_tensor, overrides: dict[int, float]):
    """Write ``overrides`` (position -> value) into a torch advantages tensor.

    No-op safe on an empty mapping so the legacy path never touches the tensor.
    """
    if advantages_tensor is None or not overrides:
        return advantages_tensor
    try:
        total = int(getattr(advantages_tensor, "numel", lambda: 0)())
        for pos, val in overrides.items():
            if 0 <= int(pos) < total:
                advantages_tensor[int(pos)] = float(val)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(
            "[grpo_traj_pool] failed writing pooled advantages (%s); skipping.", exc
        )
    return advantages_tensor


def _install_nonfinite_grad_guard(trainer, diag_state: dict) -> None:
    """Make the optimizer skip any step whose gradients are not all finite.

    Rare but fatal without this. Measured on r2 of run 20260903_002823: at inner
    step 17 (of 400) transformers logged ``'grad_norm': nan`` while the loss was
    finite (0.0002). ``Trainer`` clips before stepping, so the NaN norm scaled
    EVERY gradient to NaN, the optimizer wrote NaN into the LoRA weights, and the
    next GRPO rollout died inside ``generate()`` with

        TensorCompare.cu:109 _assert_async_cuda_kernel: Assertion `probability
        tensor contains either inf, nan or element < 0` failed
        -> torch.AcceleratorError: CUDA error: device-side assert triggered

    which aborts the whole run two hours in. Skipping the update instead costs
    one step's worth of signal and leaves the weights finite. Frequency is low
    enough that this is a guardrail, not a crutch: r1 executed 400 steps with
    zero non-finite norms (max 6.59), r2 hit one in 17.

    Deliberately NOT a reward/advantage change -- every measured invariant in the
    reward, GDPO and Δ-shaping paths is untouched. ``Trainer`` calls
    ``model.zero_grad()`` right after ``optimizer.step()`` (transformers
    ``trainer.py:2752``), so a skipped step must not zero anything itself.

    Wraps ``create_optimizer`` rather than the optimizer, because the optimizer
    does not exist until ``train()`` builds it; wrapping the bound method on the
    instance covers both the plain ``GRPOTrainer`` and the Δ-shaped subclass.

    ``opt.step`` MUST be replaced with a *bound method*, never a plain function.
    ``Trainer.create_optimizer_and_scheduler`` builds the scheduler immediately
    after the optimizer (``trainer.py:1258`` then ``:1264``), and every
    ``LRScheduler.__init__`` runs ``patch_track_step_called``, which reads
    ``func = step_fn.__func__`` and re-binds it via
    ``func.__get__(opt, opt.__class__)`` (``torch/optim/lr_scheduler.py:158-171``).
    A plain function has no ``__func__``, so assigning one makes ``train()`` die
    with ``AttributeError: 'function' object has no attribute '__func__'`` before
    the first step -- measured 2026-09-03, that took down r1 of run
    20260903_111934. ``types.MethodType`` hands torch the ``__func__``/``__self__``
    pair it expects; torch's own wrapper then sits OUTSIDE this guard, so the skip
    still happens and only torch's ``_opt_called`` bookkeeping runs on a skip.
    """
    import types                                    # local: keeps module import cheap
    import torch

    diag_state.setdefault("nonfinite_grad_skips", 0)
    _orig_create = trainer.create_optimizer

    def _create_optimizer_with_guard():
        opt = _orig_create()
        if getattr(opt, "_evoguard_finite_guard", False):
            return opt
        _orig_step = opt.step

        def _guarded_step(self, *args, **kwargs):
            grads = [p.grad for g in self.param_groups for p in g["params"]
                     if p.grad is not None]
            if grads:
                try:
                    norms = torch.stack(torch._foreach_norm(grads))
                    finite = bool(torch.isfinite(norms).all().item())
                except Exception as exc:            # noqa: BLE001
                    logger.debug("[native_grpo] grad finiteness probe failed "
                                 "(%s); stepping anyway", exc)
                    finite = True
                if not finite:
                    diag_state["nonfinite_grad_skips"] += 1
                    logger.warning(
                        "[native_grpo] non-finite gradient at optimizer step; "
                        "SKIPPING the update (skips so far this round: %d). The "
                        "weights stay finite; without this the next rollout dies "
                        "in generate() on a NaN probability tensor.",
                        diag_state["nonfinite_grad_skips"])
                    return None
            return _orig_step(*args, **kwargs)

        opt.step = types.MethodType(_guarded_step, opt)
        opt._evoguard_finite_guard = True
        return opt

    trainer.create_optimizer = _create_optimizer_with_guard


def _install_phase_timers(trainer, diag_state: dict) -> None:
    """Time the rollout+scoring phase so ``fit_seconds`` can be attributed.

    Before this, the only wall-clock number the round reported was
    ``fit_seconds``, which made the measured ~13 s/prompt unattributable between
    generation, the policy/reference logprob forwards, blocking judge HTTP and
    backward. TRL does have its own ``profiling_context``, but on this
    configuration it is a dead end: ``trl/extras/profiling.py:63-68`` forwards
    ``profiling/Time taken: ...`` to wandb/mlflow ONLY, and ``report_to=[]`` here,
    so those durations never reach ``logs`` or ``self._metrics``.

    Wraps ``_generate_and_score_completions`` as a BOUND method on the instance,
    which covers both the plain ``GRPOTrainer`` and ``_DeltaShapedGRPOTrainer``
    (whose own override is what gets wrapped, so GDPO / trajectory pooling / Δ
    shaping are all inside the measured window and none of them is perturbed --
    the wrapper only reads a clock).

    Combined with the reward function's own ``reward_seconds`` this yields a
    three-way split of the round:

        judge+reward        = reward_seconds
        generate+logprobs   = gen_score_seconds - reward_seconds
        backward+optimizer  = fit_seconds - gen_score_seconds
    """
    import types
    import time as _time

    diag_state.setdefault("gen_score_seconds", 0.0)
    diag_state.setdefault("gen_score_calls", 0)
    inner = getattr(trainer, "_generate_and_score_completions", None)
    if inner is None:                       # pragma: no cover - TRL API drift
        logger.warning("[native_grpo] no _generate_and_score_completions to time; "
                       "phase timings will be absent from the plan JSON.")
        return
    if getattr(trainer, "_evoguard_phase_timed", False):
        return
    _orig = inner

    def _timed(self, *args, **kwargs):      # noqa: ARG001 - self unused, bound for symmetry
        t0 = _time.perf_counter()
        try:
            return _orig(*args, **kwargs)
        finally:
            diag_state["gen_score_seconds"] += _time.perf_counter() - t0
            diag_state["gen_score_calls"] += 1

    trainer._generate_and_score_completions = types.MethodType(_timed, trainer)
    trainer._evoguard_phase_timed = True


def _enter_colocate_env() -> dict[str, Optional[str]]:
    """Set the env vars vLLM colocate needs; return the previous values.

    Two independent requirements, both read at engine-construction time:

    * ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` -- with multiprocessing on, the V1
      ``LLMEngine`` (``vllm/v1/engine/llm_engine.py:129-131``) does not expose
      ``.model_executor``, and TRL's ``_move_model_to_vllm`` reaches through
      ``llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner
      .model`` to push the merged LoRA weights in. Without it the per-step weight
      sync -- the only thing that keeps generation ON-POLICY -- raises.
    * ``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT`` --
      TRL passes ``distributed_executor_backend="external_launcher"``, and vLLM's
      ``ExecutorWithExternalLauncher`` initialises the process group with
      ``env://``, so it expects a launcher to have set these. world_size is 1
      here (``native_runner`` is deliberately single-process; DDP would split a
      G-sibling group across ranks and break GDPO/pooling/Δ silently), so setting
      them in-process is enough and no torchrun is involved.

    Mirrors the save/restore contract of ``_set_cuda_visible_devices`` so the
    caller can undo it in a ``finally``.
    """
    keys = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        # Fixed rather than random: world_size==1 means nothing else ever dials
        # this port, and a deterministic value keeps logs comparable.
        "MASTER_PORT": os.environ.get("EVOGUARD_COLOCATE_MASTER_PORT", "29513"),
    }
    prev: dict[str, Optional[str]] = {}
    for k, v in keys.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = v
    logger.info("[native_grpo] colocate env set: %s",
                {k: os.environ[k] for k in keys})
    return prev


def _restore_env(prev: dict[str, Optional[str]]) -> None:
    for k, v in (prev or {}).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _release_colocate_engine(trainer) -> None:
    """Drop the colocated vLLM engine's GPU memory. MUST precede group teardown.

    ``destroy_model_parallel()`` / ``destroy_distributed_environment()`` clear the
    *process groups*; they do not free a single byte of the engine's weights or KV
    cache. Those live behind ``GRPOTrainer.llm`` (``trl/trainer/grpo_trainer.py:657``
    assigns it in the constructor), and ``trainer`` stays referenced by this
    function's frame -- and by the callbacks ``_install_nonfinite_grad_guard`` /
    ``_install_phase_timers`` attach -- for the rest of the round. So the engine
    survives its own teardown, and with ``sft_then_native_grpo`` running every round
    in ONE process it survives into the next round too. Measured on run
    20260907_131715, free memory on the trainer card at the settle immediately
    before each engine was built:

        r5  64.09 GiB   (policy only, ~15 GiB resident)
        r6  29.47 GiB   (+34.6 GiB: r5's engine never went away)
        r7  ---         OOM in Trainer._move_model_to_device, 67.88 MiB free,
                        before the engine was even reached

    i.e. ~20 GiB leaked per colocate round, fatal on the third. The offline
    three-engine gate had already shown it -- free memory before each engine fell
    70.57 -> 52.79 -> 35.09 -> 17.39 GiB -- and it was wrongly dismissed as
    pre-settle measurement noise. A FOURTH engine would have failed there.

    vLLM's own ``parallel_state.cleanup_dist_env_and_memory`` (line 1905) is the
    same sequence minus this step, because its callers are tests that let the
    ``LLM`` local go out of scope first. Nothing here does, so the reference has to
    be dropped by hand. ``gc.collect()`` is required rather than optional: the
    engine is reachable through cycles, so ``del`` alone does not free it -- and
    the collect only works after ``gc.unfreeze()``, see
    ``_collect_and_empty_cache``.
    """
    if trainer is None:
        # Reached from the trainer-ctor error path, where the half-built object is
        # already unreachable. No handle to drop, but the collect still matters --
        # the ctor may have created the engine before raising.
        # The ctor may have captured CUDA graphs before raising, so the pool id
        # must be retired here too or the NEXT build inherits the same fault.
        _clear_vllm_signature_cache()
        _reset_vllm_global_graph_pool()
        _collect_and_empty_cache()
        return
    engine = getattr(trainer, "llm", None)
    if engine is None:
        return

    free_before = _cuda_free_gib()
    try:
        trainer.llm = None
    except Exception as exc:                                                # noqa: BLE001
        logger.warning("[native_grpo] could not clear trainer.llm: %s", exc)
        return
    del engine
    _clear_vllm_signature_cache()
    _reset_vllm_global_graph_pool()
    _collect_and_empty_cache()
    free_after = _cuda_free_gib()
    if free_before is not None and free_after is not None:
        logger.info("[native_grpo] released colocate vLLM engine: free %.2f -> "
                    "%.2f GiB (+%.2f GiB reclaimed)",
                    free_before, free_after, free_after - free_before)
    else:
        logger.info("[native_grpo] released colocate vLLM engine "
                    "(free-memory readback unavailable)")


def _clear_vllm_signature_cache() -> None:
    """Evict the ONE reference that outlives everything else: an lru_cache key.

    ``vllm.utils.func_utils.supports_kw`` is ``@lru_cache``-decorated and takes the
    callable as its first argument (``func_utils.py:47-54``). vLLM's model-protocol
    checks call it with **bound methods of the model instance**:
    ``_check_vllm_model_init`` passes ``model.__init__``
    (``model_executor/models/interfaces_base.py:60-61``) and
    ``_check_vllm_model_forward`` passes ``model.forward`` once per keyword in
    ``("input_ids", "positions")`` (``:76-82``). A bound method holds a strong ref
    to ``__self__``, so those three cache keys pin the entire
    ``Qwen2ForCausalLM`` -- 14.19 GiB of weights -- for the lifetime of the
    process, no matter what happens to the engine that built it.

    ``ia_leak_trace`` stage 2 measured exactly this: after release the whole chain
    down to ``GPUModelRunner`` was freed and ONLY the model survived, referred to
    by ``method[Qwen2ForCausalLM.__init__]`` (1 key tuple) and
    ``method[Qwen2ForCausalLM.forward]`` (2 key tuples), every one of them held by
    a ``functools._lru_cache_wrapper``. 1 + 2 keys is the exact call pattern above.

    This is why ``gc.unfreeze()`` + ``gc.collect()`` reclaimed only 2.40 of 20.09
    GiB: the retention is not a cycle, it is a live strong reference from a
    module-level memo table.

    Clearing it is cheap and safe -- the cache memoises ``inspect.signature``
    lookups, so the cost is a handful of introspection calls the next time an
    engine is built.
    """
    try:
        from vllm.utils.func_utils import supports_kw

        supports_kw.cache_clear()
    except Exception as exc:                                                # noqa: BLE001
        logger.warning("[native_grpo] could not clear vllm supports_kw cache "
                       "(colocate engine weights may stay resident): %s", exc)


def _reset_vllm_global_graph_pool() -> None:
    """Retire the CUDA-graph mempool id so the NEXT engine captures into a fresh one.

    This is what killed r2 of run 20260907_220147 and r3 of 20260908_000734, both
    with::

        RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
        "/pytorch/c10/cuda/CUDACachingAllocator.cpp":2731

    raised from ``torch/cuda/graphs.py:119 capture_begin`` inside
    ``GRPOTrainer.__init__``. It was first misread as out-of-memory (a foreign vLLM
    server was indeed sitting on the trainer card at the time) -- but after that
    server was stopped the second build failed again with **44.45 GiB free**, more
    than twice what the engine asks for. Free memory is not the variable.

    The assert is ``CUDACachingAllocator::beginAllocateToPool`` finding an existing
    entry in ``graph_pools`` whose ``use_count`` is already 0. torch keeps such an
    entry until every block inside it is freed (``releasePool`` only moves it to
    ``graph_pools_freeable``), so one surviving tensor allocated during the first
    capture pins it indefinitely. A second capture that asks for the SAME
    ``mempool_id`` then trips the assert -- reproducible in five seconds with no
    vLLM and no model at all, see ``tests/test_colocate_graph_pool.py``.

    vLLM asks for the same id every time: ``Platform.get_global_graph_pool()``
    (``platforms/interface.py:664``) memoises the handle on ``self.__class__`` and
    nothing in vLLM ever clears it, so every engine in the process shares one
    mempool_id. ``CUDAGraphWrapper.graph_pool`` (``compilation/cuda_graph.py:200``)
    reads it once per wrapper.

    Note what this does and does not buy. It removes the crash, so a second engine
    builds; it does NOT reclaim the first engine's private pool, which is part of
    the ~19 GiB/round that stays resident. Free-at-settle therefore still walks
    down and a long colocate run still has to be restarted eventually.
    """
    import sys

    mod = sys.modules.get("vllm.platforms")
    if mod is None:                     # engine never built in this process
        return
    plat = getattr(mod, "_current_platform", None)
    targets = []
    if plat is not None:
        targets.append(type(plat))
    base = getattr(mod, "Platform", None)
    if base is not None and base not in targets:
        targets.append(base)
    for cls in targets:
        try:
            if getattr(cls, "_global_graph_pool", None) is not None:
                cls._global_graph_pool = None
                logger.info("[native_grpo] retired vLLM global CUDA-graph pool id "
                            "on %s; the next colocate engine will capture into a "
                            "fresh mempool", cls.__name__)
        except Exception as exc:                                            # noqa: BLE001
            logger.warning("[native_grpo] could not reset _global_graph_pool on "
                           "%s (a second colocate engine will hit the "
                           "use_count assert): %s", cls, exc)


def _collect_and_empty_cache() -> None:
    """``gc.unfreeze()`` -> ``gc.collect()`` -> ``torch.cuda.empty_cache()``.

    The ``unfreeze()`` is the load-bearing line, not defensive tidying.
    ``EngineCore.__init__`` ends with ``freeze_gc_heap()``
    (``v1/engine/core.py:221`` -> ``utils/gc_utils.py:96``), which does
    ``gc.collect(0/1/2)`` then ``gc.freeze()``: every gc-tracked object alive at
    that instant -- i.e. the whole engine graph, weights included -- is moved to
    the permanent generation. Frozen objects are never scanned by ``gc.collect()``
    and never reported by ``gc.get_referrers()``. So dropping ``trainer.llm`` and
    collecting cannot break the engine's reference cycles, and the ~17.7 GiB
    stays resident.

    Measured by ``ia_leak_trace``: after release, ``LLM`` and ``LLMEngine`` were
    freed but ``EngineCore`` -> ``ExecutorWithExternalLauncher`` ->
    ``WorkerWrapperBase`` -> ``GPUModelRunner`` -> ``Qwen2ForCausalLM`` were all
    still ALIVE, and ``gc.get_referrers(EngineCore)`` returned NOTHING at all --
    not even this module's own globals dict, which is the fingerprint of a frozen
    heap rather than of a locatable holder. Only 2.30 GiB came back, the part
    allocated after the freeze.

    This is why vLLM's own ``cleanup_dist_env_and_memory`` opens with
    ``gc.unfreeze()`` (``distributed/parallel_state.py:1920``).

    Unfreezing is safe here: the freeze is purely a GC-pause optimisation for
    long-lived serving processes, and this process is about to build a fresh
    engine that will freeze again.
    """
    import gc

    gc.unfreeze()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:                                                # noqa: BLE001
        logger.warning("[native_grpo] empty_cache() after engine release failed: %s",
                       exc)


def _cuda_free_gib() -> Optional[float]:
    """Free GiB on the current device, or ``None`` if CUDA is not usable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[0] / 2**30
    except Exception:                                                       # noqa: BLE001
        return None


def _teardown_colocate_process_group() -> None:
    """Tear down ALL distributed state vLLM's colocate executor left behind.

    Two layers, and clearing only one is worse than clearing neither. A THIRD
    concern -- the engine's ~20 GiB of weights and KV cache -- is NOT handled here;
    see ``_release_colocate_engine``, which must run BEFORE this.

    Layer 1 -- torch's process group. Restoring ``RANK``/``WORLD_SIZE`` is NOT
    enough: ``ExecutorWithExternalLauncher`` calls ``init_process_group("env://")``
    inside the ``LLM`` constructor, and that group outlives both the engine and the
    env vars. It stays initialised for the remainder of the process, which breaks
    the NEXT round rather than this one:

        peft/utils/save_and_load.py:813
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                _maybe_shard_state_dict_for_tp(...)

    That branch does ``from transformers.integrations.tensor_parallel import
    ... EmbeddingParallel ...``, a name transformers 4.57.6 does not export (the
    other three in the same import DO exist), so every warm-start after the first
    colocate round dies with ``ImportError: cannot import name 'EmbeddingParallel'``.
    Measured on run 20260907_012023: r0 SFT and r1 GRPO trained, then r2-r8 each
    raised at ``PeftModel.from_pretrained(grpo_native/r1/adapter_weights)``, wrote
    no weights, and left every val replaying the frozen r1 adapter -- 7 rounds of
    silent no-op whose only symptom was an unchanged ``adapter=`` in the val line.
    A 2-round A/B cannot see this; it needs a SECOND GRPO round.

    Layer 2 -- vLLM's own cached ``GroupCoordinator``s, which live in module
    globals (``parallel_state._WORLD/_TP/_PP/_DP/_EP/...``) and do NOT notice
    torch's group going away. Destroying layer 1 alone is what killed the first
    attempt at this fix (run 20260907_102325): r2 trained, tore down torch's PG,
    and r3's ``LLM(...)`` then died in ``ensure_model_parallel_initialized`` ->
    ``get_backend`` with ``ValueError: Process group <...> is not initialized in
    the world group map`` -- it re-used the stale cached coordinator. So vLLM's
    state must go FIRST; ``destroy_distributed_environment()`` also calls
    ``torch.distributed.destroy_process_group()`` itself, and the raw call below
    is only a fallback for the case where vLLM is not importable or left the
    group up anyway.

    Torn down per round rather than once at exit because
    ``sft_then_native_grpo`` runs every round in the same process. Also silences
    the ``destroy_process_group() was not called before program exit`` warning.
    """
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except Exception as exc:                                            # noqa: BLE001
        logger.warning("[native_grpo] could not import vLLM teardown helpers "
                       "(next colocate round may fail): %s", exc)
    else:
        try:
            destroy_model_parallel()
            destroy_distributed_environment()
            logger.info("[native_grpo] destroyed vLLM colocate distributed state "
                        "(model-parallel groups + world group)")
        except Exception as exc:                                        # noqa: BLE001
            logger.warning("[native_grpo] vLLM distributed teardown failed: %s", exc)

    try:
        import torch.distributed as dist
    except Exception:                                                   # noqa: BLE001
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.destroy_process_group()
        logger.info("[native_grpo] destroyed colocate torch.distributed group "
                    "(keeps peft's TP-shard branch out of the next warm-start)")
    except Exception as exc:                                            # noqa: BLE001
        # Non-fatal: worst case the next warm-start hits the ImportError above,
        # which is loud. Never let cleanup mask the round's real outcome.
        logger.warning("[native_grpo] destroy_process_group() failed: %s", exc)


def _trainer_seed(round_label: str) -> int:
    """Seed handed to ``GRPOConfig``.

    Default mixes the wall clock in, so two runs of the same round explore
    differently. ``EVOGUARD_GRPO_SEED`` pins it instead, which is what makes a
    generation-backend A/B readable: with the clock in the seed, the two arms
    differ in prompt ORDER and sampling stream as well as in backend, and any
    reward gap is unattributable. Pinning does NOT make the arms bit-identical --
    vLLM and HF ``generate`` do not share a sampler implementation -- so the
    comparison is between reward MEANS, and a large systematic gap means the
    per-step weight resync is not landing, not that colocate is "different".

    The parenthesisation is load-bearing: ``^`` binds LOOSER than ``&``, so
    without it ``abs(hash(...))`` survives the mask and ``GRPOConfig`` rejects
    the value with "Seed must be between 0 and 2**32 - 1".
    """
    pinned = os.environ.get("EVOGUARD_GRPO_SEED")
    if pinned:
        try:
            return int(pinned) & 0xFFFFFFFF
        except ValueError:
            logger.warning("[native_grpo] EVOGUARD_GRPO_SEED=%r is not an int; ignoring.",
                           pinned)
    return (abs(hash(round_label)) ^ int(time.time())) & 0xFFFFFFFF


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


#: Env var naming the model served at the judge/progress endpoint. Both judges
#: talk to the same secondary vLLM (:8003 in every shipped config), which serves
#: exactly one model, and an OpenAI-compatible server 404s on a name it does not
#: host -- so this MUST match the served name. The default tracks the model in
#: `configs/*.yaml`; override when serving something else.
_JUDGE_MODEL_ENV = "EVOGUARD_JUDGE_LLM_MODEL"
_JUDGE_MODEL_DEFAULT = "qwen2.5-7b-it"


def _judge_model_name() -> str:
    return (os.environ.get(_JUDGE_MODEL_ENV, "").strip() or _JUDGE_MODEL_DEFAULT)


#: Fan-out used when scoring a generation batch against the judges. 16 matches
#: what `stepwise_eval` sustains against the same vLLM without queueing.
_REWARD_WORKERS_ENV = "EVOGUARD_REWARD_JUDGE_WORKERS"


def _reward_judge_workers() -> int:
    try:
        n = int(os.environ.get(_REWARD_WORKERS_ENV, "16"))
    except ValueError:
        return 16
    return max(1, min(64, n))


def _build_judge_callable(judge_client_endpoint_url: Optional[str]) -> Optional[Callable]:
    """Construct the ``judge_call`` closure consumed by ``grpo_reward``.

    Signature ``(completion_text:str, meta:PromptMeta) -> str`` returning one of
    ``held`` / ``held_but_fired`` / ``baited`` / ``unclear`` (or ``""`` meaning
    "no verdict"). The implementation is
    :func:`evoguard.training.step_safety_judge.build_step_safety_callable`.

    This replaces (2026-09-02) a closure that reused the TRAJECTORY judge
    ``AttackJudge`` on a synthetic one-action trajectory, with every guardrail in
    that judge disabled by construction: the action's turn was
    ``injection_point + 2`` while ``injection_visible_turn`` was
    ``injection_point``, so the temporal gate was a no-op; ``arguments={}``
    discarded the parsed tool arguments, making argument-hijack attacks invisible;
    no ``benign_reference`` was passed, so the clean-twin rule could never fire;
    and with one action and no observations the model was asked to judge a step in
    the absence of the context that makes it judgeable. The evidence it needed was
    already computed at prompt-build time and simply not carried -- it now travels
    on ``PromptMeta`` (``prefix_timeline`` / ``benign_reference_timeline`` /
    ``payload_visible_in_prefix``), so nothing here has to reconstruct a
    ``Trajectory`` at all.
    """
    if not judge_client_endpoint_url:
        return None

    from evoguard.training.step_safety_judge import build_step_safety_callable

    return build_step_safety_callable(judge_client_endpoint_url, _judge_model_name())


# --------------------------------------------------------------------------- #
# Progress judge (S1)                                                          #
# --------------------------------------------------------------------------- #
# Prompt, parser and builder live in evoguard.training.progress_judge (mirrors
# the step_safety_judge extraction) so the eval harness
# (scripts/eval_progress_judge.py) and production can never drift. The shipped
# configuration is the measured winner of the 2026-09-04 matrix; every knob and
# its measurement history are documented on build_progress_callable itself.
#
# Drop-in hazard, measured: the PREVIOUS inline implementation (thin one-word
# prompt, max_tokens=8) reads 0% of verdicts against a thinking-native judge
# (qwen3.5-9b emits a reasoning preamble first), silently putting r_progress
# into permanent degraded fallback. The module's no_think default fixes this.
from evoguard.training.progress_judge import (  # noqa: E402
    parse_progress_verdict as _parse_progress_verdict,  # re-export for tests
)


def _build_progress_callable(progress_endpoint_url: Optional[str]) -> Optional[Callable]:
    """Construct the ``progress_call`` closure consumed by ``grpo_reward``.

    Signature ``(completion_text:str, meta:PromptMeta) -> str`` returning one of
    ``"advance"``/``"neutral"``/``"waste"`` (or ``""`` meaning "no verdict").
    Without it the reward runs in a degraded mode that can never award
    ``R_PROGRESS_ADVANCE`` -- doing the user's job correctly becomes
    indistinguishable from stalling -- so the caller warns loudly when the
    endpoint is unset.
    """
    if not progress_endpoint_url:
        return None

    from evoguard.training.progress_judge import build_progress_callable

    return build_progress_callable(progress_endpoint_url, _judge_model_name())


# --------------------------------------------------------------------------- #
# Reward function builder                                                      #
# --------------------------------------------------------------------------- #
def build_evoguard_reward_callable(
    metas_by_prompt_idx: dict[int, Any],
    reward_trace_sink: Optional[list] = None,
    disable_safety: bool = False,
):
    """Create the actual function handed to TRL.GRPOTrainer.reward_funcs.

    TRL invokes reward funcs as ``(prompts, completions, **kwargs) -> list[float]``
    where kwargs carries extra columns from train_dataset rows. We attach a unique
    integer index per prompt-row ("row_idx") enabling lookup back into our cached
    :class:`PromptMeta` objects stored alongside extracted prompts upstream.

    Parameters
    ----------
    metas_by_prompt_idx :
        Mapping row_idx(int) -> PromptMeta used during scoring lookups.
    reward_trace_sink :
        Optional single-slot list. When provided, element 0 is REPLACED on every
        call with ``(rewards, row_indices, components)`` for the batch just
        scored, where ``components[i]`` is the SIGNED per-term tuple
        ``(r_safety, r_progress, -p_drift)`` summing to ``rewards[i]``. This
        exists because TRL's ``_generate_and_score_completions`` returns only
        ``prompt_ids/prompt_mask/completion_ids/completion_mask/advantages/
        num_items_in_batch`` (``grpo_trainer.py`` ~:2128) -- the raw rewards are
        consumed internally and never surfaced, so the trajectory-pooled baseline
        below has no other way to see them, and GDPO additionally needs the
        pre-sum decomposition that never leaves the reward function at all.
        Stashing rather than recomputing keeps the views of the reward guaranteed
        identical (and costs no extra judge calls). Consumers must INDEX the
        tuple rather than unpack it, so widening it again stays backwards
        compatible.

    The returned function carries a ``safety_tally`` attribute: a dict counting
    ``(safety_source, safety_label)`` pairs over every completion scored in the
    round. It exists because a safety-judge outage is otherwise INVISIBLE. Since
    ``took_bait`` was deleted, the fallback value ``R_SAFETY_UNCLEAR`` is the same
    constant for every attacked row, and GRPO advantages are group-relative, so an
    outage does not bias ``r_safety`` -- it deletes the safety gradient while the
    reward mean, the loss and the gradient norm all keep looking healthy. A round
    whose tally is dominated by ``fallback`` trained progress only.
    The returned function also carries ``reward_seconds``: a ONE-ELEMENT LIST
    accumulating the wall-clock seconds spent inside the reward function over the
    whole round. A list rather than a float because the caller holds the reference
    before the first call and floats are immutable. This is the only way to split
    ``fit_seconds`` into "the generator is slow" vs "the judges are slow": both
    judges are BLOCKING HTTP called from here, so this counter is the
    judge-attributable share of training time.
    """

    import threading

    from evoguard.training.grpo_reward import compute_evoguard_reward

    #: ``"{safety_source}:{safety_label}"`` -> count, over the whole round.
    safety_tally: dict[str, int] = {}
    _tally_lock = threading.Lock()

    #: Accumulated seconds inside ``_evoguard_reward_func``. Mutated from the
    #: main thread only (TRL calls reward funcs serially per generation batch),
    #: so it needs no lock even though scoring fans out internally.
    _reward_seconds: list[float] = [0.0]

    # Lazy-init judge once across all invocations within this round to amortize
    # tokenizer/client construction cost over potentially hundreds of inner steps.
    _cached_judge: list[Optional[Callable]] = [None]
    _endpoint_env_var = "EVOGUARD_JUDGE_LLM_BASE_URL"
    # The progress judge defaults onto the SAME endpoint as the safety judge --
    # every shipped config puts both on the secondary vLLM (:8003) -- but keeps
    # its own env var so it can be split onto a third server if the extra
    # G x N_prompts calls per step ever saturate that one.
    _cached_progress: list[Optional[Callable]] = [None]
    _progress_endpoint_env_var = "EVOGUARD_PROGRESS_LLM_BASE_URL"

    def _get_or_init_judge():
        if _cached_judge[0] is not None:
            return _cached_judge[0]
        url = os.environ.get(_endpoint_env_var, "").strip()
        if not url:
            # No endpoint: r_safety cannot be scored at all this round.
            logger.warning(
                "[native_grpo][reward_fn] %s is unset -> the SAFETY term is DEAD "
                "for this entire round: every attacked row scores the same "
                "UNCLEAR/-0.50 constant, and a per-prompt constant contributes "
                "exactly zero gradient under group-relative advantages, so only "
                "r_progress is trained. Nothing in the loss or reward mean will "
                "show this. Export it (e.g. http://127.0.0.1:8003/v1) to train "
                "against the judge.",
                _endpoint_env_var,
            )
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

    def _get_or_init_progress():
        if _cached_progress[0] is not None:
            return _cached_progress[0]
        url = (os.environ.get(_progress_endpoint_env_var, "").strip()
               or os.environ.get(_endpoint_env_var, "").strip())
        if not url:
            logger.warning(
                "[native_grpo][reward_fn] neither %s nor %s is set -> the "
                "PROGRESS term runs in degraded mode for this entire round: it "
                "can only tell no-action/repeat (waste) from everything else "
                "(neutral) and can NEVER award advance/+1.20, so doing the "
                "user's job correctly scores the same as stalling.",
                _progress_endpoint_env_var, _endpoint_env_var,
            )
            _cached_progress[0] = False
            return None
        try:
            pc = _build_progress_callable(url)
            _cached_progress[0] = pc
            logger.info(
                "[native_grpo][reward_fn] progress judge active at %s (model=%s).",
                url, _judge_model_name(),
            )
            return pc
        except Exception as exc:                                       # noqa: BLE001
            logger.warning(
                "[native_grpo][reward_fn] progress_callable setup failed (%s); "
                "r_progress falls back to the never-advance heuristic for this run.",
                exc,
            )
            _cached_progress[0] = False
            return None

    def _evoguard_reward_func(prompts, completions, **kwargs):
        _t_rf0 = time.perf_counter()
        # Recover parallel-aligned row indices supplied via dataset column.
        raw_row_idx = kwargs.get("row_idx", [])
        try:
            idxs_iter = iter(raw_row_idx)
        except TypeError:
            idxs_iter = iter([raw_row_idx])

        # Ablation arm (`grpo_disable_safety_term`): r_safety is pinned to 0.0
        # inside the reward, so a verdict would be discarded. Skip CONSTRUCTING
        # the judge, not just calling it -- otherwise the round pays the client
        # setup and, when the endpoint env var is unset, logs the "SAFETY term is
        # DEAD" warning that exists to catch accidents, which here would be noise.
        judge_cb_ref = None if disable_safety else _get_or_init_judge()
        progress_cb_ref = _get_or_init_progress()
        effective_jcb = judge_cb_ref if callable(judge_cb_ref) else None
        effective_pcb = progress_cb_ref if callable(progress_cb_ref) else None
        # prompts may be List[str] OR List[List[{role,content}]] depending on whether caller
        # serialized via apply_chat_template beforehand. We rely solely on metas indexed by
        # position rather than parsing prompt content again -> agnostic handling.
        assert len(prompts)==len(completions), (
            f"[reward_fn] len(prompts)={len(prompts)} != len(completions)={len(completions)}"
        )

        pairs: list[tuple[str, Any]] = []
        row_indices: list[int] = []
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
            pairs.append((comp_txt, meta))
            row_indices.append(ri)

        def _score_one(pair):
            """Score one completion -> ``(total, signed_components)``.

            The components are the SIGNED per-term contributions in a fixed
            order ``(r_safety, r_progress, -p_drift)``, so ``sum(components)``
            equals ``total`` by construction. Only GDPO reads them; TRL still
            sees nothing but the scalar total, which keeps the legacy path
            bit-identical. ``p_drift`` is stored on the breakdown as a positive
            magnitude and SUBTRACTED inside ``total``, hence the negation here --
            GDPO normalises each term independently and must therefore be handed
            contributions that already carry their sign.
            """
            comp_txt, meta = pair
            if meta is None:
                # Unknown row: the historical neutral default. Expressed as a
                # single-term-carrying tuple so the sum invariant still holds.
                # Under the safety ablation the same -0.5 rides in the PROGRESS
                # slot instead, so the safety slot stays identically 0.0 across
                # the batch and GDPO's zero-std branch keeps skipping it -- a
                # -0.5 leaking into that slot would make the "deleted" term
                # non-unanimous and hand it a gradient again.
                if disable_safety:
                    return (-0.5, (0.0, -0.5, 0.0))
                return (-0.5, (-0.5, 0.0, 0.0))
            bd = compute_evoguard_reward(
                completion_text=comp_txt,
                meta=meta,
                judge_call=effective_jcb,
                progress_call=effective_pcb,
                disable_safety=disable_safety,
            )
            # Scoring is fanned out across threads, so the tally needs the lock;
            # it is contended for nanoseconds against an HTTP round-trip.
            key = f"{bd.safety_source or 'none'}:{bd.safety_label or 'none'}"
            with _tally_lock:
                safety_tally[key] = safety_tally.get(key, 0) + 1
            return (
                float(bd.total),
                (float(bd.r_safety), float(bd.r_progress), -float(bd.p_drift)),
            )

        # With both judges live, scoring costs TWO HTTP round-trips per
        # completion -- serially that is minutes per optimizer step. The judge
        # closures hold no mutable state (each call builds its own request), so
        # fan them out. Falls back to a plain loop when no judge is active,
        # keeping the pure-heuristic path allocation-free and deterministic.
        n_workers = _reward_judge_workers() if (effective_jcb or effective_pcb) else 0
        if n_workers > 1 and len(pairs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(n_workers, len(pairs))) as pool:
                scored = list(pool.map(_score_one, pairs))
        else:
            scored = [_score_one(p) for p in pairs]
        results_floats = [float(t) for t, _c in scored]
        components = [tuple(c) for _t, c in scored]
        if reward_trace_sink is not None:
            # Single-slot buffer, overwritten on every call: the trainer subclass reads
            # it immediately after super()._generate_and_score_completions()
            # returns, so no history is needed and the memory stays O(batch).
            try:
                trace = (list(results_floats), list(row_indices), list(components))
                if reward_trace_sink:
                    reward_trace_sink[0] = trace
                else:
                    reward_trace_sink.append(trace)
            except Exception:                                          # noqa: BLE001
                pass
        _reward_seconds[0] += time.perf_counter() - _t_rf0
        return results_floats

    _evoguard_reward_func.__name__ = "evoguard_defense_rl_reward"
    _evoguard_reward_func.safety_tally = safety_tally     # type: ignore[attr-defined]
    _evoguard_reward_func.reward_seconds = _reward_seconds  # type: ignore[attr-defined]
    return _evoguard_reward_func


def _extraction_seed(training_cfg: TrainingConfig, round_label: str) -> int:
    """Deterministic per-round seed for the prompt sampler's tie-breaking RNG.

    The call site used to read ``getattr(training_cfg, "_seed_for_extraction", 0)``
    but **nothing ever assigns that attribute** -- a repo-wide grep finds exactly
    one occurrence, the read itself -- so every round of every run so far sampled
    with seed 0. That is not merely cosmetic: capping is heavy (r11 discarded 484
    of 495 attacked candidates), and with a fixed seed the RNG's tie-breaks are
    identical each round, which biases which tasks get retried.

    Precedence: an explicitly configured ``_seed_for_extraction`` still wins (so
    a caller can pin it), otherwise the round's trailing digits are used, so r0..
    r11 each get their own stream while a re-run of the same round reproduces.
    """
    explicit = getattr(training_cfg, "_seed_for_extraction", None)
    if explicit is not None:
        try:
            return int(explicit)
        except Exception:                                              # noqa: BLE001
            pass
    digits = "".join(ch for ch in str(round_label or "") if ch.isdigit())
    base = int(getattr(training_cfg, "seed", 0) or 0)
    return base * 1000 + (int(digits) if digits else 0)


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
    k_traj_steps = max(1, int(getattr(training_cfg, "grpo_traj_group_size", 1) or 1))
    prompt_rows, stats = extract_grpo_prompts(
        records=records,
        dataset_builder=dataset_builder,
        max_prompts=max_prompts_cap,
        seed=_extraction_seed(training_cfg, round_label),
        clean_ratio=float(getattr(training_cfg, "grpo_clean_prompt_ratio", 0.0) or 0.0),
        steps_per_trajectory=k_traj_steps,
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
                "grpo_clean_prompt_ratio","grpo_advantage_curriculum_lambda",
                "grpo_traj_group_size",
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
        from peft import PeftModel                                    # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        from evoguard.training._trl_compat import patch_trl_probes
        # Shim 3 decides what ``trl.trainer.grpo_trainer`` imports at MODULE scope,
        # so the colocate flag has to be known here -- before the ``from trl``
        # below -- not down at B3 where the rest of the GRPOConfig knobs are read.
        # A False return means the vLLM aliases could not be installed; B3 honours
        # it and stays on HF ``generate`` rather than letting GRPOTrainer.__init__
        # raise ImportError.
        want_vllm_colocate = bool(getattr(training_cfg, "grpo_use_vllm_colocate", False))
        vllm_backend_live = patch_trl_probes(enable_vllm=want_vllm_colocate)
        if want_vllm_colocate and not vllm_backend_live:
            logger.warning(
                "[native_grpo] grpo_use_vllm_colocate=True but TRL's vLLM backend "
                "could not be bridged onto this vLLM; generating with HF generate."
            )
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
        use_vllm_colocate=want_vllm_colocate and vllm_backend_live
        vllm_mem_util=float(getattr(training_cfg,"grpo_vllm_gpu_memory_utilization",0.25) or 0.25)

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
            # Generation backend. False = HF `generate`, one unbatched decode per
            # completion, which is what made a round cost ~13 s/prompt. True +
            # colocate = an in-process vllm.LLM on the trainer's own device with a
            # per-step weight resync (GRPO must stay on-policy, so the long-lived
            # :8000 server -- adapter reloaded once per ROUND -- cannot serve it).
            # vllm_mode="server" is unavailable on vllm 0.19.1: importing
            # trl.scripts.vllm_serve raises ImportError on GuidedDecodingParams.
            use_vllm=use_vllm_colocate,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=vllm_mem_util,
            # 轨C.3: K>1 makes ONE generation batch cover exactly one trajectory
            # (G siblings x K steps). Pinned together with shuffle_dataset=False
            # because TRL's RepeatSampler with shuffle=False walks range(N) in
            # order and chunks it by generation_batch_size//num_generations == K
            # (trl/trainer/utils.py ~:895) -- with shuffling on, a "trajectory"
            # batch would be K unrelated rows and the pooled baseline would be
            # nonsense. K==1 keeps steps_per_generation=None (TRL then defaults it
            # to gradient_accumulation_steps) and shuffling on, i.e. legacy.
            steps_per_generation=(g_size*k_traj_steps) if k_traj_steps>1 else None,
            shuffle_dataset=False if k_traj_steps>1 else True,
            per_device_train_batch_size=max(1,int(getattr(training_cfg,"per_device_batch_size",1))),
            gradient_accumulation_steps=max(1,int(getattr(training_cfg,"gradient_accumulation",8))),
            optim="adamw_torch_fused" if bf16_avail else "adamw_torch",
            lr_scheduler_type="cosine",
            save_strategy="no",     # trainer checkpoints never resumed (resume_from_checkpoint=False);
                                    # end-of-training final save still wrote ~940M optimizer-state checkpoint
                                    # per round. Final adapter is saved explicitly via save_pretrained()
                                    # to <out_root>/adapter_weights below.
            save_total_limit=1,
            report_to=[],
            disable_tqdm=True,
            dataloader_num_workers=0,
            seed=_trainer_seed(round_label),
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
        # Name the dropped keys. This filter is silent by construction, so a kwarg
        # that TRL renamed or never had (e.g. vllm_gpu_memory_utilization) simply
        # vanishes and the run looks configured when it is not.
        dropped_keys=sorted(set(st_args) - set(cleaned_st_args))
        if dropped_keys:
            logger.warning(
                "[native_grpo] GRPOConfig does NOT accept %d kwarg(s), DROPPED: %s "
                "-- whatever they were meant to switch on is OFF this round.",
                len(dropped_keys), dropped_keys,
            )
        if use_vllm_colocate:
            missing=[k for k in ("use_vllm","vllm_mode","vllm_gpu_memory_utilization")
                     if k not in cleaned_st_args]
            if missing:
                raise RuntimeError(
                    "grpo_use_vllm_colocate=True but this TRL build does not accept "
                    f"{missing}; refusing to run, because the round would silently "
                    "fall back to HF generate and its timings would be reported as "
                    "the colocate numbers."
                )
            logger.info(
                "[native_grpo] vLLM COLOCATE generation ENABLED "
                "(vllm_gpu_memory_utilization=%.2f of the whole trainer card). "
                "Weights resync into the engine every optimizer step; advantage "
                "maths is untouched, so a change in mean_reward vs use_vllm=False "
                "is a bug, not a speedup.", vllm_mem_util,
            )

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
        reward_trace_sink: list = []
        disable_safety_term = bool(
            getattr(training_cfg, "grpo_disable_safety_term", False)
        )
        if disable_safety_term:
            logger.warning(
                "[native_grpo] ABLATION grpo_disable_safety_term=True: r_safety is "
                "pinned to 0.0 on BOTH arms and the step-safety judge is never "
                "called, so the trained reward is R = r_progress - p_drift. Expect "
                "safety_source_tally == {'disabled:disabled': N}; a 'fallback:' "
                "tally instead would mean the flag did not reach the reward."
            )
        reward_fn_closure=build_evoguard_reward_callable(
            metas_lookup_table, reward_trace_sink=reward_trace_sink,
            disable_safety=disable_safety_term,
        )

        diag_state={"step_counter":0,"first_rewards":[],"last_rewards":[],"kl_trace":[],
                    "reward_std_trace":[],"frac_reward_zero_std_trace":[],
                    "gen_score_seconds":0.0,"gen_score_calls":0,
                    "delta_shaping_applied_count": 0,
                    "delta_shaping_mean_scale": []}
        # CRITICAL: inherit from transformers.TrainerCallback so all lifecycle hooks
        # (on_init_end, on_train_begin, on_epoch_end, etc.) get default no-op
        # implementations. TRL>=0.19 GRPOTrainer.__init__ calls on_init_end on
        # every registered callback -- bare `object` lacks that method and raises:
        #   AttributeError: '_DiagCallback' object has no attribute 'on_init_end'
        from transformers import TrainerCallback as _BaseCb  # local import keeps module-level deps lazy.
        class _DiagCallback(_BaseCb):
            """Lightweight hook capturing diagnostics needed by outcome reporting.

            Reward/KL collection lives in ``on_log``, NOT ``on_step_end``. Two bugs
            were stacked here before (2026-09-07): ``Trainer`` invokes
            ``on_step_end(args, state, control)`` WITHOUT a ``logs`` kwarg -- ``logs``
            is only ever passed to ``on_log`` -- so the whole body was unreachable and
            every ``mean_reward_before`` / ``mean_reward_after`` / ``kl_estimate_avg``
            in the plan JSON came out ``null``; and the keys it looked for
            (``rewards/mean``) are not the ones TRL emits. ``GRPOTrainer.log()``
            merges ``self._metrics[mode]`` into ``logs`` before delegating to
            ``Trainer.log`` (grpo_trainer.py ~:1446-1458), so the real key names are
            ``reward`` / ``reward_std`` / ``frac_reward_zero_std`` / ``kl``.

            ``frac_reward_zero_std`` is the saturation observable: it is the share of
            prompt groups whose G siblings all scored identically, i.e. rows that
            contribute exactly zero gradient. Historically 62-73%; the temperature
            knob is only worth touching when this stays above ~0.5.
            """
            def __init__(self,state): super().__init__() ; self._st=state
            def on_step_end(self,args=None,state=None,control=None,model=None,**kw):  # noqa: ARG002,D401
                self._st["step_counter"]+=1

            def on_log(self,args=None,state=None,control=None,logs=None,**kw):  # noqa: ARG002,D401
                if not logs:
                    return
                def _f(key):
                    try:
                        return float(logs[key])
                    except (KeyError, TypeError, ValueError):
                        return None
                r=_f("reward")
                if r is not None:
                    # "before" is the FIRST logged step (logging_first_step=True,
                    # logging_steps=1), "after" is every step -- kept as a running
                    # list so the reported mean is over the round, matching the
                    # pre-existing field semantics.
                    if not self._st["first_rewards"]:
                        self._st["first_rewards"].append(r)
                    self._st["last_rewards"].append(r)
                for key,slot in (("reward_std","reward_std_trace"),
                                 ("frac_reward_zero_std","frac_reward_zero_std_trace"),
                                 ("kl","kl_trace")):
                    v=_f(key)
                    if v is not None:
                        self._st[slot].append(v)

        cb=_DiagCallback(diag_state)
        lam_curriculum = float(getattr(training_cfg, "grpo_advantage_curriculum_lambda", 0.0) or 0.0)
        use_delta_shaping = (lam_curriculum > 0.0)
        use_traj_pool = (k_traj_steps > 1)
        use_gdpo = bool(getattr(training_cfg, "grpo_gdpo", False))
        use_adv_hook = use_delta_shaping or use_traj_pool or use_gdpo

        if use_gdpo:
            logger.info(
                "[native_grpo] GDPO ENABLED: each of the three reward terms "
                "(r_safety, r_progress, -p_drift) is standardised inside its own "
                "group of G=%d siblings and the per-term advantages are then "
                "summed and rescaled batch-wise, instead of GRPO's "
                "sum-then-standardise. A term the group agrees on contributes "
                "exactly 0 without flattening the terms it disagrees on.",
                g_size,
            )

        if use_traj_pool:
            logger.info(
                "[native_grpo] trajectory-pooled advantage baseline ENABLED "
                "(K=%d steps/trajectory, steps_per_generation=%s, "
                "shuffle_dataset=False): a prompt group whose G siblings all "
                "score identically borrows a baseline pooled over its own "
                "trajectory instead of contributing zero gradient.",
                k_traj_steps, cleaned_st_args.get("steps_per_generation"),
            )
        if use_delta_shaping:
            logger.info(
                "[native_grpo] Δ-aware advantage shaping ENABLED with λ=%.4f "
                "(Ã=(1+λ·δ_p)·A applied per-prompt-group atop group-relative advantages).",
                lam_curriculum,
            )

        if use_adv_hook:
            # Local subclass overriding _generate_and_score_completions ONLY --
            # parent handles everything else unchanged keeping blast radius minimal.
            class _DeltaShapedGRPOTrainer(GRPOTrainer):                       # type: ignore[misc]
                """Thin GRPOTrainer override rewriting advantages post-scoring.

                Three independent, composable interventions, in this order:

                  0. GDPO -- recomputes the ENTIRE advantage vector by
                     standardising each reward term inside its group and summing
                     the per-term advantages (see :func:`_gdpo_advantages`).
                     Skipped unless ``grpo_gdpo`` is set.
                  1. trajectory POOLING -- replaces the advantage of positions
                     whose own prompt group is degenerate (``reward_std == 0``,
                     62.8% of steps on the plan_abc run, i.e. no gradient at all)
                     with one standardised against that trajectory's pooled
                     rewards. Skipped entirely when K == 1.
                  2. Δ-aware SHAPING -- multiplies by ``(1 + λ·δ_p)``.

                The order is load-bearing in both places. GDPO must run FIRST
                because it overwrites the whole vector, so anything applied
                before it is discarded; it also leaves fewer groups degenerate,
                which makes pooling the narrower fallback it is meant to be
                (pooling still keys off the SCALAR total's group std, so it can
                still fire on a group where all three terms are unanimous).
                Pooling must in turn precede shaping: shaping is a multiplicative
                curriculum on whatever advantage the step ended up with, so
                applying it to a value that pooling is about to overwrite would
                silently drop the curriculum on exactly the pooled positions.
                """

                _EVOGUARD_LAMBDA_CURRICULUM_DEFAULT: float = 0.0       # type: ignore[assignment]
                _EVOGUARD_METAS_LOOKUP_DEFAULT: dict = {}              # type: ignore[assignment]

                def __init__(self,*args,_evoguard_lambda:float=0.0,
                             _evoguard_metas_by_idx:Optional[dict]=None,
                             _evoguard_reward_trace:Optional[list]=None,
                             _evoguard_num_generations:int=1,
                             _evoguard_traj_pool:bool=False,
                             _evoguard_gdpo:bool=False,
                             _evoguard_diag:Optional[dict]=None,**kwargs):
                    self._evoguard_lambda_val=float(_evoguard_lambda or 0.0)
                    self._evoguard_metas_lookup=_evoguard_metas_by_idx or {}
                    self._evoguard_reward_trace=_evoguard_reward_trace
                    self._evoguard_num_gen=max(1,int(_evoguard_num_generations or 1))
                    self._evoguard_traj_pool=bool(_evoguard_traj_pool)
                    self._evoguard_gdpo=bool(_evoguard_gdpo)
                    # Share the caller's diag dict when given so the counters
                    # actually reach the round log (they were write-only before).
                    self._diag_state_ref:dict[str,Any]=(
                        _evoguard_diag if _evoguard_diag is not None else {})
                    for _k,_v in (("delta_shaping_applied_count",0),
                                  ("delta_shaping_mean_scale",[]),
                                  ("last_n_shaped_slots",0),
                                  ("traj_pool_batches",0),
                                  ("traj_pool_slots",0),
                                  ("gdpo_batches",0),
                                  ("gdpo_slots",0)):
                        self._diag_state_ref.setdefault(_k,_v)
                    # Strip our private kwargs then forward normally.
                    super().__init__(*args,**kwargs)

                def _evoguard_row_idxs(self,inputs)->list[Any]:
                    row_idxs:list[Any]=[]
                    for x in (inputs or []):
                        ri:Any=None
                        try:
                            if hasattr(x,"get"): ri=x.get("row_idx")
                        except Exception:                              # noqa: BLE001
                            ri=None
                        row_idxs.append(ri)
                    return row_idxs

                def _evoguard_trace_view(self,row_idxs):
                    """Alignment-checked ``(rewards, components)`` for this batch.

                    Thin delegation to the module-level, unit-tested
                    :func:`_aligned_trace_view`.
                    """
                    return _aligned_trace_view(self._evoguard_reward_trace,row_idxs)

                def _evoguard_apply_gdpo(self,out_dict,row_idxs)->None:
                    """Replace TRL's advantages with GDPO's per-reward ones.

                    Runs FIRST of the three interventions because it rewrites the
                    whole vector; pooling then acts as a fallback for groups that
                    GDPO still leaves degenerate (all three terms unanimous), and
                    Δ shaping multiplies whatever survives.
                    """
                    if not self._evoguard_gdpo:
                        return
                    _rewards,comps=self._evoguard_trace_view(row_idxs)
                    if not comps:
                        return
                    adv=_gdpo_advantages(
                        comps,num_generations=self._evoguard_num_gen)
                    if not adv:
                        return
                    _apply_advantage_overrides_inplace(
                        out_dict.get("advantages"),
                        {i:v for i,v in enumerate(adv)})
                    self._diag_state_ref["gdpo_batches"]+=1
                    self._diag_state_ref["gdpo_slots"]+=len(adv)

                def _evoguard_apply_traj_pool(self,out_dict,row_idxs)->None:
                    """Substitute pooled advantages for zero-std prompt groups."""
                    if not self._evoguard_traj_pool:
                        return
                    rewards_seq,_comps=self._evoguard_trace_view(row_idxs)
                    if not rewards_seq:
                        return
                    metas_lut=self._evoguard_metas_lookup or {}
                    traj_ids:list[str]=[]
                    for ri in row_idxs:
                        tid=""
                        try:
                            m=metas_lut.get(int(ri))
                            tid=str(getattr(m,"traj_group_id","") or "")
                        except Exception:                              # noqa: BLE001
                            tid=""
                        traj_ids.append(tid)
                    overrides=_traj_pooled_advantage_overrides(
                        [float(r) for r in rewards_seq],
                        traj_ids,
                        num_generations=self._evoguard_num_gen,
                    )
                    if not overrides:
                        return
                    _apply_advantage_overrides_inplace(out_dict.get("advantages"),overrides)
                    self._diag_state_ref["traj_pool_batches"]+=1
                    self._diag_state_ref["traj_pool_slots"]+=len(overrides)

                def _generate_and_score_completions(self,inputs):
                    out_dict=super()._generate_and_score_completions(inputs)
                    try:
                        if not (isinstance(out_dict,dict) and "advantages" in out_dict and inputs):
                            return out_dict
                        row_idxs=self._evoguard_row_idxs(inputs)
                        self._evoguard_apply_gdpo(out_dict,row_idxs)
                        self._evoguard_apply_traj_pool(out_dict,row_idxs)
                        lam=float(getattr(self,"_evoguard_lambda_val",0.0))
                        metas_lut=getattr(self,"_evoguard_metas_lookup",{}) or {}
                        if lam>0.0:
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
        if use_adv_hook:
            ctor_kwargs["_evoguard_lambda"]=lam_curriculum if use_delta_shaping else 0.0
            ctor_kwargs["_evoguard_metas_by_idx"]=metas_lookup_table
            ctor_kwargs["_evoguard_reward_trace"]=reward_trace_sink
            ctor_kwargs["_evoguard_num_generations"]=g_size
            ctor_kwargs["_evoguard_traj_pool"]=use_traj_pool
            ctor_kwargs["_evoguard_gdpo"]=use_gdpo
            ctor_kwargs["_evoguard_diag"]=diag_state

        # The colocated vLLM engine is built INSIDE GRPOTrainer.__init__ --
        # trl/trainer/grpo_trainer.py does `self.llm = LLM(..., distributed_executor_backend
        # ="external_launcher")` there, not lazily at first generation. So the env has to be
        # live before the CONSTRUCTOR, not merely before train(); and it has to stay live
        # through train(), because ExecutorWithExternalLauncher keeps using the
        # torch.distributed group it initialised from `env://`. Restored afterwards so
        # RANK/WORLD_SIZE cannot leak into the next round's SFT/accelerate path.
        colocate_env_prev=_enter_colocate_env() if use_vllm_colocate else None

        try:
            if use_adv_hook:
                trainer=_DeltaShapedGRPOTrainer(**ctor_kwargs)               # type: ignore[arg-type,misc]
            else:
                clean_kwargs={k:v for k,v in ctor_kwargs.items()
                              if not k.startswith("_evoguard_")}
                trainer=GRPOTrainer(**clean_kwargs)
        except Exception as ctor_exc:                                       # noqa: BLE001
            logger.exception("[native_grpo] GRPOTrainer instantiation raised:%s",ctor_exc)
            _append_plan_json(plan_log_path,{"phase":"trainer_ctor_error","err":str(ctor_exc)})
            if colocate_env_prev is not None:
                _restore_env(colocate_env_prev)
                # locals() because the ctor is what raised: `trainer` may not exist.
                _release_colocate_engine(locals().get("trainer"))
                _teardown_colocate_process_group()
            # Re-raise for the same fail-fast reason as the fit() crash above.
            raise

        _install_nonfinite_grad_guard(trainer, diag_state)
        _install_phase_timers(trainer, diag_state)

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
            # Re-raise: a crashed trainer must abort the co-evolution loop.
            # Returning a soft error outcome here previously let the pipeline
            # continue for 11 more rounds against a frozen defender (silent
            # per-round OOMs, observed 2026-08-16).
            raise
        finally:
            if colocate_env_prev is not None:
                _restore_env(colocate_env_prev)
                # Engine memory FIRST: destroying the process groups frees no bytes,
                # and the next round loads its policy before it builds its engine.
                _release_colocate_engine(trainer)
                _teardown_colocate_process_group()

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

        def _mean(trace_key):
            vals = diag_state.get(trace_key) or []
            return (sum(vals)/len(vals)) if vals else None

        reward_std_avg = _mean("reward_std_trace")
        frac_zero_std_avg = _mean("frac_reward_zero_std_trace")

        # Phase attribution. reward_seconds is judge HTTP + reward arithmetic;
        # gen_score_seconds is that PLUS generation and the policy/reference
        # logprob forwards; whatever fit_seconds has left over is
        # backward+optimizer. See _install_phase_timers.
        reward_secs = float((getattr(reward_fn_closure, "reward_seconds", [0.0]) or [0.0])[0])
        gen_score_secs = float(diag_state.get("gen_score_seconds", 0.0) or 0.0)
        timing = {
            "fit_seconds": round(fit_secs, 2),
            "seconds_per_prompt": round(fit_secs/max(1,n_samples), 3),
            "gen_score_seconds": round(gen_score_secs, 2),
            "reward_judge_seconds": round(reward_secs, 2),
            "generate_and_logprob_seconds": round(max(0.0, gen_score_secs - reward_secs), 2),
            "backward_seconds": round(max(0.0, fit_secs - gen_score_secs), 2),
            "gen_score_calls": int(diag_state.get("gen_score_calls", 0) or 0),
        }
        logger.info(
            "[native_grpo] phase split over %.1fs: generate+logprobs %.1fs | "
            "judge+reward %.1fs | backward+optim %.1fs (%.2fs/prompt)",
            fit_secs, timing["generate_and_logprob_seconds"],
            timing["reward_judge_seconds"], timing["backward_seconds"],
            timing["seconds_per_prompt"],
        )
        if frac_zero_std_avg is not None:
            logger.info(
                "[native_grpo] reward_std=%.4f frac_reward_zero_std=%.3f -- the "
                "latter is the share of prompt groups contributing ZERO gradient; "
                "above ~0.5 the rollout temperature is the knob to reach for, not "
                "the reward.", reward_std_avg if reward_std_avg is not None else float("nan"),
                frac_zero_std_avg,
            )
        n_pool_batches=int(diag_state.get("traj_pool_batches",0) or 0)
        n_pool_slots=int(diag_state.get("traj_pool_slots",0) or 0)
        if use_traj_pool:
            # This is the intervention's only observable: zero batches means the
            # pooling never fired (either no degenerate group, or the alignment
            # cross-check rejected the batch) and the round is legacy GRPO.
            logger.info(
                "[native_grpo] trajectory pooling fired on %d generation batches "
                "(%d completion slots re-based).", n_pool_batches, n_pool_slots,
            )
        if use_gdpo:
            # Same reasoning as above: zero batches means GDPO never actually
            # replaced an advantage (missing components in the trace, or the
            # alignment cross-check rejected every batch) and the round ran
            # legacy GRPO despite the flag.
            logger.info(
                "[native_grpo] GDPO fired on %d generation batches "
                "(%d completion slots re-normalised).",
                int(diag_state.get("gdpo_batches",0) or 0),
                int(diag_state.get("gdpo_slots",0) or 0),
            )

        # The safety term's only observable. A round whose r_safety came from the
        # fallback trained r_progress alone -- see build_evoguard_reward_callable.
        safety_tally = dict(getattr(reward_fn_closure, "safety_tally", {}) or {})
        n_scored = sum(safety_tally.values())
        n_fallback = sum(v for k, v in safety_tally.items()
                         if k.startswith("fallback:"))
        if n_scored:
            logger.info("[native_grpo] r_safety sources over %d completions: %s",
                         n_scored, sorted(safety_tally.items()))
            if n_fallback:
                logger.warning(
                    "[native_grpo] %d/%d completions (%.1f%%) scored r_safety from "
                    "the FALLBACK constant -- no safety gradient on those rows.",
                    n_fallback, n_scored, 100.0 * n_fallback / n_scored,
                )

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
            **timing,
            "adapter_saved_at":os.path.abspath(saved_adapters_dir),
            "mean_reward_before":mr_before,
            "mean_reward_after":mr_after,
            "reward_std_avg":reward_std_avg,
            "frac_reward_zero_std_avg":frac_zero_std_avg,
            "kl_estimate_avg":kl_est,
            "steps_executed":diag_state["step_counter"],
            "traj_group_size_k":k_traj_steps,
            "traj_pool_batches":n_pool_batches,
            "traj_pool_slots":n_pool_slots,
            "delta_shaping_applied_count":int(diag_state.get("delta_shaping_applied_count",0) or 0),
            "safety_source_tally":safety_tally,
            "n_safety_fallback":n_fallback,
            "nonfinite_grad_skips":int(diag_state.get("nonfinite_grad_skips",0) or 0),
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
