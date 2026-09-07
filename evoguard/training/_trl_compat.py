"""Compatibility shims applied immediately before importing TRL trainers.

This environment runs TRL 0.27 against transformers 5.x, a pairing TRL predates.
``patch_trl_probes()`` applies every shim needed to make that combination work.
It is idempotent and each individual shim no-ops on versions that don't need it.

Shim 1 -- tuple-returning package probes. TRL 0.27 has a truthiness bug in
``trl/extras/vllm_client.py``::

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import ...

``trl.import_utils.is_vllm_ascend_available`` returns
``_is_package_available("vllm_ascend")`` -- a ``(bool, version)`` *tuple*, not a
bool. ``(False, None)`` is truthy, so on any CUDA box that has ``vllm``
installed but not ``vllm_ascend`` the guard fires and the import explodes with
``ModuleNotFoundError: No module named 'vllm_ascend'``. That takes down
``from trl import GRPOTrainer`` entirely. The probes are rebound to plain bools
before ``trl.extras.vllm_client`` is first imported.

Shim 2 -- ``PreTrainedModel.warnings_issued``. ``GRPOTrainer.__init__`` does::

    model.warnings_issued["estimate_tokens"] = True

``warnings_issued`` was a per-instance dict on transformers 4.x models and is
gone in 5.x, so the assignment raises ``AttributeError`` (routed through PEFT's
``__getattr__`` delegation chain, which makes the traceback misleading). A
lazily-populated property is installed on ``PreTrainedModel`` so the dict exists
per instance again.

Shim 3 -- TRL's optional vLLM generation backend, in one of TWO directions
depending on ``patch_trl_probes(enable_vllm=...)``. ``trl/trainer/grpo_trainer.py``
does, at module scope::

    if is_vllm_available():
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

``GuidedDecodingParams`` was renamed ``StructuredOutputsParams`` in vLLM 0.19, so
on a box where the trainer env also carries a modern vLLM that guard fires and
``from trl import GRPOTrainer`` dies with ``ImportError``.

*Default (``enable_vllm=False``)* -- the probe is pinned to ``False``. HF
``generate`` runs in-process, TRL's vLLM paths are unreachable, and the shim stays
independent of any future vLLM rename.

*``enable_vllm=True``* -- needed for ``vllm_mode="colocate"``, and it takes TWO
aliases, not one, because vLLM 0.19 broke the API in two places:

1. ``vllm.sampling_params.GuidedDecodingParams`` is re-pointed at
   ``StructuredOutputsParams`` so the module-scope import above resolves. TRL only
   *constructs* it when ``guided_decoding_regex`` is set, which EvoGuard never
   sets, so this alias exists purely to satisfy the import.
2. ``SamplingParams`` no longer accepts a ``guided_decoding`` kwarg (the field is
   ``structured_outputs``), and TRL's colocate branch passes it UNCONDITIONALLY --
   ``guided_decoding=None`` alone is a ``TypeError``. TRL's own module namespace
   gets a wrapper that drops a ``None`` and forwards anything else as
   ``structured_outputs``. Patching ``trl.trainer.grpo_trainer.SamplingParams``
   rather than ``vllm.SamplingParams`` keeps every other vLLM caller in the
   process (the colocated engine itself included) on the real class.

Both aliases have to be in place before ``trl.trainer.grpo_trainer`` is first
imported, so ``_enable_trl_vllm_backend`` forces that import itself and then
rebinds inside it. If either step fails, it falls back to pinning the probe
``False`` and reports that by returning ``False`` -- callers must honour the
return value and drop to ``use_vllm=False`` rather than let ``GRPOTrainer``
raise ``ImportError("vLLM is not available and `use_vllm` is set to True")``.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("evoguard.training.trl_compat")

#: ``None`` until the first ``patch_trl_probes()`` call, then the ``enable_vllm``
#: value that call settled on. Shim 3 is NOT re-runnable: it decides what
#: ``trl.trainer.grpo_trainer`` imports at module scope, and that import happens
#: once per process.
_APPLIED_VLLM: Optional[bool] = None


def patch_trl_probes(*, enable_vllm: bool = False) -> bool:
    """Apply every TRL-0.27-on-transformers-5.x shim. Idempotent.

    Returns whether TRL's vLLM generation backend is live in this process. A
    caller that asked for ``enable_vllm=True`` and gets ``False`` back MUST drop
    to ``use_vllm=False``; ``GRPOTrainer.__init__`` turns the mismatch into
    ``ImportError("vLLM is not available and `use_vllm` is set to True")``.
    """
    global _APPLIED_VLLM

    if _APPLIED_VLLM is not None:
        if _APPLIED_VLLM != enable_vllm:
            logger.warning(
                "[trl_compat] patch_trl_probes(enable_vllm=%s) ignored: shim 3 already "
                "settled on %s earlier in this process and trl.trainer.grpo_trainer "
                "resolves the vLLM guard once, at module import. Run the two "
                "generation backends in separate processes.",
                enable_vllm, _APPLIED_VLLM,
            )
        return _APPLIED_VLLM

    _coerce_tuple_probes()
    _restore_warnings_issued()
    _APPLIED_VLLM = _enable_trl_vllm_backend() if enable_vllm else False
    if not _APPLIED_VLLM:
        _disable_trl_vllm_backend()
    return _APPLIED_VLLM


def _coerce_tuple_probes() -> None:
    """Normalise TRL's tuple-returning package probes to plain bools."""
    try:
        from trl import import_utils as _iu  # type: ignore
    except Exception as exc:  # pragma: no cover - trl absent (dry-run paths)
        logger.debug("[trl_compat] trl.import_utils unavailable (%s); skipping", exc)
        return

    coerced_names = []
    for name in dir(_iu):
        if not (name.startswith("is_") and name.endswith("_available")):
            continue
        fn = getattr(_iu, name, None)
        if not callable(fn):
            continue
        try:
            probed = fn()
        except Exception as exc:
            logger.debug("[trl_compat] probe %s() raised %s; leaving as-is", name, exc)
            continue
        if not isinstance(probed, tuple):
            continue  # already well-behaved on this TRL version
        coerced = bool(probed[0])
        setattr(_iu, name, lambda _v=coerced: _v)
        coerced_names.append(f"{name}={coerced}")

    if coerced_names:
        logger.info("[trl_compat] coerced tuple-returning probes -> bool: %s",
                    ", ".join(coerced_names))


_WARNINGS_ISSUED_SLOT = "_evoguard_warnings_issued"


def _restore_warnings_issued() -> None:
    """Re-add the ``warnings_issued`` dict transformers 5.x dropped."""
    try:
        from transformers.modeling_utils import PreTrainedModel  # type: ignore
    except Exception as exc:  # pragma: no cover - transformers absent
        logger.debug("[trl_compat] transformers unavailable (%s); skipping", exc)
        return

    if hasattr(PreTrainedModel, "warnings_issued"):
        return  # transformers 4.x, or already patched

    def _get(self):
        store = self.__dict__.get(_WARNINGS_ISSUED_SLOT)
        if store is None:
            store = {}
            self.__dict__[_WARNINGS_ISSUED_SLOT] = store
        return store

    def _set(self, value):
        self.__dict__[_WARNINGS_ISSUED_SLOT] = value

    PreTrainedModel.warnings_issued = property(_get, _set)
    logger.info("[trl_compat] installed PreTrainedModel.warnings_issued shim "
                "(transformers 5.x dropped it; TRL still writes to it)")


def _disable_trl_vllm_backend() -> None:
    """Pin ``trl.import_utils.is_vllm_available`` to ``False``.

    Must run before ``trl.trainer.grpo_trainer`` is first imported: that module
    resolves the probe with a ``from ..import_utils import ...`` at module scope
    and then imports vLLM symbols under it. EvoGuard generates in-process
    (``use_vllm=False``), so nothing downstream of the guard is reachable.
    """
    try:
        from trl import import_utils as _iu  # type: ignore
    except Exception as exc:  # pragma: no cover - trl absent (dry-run paths)
        logger.debug("[trl_compat] trl.import_utils unavailable (%s); skipping", exc)
        return

    if not hasattr(_iu, "is_vllm_available"):
        return
    _iu.is_vllm_available = lambda: False
    logger.info("[trl_compat] pinned trl.import_utils.is_vllm_available()=False "
                "(EvoGuard uses use_vllm=False; TRL 0.19 targets the pre-0.19 "
                "vLLM sampling-params API)")


def _enable_trl_vllm_backend() -> bool:
    """Bridge TRL 0.19's vLLM call sites onto vLLM 0.19's renamed API.

    Returns ``False`` -- caller then pins the probe off -- if any step of the
    bridge is unavailable, so a missing alias degrades to HF ``generate`` instead
    of aborting the round.
    """
    try:
        from trl import import_utils as _iu  # type: ignore
        from vllm import sampling_params as _sp  # type: ignore
    except Exception as exc:
        logger.warning("[trl_compat] cannot enable TRL's vLLM backend (%s); "
                       "falling back to HF generate.", exc)
        return False

    # Alias 1: satisfy grpo_trainer's module-scope `from vllm.sampling_params
    # import GuidedDecodingParams`. Never constructed on EvoGuard's path.
    if not hasattr(_sp, "GuidedDecodingParams"):
        renamed = getattr(_sp, "StructuredOutputsParams", None)
        if renamed is None:
            logger.warning("[trl_compat] vllm.sampling_params has neither "
                           "GuidedDecodingParams nor StructuredOutputsParams; "
                           "falling back to HF generate.")
            return False
        _sp.GuidedDecodingParams = renamed
        logger.info("[trl_compat] aliased vllm.sampling_params.GuidedDecodingParams "
                    "-> StructuredOutputsParams (renamed in vLLM 0.19)")

    if hasattr(_iu, "is_vllm_available"):
        _iu.is_vllm_available = lambda: True

    # Import HERE, while alias 1 is in place, so the guarded block resolves.
    try:
        from trl.trainer import grpo_trainer as _gt  # type: ignore
    except Exception as exc:
        logger.warning("[trl_compat] trl.trainer.grpo_trainer failed to import with "
                       "vLLM enabled (%s); falling back to HF generate.", exc)
        return False

    # Alias 2: absorb the `guided_decoding` kwarg TRL's colocate branch always
    # passes. Scoped to TRL's namespace so the colocated engine keeps the real
    # class.
    real_sampling_params = getattr(_gt, "SamplingParams", None)
    if real_sampling_params is None:
        logger.warning("[trl_compat] trl.trainer.grpo_trainer exposes no "
                       "SamplingParams; falling back to HF generate.")
        return False

    def _sampling_params_compat(*args, **kwargs):
        guided = kwargs.pop("guided_decoding", None)
        if guided is not None:
            kwargs["structured_outputs"] = guided
        return real_sampling_params(*args, **kwargs)

    _gt.SamplingParams = _sampling_params_compat
    logger.info("[trl_compat] TRL vLLM backend ENABLED: is_vllm_available()=True and "
                "grpo_trainer.SamplingParams wrapped to drop the removed "
                "'guided_decoding' kwarg (vLLM 0.19 calls it 'structured_outputs')")

    # Alias 3: settle the CUDA caching allocator in the last instant before the
    # colocated engine is built, and retry if vLLM's memory profiling still
    # races. See _build_llm_with_settled_memory.
    real_llm = getattr(_gt, "LLM", None)
    if real_llm is None:
        logger.warning("[trl_compat] trl.trainer.grpo_trainer exposes no LLM; "
                       "colocate engines will profile against unsettled memory.")
    else:
        def _llm_compat(*args, **kwargs):
            return _build_llm_with_settled_memory(real_llm, *args, **kwargs)

        _gt.LLM = _llm_compat
        logger.info("[trl_compat] grpo_trainer.LLM wrapped to settle the CUDA "
                    "allocator before vLLM's memory-profiling snapshot and to "
                    "retry if it still races")
    return True


#: vLLM's own words in the assertion this retries on.
_MEMORY_PROFILING_MARKER = "Error in memory profiling"


def _build_llm_with_settled_memory(real_llm, *args, **kwargs):
    """Construct a colocated vLLM engine, tolerating its memory-profiling race.

    ``vllm/v1/worker/gpu_worker.py:408`` asserts free GPU memory never GREW
    between the snapshot taken in ``init_device`` and the end of profiling:

        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. ... This happens when other processes
             sharing the same container release GPU memory while vLLM is
             profiling during initialization." )

    The message blames another process; with colocate the culprit is this one --
    something in the trainer released ~0.6 GiB mid-profile. It killed r4 of run
    20260907_114433 (29.47 -> 30.07 GiB) and then r5 of run 20260907_123628
    (29.45 -> 30.05 GiB), the same delta both times, always on the round AFTER a
    colocate round and never on the first.

    The exact releaser is NOT established. A synthetic dirty allocator cache does
    not reproduce it (see the control arm of the offline gate), so the earlier
    "vLLM's closing empty_cache() hands back the trainer's cached blocks" story
    is unproven and the previous round's residency is the remaining suspect: at
    r5's settle 35.6 GiB was still held AFTER a full gc.collect(), i.e. r4's
    engine and models were not actually released.

    So this does two things rather than pretend to know one:
      * settles the allocator in the last instant before construction -- and it
        must be HERE, not in ``_enter_colocate_env``, which runs before
        ``GRPOTrainer.__init__`` puts policy+reference on the card (r5's log:
        43.54 GiB free at that point, 29.45 GiB by the engine's snapshot);
      * retries on that one assertion. The retry is sound regardless of the
        mechanism: whatever freed the memory has already freed it, so the next
        attempt snapshots the settled value. Bounded, and only this assertion is
        swallowed -- an OOM or any other failure propagates untouched.
    """
    attempts = 3
    for attempt in range(1, attempts + 1):
        _settle_cuda_allocator(f"before vLLM colocate LLM() attempt {attempt}")
        try:
            return real_llm(*args, **kwargs)
        except AssertionError as exc:
            if _MEMORY_PROFILING_MARKER not in str(exc) or attempt == attempts:
                raise
            logger.warning(
                "[trl_compat] vLLM memory profiling raced on attempt %d/%d (%s); "
                "the memory it saw released is now released, retrying.",
                attempt, attempts, str(exc).splitlines()[0],
            )
    raise AssertionError("unreachable")  # pragma: no cover


def _settle_cuda_allocator(why: str) -> None:
    """``gc.collect()`` + ``empty_cache()`` so the engine profiles settled memory.

    The collect matters on its own: the previous round's engine is reachable only
    through the finished round's ``GRPOTrainer``, whose object graph has cycles,
    so without an explicit collect it can be freed part-way through this round's
    profiling -- which is one candidate for the race documented above.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            logger.info("[trl_compat] settled CUDA allocator %s: free %.2f GiB "
                        "of %.2f GiB", why, free_b / 2**30, total_b / 2**30)
    except Exception as exc:                                            # noqa: BLE001
        logger.warning("[trl_compat] empty_cache() failed: %s", exc)
