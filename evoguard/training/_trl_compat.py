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
"""

from __future__ import annotations

import logging

logger = logging.getLogger("evoguard.training.trl_compat")

_APPLIED = False


def patch_trl_probes() -> None:
    """Apply every TRL-0.27-on-transformers-5.x shim. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _coerce_tuple_probes()
    _restore_warnings_issued()


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
