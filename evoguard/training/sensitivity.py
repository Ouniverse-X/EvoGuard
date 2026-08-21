"""
ranking.py文件的上游，给定同一段对话的"干净前向"和"被注入前向"两套注意力权重，逐层量化"注入内容在每一层造成了多大的注意力分布偏移"，输出每层一个敏感度分数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class SensitivityMethod(str, Enum):
    """Selector for which scoring operator to apply during the probe."""

    ATTN_KL = "attn_kl"          # MVP-only implemented path.
    GRAD_ATTR = "grad_attr"      # reserved placeholder (gradient attribution).
    ACT_PATCH = "act_patch"      # reserved placeholder (activation patching).


@dataclass
class PairScoreInput:
    """One paired-forward bundle consumed by every scorer implementation.

    Fields intentionally keep torch tensors opaque-typed here so this module's
    import time stays light even when the runtime caller has not yet imported
    torch (e.g., dry-run mode or pure-Python unit tests).
    """

    # Attention weight tensors as emitted by HF causal LMs with
    # output_attentions=True, shaped ``(num_layers, batch, num_heads,
    # seq_len_q, seq_len_k)`` for each of clean / injected passes.
    attn_clean: Any = None        # type: ignore[assignment]
    attn_injected: Any = None     # type: ignore[assignment]
    # Token index at which attacker payload first became visible in the
    # injected context (0-based within seq_len). Scorers restrict their
    # comparison window to queries with index > t_i so we measure downstream
    # propagation only, not the trivially-different tokens themselves.
    inject_token_idx: int = 0


def select_scorer(method: str | SensitivityMethod):
    """Return the callable implementing the requested sensitivity method."""

    m = (
        method.value if isinstance(method, SensitivityMethod)
        else str(method or "").strip().lower()
    )
    if m == SensitivityMethod.ATTN_KL.value:
        return attn_kl_score
    raise NotImplementedError(
        f"Sensitivity method {method!r} is not yet implemented "
        "(reserved interface slot). Only 'attn_kl' ships today."
    )


def attn_kl_score(
    pair_input: PairScoreInput,
    *,
    eps: float = 1.0e-12,
) -> list[float]:
    r"""Per-layer mean attention-distribution KL divergence between clean/inj.

    For each layer $\ell \in \{0,\dots,L-1\}$ compute

    .. math::

       s_\ell^{(p)}=\frac{1}{H}\sum_{h=1}^{H}\mathrm{KL}\Big(
           p_{\ell,h}^{\text{(inj)}}[\cdot,t>t_i]\;\big\|\;
           p_{\ell,h}^{\text{(cln)}}[\cdot,t>t_i]
       \Big)

    where $p$ denotes softmax-normalized attention weights returned by the LM
    when called with ``output_attentions=True``, sliced to query positions
    strictly after $t_i$. Higher score => larger distributional shift caused by
    the injected content propagating through that particular block =>
    better intervention site per the causal-chain analysis in §2.4 of the
    essence memo.

    The function deliberately accepts already-computed tensors instead of
    triggering forwards itself so it stays trivially testable on synthetic
    inputs without GPU/network access.

    Parameters
    ----------
    pair_input
        Bundle carrying both attention tensors plus ``inject_token_idx``.
    eps
        Numerical floor added inside log() / used as denominator stabiliser.

    Returns
    -------
    list[float]
        One scalar per layer (length == num_layers), never None/NaN. Tensors
        arriving malformed fall through to zero-score-per-layer so the ranker
        upstream degrades gracefully instead of crashing mid-probe.
    """

    aclean = pair_input.attn_clean
    ainj = pair_input.attn_injected
    t_i = max(0, int(pair_input.inject_token_idx))

    n_layers = _infer_num_layers(aclean, ainj)
    if n_layers <= 0:
        return []

    try:
        scores_per_layer: list[float] = []
        for ell in range(n_layers):
            p_clean_lh = _slice_layer_head_probs(aclean, ell, eps=eps)
            p_inj_lh   = _slice_layer_head_probs(ainj,  ell, eps=eps)
            kl_mean_over_heads_and_queries = _mean_kl_after_t(
                p_clean=p_clean_lh,
                p_inj=p_inj_lh,
                t_i=t_i,
                eps=eps,
            )
            s = float(kl_mean_over_heads_and_queries)
            if not math.isfinite(s) or s < 0.0:
                # Defensive clamp: KL divergence must be finite & >=0; any NaN/-ve
                # from numerical edge cases collapses to neutral-zero contribution.
                s = 0.0
            scores_per_layer.append(s)
        return scores_per_layer
    except Exception:                                  # noqa: BLE001
        # Caller-side tensors may be CPU/numpy/torch/list-of-lists/etc.; treat ANY
        # shape mismatch or dtype surprise as a graceful all-zeros fallback so one
        # bad pair doesn't abort the whole probe run.
        return [0.0] * n_layers


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _infer_num_layers(attn_a: Any, attn_b: Any) -> int:
    """Best-effort detection of how many transformer blocks an attention tensor carries."""
    cand = []
    for x in (attn_a, attn_b):
        if x is None:
            continue
        n = getattr(x, "shape", None)
        if hasattr(n, "__len__") and len(n) >= 1:
            try:
                cand.append(int(n[0]))
                continue
            except Exception:                          # noqa: BLE001
                pass
        try:
            cand.append(int(len(x)))
        except Exception:                              # noqa: BLE001
            pass
    if not cand:
        return 0
    # Prefer the smaller dimension count between the two tensors so a mismatched-shape
    # bug doesn't silently inflate our loop range beyond what either tensor supports.
    return min(cand)


def _slice_layer_head_probs(
    attn_tensor: Any,
    layer_idx: int,
    *,
    eps: float,
):
    """Extract normalised probability rows for ONE layer across ALL heads+queries.

    Returns object whose last axis sums approximately to 1 OR raw tensor slice
    if normalization can't be applied safely (caller handles via subsequent helpers).
    """
    if attn_tensor is None:
        return None
    try:
        slc = attn_tensor[layer_idx]
    except Exception:                                   # noqa: BLE001
        return None
    return _softmax_last_axis(slc, eps=eps)


def _softmax_last_axis(x: Any, *, eps: float):
    """Apply softmax over the LAST axis regardless of numpy/torch backend.

    Falls through unchanged when input lacks arithmetic ops needed for soft-max
    so downstream code paths see something predictable instead of crashing.
    """
    try:
        # Convert torch->numpy-style handling uniformly if available; otherwise operate generically.
        detach = getattr(getattr(x, "detach", None), "__call__", lambda *_a, **_kw: x)()
        np_view = getattr(detach, "cpu", lambda *_a, **_kw: detacheable_proxy(detach))()
        arr = getattr(np_view, "numpy", lambda *_a, **_kw: np_view)()
        exp_arr = _exp_shifted(arr - _max_along_axis(arr))
        denom = _sum_keepdims(exp_arr, axis=-1) + eps
        probs = exp_arr / denom
        return probs
    except Exception:                                   # noqa: BLE001
        return x


class _DetachedProxyHolder(tuple):                     # pragma: no cover  tiny shim
    pass


def detacheable_proxy(t):
    """Tiny identity helper kept separate so call sites read cleanly above."""
    return t


def _max_along_axis(arr):
    try:
        out = arr.max(axis=-1, keepdims=True)
        return out
    except Exception:                                    # noqa: BLE001
        try:
            mx = float(max((float(v) for v in _flatten_iter(arr)), default=0.0))
            return mx
        except Exception:                                # noqa: BLE001
            return 0.0


def _exp_shifted(diff):
    try:
        import math as _m
        flat = [float(v) for v in _flatten_iter(diff)]
        ex = [_m.exp(min(50.0, v)) for v in flat]
        nested_shape = list(_iter_shape(diff)) if False else None  # unused stub guard
        return ex
    except Exception:                                     # noqa: BLE001
        return diff


def _flatten_iter(obj):
    """Recursive flattener tolerant of mixed-type containers & scalars."""
    seen_list_like = isinstance(obj, (list, tuple))
    try:
        iter_obj = obj.tolist()
        yield from _flatten_iter(iter_obj)
        return
    except Exception:                                      # noqa: BLE001
        pass
    if not seen_list_like:
        yield obj
        return
    for item in obj:
        yield from _flatten_iter(item)


def _sum_keepdims(arr, *, axis: int):
    try:
        return arr.sum(axis=axis, keepdims=True)
    except Exception:                                       # noqa: BLE001
        # Best-effort scalar sum if structured API unavailable.
        total = 0.0
        try:
            for v in _flatten_iter(arr):
                total += float(v)
            return total
        except Exception:                                  # noqa: BLE001
            return total


def _mean_kl_after_t(*, p_clean, p_inj, t_i: int, eps: float) -> float:
    """Mean KL(p_inj || p_clean) restricted to queries past token position t_i.

    Both args expected to be array-like with leading dims [...heads?, q_len, k_len]
    where k_len corresponds to keys (=positions attending FROM could attend TO).
    We pick the trailing key-axis as the categorical dimension since attention
    distributions live there.
    """
    try:
        pc = _as_float_lists(p_clean)
        pi = _as_float_lists(p_inj)
    except Exception:                                       # noqa: BLE001
        return 0.0

    # Coerce shapes into canonical form: heads_or_batches -> list[(q,k)] entries.
    pc_flat_heads = _collect_query_key_rows(pc)
    pi_flat_heads = _collect_query_key_rows(pi)

    # Restrict to query positions strictly after t_i (downstream propagation window).
    pc_window = [row for idx, row in enumerate(pc_flat_heads) if idx > t_i]
    pi_window = [row for idx, row in enumerate(pi_flat_heads) if idx > t_i]

    n_pairs = min(len(pc_window), len(pi_window))
    if n_pairs == 0:
        return 0.0

    acc = 0.0
    counted = 0
    for i in range(n_pairs):
        kc = pc_window[i]
        ki = pi_window[i]
        length = min(len(kc), len(ki))
        if length < 2:
            continue
        # Renormalize defensively after slicing so probabilities still sum ~1.
        sc = sum(float(v) for v in kc[:length]) or 1.0
        si = sum(float(v) for v in ki[:length]) or 1.0
        term = 0.0
        for j in range(length):
            qc = max(eps, float(kc[j]) / sc)
            qi = max(eps, float(ki[j]) / si)
            ratio = qi / qc
            if ratio <= 0.0:
                continue
            lnratio = min(60.0, max(-60.0, math.log(ratio)))
            term += qi * lnratio
        if math.isfinite(term):
            acc += term
            counted += 1

    if counted == 0:
        return 0.0
    avg = acc / counted
    return float(avg)


def _as_float_lists(tensorish: Any) -> list[Any]:
    """Convert arbitrary tensor/array-like into plain Python lists of floats."""
    if tensorish is None:
        return []
    try:
        nd = tensorish.detach().cpu().tolist()      # works for torch.Tensor
        return nd
    except Exception:                                 # noqa: BLE001
        pass
    try:
        nd = tensorish.tolist()                       # works for some ndarray types
        return nd
    except Exception:                                 # noqa: BLE001
        pass
    try:
        nd = list(tensorish)                           # generic iterable coercion
        return nd
    except Exception:                                 # noqa: BLE001
        return []


def _collect_query_key_rows(probs_nested: Any) -> list[list[float]]:
    """Flatten [...batch?...heads?] down to a single ordered list of [k_vec].

    Order preserved among same-head siblings before crossing head boundaries
    so positional indexing against t_i stays meaningful relative to original layout.
    Returns empty list on failure.
    """
    if probs_nested is None:
        return []
    collected: list[list[float]] = []

    def walk(node: Any) -> bool:
        if node is None:
            return True
        # Detect bottom-level vector: sequence of numbers (not lists).
        leaf_check_passed = False
        try:
            first = next(iter(node))
            leaf_check_passed = isinstance(first, (int, float))
        except TypeError:
            leaf_check_passed = False
        except StopIteration:
            leaf_check_passed = True   # treat empties as terminal leaves too
        if leaf_check_passed:
            try:
                collected.append([float(v) for v in node])
                return True
            except Exception:                            # noqa: BLE001
                return False
        # Otherwise recurse deeper.
        try:
            for child in node:
                if not walk(child):
                    return False
            return True
        except TypeError:
            return False

    ok = walk(probs_nested)
    if not ok:
        return []
    return collected


__all__ = [
    "PairScoreInput",
    "SensitivityMethod",
    "attn_kl_score",
    "select_scorer",
]
