"""Top-K block ranking + JSON-artifact emission for the LoRA probe.

Consumes per-layer sensitivity scores produced by
:mod:`evoguard.training.sensitivity` and emits a self-describing JSON file that
:mod:`evoguard.training.native_runner` later reads to override the static
``lora_target_modules`` defaults during cold-start SFT.

Schema (v1, written by :func:`emit_probe_artifact`):

    {
      "schema_version": "1",
      "method": "attn_kl",
      "base_model": "...",
      "n_pairs_used": <int>,
      "inject_position_avg_tokens": <float>,
      "scores_by_block": {"0": 0.031, ...},
      "selected_blocks_sorted_desc": [<int>,...],
      "top_k_value": <int>,
      "recommended_target_modules": ["model.layers.<L>.self_attn.q_proj", ...],
      "created_ts_unix": <int>
    }

The ``recommended_target_modules`` list is the only field native_runner.py
actually consumes; everything else exists for auditability/reproducibility.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Lazy logger handle so this module stays light to import even before
# evoguard.utils.logging is configured by the caller.
try:
    from evoguard.utils.logging import get_logger as _get_logger   # noqa: F401
    _LOG = _get_logger("training.ranking")
except Exception:                                                  # noqa: BLE001
    class _NullLogger:
        def debug(self, *_a, **_kw): pass
        def info(self,  *_a, **_kw): pass
        def warning(self, *_a, **_kw): pass
        def error(self,   *_a, **_kw): pass
    _LOG = _NullLogger()

# Canonical four-projection set mounted on every selected Transformer block.
# Mirrors the project's existing default target_modules shape so swapping in a
# probe-selected subset keeps the same trainable-parameter family.
DEFAULT_PROJECTIONS_PER_BLOCK: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

SCHEMA_VERSION = "1"


@dataclass
class ProbeArtifact:
    """In-memory representation of the JSON artifact emitted at probe end."""

    method: str = "attn_kl"
    base_model: str = ""
    n_pairs_used: int = 0
    inject_position_avg_tokens: float = 0.0
    scores_by_block: dict[int, float] = field(default_factory=dict)
    selected_blocks_sorted_desc: list[int] = field(default_factory=list)
    top_k_value: int = 0
    recommended_target_modules: list[str] = field(default_factory=list)
    created_ts_unix: int = 0


def rank_blocks_by_score(
    scores_by_block: list[float] | dict[Any, Any],
) -> list[tuple[int, float]]:
    """Return ``(block_idx, score)`` pairs sorted strictly descending.

    Accepts either:
        * a plain list of floats indexed [0..N-1] (canonical form from the scorer),
          OR
        * a mapping {block_idx_or_strkey -> score} for caller convenience when
          blocks come pre-aggregated across multiple pairs already.

    Ties broken by lower index first so output ordering stays deterministic
    regardless of insertion order. Negative or NaN scores are clamped to zero;
    they still occupy their slot but never outrank positive contributors.
    """

    if isinstance(scores_by_block, dict):
        items: list[tuple[int, float]] = []
        for k, v in scores_by_block.items():
            try:
                idx_int = int(k)
            except Exception as exc:                              # noqa: BLE001
                raise TypeError(f"rank_blocks_by_score: non-integer key {k!r}") from exc
            s = _safe_float(v)
            items.append((idx_int, s))
    else:
        try:
            seq = list(scores_by_block)
        except TypeError as exc:
            raise TypeError("rank_blocks_by_score expects list-or-dict input") from exc
        items = [(idx, _safe_float(v)) for idx, v in enumerate(seq)]

    # Sort: primary key score desc; secondary key block_idx asc (deterministic tie-break).
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items


def select_top_k_blocks(
    ranked_pairs: list[tuple[int, float]],
    *,
    top_k: int,
) -> tuple[list[int], int]:
    """Pick top-K block indices given ranked ``(idx,score)`` pairs.

    Returns ``(selected_indices_in_ranked_order, effective_top_k)`` where
    *effective_top_k* equals min(top_k, len(ranked)) -- callers should record it
    into the artifact's ``top_k_value`` rather than echoing back the requested k,
    because selecting more than available silently truncates and downstream code
    must know which actually happened.
    """
    n_total = len(ranked_pairs)
    eff_k = max(0, min(int(top_k), n_total))
    return [int(idx) for idx, _score in ranked_pairs[:eff_k]], eff_k


def build_target_modules(
    selected_block_idxs: list[int],
    *,
    projections_per_block: Optional[list[str] | tuple[str, ...]] = None,
    module_name_template: str = "model.layers.{layer}.self_attn.{proj}",
) -> list[str]:
    """Expand chosen block indices into fully-qualified PEFT target-module names.

    Output order matches plan.md example structure::

       [
         "model.layers.<b0>.self_attn.q_proj",
         "model.layers.<b0>.self_attn.k_proj",
         ...
         "model.layers.<bk>.self_attn.o_proj",
       ]

    Each selected block contributes all four projection names by default since
    partial mounting risks expression-power loss against full-attention updates
    during co-evolution training cycles.
    """
    projs = (
        tuple(projections_per_block)
        if projections_per_block is not None
        else DEFAULT_PROJECTIONS_PER_BLOCK
    )
    out: list[str] = []
    seen: set[str] = set()
    for blk in selected_block_idxs:
        b = int(blk)
        for p in projs:
            name = module_name_template.format(layer=b, proj=p)
            # Dedup defensively even though inputs are unique-blocks-only typically.
            if name not in seen:
                out.append(name)
                seen.add(name)
    return out


def emit_probe_artifact(
    artifact: ProbeArtifact,
    *,
    path: str | Path,
) -> str:
    """Serialise + write :class:`ProbeArtifact` to disk as pretty-printed JSON.

    Validates required fields via :func:`validate_artifact_dict` before writing
    so malformed artifacts cannot reach disk where they would crash native_runner
    mid-training instead of here at construction time.

    Returns absolute resolved path string for convenience / logging.
    """
    payload = artifact_to_dict(artifact)
    validate_artifact_dict(payload)

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp_path, p)
    return str(p)


def artifact_to_dict(a: ProbeArtifact) -> dict[str, Any]:
    """Convert ProbeArtifact to canonical schema-conformant dict."""
    scores_str_keyed = {
        str(k): (_safe_float(v))
        for k, v in sorted((a.scores_by_block or {}).items(), key=lambda kv: int(kv[0]))
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": str(a.method or ""),
        "base_model": str(a.base_model or ""),
        "n_pairs_used": int(a.n_pairs_used),
        "inject_position_avg_tokens": float(a.inject_position_avg_tokens or 0.0),
        "scores_by_block": scores_str_keyed,
        "selected_blocks_sorted_desc": [int(b) for b in a.selected_blocks_sorted_desc],
        "top_k_value": int(a.top_k_value),
        "recommended_target_modules": [str(m) for m in a.recommended_target_modules],
        "created_ts_unix": int(a.created_ts_unix or time.time()),
    }


def load_artifact_dict(path: str | Path) -> dict[str, Any]:
    """Read+parse an artifact JSON file. Raises FileNotFoundError/JSONDecodeError."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return json.loads(text)


REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "method",
    "base_model",
    "n_pairs_used",
    "selected_blocks_sorted_desc",
    "recommended_target_modules",
)


def validate_artifact_dict(payload: dict[str, Any]) -> None:
    """Lightweight structural check used both before writing AND after reading."""

    if not isinstance(payload, dict):
        raise ValueError(f"probe artifact must be a JSON object, got type={type(payload).__name__}")

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"probe artifact missing required fields: {missing}")

    sv = str(payload.get("schema_version", ""))
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported probe-artifact schema_version={sv!r}; expected {SCHEMA_VERSION!r}"
        )

    sel = payload.get("selected_blocks_sorted_desc")
    if not isinstance(sel, list):
        raise ValueError("'selected_blocks_sorted_desc' must be a list")
    for x in sel:
        if not isinstance(x, int):
            raise ValueError(f"'selected_blocks_sorted_desc' entries must be ints, got {x!r}")

    mods = payload.get("recommended_target_modules")
    if not isinstance(mods, list):
        raise ValueError("'recommended_target_modules' must be a list")
    for m in mods:
        if not isinstance(m, str) or not m.strip():
            raise ValueError(f"'recommended_target_modules' entry invalid: {m!r}")

    sbk = payload.get("scores_by_block", {})
    if not isinstance(sbk, dict):
        raise ValueError("'scores_by_block' must be object<int-key,float-val>")

    npairs = payload.get("n_pairs_used")
    if isinstance(npairs, bool) or not isinstance(npairs, int) or npairs < 0:
        raise ValueError(f"'n_pairs_used' must be non-negative integer, got {npairs!r}")


def aggregate_pair_scores(
    per_pair_scores: list[list[float]],
    *,
    weights: Optional[list[float]] = None,
) -> list[float]:
    """Element-wise aggregate over per-pair layer-score lists.

    When ``weights`` is None (default) the function performs the legacy
    equal-weight arithmetic mean — preserving backwards compatibility with
    every existing yaml-driven probe run. When ``weights`` is supplied it must
    be a sequence of non-negative floats with length matching the number of
    rows in ``per_pair_scores``; the result then becomes the normalised
    weighted mean

    .. math::

        \\bar{s}_\\ell = \\frac{\\sum_p w_p\\, s_\\ell^{(p)}}{\\sum_p w_p}

    so callers can pass raw Δ values directly without pre-normalising.

    Defensive fallbacks (any of which collapses back to equal-mean behaviour):

      * length mismatch between weights and rows,
      * all-zero / all-negative / NaN / Inf weight vector,
      * ragged row widths are still padded-with-zeros as before.

    Returns empty list iff input was entirely empty.
    """
    cleaned_rows: list[list[float]] = []
    maxlen = 0
    for r in per_pair_scores:
        rr = [_safe_float(v) for v in (r or [])]
        cleaned_rows.append(rr)
        if len(rr) > maxlen:
            maxlen = len(rr)
    if maxlen == 0 or len(cleaned_rows) == 0:
        return []

    n_rows = len(cleaned_rows)

    # Resolve effective weighting scheme, falling back to uniform when invalid.
    eff_weights: Optional[list[float]]
    use_weighted = False
    if weights is not None:
        try:
            raw_w = [float(w) for w in weights]
        except (TypeError, ValueError):
            eff_weights = None
        else:
            # Sanitise NaN/Inf/negative entries to zero.
            sane_w = [
                (w if math.isfinite(w) and w > 0.0 else 0.0)
                for w in raw_w
            ]
            total = sum(sane_w)
            if (
                len(sane_w) == n_rows          # length must match exactly
                and total > 0.0                 # at least one positive entry survives
            ):
                use_weighted = True
                eff_weights = sane_w
            elif _LOG is not None:
                _LOG.debug(
                    "[aggregate_pair_scores] weights rejected "
                    "(len=%d expected %d, sum=%.4g); falling back to equal mean.",
                    len(sane_w), n_rows, float(total),
                )

    sums = [0.0] * maxlen
    norm_divisor_per_layer = [0.0] * maxlen   # tracks ∑w contributing valid scores per layer slot
    counts_uniform = [0] * maxlen             # used only on the fallback path

    for r_idx, rr in enumerate(cleaned_rows):
        w_row = eff_weights[r_idx] if (use_weighted and eff_weights is not None) else 1.0
        for i, v in enumerate(rr):
            sums[i] += v * w_row
            norm_divisor_per_layer[i] += w_row
            counts_uniform[i] += 1

    means: list[float] = []
    for i in range(maxlen):
        c_or_wsum = norm_divisor_per_layer[i] if use_weighted \
                    else max(1, counts_uniform[i])
        means.append(sums[i] / c_or_wsum if c_or_wsum > 0 else 0.0)
    return means


def make_artifact_from_run(
    *,
    method: str,
    base_model: str,
    per_pair_layer_scores: list[list[float]],
    inject_token_positions: list[int],
    inject_position_avg_tokens: float,
    top_k_requested: int,
    projections_per_block: Optional[list[str] | tuple[str, ...]] = None,
    per_pair_weights: Optional[list[float]] = None,
) -> ProbeArtifact:
    """Convenience builder aggregating raw pair-level outputs into final Artifact.

    Centralises rank->select->expand pipeline so the CLI entry point can stay thin.
    When ``per_pair_weights`` is provided (e.g. normalised Δ values from
    :mod:`evoguard.process.signals`) the aggregation step uses them as importance
    weights so latent-attack pairs exert proportionally more influence on layer
    ranking — implementing the explicit coupling described in §2.4 of
    ``docs/delta_signal_essence.md``. ``None`` keeps legacy equal-mean behaviour.
    """
    agg = aggregate_pair_scores(per_pair_layer_scores, weights=per_pair_weights)
    ranked = rank_blocks_by_score(agg)
    selected, eff_k = select_top_k_blocks(ranked, top_k=top_k_requested)
    mods = build_target_modules(selected, projections_per_block=projections_per_block)
    scores_map = {i: agg[i] if i < len(agg) else 0.0 for i in range(max(len(agg), 0))}
    avg_pos_val = float(inject_position_avg_tokens)
    if avg_pos_val == 0.0 and inject_token_positions:
        avg_pos_val = sum(inject_token_positions) / len(inject_token_positions)
    art = ProbeArtifact(
        method=str(method or "attn_kl"),
        base_model=str(base_model or ""),
        n_pairs_used=len(per_pair_layer_scores),
        inject_position_avg_tokens=avg_pos_val,
        scores_by_block=scores_map,
        selected_blocks_sorted_desc=selected,
        top_k_value=eff_k,
        recommended_target_modules=mods,
        created_ts_unix=int(time.time()),
    )
    return art


def _safe_float(x: Any) -> float:
    """Coerce arbitrary numeric-ish value to finite >=0 float; NaN/neg collapse to 0."""
    try:
        f = float(x)
    except Exception:                                  # noqa: BLE001
        return 0.0
    if f != f or f != f or f > 1e18 or f < -1e18:     # NaN check + sanity bounds
        return 0.0
    if f < 0.0:
        return 0.0                                     # KL-style divergences are inherently >=0
    return f


__all__ = [
    "ProbeArtifact",
    "DEFAULT_PROJECTIONS_PER_BLOCK",
    "SCHEMA_VERSION",
    "REQUIRED_FIELDS",
    "aggregate_pair_scores",
    "artifact_to_dict",
    "build_target_modules",
    "emit_probe_artifact",
    "load_artifact_dict",
    "make_artifact_from_run",
    "rank_blocks_by_score",
    "select_top_k_blocks",
    "validate_artifact_dict",
]
