"""Injection point, behavior turning point and delta (``docs/plan.md``).

Given a clean trajectory A and an attacked trajectory B for the same task plus
the :class:`AttackSpec` that produced B, this module computes the three signals
that drive the whole method:

* **injection_point** -- the turn the attacker injected at (from the spec).
* **turning_point** -- the turn of B's first significant behavioral divergence
  from A, found by aligning their tool-call signature sequences.
* **delta = turning_point - injection_point** -- large delta => latent/stealthy
  attack (attacker's objective), small delta => immediate-trigger (defender's
  objective). ``delta_normalized`` in ``[0, 1]`` is used as genetic fitness.

The turning point is expressed in *turn* units. Because alignment happens over
the sub-sequence of tool calls, we map the diverging tool-call index back to the
actual ``turn`` recorded on the attacked action.
"""

from __future__ import annotations

from typing import Optional

from evoguard.config import ProcessConfig
from evoguard.core.types import AttackSpec, Signals, Trajectory
from evoguard.process.edit_distance import align, first_divergence_index_b


def compute_signals(
    clean: Trajectory,
    attacked: Trajectory,
    attack: AttackSpec,
    config: Optional[ProcessConfig] = None,
) -> Signals:
    """Compute :class:`Signals` for an attacked trajectory versus its clean twin."""

    config = config or ProcessConfig()
    seq_a = clean.action_signatures()
    seq_b = attacked.action_signatures()
    b_turns = attacked.tool_call_turns()

    alignment = align(seq_a, seq_b)
    div_idx = first_divergence_index_b(seq_a, seq_b, config.divergence_threshold)

    injection_point = attack.target_turn
    turning_point: Optional[int] = None
    if div_idx is not None and 0 <= div_idx < len(b_turns):
        turning_point = b_turns[div_idx]
    elif div_idx is not None and b_turns:
        turning_point = b_turns[-1] + 1

    delta: Optional[int] = None
    if turning_point is not None:
        delta = turning_point - injection_point

    delta_normalized = _normalize_delta(delta, clean, attacked, config)

    return Signals(
        injection_point=injection_point,
        turning_point=turning_point,
        delta=delta,
        delta_normalized=delta_normalized,
        edit_distance=int(round(alignment.distance)),
        metadata={
            "clean_len": len(seq_a),
            "attacked_len": len(seq_b),
            "divergence_index_b": div_idx,
        },
    )


def _normalize_delta(
    delta: Optional[int],
    clean: Trajectory,
    attacked: Trajectory,
    config: ProcessConfig,
) -> float:
    """Scale delta into ``[0, 1]`` for use as genetic fitness.

    Only non-negative deltas contribute positive fitness (a turning point before
    the injection is not attributable to the attack). The denominator is the
    clean-trajectory length (default) or the max of the two lengths.
    """

    if delta is None or delta < 0:
        return 0.0
    clean_len = max(1, len(clean.action_signatures()))
    attacked_len = max(1, len(attacked.action_signatures()))
    if config.normalize_by == "max_length":
        denom = max(clean_len, attacked_len)
    else:
        denom = clean_len
    return min(1.0, delta / float(denom))
