"""Rollout strategies and the dual-trajectory collection driver.

Public surface:

* :class:`CleanRollout`, :class:`AttackedRollout` --
  single-trajectory strategies (``rollouts/base.py``).
* :func:`collect_tri_rollouts` -- per-round driver that, for every task,
  produces the clean trajectory A, then one attacked record B/C for each
  attack individual in the current population
  (``rollouts/rollout.py``).
* :class:`RoundRollouts` -- the bundle returned by ``collect_tri_rollouts``
  carrying records + per-task GA evaluations.
"""

from __future__ import annotations

from evoguard.rollouts.base import AttackedRollout, CleanRollout, RolloutStrategy
from evoguard.rollouts.rollout import RoundRollouts, collect_tri_rollouts

__all__ = [
    "RolloutStrategy",
    "CleanRollout",
    "AttackedRollout",
    "RoundRollouts",
    "collect_tri_rollouts",
]
