"""
The defense agent is the entity being hardened by the co-evolution loop. It is
decomposed into a single-step decision function so the *rollout driver* (which
owns the interaction loop) can splice attacker-controlled content into tool
observations between steps. This separation keeps injection logic in the
controller while the agent stays a pure policy ``history -> next action``.
"""

from __future__ import annotations

import abc
from typing import Sequence

from evoguard.core.types import Action, Task, ToolSpec


class DefenseAgent(abc.ABC):
    """A tool-using policy that decides one action at a time."""

    #: Identifier used when persisting which agent produced a trajectory.
    name: str = "base"

    @abc.abstractmethod
    def decide(
        self,
        task: Task,
        tools: Sequence[ToolSpec],
        history: Sequence[Action],
    ) -> Action:
        """Return the next :class:`Action` given the interaction ``history``.

        ``history`` contains completed turns, each with the *possibly poisoned*
        observation the environment/controller produced. The returned action's
        ``turn`` should equal ``len(history)``.
        """

    def reset(self) -> None:
        """Hook for stateful agents to clear per-episode state (no-op by default)."""
