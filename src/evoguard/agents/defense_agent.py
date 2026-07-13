"""Defense agent wrapper for pre-execution tool-call decisions."""

from __future__ import annotations

from typing import Any

from evoguard.agents.safety_head import RuleBasedSafetyHead
from evoguard.types import DefenseDecision, ToolCall, TrajectoryRecord


class DefenseAgent:
    """Thin wrapper around a replaceable safety head."""

    def __init__(self, safety_head: RuleBasedSafetyHead | None = None) -> None:
        self.safety_head = safety_head or RuleBasedSafetyHead()
        self.update_steps = 0

    def decide(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> DefenseDecision:
        return self.safety_head.decide(user_task, context, tool_call, risky_tool)

    def mark_updated(self) -> None:
        self.update_steps += 1

    def update(self, records: list[TrajectoryRecord], rewards: list[dict[str, float]] | None = None) -> int:
        update_fn = getattr(self.safety_head, "update", None)
        if update_fn is None:
            self.mark_updated()
            return 0
        updates = int(update_fn(records, rewards))
        self.mark_updated()
        return updates

    def last_update_stats(self) -> dict[str, Any]:
        stats = getattr(self.safety_head, "last_update_stats", None)
        if stats is None:
            return {}
        if hasattr(stats, "__dataclass_fields__"):
            return {field: getattr(stats, field) for field in stats.__dataclass_fields__}
        if isinstance(stats, dict):
            return dict(stats)
        return {}
