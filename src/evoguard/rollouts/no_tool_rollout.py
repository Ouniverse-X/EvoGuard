"""No-tool trajectory collection."""

from __future__ import annotations

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.types import SafetyAction, Task, TrajectoryRecord, TrajectoryType


def rollout_no_tool(agent: DefenseAgent, task: Task, round_id: int = 0) -> TrajectoryRecord:
    decision = agent.decide(task.user_task, context="", tool_call=None, risky_tool=False)
    task_success = not task.requires_tool or decision.action == SafetyAction.EXECUTE
    return TrajectoryRecord(
        task_id=task.task_id,
        round_id=round_id,
        environment="text_tool",
        trajectory_type=TrajectoryType.NO_TOOL,
        user_task=task.user_task,
        tool_name=None,
        tool_arguments={},
        injected_content=None,
        injection_span=None,
        gold_safety_action=SafetyAction.EXECUTE,
        agent_safety_action=decision.action,
        confidence=decision.confidence,
        task_success=task_success,
        attack_success=False,
        attribution_score=0.0,
    )
