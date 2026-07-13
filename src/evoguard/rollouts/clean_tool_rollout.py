"""Clean tool trajectory collection."""

from __future__ import annotations

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.envs.text_tool_env import TextToolEnv
from evoguard.types import SafetyAction, Task, TrajectoryRecord, TrajectoryType


def rollout_clean_tool(agent: DefenseAgent, env: TextToolEnv, task: Task, round_id: int = 0) -> TrajectoryRecord:
    tool_call = env.propose_tool_call(task)
    risky_tool = env.is_tool_risky(tool_call.tool_name if tool_call else None)
    decision = agent.decide(task.user_task, context="", tool_call=tool_call, risky_tool=risky_tool)

    task_success = False
    if tool_call is None:
        task_success = not task.requires_tool
    elif decision.action in (SafetyAction.EXECUTE, SafetyAction.ASK_CONFIRMATION):
        task_success = env.execute(tool_call).success

    return TrajectoryRecord(
        task_id=task.task_id,
        round_id=round_id,
        environment=env.name,
        trajectory_type=TrajectoryType.CLEAN_TOOL,
        user_task=task.user_task,
        tool_name=tool_call.tool_name if tool_call else None,
        tool_arguments=dict(tool_call.arguments) if tool_call else {},
        injected_content=None,
        injection_span=None,
        gold_safety_action=task.gold_safety_action,
        agent_safety_action=decision.action,
        confidence=decision.confidence,
        task_success=task_success,
        attack_success=False,
        attribution_score=0.0,
    )
