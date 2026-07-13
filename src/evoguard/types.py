"""Shared data structures for the EvoGuard MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class SafetyAction(str, Enum):
    """Pre-execution safety action for a proposed tool call."""

    EXECUTE = "execute"
    ASK_CONFIRMATION = "ask_confirmation"
    REFUSE = "refuse"


class TrajectoryType(str, Enum):
    """EvoGuard rollout trajectory variants."""

    NO_TOOL = "no_tool"
    CLEAN_TOOL = "clean_tool"
    ATTACKED_TOOL = "attacked_tool"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    risky: bool = False
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    task_id: str
    user_task: str
    preferred_tool: str | None
    tool_arguments: Mapping[str, Any] = field(default_factory=dict)
    requires_tool: bool = True
    risky: bool = False
    gold_safety_action: SafetyAction = SafetyAction.EXECUTE


@dataclass(frozen=True)
class AttackSample:
    attack_id: str
    task: Task
    round_id: int
    attack_type: str
    target_tool: str
    injected_content: str
    insertion_offset: int
    gold_safety_action: SafetyAction = SafetyAction.REFUSE


@dataclass(frozen=True)
class DefenseDecision:
    action: SafetyAction
    confidence: float
    attribution_span: tuple[int, int] | None = None


@dataclass
class TrajectoryRecord:
    task_id: str
    round_id: int
    environment: str
    trajectory_type: TrajectoryType
    user_task: str
    tool_name: str | None
    tool_arguments: Mapping[str, Any]
    injected_content: str | None
    injection_span: tuple[int, int] | None
    gold_safety_action: SafetyAction
    agent_safety_action: SafetyAction
    confidence: float
    task_success: bool
    attack_success: bool
    attribution_score: float
    attack_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trajectory_type"] = self.trajectory_type.value
        data["gold_safety_action"] = self.gold_safety_action.value
        data["agent_safety_action"] = self.agent_safety_action.value
        return data
