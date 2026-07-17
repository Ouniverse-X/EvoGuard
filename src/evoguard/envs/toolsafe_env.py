"""ToolSafe/TS-Bench dataset adapter for EvoGuard trajectories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evoguard.types import SafetyAction, Task, ToolCall, ToolSpec, TrajectoryRecord, TrajectoryType


@dataclass(frozen=True)
class ToolSafeSample:
    """Normalized external tool-safety sample."""

    row: dict[str, Any]
    source: str
    index: int


class ToolSafeEnv:
    """Adapter that converts ToolSafe/TraceSafe-style JSON records into EvoGuard records.

    The public ToolSafe TS-Bench files are JSON files hosted through Git LFS. Once
    downloaded locally, this adapter accepts either JSON arrays, objects with
    common data-list keys, or JSONL files. Field extraction is intentionally
    permissive because TS-Bench subsets use different source schemas.
    """

    name = "toolsafe"

    def __init__(self, samples: Iterable[ToolSafeSample] | None = None) -> None:
        self.samples = list(samples or [])
        self.tools = {tool.name: tool for tool in self._collect_tools(self.samples)}

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "ToolSafeEnv":
        samples: list[ToolSafeSample] = []
        for path in paths:
            samples.extend(read_toolsafe_samples(Path(path)))
        return cls(samples)

    def get_tasks(self) -> list[Task]:
        return [sample_to_task(sample, environment=self.name) for sample in self.samples]

    def to_records(self, *, round_id: int = 0, split: str = "train") -> list[TrajectoryRecord]:
        return [
            sample_to_trajectory_record(sample, round_id=round_id, split=split, environment=self.name)
            for sample in self.samples
        ]

    def split_records(
        self,
        *,
        heldout_ratio: float = 0.2,
        round_id: int = 0,
    ) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
        records = self.to_records(round_id=round_id)
        return split_records_stratified(records, heldout_ratio=heldout_ratio)

    def propose_tool_call(self, task: Task) -> ToolCall | None:
        if task.preferred_tool is None:
            return None
        return ToolCall(task.preferred_tool, dict(task.tool_arguments))

    def execute(self, tool_call: ToolCall) -> Any:
        return _ToolSafeResult(success=True, output=f"Dataset replay for {tool_call.tool_name}")

    def is_tool_risky(self, tool_name: str | None) -> bool:
        if tool_name is None:
            return False
        tool = self.tools.get(tool_name)
        return bool(tool and tool.risky)

    def risk_type(self, tool_name: str | None) -> str | None:
        if tool_name is None:
            return None
        return self.tools.get(tool_name, ToolSpec(tool_name, "")).description or None

    @staticmethod
    def _collect_tools(samples: list[ToolSafeSample]) -> list[ToolSpec]:
        names = sorted({extract_tool_name(sample.row) for sample in samples if extract_tool_name(sample.row)})
        return [ToolSpec(name=name, description="Imported ToolSafe dataset tool.", risky=True) for name in names]


@dataclass(frozen=True)
class _ToolSafeResult:
    success: bool
    output: str
    risk_triggered: bool = False


def read_toolsafe_samples(path: Path) -> list[ToolSafeSample]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _read_rows(path)
    return [ToolSafeSample(row=row, source=str(path), index=index) for index, row in enumerate(rows)]


def sample_to_task(sample: ToolSafeSample, *, environment: str = "toolsafe") -> Task:
    row = sample.row
    gold = infer_gold_action(row, source=sample.source)
    return Task(
        task_id=extract_task_id(row, sample),
        user_task=extract_user_task(row),
        preferred_tool=extract_tool_name(row),
        tool_arguments=extract_tool_arguments(row),
        requires_tool=extract_tool_name(row) is not None,
        risky=gold != SafetyAction.EXECUTE,
        gold_safety_action=gold,
    )


def sample_to_trajectory_record(
    sample: ToolSafeSample,
    *,
    round_id: int,
    split: str,
    environment: str = "toolsafe",
) -> TrajectoryRecord:
    return transform_record(sample, round_id=round_id, split=split, environment=environment)


def transform_record(
    sample: ToolSafeSample,
    *,
    round_id: int,
    split: str,
    environment: str = "toolsafe",
) -> TrajectoryRecord:
    row = sample.row
    is_harmful = infer_is_harmful(row, source=sample.source)
    gold = SafetyAction.REFUSE if is_harmful else SafetyAction.EXECUTE
    tool_name = extract_tool_name(row)
    user_task = extract_user_task(row)
    injected_content = extract_injected_content(row)
    if is_harmful and injected_content is None:
        injected_content = user_task
    if is_harmful:
        trajectory_type = TrajectoryType.ATTACKED_TOOL
    elif tool_name:
        trajectory_type = TrajectoryType.CLEAN_TOOL
    else:
        trajectory_type = TrajectoryType.NO_TOOL
    agent_action = parse_action(
        first_present(row, ("agent_safety_action", "prediction", "predicted_label", "guard_action", "model_action")),
        default=gold,
    )
    injection_span = infer_injection_span(user_task, injected_content)
    attack_success = trajectory_type == TrajectoryType.ATTACKED_TOOL and agent_action == SafetyAction.EXECUTE
    task_success = infer_task_success(trajectory_type, agent_action, attack_success)

    return TrajectoryRecord(
        task_id=extract_task_id(row, sample),
        round_id=round_id,
        environment=environment,
        trajectory_type=trajectory_type,
        user_task=user_task,
        tool_name=tool_name,
        tool_arguments=extract_tool_arguments(row),
        injected_content=injected_content,
        injection_span=injection_span,
        gold_safety_action=gold,
        agent_safety_action=agent_action,
        confidence=float_or_default(first_present(row, ("confidence", "score", "probability")), 1.0),
        task_success=task_success,
        attack_success=attack_success,
        attribution_score=1.0 if injection_span else 0.0,
        attack_type=extract_attack_type(row),
        metadata={
            "source": sample.source,
            "source_index": sample.index,
            "split": split,
            "raw_keys": sorted(row.keys()),
        },
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("version https://git-lfs.github.com/spec"):
        raise ValueError(f"{path} is a Git LFS pointer. Download the LFS file content before importing it.")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        for key in ("data", "samples", "examples", "instances", "records", "trajectories"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [parsed]
    raise ValueError(f"Unsupported ToolSafe file structure: {path}")


def extract_task_id(row: dict[str, Any], sample: ToolSafeSample) -> str:
    value = first_present(row, ("task_id", "id", "sample_id", "qid", "uid", "trajectory_id"))
    return str(value) if value not in (None, "") else f"toolsafe_{sample.index:06d}"


def extract_user_task(row: dict[str, Any]) -> str:
    value = first_present(
        row,
        (
            "user_task",
            "instruction",
            "task",
            "query",
            "prompt",
            "goal",
            "user_request",
            "request",
        ),
    )
    if isinstance(value, dict):
        value = first_present(value, ("content", "text", "instruction", "query"))
    if value is None:
        value = extract_trace_text(row)
    return str(value or "")


def extract_trace_text(row: dict[str, Any]) -> str:
    trace = first_present(row, ("trajectory", "trace", "history", "messages", "conversation", "steps"))
    if isinstance(trace, list):
        parts: list[str] = []
        for step in trace:
            if isinstance(step, dict):
                role = str(first_present(step, ("role", "type", "name")) or "step")
                content = first_present(step, ("content", "text", "message", "observation", "thought"))
                if content is None:
                    content = json.dumps(step, ensure_ascii=False, sort_keys=True)
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(step))
        return "\n".join(parts)
    if isinstance(trace, dict):
        return json.dumps(trace, ensure_ascii=False, sort_keys=True)
    return str(trace or "")


def extract_tool_name(row: dict[str, Any]) -> str | None:
    parsed_call = _parse_tool_call(first_present(row, ("tool_call", "action", "current_action", "proposed_action")))
    if parsed_call:
        return parsed_call.tool_name
    value = first_present(row, ("tool_name", "tool", "function", "api", "action_name"))
    if isinstance(value, dict):
        value = first_present(value, ("name", "tool_name", "function", "api"))
    return str(value) if value not in (None, "") else None


def extract_tool_arguments(row: dict[str, Any]) -> dict[str, Any]:
    parsed_call = _parse_tool_call(first_present(row, ("tool_call", "action", "current_action", "proposed_action")))
    if parsed_call:
        return dict(parsed_call.arguments)
    value = first_present(row, ("tool_arguments", "arguments", "args", "parameters", "params"))
    return normalize_tool_arguments(value)


def _parse_tool_call(value: Any) -> ToolCall | None:
    if isinstance(value, ToolCall):
        return value
    if isinstance(value, dict):
        name = first_present(value, ("tool_name", "tool", "name", "function", "api", "action"))
        arguments = first_present(value, ("arguments", "args", "parameters", "params", "action_input", "input"))
        if name in (None, ""):
            return None
        return ToolCall(str(name), normalize_tool_arguments(arguments))
    if not isinstance(value, str):
        return None

    action_match = re.search(r"(?:^|\n)\s*(?:\(\d+\)\s*)?Action\s*:\s*([^\n\r]+)", value, flags=re.IGNORECASE)
    if not action_match:
        return None
    tool_name = action_match.group(1).strip().strip("` ")
    tool_name = tool_name.split()[0] if tool_name else ""
    input_match = re.search(
        r"(?:^|\n)\s*(?:\(\d+\)\s*)?Action\s+Input\s*:\s*(.*?)(?=\n\s*(?:\(\d+\)\s*)?[A-Z][A-Za-z ]+\s*:|\Z)",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    arguments = normalize_tool_arguments(input_match.group(1).strip() if input_match else None)
    return ToolCall(tool_name, arguments) if tool_name else None


def normalize_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
    return {}


def extract_injected_content(row: dict[str, Any]) -> str | None:
    value = first_present(
        row,
        (
            "injected_content",
            "attack",
            "attack_prompt",
            "attack_instruction",
            "adversarial_content",
            "malicious_instruction",
            "indirect_prompt",
        ),
    )
    return str(value) if value not in (None, "") else None


def extract_attack_type(row: dict[str, Any]) -> str | None:
    value = first_present(row, ("attack_type", "risk_category", "category", "harm_type", "label_type"))
    return str(value) if value not in (None, "") else None


def infer_gold_action(row: dict[str, Any], *, source: str | None = None) -> SafetyAction:
    if infer_is_harmful(row, source=source):
        return SafetyAction.REFUSE
    if infer_is_benign(row, source=source):
        return SafetyAction.EXECUTE
    if "gold_safety_action" in row:
        return parse_action(row["gold_safety_action"], default=SafetyAction.EXECUTE)
    if "score" in row:
        return score_to_action(row["score"])
    for key in ("is_safe", "safe"):
        if key in row:
            return SafetyAction.EXECUTE if bool(row[key]) else SafetyAction.REFUSE
    for key in ("is_harmful", "harmful", "unsafe", "attack_success"):
        if key in row:
            return SafetyAction.REFUSE if bool(row[key]) else SafetyAction.EXECUTE
    value = first_present(row, ("safety_label", "label", "judgement", "judgment", "target", "answer"))
    return parse_action(value, default=SafetyAction.EXECUTE)


def infer_is_harmful(row: dict[str, Any], *, source: str | None = None) -> bool:
    source_lower = str(source or "").lower()
    if "harmful" in source_lower or "unsafe" in source_lower:
        return True
    for key in ("is_harmful", "harmful", "unsafe"):
        if key in row:
            return bool(row[key])
    if "score" in row:
        return score_to_action(row["score"]) == SafetyAction.REFUSE
    value = first_present(row, ("safety_label", "label", "judgement", "judgment", "target", "answer"))
    return parse_action(value, default=SafetyAction.EXECUTE) == SafetyAction.REFUSE


def infer_is_benign(row: dict[str, Any], *, source: str | None = None) -> bool:
    source_lower = str(source or "").lower()
    if "benign" in source_lower:
        return True
    for key in ("is_safe", "safe"):
        if key in row:
            return bool(row[key])
    if "score" in row:
        return score_to_action(row["score"]) == SafetyAction.EXECUTE
    value = first_present(row, ("safety_label", "label", "judgement", "judgment", "target", "answer"))
    return parse_action(value, default=SafetyAction.REFUSE) == SafetyAction.EXECUTE


def infer_trajectory_type(
    row: dict[str, Any],
    gold: SafetyAction,
    tool_name: str | None,
    injected_content: str | None,
) -> TrajectoryType:
    value = first_present(row, ("trajectory_type", "traj_type", "rollout_type"))
    if value is not None:
        try:
            return TrajectoryType(str(value))
        except ValueError:
            pass
    if injected_content or gold == SafetyAction.REFUSE:
        return TrajectoryType.ATTACKED_TOOL
    if tool_name is None:
        return TrajectoryType.NO_TOOL
    return TrajectoryType.CLEAN_TOOL


def infer_injection_span(user_task: str, injected_content: str | None) -> tuple[int, int] | None:
    if not injected_content:
        return None
    start = user_task.find(injected_content)
    if start < 0:
        start = max(0, len(user_task) - len(injected_content))
    return start, start + len(injected_content)


def infer_task_success(
    trajectory_type: TrajectoryType,
    agent_action: SafetyAction,
    attack_success: bool,
) -> bool:
    if trajectory_type == TrajectoryType.ATTACKED_TOOL:
        return not attack_success
    if trajectory_type == TrajectoryType.CLEAN_TOOL:
        return agent_action in (SafetyAction.EXECUTE, SafetyAction.ASK_CONFIRMATION)
    return agent_action == SafetyAction.EXECUTE


def split_records_stratified(
    records: list[TrajectoryRecord],
    *,
    heldout_ratio: float,
) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
    if not records:
        return [], []
    train: list[TrajectoryRecord] = []
    heldout: list[TrajectoryRecord] = []
    for trajectory_type in TrajectoryType:
        group = [record for record in records if record.trajectory_type == trajectory_type]
        if not group:
            continue
        heldout_count = max(1, int(round(len(group) * heldout_ratio))) if len(group) > 1 else 0
        split_at = max(0, len(group) - heldout_count)
        train.extend(group[:split_at])
        heldout.extend(group[split_at:])
    return train, heldout


def parse_action(value: Any, *, default: SafetyAction) -> SafetyAction:
    normalized = str(value or "").strip().lower()
    if normalized in {"execute", "allow", "allowed", "safe", "benign", "0"}:
        return SafetyAction.EXECUTE
    if normalized in {"ask_confirmation", "ask-confirmation", "confirm", "confirmation", "risky"}:
        return SafetyAction.ASK_CONFIRMATION
    if normalized in {"refuse", "block", "blocked", "unsafe", "harmful", "malicious", "attack", "1"}:
        return SafetyAction.REFUSE
    return default


def score_to_action(value: Any) -> SafetyAction:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return parse_action(value, default=SafetyAction.EXECUTE)
    return SafetyAction.EXECUTE if score == 0.0 else SafetyAction.REFUSE


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
