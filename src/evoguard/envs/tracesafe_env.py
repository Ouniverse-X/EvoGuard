"""TraceSafe-Bench adapter for EvoGuard trajectory records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evoguard.types import SafetyAction, ToolCall, ToolSpec, TrajectoryRecord, TrajectoryType


TRACE_SAFE_CATEGORIES = {
    "golden_0_benign": ("BENIGN", "benign"),
    "golden_1_PromptInjectionIn": ("PROMPT_INJECTION", "PromptInjectionIn"),
    "golden_2_PromptInjectionOut": ("PROMPT_INJECTION", "PromptInjectionOut"),
    "golden_3_UserInfoLeak": ("PRIVACY_LEAKAGE", "UserInfoLeak"),
    "golden_4_ApiKeyLeak": ("PRIVACY_LEAKAGE", "ApiKeyLeak"),
    "golden_5_DataLeak": ("PRIVACY_LEAKAGE", "DataLeak"),
    "golden_6_AmbiguousArg": ("HALLUCINATION", "AmbiguousArg"),
    "golden_7_HallucinatedTool": ("HALLUCINATION", "HallucinatedTool"),
    "golden_8_HallucinatedArgValue": ("HALLUCINATION", "HallucinatedArgValue"),
    "golden_9_RedundantArg": ("HALLUCINATION", "RedundantArg"),
    "golden_10_MissingTypeHint": ("HALLUCINATION", "MissingTypeHint"),
    "golden_11_VersionConflict": ("INTERFACE_INCONSISTENCIES", "VersionConflict"),
    "golden_12_DescriptionMismatch": ("INTERFACE_INCONSISTENCIES", "DescriptionMismatch"),
}


@dataclass(frozen=True)
class TraceSafeSample:
    row: dict[str, Any]
    source: str
    index: int


class TraceSafeEnv:
    """Convert TraceSafe-Bench JSONL records into EvoGuard safety rollouts."""

    name = "tracesafe"

    def __init__(self, samples: Iterable[TraceSafeSample] | None = None) -> None:
        self.samples = list(samples or [])
        self.tools = {tool.name: tool for tool in self._collect_tools(self.samples)}

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "TraceSafeEnv":
        samples: list[TraceSafeSample] = []
        for path in paths:
            samples.extend(read_tracesafe_samples(Path(path)))
        return cls(samples)

    def to_records(self, *, round_id: int = 0, split: str = "tracesafe") -> list[TrajectoryRecord]:
        return [
            sample_to_trajectory_record(sample, round_id=round_id, split=split)
            for sample in self.samples
        ]

    def propose_tool_call(self, record: TrajectoryRecord) -> ToolCall | None:
        if record.tool_name is None:
            return None
        return ToolCall(record.tool_name, dict(record.tool_arguments))

    def is_tool_risky(self, tool_name: str | None) -> bool:
        return tool_name in self.tools if tool_name else False

    @staticmethod
    def _collect_tools(samples: list[TraceSafeSample]) -> list[ToolSpec]:
        tools: dict[str, str] = {}
        for sample in samples:
            trace = new_trace(sample.row)
            for spec in trace.get("tool_lists") or []:
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("name") or "").strip()
                if name:
                    tools[name] = str(spec.get("description") or spec.get("prototype") or "")
            target = mutation_metadata(sample.row).get("target_tool")
            if target:
                tools.setdefault(str(target), "TraceSafe mutation target tool.")
        return [ToolSpec(name=name, description=description, risky=True) for name, description in sorted(tools.items())]


def read_tracesafe_samples(path: Path) -> list[TraceSafeSample]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [TraceSafeSample(row=row, source=str(path), index=index) for index, row in enumerate(rows)]


def sample_to_trajectory_record(
    sample: TraceSafeSample,
    *,
    round_id: int,
    split: str,
) -> TrajectoryRecord:
    row = sample.row
    trace = new_trace(row)
    meta = mutation_metadata(row)
    golden = golden_meta(row)
    category = category_name(row, sample)
    root_vector = root_vector_name(row, sample)
    is_benign = str(golden.get("type") or "").lower() == "pure_benign" or category.lower() == "benign"
    gold = SafetyAction.EXECUTE if is_benign else SafetyAction.REFUSE
    trajectory_type = TrajectoryType.CLEAN_TOOL if is_benign else TrajectoryType.ATTACKED_TOOL
    user_query = str(trace.get("user_query") or "")
    trace_text = format_trace_context(trace)
    diff_text = format_difference(row.get("difference"))
    rationale = str(meta.get("rationale") or "")
    injected_content = None if is_benign else "\n\n".join(part for part in (diff_text, rationale) if part)
    if not injected_content and not is_benign:
        injected_content = trace_text
    user_task = "\n\n".join(
        part
        for part in (
            f"User query: {user_query}" if user_query else "",
            "Tool-use trace:\n" + trace_text if trace_text else "",
            "Mutation evidence:\n" + injected_content if injected_content else "",
        )
        if part
    )
    tool_name = infer_tool_name(row, trace)
    tool_arguments = infer_tool_arguments(row, trace)
    injection_span = infer_span(user_task, injected_content)
    task_id = str(golden.get("id") or row.get("id") or f"tracesafe_{Path(sample.source).stem}_{sample.index:04d}")

    return TrajectoryRecord(
        task_id=task_id,
        round_id=round_id,
        environment="tracesafe",
        trajectory_type=trajectory_type,
        user_task=user_task,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        injected_content=injected_content,
        injection_span=injection_span,
        gold_safety_action=gold,
        agent_safety_action=gold,
        confidence=1.0,
        task_success=True,
        attack_success=False,
        attribution_score=1.0 if injection_span else 0.0,
        attack_type=category,
        metadata={
            "source": sample.source,
            "source_index": sample.index,
            "split": split,
            "root_vector": root_vector,
            "category": category,
            "mutation_category": row.get("mutation_category"),
            "mutator_name": meta.get("mutator_name"),
            "source_model": golden.get("source_model") or trace.get("agent_model"),
            "target_tool": meta.get("target_tool"),
        },
    )


def new_trace(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("new_trace") or row.get("trace") or row.get("original_trace") or {}
    return trace if isinstance(trace, dict) else {}


def mutation_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("mutation_metadata") or {}
    return value if isinstance(value, dict) else {}


def golden_meta(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("golden_meta") or {}
    return value if isinstance(value, dict) else {}


def category_name(row: dict[str, Any], sample: TraceSafeSample) -> str:
    golden = golden_meta(row)
    meta = mutation_metadata(row)
    value = golden.get("category") or meta.get("mutator_name")
    if value:
        return str(value)
    return TRACE_SAFE_CATEGORIES.get(Path(sample.source).stem, ("", Path(sample.source).stem))[1]


def root_vector_name(row: dict[str, Any], sample: TraceSafeSample) -> str:
    value = row.get("mutation_category")
    if value:
        return str(value)
    return TRACE_SAFE_CATEGORIES.get(Path(sample.source).stem, ("UNKNOWN", ""))[0]


def format_trace_context(trace: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("domain", "environment", "scenario_description"):
        value = trace.get(key)
        if value:
            parts.append(f"{key}: {value}")
    tool_lines = []
    for tool in trace.get("tool_lists") or []:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description") or tool.get("prototype")
            if name:
                tool_lines.append(f"- {name}: {description}")
    if tool_lines:
        parts.append("tools:\n" + "\n".join(tool_lines))
    steps = []
    for step in trace.get("trace") or []:
        if isinstance(step, dict):
            role = step.get("role") or step.get("type") or "step"
            content = step.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, sort_keys=True)
            steps.append(f"{role}: {content}")
        else:
            steps.append(str(step))
    if steps:
        parts.append("trace:\n" + "\n".join(steps))
    return "\n\n".join(parts)


def format_difference(value: Any) -> str:
    if value in (None, {}, []):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def infer_tool_name(row: dict[str, Any], trace: dict[str, Any]) -> str | None:
    target = mutation_metadata(row).get("target_tool")
    if target:
        return str(target)
    for step in reversed(trace.get("trace") or []):
        text = json.dumps(step, ensure_ascii=False, sort_keys=True) if isinstance(step, dict) else str(step)
        match = re.search(r'"(?:tool_name|name|function)"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    tools = trace.get("tool_lists") or []
    if tools and isinstance(tools[0], dict) and tools[0].get("name"):
        return str(tools[0]["name"])
    return None


def infer_tool_arguments(row: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    meta = mutation_metadata(row)
    args: dict[str, Any] = {}
    if meta.get("internal_meta") is not None:
        args["internal_meta"] = meta["internal_meta"]
    diff = row.get("difference")
    if diff:
        args["difference"] = diff
    for step in reversed(trace.get("trace") or []):
        parsed = parse_jsonish(step.get("content") if isinstance(step, dict) else step)
        if isinstance(parsed, dict):
            for key in ("arguments", "args", "parameters", "params"):
                if isinstance(parsed.get(key), dict):
                    args.setdefault("current_action_arguments", parsed[key])
                    return args
    return args


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def infer_span(text: str, needle: str | None) -> tuple[int, int] | None:
    if not needle:
        return None
    start = text.find(needle)
    if start < 0:
        return None
    return start, start + len(needle)
