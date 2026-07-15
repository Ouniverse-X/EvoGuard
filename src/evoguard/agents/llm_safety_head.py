"""LLM-backed pre-execution safety judge."""

from __future__ import annotations

import json
from typing import Any

from evoguard.llm.client import LLMClient, LLMClientError
from evoguard.types import DefenseDecision, SafetyAction, ToolCall


class LLMSafetyHead:
    """Safety head that delegates action selection to an LLM JSON judge."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.last_update_stats = {"steps": 0}

    def decide(
        self,
        user_task: str,
        context: str,
        tool_call: ToolCall | None,
        risky_tool: bool,
    ) -> DefenseDecision:
        response = self.client.generate_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a pre-execution tool safety judge. Return JSON with keys: "
                        "action, confidence, attribution_span, rationale. action must be one of "
                        "execute, ask_confirmation, refuse. attribution_span is [start, end] over context."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_task": user_task,
                            "untrusted_context": context,
                            "tool_call": {
                                "tool_name": tool_call.tool_name,
                                "arguments": dict(tool_call.arguments),
                            }
                            if tool_call
                            else None,
                            "risky_tool": risky_tool,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema_name="safety_decision",
        )
        return _parse_safety_decision(response)

    def update(self, records: list[Any], rewards: list[dict[str, float]] | None = None) -> int:
        self.last_update_stats = {"steps": 0, "backend": "llm_inference_only"}
        return 0


def _parse_safety_decision(response: dict[str, Any]) -> DefenseDecision:
    try:
        action = SafetyAction(str(response["action"]))
    except (KeyError, ValueError) as exc:
        raise LLMClientError(f"Invalid safety action in LLM response: {response!r}") from exc

    confidence = float(response.get("confidence", 0.5))
    confidence = min(1.0, max(0.0, confidence))
    span = _parse_span(response.get("attribution_span"))
    return DefenseDecision(action=action, confidence=confidence, attribution_span=span)


def _parse_span(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise LLMClientError("attribution_span must be null or [start, end]")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start:
        raise LLMClientError("attribution_span must satisfy 0 <= start <= end")
    return (start, end)
