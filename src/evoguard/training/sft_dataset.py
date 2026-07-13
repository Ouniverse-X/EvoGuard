"""Export trajectory records into safety-judge SFT examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evoguard.types import TrajectoryRecord


def trajectory_to_safety_sft_example(record: TrajectoryRecord) -> dict[str, Any]:
    """Convert one rollout record into a chat-style supervised example."""

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a pre-execution tool safety judge. Return compact JSON with "
                    "action, confidence, and attribution_span."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_task": record.user_task,
                        "untrusted_context": record.injected_content,
                        "tool_call": {
                            "tool_name": record.tool_name,
                            "arguments": dict(record.tool_arguments),
                        }
                        if record.tool_name
                        else None,
                        "trajectory_type": record.trajectory_type.value,
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action": record.gold_safety_action.value,
                        "confidence": 1.0,
                        "attribution_span": list(record.injection_span) if record.injection_span else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        "metadata": {
            "task_id": record.task_id,
            "round_id": record.round_id,
            "trajectory_type": record.trajectory_type.value,
            "attack_type": record.attack_type,
        },
    }


def write_safety_sft_jsonl(records: list[TrajectoryRecord], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(trajectory_to_safety_sft_example(record), ensure_ascii=False) + "\n")
    return output_path
