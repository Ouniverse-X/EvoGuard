#!/usr/bin/env python3
"""Convert generated attack samples into EvoGuard attacked-tool rollout JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/attacks/llm_generated_round1.jsonl"
DEFAULT_OUTPUT = "data/attacks/llm_generated_round1_rollouts.jsonl"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = [attack_to_rollout(row, args) for row in read_jsonl(input_path)]
    write_jsonl(output_path, records)
    print(f"converted {len(records)} attacks: {input_path} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AttackSample JSONL to attacked-tool TrajectoryRecord JSONL.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input AttackSample JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output TrajectoryRecord JSONL path.")
    parser.add_argument("--environment", default=None, help="Override output environment. Defaults to source metadata/task.")
    parser.add_argument("--agent-action", default="refuse", choices=("execute", "ask_confirmation", "refuse"))
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--round-id", type=int, default=None, help="Override source round_id.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def attack_to_rollout(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    task = row.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"attack {row.get('attack_id', '<unknown>')} is missing object field 'task'")

    injected_content = str(row.get("injected_content") or "")
    task_id = str(task.get("task_id") or row.get("task_id") or row.get("attack_id"))
    tool_name = str(row.get("target_tool") or task.get("preferred_tool") or row.get("tool_name") or "")
    insertion_offset = int(row.get("insertion_offset") or len(str(task.get("user_task") or "")) + 1)

    return {
        "task_id": task_id,
        "round_id": int(args.round_id if args.round_id is not None else row.get("round_id", 0)),
        "environment": args.environment or _environment(row, task),
        "trajectory_type": "attacked_tool",
        "user_task": str(task.get("user_task") or row.get("user_task") or ""),
        "tool_name": tool_name or None,
        "tool_arguments": dict(task.get("tool_arguments") or row.get("tool_arguments") or {}),
        "injected_content": injected_content,
        "injection_span": [insertion_offset, insertion_offset + len(injected_content)],
        "gold_safety_action": "refuse",
        "agent_safety_action": args.agent_action,
        "confidence": args.confidence,
        "task_success": True,
        "attack_success": False,
        "attribution_score": 1.0 if injected_content else 0.0,
        "attack_type": row.get("attack_type"),
        "metadata": {
            "attack_id": row.get("attack_id"),
            "target_tool": row.get("target_tool"),
            "source_gold_safety_action": row.get("gold_safety_action"),
            "source_task_gold_safety_action": task.get("gold_safety_action"),
        },
    }


def _environment(row: dict[str, Any], task: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("environment"):
        return str(metadata["environment"])
    task_metadata = task.get("metadata")
    if isinstance(task_metadata, dict) and task_metadata.get("environment"):
        return str(task_metadata["environment"])
    task_id = str(task.get("task_id") or row.get("task_id") or "")
    return "toolsafe" if task_id.startswith("toolsafe_") else "text_tool"


if __name__ == "__main__":
    main()
