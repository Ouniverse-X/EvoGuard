#!/usr/bin/env python
"""Sample API-backed EvoGuard attack injections for human review."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.attacks.llm_attack_generator import (
    DEFAULT_ATTACK_STYLES,
    build_api_attack_generator,
)
from evoguard.envs.text_tool_env import TOOLS, TextToolEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--attacks-per-task", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    env = TextToolEnv()
    tool_catalog = {tool.name: tool for tool in TOOLS}
    tasks = [task for task in env.get_tasks() if task.preferred_tool]
    task = tasks[args.task_index]

    generator = build_api_attack_generator(
        attacks_per_task=args.attacks_per_task,
        tool_catalog=tool_catalog,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    samples = generator.generate([task], round_id=args.round_id)
    style_counts = Counter(_extract_style(sample.attack_id) for sample in samples)

    print(
        json.dumps(
            {
                "model": "env:EVOGUARD_ATTACK_MODEL",
                "task_id": task.task_id,
                "task": task.user_task,
                "target_tool": task.preferred_tool,
                "style_distribution": dict(style_counts),
                "samples": [
                    {
                        "attack_id": sample.attack_id,
                        "attack_type": sample.attack_type,
                        "target_tool": sample.target_tool,
                        "injected_content": sample.injected_content,
                    }
                    for sample in samples
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

def _extract_style(attack_id: str) -> str:
    for style in DEFAULT_ATTACK_STYLES:
        if attack_id.endswith(f"_{style}"):
            return style
    return "unspecified"


if __name__ == "__main__":
    main()
