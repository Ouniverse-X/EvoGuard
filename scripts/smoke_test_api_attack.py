#!/usr/bin/env python
"""Smoke-test API-backed red-team attack generation on 10 EvoGuard tasks."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.attacks.llm_attack_generator import build_api_attack_generator
from evoguard.envs.text_tool_env import TOOLS, TextToolEnv
from evoguard.llm.client import LLMClientError
from evoguard.types import AttackSample


OUTPUT_PATH = Path("data/attacks/api_round1.jsonl")


def main() -> None:
    missing = [
        name
        for name in ("EVOGUARD_ATTACK_API_KEY", "EVOGUARD_ATTACK_BASE_URL", "EVOGUARD_ATTACK_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(
            "Missing API environment variables: "
            + ", ".join(missing)
            + ". Set them and rerun scripts/smoke_test_api_attack.py."
        )

    env = TextToolEnv()
    tasks = [task for task in env.get_tasks() if task.preferred_tool][:10]
    generator = build_api_attack_generator(
        split="api",
        attacks_per_task=5,
        tool_catalog={tool.name: tool for tool in TOOLS},
        max_tokens=1800,
        temperature=0.8,
    )

    attacks: list[AttackSample] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[api-smoke] generating {index}/{len(tasks)} task_id={task.task_id}", flush=True)
        attacks.extend(generator.generate([task], round_id=1))

    write_jsonl(OUTPUT_PATH, [_attack_to_dict(attack) for attack in attacks])
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "tasks": len(tasks),
                "attacks": len(attacks),
                "model": os.environ.get("EVOGUARD_ATTACK_MODEL", ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _attack_to_dict(attack: AttackSample) -> dict[str, object]:
    data = asdict(attack)
    data["gold_safety_action"] = attack.gold_safety_action.value
    data["task"]["gold_safety_action"] = attack.task.gold_safety_action.value
    return data


if __name__ == "__main__":
    try:
        main()
    except LLMClientError as exc:
        raise SystemExit(f"API attack smoke test failed: {exc}") from exc
