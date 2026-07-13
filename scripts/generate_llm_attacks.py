#!/usr/bin/env python
"""Generate local-Qwen attack samples and attacked rollouts for EvoGuard eval."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.attacks.llm_attack_generator import DEFAULT_LOCAL_QWEN_PATH, build_local_qwen_attack_generator
from evoguard.envs.text_tool_env import TOOLS, TextToolEnv
from evoguard.training.llm_rl import read_rollout_jsonl
from evoguard.types import AttackSample, SafetyAction, TrajectoryRecord, TrajectoryType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_LOCAL_QWEN_PATH)
    parser.add_argument("--output", default="data/attacks/llm_generated_round1.jsonl")
    parser.add_argument("--rollouts-output", default="data/eval/llm_generated_round1_attacked_rollouts.jsonl")
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--attacks-per-task", type=int, default=8)
    parser.add_argument("--min-total", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--task-indices", default="", help="Comma-separated preferred-tool task indices. Empty means all.")
    parser.add_argument("--failure-cases-jsonl", default="")
    parser.add_argument("--blocked-cases-jsonl", default="")
    parser.add_argument("--history-limit", type=int, default=12)
    args = parser.parse_args()

    env = TextToolEnv()
    tool_catalog = {tool.name: tool for tool in TOOLS}
    tasks = [task for task in env.get_tasks() if task.preferred_tool]
    if args.task_indices.strip():
        selected = [tasks[int(index)] for index in args.task_indices.split(",") if index.strip()]
    else:
        selected = tasks

    generator = build_local_qwen_attack_generator(
        args.model,
        attacks_per_task=args.attacks_per_task,
        tool_catalog=tool_catalog,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    successful_cases = _load_history(args.failure_cases_jsonl, args.history_limit)
    blocked_cases = _load_history(args.blocked_cases_jsonl, args.history_limit)
    attacks: list[AttackSample] = []
    failures: list[dict[str, str]] = []
    for index, task in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] generating task_id={task.task_id}", flush=True)
        try:
            task_attacks = generator.generate(
                [task],
                round_id=args.round_id,
                failure_cases=successful_cases,
                blocked_cases=blocked_cases,
            )
            attacks.extend(task_attacks)
            print(f"[{index}/{len(selected)}] generated={len(task_attacks)} total={len(attacks)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - batch generation should preserve successful tasks.
            failures.append({"task_id": task.task_id, "error": str(exc)})
            print(f"[{index}/{len(selected)}] failed task_id={task.task_id}: {exc}", flush=True)
    if len(attacks) < args.min_total:
        _write_jsonl(Path(args.output), [_attack_to_dict(attack) for attack in attacks])
        _write_jsonl(Path(args.rollouts_output), [_attack_to_rollout(env.name, attack).to_dict() for attack in attacks])
        raise RuntimeError(
            f"Generated {len(attacks)} attacks, expected at least {args.min_total}; failures={failures}"
        )

    attack_path = Path(args.output)
    rollout_path = Path(args.rollouts_output)
    _write_jsonl(attack_path, [_attack_to_dict(attack) for attack in attacks])
    _write_jsonl(rollout_path, [_attack_to_rollout(env.name, attack).to_dict() for attack in attacks])
    print(
        json.dumps(
            {
                "model": args.model,
                "attacks": len(attacks),
                "output": str(attack_path),
                "rollouts_output": str(rollout_path),
                "failed_tasks": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _attack_to_dict(attack: AttackSample) -> dict[str, object]:
    data = asdict(attack)
    data["gold_safety_action"] = attack.gold_safety_action.value
    data["task"]["gold_safety_action"] = attack.task.gold_safety_action.value
    return data


def _load_history(path: str, limit: int) -> list[TrajectoryRecord]:
    if not path:
        return []
    records = read_rollout_jsonl(Path(path))
    return records[-limit:]


def _attack_to_rollout(environment: str, attack: AttackSample) -> TrajectoryRecord:
    task = attack.task
    context = f"{task.user_task}\n{attack.injected_content}"
    true_span = (attack.insertion_offset, attack.insertion_offset + len(attack.injected_content))
    return TrajectoryRecord(
        task_id=task.task_id,
        round_id=attack.round_id,
        environment=environment,
        trajectory_type=TrajectoryType.ATTACKED_TOOL,
        user_task=context,
        tool_name=task.preferred_tool,
        tool_arguments=dict(task.tool_arguments),
        injected_content=attack.injected_content,
        injection_span=true_span,
        gold_safety_action=SafetyAction.REFUSE,
        agent_safety_action=SafetyAction.REFUSE,
        confidence=1.0,
        task_success=True,
        attack_success=False,
        attribution_score=0.0,
        attack_type=attack.attack_type,
        metadata={"attack_id": attack.attack_id, "target_tool": attack.target_tool},
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
