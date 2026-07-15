#!/usr/bin/env python
"""Run API-backed online adaptive red-team generation against a defender."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.agents.defense_agent import DefenseAgent
from evoguard.attacks.attack_generator import AttackMemory
from evoguard.attacks.llm_attack_generator import build_api_attack_generator
from evoguard.envs.text_tool_env import TOOLS, TextToolEnv
from evoguard.evaluation.baselines import BASELINES, build_baseline_agent
from evoguard.evaluation.metrics import compute_metrics
from evoguard.rewards.combined_reward import RewardWeights, combined_reward
from evoguard.rollouts.clean_tool_rollout import rollout_clean_tool
from evoguard.rollouts.no_tool_rollout import rollout_no_tool
from evoguard.types import AttackSample, Task, TrajectoryRecord


def main() -> None:
    args = parse_args()
    env = TextToolEnv()
    tasks = [task for task in env.get_tasks() if task.preferred_tool]
    if args.task_indices.strip():
        tasks = [tasks[int(index)] for index in args.task_indices.split(",") if index.strip()]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    agent = build_baseline_agent(args.defender)
    generator = build_api_attack_generator(
        split=args.split,
        attacks_per_task=args.attacks_per_task,
        tool_catalog={tool.name: tool for tool in TOOLS},
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    memory = AttackMemory()
    successful_attacks: list[AttackSample] = []
    evaluated_records: list[TrajectoryRecord] = []
    failures: list[dict[str, str]] = []
    round_summaries: list[dict[str, object]] = []

    for round_id in range(args.start_round, args.start_round + args.rounds):
        print(f"[online-redteam] round={round_id} tasks={len(tasks)}", flush=True)
        try:
            result = generator.generate_with_feedback(
                tasks,
                round_id=round_id,
                attack_memory=memory,
                defense_agent=agent,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001 - keep long online runs from losing completed rounds.
            failures.append({"round_id": str(round_id), "task_id": "*", "error": str(exc)})
            print(f"[online-redteam] failed round={round_id}: {exc}", flush=True)
            continue

        successful_attacks.extend(result.samples)
        attacked_records = result.successful_cases + result.blocked_cases
        evaluated_records.extend(attacked_records)
        round_successes = len(result.successful_cases)
        round_blocked = len(result.blocked_cases)
        round_candidates = round_successes + round_blocked
        train_pool = build_train_pool(
            agent=agent,
            env=env,
            tasks=tasks,
            attacked_records=attacked_records,
            round_id=round_id,
            include_no_tool=not args.no_no_tool,
            include_clean_tool=not args.no_clean_tool,
            train_on_successful_attacks_only=args.train_on_successful_attacks_only,
        )
        reward_batch = [
            combined_reward(record, current_round=round_id, max_round=max(1, args.rounds), weights=RewardWeights())
            for record in train_pool
        ]
        policy_updates = 0 if args.no_defender_update else agent.update(train_pool, reward_batch)
        update_stats = agent.last_update_stats()
        metrics = compute_metrics(train_pool)
        print(
            "[online-redteam] "
            f"round={round_id} candidates={round_candidates} "
            f"successes={round_successes} blocked={round_blocked} "
            f"policy_updates={policy_updates}",
            flush=True,
        )

        round_summaries.append(
            {
                "round_id": round_id,
                "candidates": round_candidates,
                "successful_attacks": round_successes,
                "blocked_attacks": round_blocked,
                "attack_success_rate": round_successes / round_candidates if round_candidates else 0.0,
                "train_records": len(train_pool),
                "policy_updates": policy_updates,
                "update_stats": update_stats,
                "metrics": metrics,
            }
        )

    write_jsonl(Path(args.attacks_output), [attack_to_dict(attack) for attack in successful_attacks])
    write_jsonl(Path(args.rollouts_output), [record.to_dict() for record in evaluated_records])
    summary = {
        "method": "api_online_adaptive_redteam",
        "defender": args.defender,
        "model": "env:EVOGUARD_ATTACK_MODEL",
        "rounds": args.rounds,
        "tasks": len(tasks),
        "attacks_per_task": args.attacks_per_task,
        "successful_attacks": len(successful_attacks),
        "evaluated_candidates": len(evaluated_records),
        "attack_success_rate": len(successful_attacks) / len(evaluated_records) if evaluated_records else 0.0,
        "defender_update": not args.no_defender_update,
        "rounds_summary": round_summaries,
        "failures": failures,
        "outputs": {
            "attacks": args.attacks_output,
            "rollouts": args.rollouts_output,
            "summary": args.summary_output,
        },
    }
    write_json(Path(args.summary_output), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--attacks-per-task", type=int, default=6)
    parser.add_argument("--split", default="online_redteam")
    parser.add_argument("--defender", choices=BASELINES, default="rule_based_guard")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit preferred-tool tasks. 0 means all.")
    parser.add_argument("--task-indices", default="", help="Comma-separated preferred-tool task indices. Empty means all.")
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-defender-update", action="store_true")
    parser.add_argument("--no-no-tool", action="store_true")
    parser.add_argument("--no-clean-tool", action="store_true")
    parser.add_argument("--train-on-successful-attacks-only", action="store_true")
    parser.add_argument("--attacks-output", default="data/attacks/api_online_redteam_successes.jsonl")
    parser.add_argument("--rollouts-output", default="data/eval/api_online_redteam_rollouts.jsonl")
    parser.add_argument("--summary-output", default="outputs/logs/api_online_redteam_summary.json")
    return parser.parse_args()


def build_train_pool(
    *,
    agent: DefenseAgent,
    env: TextToolEnv,
    tasks: list[Task],
    attacked_records: list[TrajectoryRecord],
    round_id: int,
    include_no_tool: bool,
    include_clean_tool: bool,
    train_on_successful_attacks_only: bool,
) -> list[TrajectoryRecord]:
    train_pool: list[TrajectoryRecord] = []
    for task in tasks:
        if include_no_tool:
            train_pool.append(rollout_no_tool(agent, task, round_id=round_id))
        if include_clean_tool:
            train_pool.append(rollout_clean_tool(agent, env, task, round_id=round_id))
    if train_on_successful_attacks_only:
        train_pool.extend(record for record in attacked_records if record.attack_success)
    else:
        train_pool.extend(attacked_records)
    return train_pool


def attack_to_dict(attack: AttackSample) -> dict[str, object]:
    data = asdict(attack)
    data["gold_safety_action"] = attack.gold_safety_action.value
    data["task"]["gold_safety_action"] = attack.task.gold_safety_action.value
    return data


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
