#!/usr/bin/env python
"""Train a Qwen-LoRA defender with API-backed online red-team co-evolution."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoguard.attacks.attack_filter import extract_failures
from evoguard.attacks.attack_generator import AttackMemory, build_attack_generator
from evoguard.attacks.llm_attack_generator import build_api_attack_generator
from evoguard.envs.text_tool_env import TOOLS, TextToolEnv
from evoguard.evaluation.metrics import compute_metrics
from evoguard.rewards.combined_reward import RewardWeights, combined_reward
from evoguard.training.llm_rl import (
    LoraRLConfig,
    apply_generated_decision,
    build_safety_prompt,
    parse_generated_decision,
    train_lora_reward_rl,
)
from evoguard.types import AttackSample, SafetyAction, Task, TrajectoryRecord, TrajectoryType


DEFAULT_MODEL = "/mnt/sata1/beihang_toolsafe/models/Qwen2.5-1.5B-Instruct"


def main() -> None:
    args = parse_args()
    env = TextToolEnv()
    tasks = [task for task in env.get_tasks() if task.preferred_tool]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    output_dir = Path(args.output_dir)
    rollout_dir = output_dir / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    attack_generator = build_api_attack_generator(
        split="coevolution_redteam",
        attacks_per_task=args.attacks_per_task,
        tool_catalog={tool.name: tool for tool in TOOLS},
        max_tokens=args.attack_max_tokens,
        temperature=args.attack_temperature,
    )
    memory = AttackMemory()
    current_adapter = args.sft_adapter
    all_successful_attacks: list[AttackSample] = []

    for round_id in range(1, args.rounds + 1):
        print(f"[coevo-llm] round={round_id} loading defender adapter={current_adapter}", flush=True)
        model, tokenizer, torch = load_defender(args.model, current_adapter)
        try:
            candidate_attacks = attack_generator.generate(
                tasks,
                round_id=round_id,
                attack_memory=memory,
            )
            attacked_records = evaluate_attacks(
                model,
                tokenizer,
                candidate_attacks,
                env,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
            )
            successful_records = extract_failures(attacked_records)
            blocked_records = [record for record in attacked_records if record not in successful_records]
            memory.add_success_cases(round_id, successful_records)
            memory.add_blocked_cases(round_id, blocked_records)
            successful_ids = {
                str(record.metadata.get("attack_id"))
                for record in successful_records
                if record.metadata.get("attack_id")
            }
            successful_attacks = [attack for attack in candidate_attacks if attack.attack_id in successful_ids]
            all_successful_attacks.extend(successful_attacks)

            no_tool_records = evaluate_no_tool_records(
                model,
                tokenizer,
                tasks,
                env.name,
                round_id,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
            )
            clean_records = evaluate_clean_records(
                model,
                tokenizer,
                tasks,
                env,
                round_id,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
            )
            train_records = no_tool_records + clean_records + attacked_records
            reward_batch = [
                combined_reward(
                    record,
                    current_round=round_id,
                    max_round=max(1, args.rounds),
                    weights=RewardWeights(),
                )
                for record in train_records
            ]
            pre_update_metrics = compute_metrics(train_records)
            mean_reward = sum(item["total"] for item in reward_batch) / max(1, len(reward_batch))

            rollout_path = rollout_dir / f"round_{round_id}_train_rollouts.jsonl"
            attacks_path = rollout_dir / f"round_{round_id}_successful_attacks.jsonl"
            write_jsonl(rollout_path, [record.to_dict() for record in train_records])
            write_jsonl(attacks_path, [attack_to_dict(attack) for attack in successful_attacks])
        finally:
            del model
            del tokenizer
            gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        round_output = output_dir / f"round_{round_id}"
        rl_config = LoraRLConfig(
            model_name_or_path=args.model,
            rollout_jsonl=str(rollout_path),
            sft_adapter_path=current_adapter,
            output_dir=str(round_output),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.rl_epochs,
            temperature=args.rl_temperature,
            top_p=args.rl_top_p,
            advantage_baseline=args.advantage_baseline,
            gradient_checkpointing=args.gradient_checkpointing,
            bf16=not args.no_bf16,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        print(f"[coevo-llm] round={round_id} training LoRA -> {round_output}", flush=True)
        rl_stats = train_lora_reward_rl(rl_config)
        current_adapter = str(round_output)

        heldout_metrics = None
        if args.eval_heldout:
            model, tokenizer, torch = load_defender(args.model, current_adapter)
            try:
                heldout_metrics = evaluate_heldout(
                    model,
                    tokenizer,
                    env,
                    round_id=args.heldout_round,
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                )
            finally:
                del model
                del tokenizer
                gc.collect()
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        summary = {
            "round_id": round_id,
            "model": args.model,
            "input_adapter": rl_config.sft_adapter_path,
            "output_adapter": current_adapter,
            "candidate_attacks": len(candidate_attacks),
            "successful_attacks": len(successful_records),
            "blocked_attacks": len(blocked_records),
            "attack_success_rate": len(successful_records) / max(1, len(attacked_records)),
            "train_records": len(train_records),
            "mean_reward": mean_reward,
            "pre_update_metrics": pre_update_metrics,
            "rl_stats": summarize_rl_stats(rl_stats),
            "heldout_metrics": heldout_metrics,
            "artifacts": {
                "train_rollouts": str(rollout_path),
                "successful_attacks": str(attacks_path),
            },
        }
        append_jsonl(metrics_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    write_jsonl(
        output_dir / "all_successful_attacks.jsonl",
        [attack_to_dict(attack) for attack in all_successful_attacks],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--attacks-per-task", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--output-dir", default="outputs/checkpoints/coevolving_llm_defender")
    parser.add_argument("--metrics-output", default="outputs/logs/coevolution_llm_metrics.jsonl")
    parser.add_argument("--eval-heldout", action="store_true")
    parser.add_argument("--heldout-round", type=int, default=99)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--attack-max-tokens", type=int, default=1400)
    parser.add_argument("--attack-temperature", type=float, default=0.8)
    parser.add_argument("--max-prompt-length", type=int, default=LoraRLConfig.max_prompt_length)
    parser.add_argument("--max-new-tokens", type=int, default=LoraRLConfig.max_new_tokens)
    parser.add_argument("--per-device-train-batch-size", type=int, default=LoraRLConfig.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=LoraRLConfig.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=LoraRLConfig.learning_rate)
    parser.add_argument("--rl-epochs", type=float, default=LoraRLConfig.num_train_epochs)
    parser.add_argument("--rl-temperature", type=float, default=LoraRLConfig.temperature)
    parser.add_argument("--rl-top-p", type=float, default=LoraRLConfig.top_p)
    parser.add_argument("--advantage-baseline", choices=("zero", "batch_mean"), default=LoraRLConfig.advantage_baseline)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=LoraRLConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=LoraRLConfig.lora_alpha)
    return parser.parse_args()


def load_defender(model_name_or_path: str, adapter_path: str) -> tuple[Any, Any, Any]:
    torch, transformers, peft = import_training_deps()
    tokenizer = transformers.AutoTokenizer.from_pretrained(adapter_path or model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = peft.PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer, torch


def import_training_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        import peft
    except ImportError as exc:
        raise RuntimeError("Qwen-LoRA co-evolution requires torch, transformers, and peft.") from exc
    return torch, transformers, peft


def evaluate_attacks(
    model: Any,
    tokenizer: Any,
    attacks: list[AttackSample],
    env: TextToolEnv,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
) -> list[TrajectoryRecord]:
    return [
        evaluate_record(
            model,
            tokenizer,
            attack_to_record(env, attack),
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
        )
        for attack in attacks
    ]


def evaluate_no_tool_records(
    model: Any,
    tokenizer: Any,
    tasks: list[Task],
    environment: str,
    round_id: int,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
) -> list[TrajectoryRecord]:
    records = [
        TrajectoryRecord(
            task_id=task.task_id,
            round_id=round_id,
            environment=environment,
            trajectory_type=TrajectoryType.NO_TOOL,
            user_task=task.user_task,
            tool_name=None,
            tool_arguments={},
            injected_content=None,
            injection_span=None,
            gold_safety_action=SafetyAction.EXECUTE,
            agent_safety_action=SafetyAction.EXECUTE,
            confidence=0.0,
            task_success=False,
            attack_success=False,
            attribution_score=0.0,
            metadata={"source": "coevolution_no_tool"},
        )
        for task in tasks
    ]
    return [
        evaluate_record(model, tokenizer, record, max_prompt_length=max_prompt_length, max_new_tokens=max_new_tokens)
        for record in records
    ]


def evaluate_clean_records(
    model: Any,
    tokenizer: Any,
    tasks: list[Task],
    env: TextToolEnv,
    round_id: int,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for task in tasks:
        tool_call = env.propose_tool_call(task)
        records.append(
            TrajectoryRecord(
                task_id=task.task_id,
                round_id=round_id,
                environment=env.name,
                trajectory_type=TrajectoryType.CLEAN_TOOL,
                user_task=task.user_task,
                tool_name=tool_call.tool_name if tool_call else None,
                tool_arguments=dict(tool_call.arguments) if tool_call else {},
                injected_content=None,
                injection_span=None,
                gold_safety_action=task.gold_safety_action,
                agent_safety_action=SafetyAction.EXECUTE,
                confidence=0.0,
                task_success=False,
                attack_success=False,
                attribution_score=0.0,
                metadata={"source": "coevolution_clean_tool"},
            )
        )
    return [
        evaluate_record(model, tokenizer, record, max_prompt_length=max_prompt_length, max_new_tokens=max_new_tokens)
        for record in records
    ]


def evaluate_record(
    model: Any,
    tokenizer: Any,
    record: TrajectoryRecord,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
) -> TrajectoryRecord:
    torch, _, _ = import_training_deps()
    prompt = build_safety_prompt(record, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    completion_ids = output_ids[inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return apply_generated_decision(record, parse_generated_decision(raw_text))


def attack_to_record(env: TextToolEnv, attack: AttackSample) -> TrajectoryRecord:
    task = attack.task
    tool_call = env.propose_tool_call(task)
    true_span = (attack.insertion_offset, attack.insertion_offset + len(attack.injected_content))
    return TrajectoryRecord(
        task_id=task.task_id,
        round_id=attack.round_id,
        environment=env.name,
        trajectory_type=TrajectoryType.ATTACKED_TOOL,
        user_task=f"{task.user_task}\n{attack.injected_content}",
        tool_name=tool_call.tool_name if tool_call else task.preferred_tool,
        tool_arguments=dict(tool_call.arguments) if tool_call else dict(task.tool_arguments),
        injected_content=attack.injected_content,
        injection_span=true_span,
        gold_safety_action=SafetyAction.REFUSE,
        agent_safety_action=SafetyAction.EXECUTE,
        confidence=0.0,
        task_success=False,
        attack_success=False,
        attribution_score=0.0,
        attack_type=attack.attack_type,
        metadata={"attack_id": attack.attack_id, "target_tool": attack.target_tool},
    )


def evaluate_heldout(
    model: Any,
    tokenizer: Any,
    env: TextToolEnv,
    round_id: int,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
) -> dict[str, float]:
    attacks = build_attack_generator("heldout").generate(env.get_tasks(), round_id=round_id)
    attacked = evaluate_attacks(
        model,
        tokenizer,
        attacks,
        env,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
    )
    clean = evaluate_clean_records(
        model,
        tokenizer,
        [task for task in env.get_tasks() if task.preferred_tool],
        env,
        round_id,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
    )
    return compute_metrics(clean + attacked)


def summarize_rl_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keys = ("updates", "samples", "valid_json_rate", "mean_reward", "mean_advantage", "mean_loss")
    return {key: stats.get(key) for key in keys if key in stats}


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


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
