"""Reward-based LoRA training over EvoGuard tri-trajectory rollouts.

This is an intentionally compact RL stage for the project: the model samples a
safety decision for each rollout prompt, EvoGuard computes the trajectory reward,
and the LoRA adapter is updated with an advantage-weighted policy-gradient loss
on the sampled completion. It is not a full PPO/GRPO replacement yet, but it
connects Qwen LoRA training to the actual three-trajectory reward objective.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from evoguard.rewards.combined_reward import RewardWeights, combined_reward
from evoguard.training.llm_sft import _format_messages, _import_training_deps
from evoguard.training.sft_dataset import trajectory_to_safety_sft_example
from evoguard.types import SafetyAction, TrajectoryRecord, TrajectoryType


@dataclass(frozen=True)
class LoraRLConfig:
    model_name_or_path: str
    rollout_jsonl: str = "data/rollouts/train_tri_rollouts_round0.jsonl"
    sft_adapter_path: str | None = None
    output_dir: str = "outputs/checkpoints/llm_safety_lora_rl"
    max_prompt_length: int = 1024
    max_new_tokens: int = 96
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 7
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    max_advantage_abs: float = 5.0
    entropy_bonus: float = 0.0
    advantage_baseline: str = "zero"
    log_samples: int = 5


@dataclass(frozen=True)
class GeneratedDecision:
    action: SafetyAction
    confidence: float
    attribution_span: tuple[int, int] | None
    raw_text: str
    valid_json: bool


VALID_GENERATED_ACTIONS = {action.value for action in SafetyAction}
BYPASS_ACTION_ALIASES = {"bypass"}
INVALID_ACTION_REWARD_PENALTY = 3.0


def train_lora_reward_rl(config: LoraRLConfig) -> dict[str, Any]:
    torch, transformers, peft = _import_training_deps()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    records = read_rollout_jsonl(Path(config.rollout_jsonl))
    if not records:
        raise ValueError(f"No rollout records found in {config.rollout_jsonl}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.sft_adapter_path or config.model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if config.bf16 else None,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if config.sft_adapter_path:
        model = peft.PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    else:
        lora_config = peft.LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[module.strip() for module in config.target_modules.split(",") if module.strip()],
        )
        model = peft.get_peft_model(model, lora_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=config.learning_rate)

    total_steps = max(1, math.ceil(len(records) * config.num_train_epochs / config.per_device_train_batch_size))
    target_examples = max(1, int(math.ceil(len(records) * config.num_train_epochs)))
    sampled_records = [records[index % len(records)] for index in range(target_examples)]
    random.shuffle(sampled_records)

    stats: dict[str, Any] = {
        "num_rollouts": len(records),
        "target_examples": target_examples,
        "updates": 0,
        "samples": 0,
        "valid_json": 0,
        "mean_reward": 0.0,
        "mean_advantage": 0.0,
        "mean_loss": 0.0,
        "sample_decisions": [],
    }
    reward_total = 0.0
    advantage_total = 0.0
    loss_total = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch_start in range(0, len(sampled_records), config.per_device_train_batch_size):
        batch_records = sampled_records[batch_start : batch_start + config.per_device_train_batch_size]
        batch_samples = [_sample_decision(model, tokenizer, record, config, device) for record in batch_records]
        rewards = [
            reward_for_generated_decision(record, sample, current_round=record.round_id, max_round=99)["total"]
            for record, sample in zip(batch_records, batch_samples, strict=True)
        ]
        baseline = _advantage_baseline(rewards, config)
        advantages = [
            max(-config.max_advantage_abs, min(config.max_advantage_abs, reward - baseline))
            for reward in rewards
        ]

        for record, sample, reward, advantage in zip(batch_records, batch_samples, rewards, advantages, strict=True):
            if len(stats["sample_decisions"]) < config.log_samples:
                stats["sample_decisions"].append(
                    {
                        "task_id": record.task_id,
                        "trajectory_type": record.trajectory_type.value,
                        "attack_type": record.attack_type,
                        "raw_text": sample.raw_text,
                        "parsed_action": sample.action.value,
                        "valid_json": sample.valid_json,
                        "reward": reward,
                        "advantage": advantage,
                    }
                )
            loss = _policy_gradient_loss(model, tokenizer, record, sample.raw_text, advantage, config, device)
            scaled_loss = loss / max(1, config.gradient_accumulation_steps)
            scaled_loss.backward()
            loss_value = float(loss.detach().cpu().item())
            stats["samples"] += 1
            stats["valid_json"] += int(sample.valid_json)
            reward_total += float(reward)
            advantage_total += float(advantage)
            loss_total += loss_value

            if stats["samples"] % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                stats["updates"] += 1

        if total_steps and (batch_start // max(1, config.per_device_train_batch_size)) % 10 == 0:
            _write_progress(config, stats, reward_total, advantage_total, loss_total)

    if stats["samples"] % config.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        stats["updates"] += 1

    stats["mean_reward"] = reward_total / max(1, stats["samples"])
    stats["mean_advantage"] = advantage_total / max(1, stats["samples"])
    stats["mean_loss"] = loss_total / max(1, stats["samples"])
    stats["valid_json_rate"] = stats["valid_json"] / max(1, stats["samples"])
    stats["config"] = asdict(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "rl_metrics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def read_rollout_jsonl(path: Path) -> list[TrajectoryRecord]:
    return [record_from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_from_dict(row: dict[str, Any]) -> TrajectoryRecord:
    return TrajectoryRecord(
        task_id=str(row["task_id"]),
        round_id=int(row["round_id"]),
        environment=str(row["environment"]),
        trajectory_type=TrajectoryType(str(row["trajectory_type"])),
        user_task=str(row["user_task"]),
        tool_name=row.get("tool_name"),
        tool_arguments=dict(row.get("tool_arguments", {})),
        injected_content=row.get("injected_content"),
        injection_span=_tuple_or_none(row.get("injection_span")),
        gold_safety_action=SafetyAction(str(row["gold_safety_action"])),
        agent_safety_action=SafetyAction(str(row["agent_safety_action"])),
        confidence=float(row.get("confidence", 0.0)),
        task_success=bool(row.get("task_success", False)),
        attack_success=bool(row.get("attack_success", False)),
        attribution_score=float(row.get("attribution_score", 0.0)),
        attack_type=row.get("attack_type"),
        metadata=dict(row.get("metadata", {})),
    )


def reward_for_generated_decision(
    record: TrajectoryRecord,
    decision: GeneratedDecision,
    *,
    current_round: int,
    max_round: int,
) -> dict[str, float]:
    updated = apply_generated_decision(record, decision)
    reward = combined_reward(updated, current_round=current_round, max_round=max_round, weights=RewardWeights())
    if not decision.valid_json:
        reward = {
            **reward,
            "format": -INVALID_ACTION_REWARD_PENALTY,
            "total": reward["total"] - INVALID_ACTION_REWARD_PENALTY,
        }
    else:
        reward = {**reward, "format": 0.0}
    return reward


def apply_generated_decision(record: TrajectoryRecord, decision: GeneratedDecision) -> TrajectoryRecord:
    attack_success = (
        record.trajectory_type == TrajectoryType.ATTACKED_TOOL
        and record.gold_safety_action != SafetyAction.EXECUTE
        and decision.action == SafetyAction.EXECUTE
    )
    if record.trajectory_type == TrajectoryType.ATTACKED_TOOL:
        task_success = not attack_success if decision.action == SafetyAction.EXECUTE else True
    elif record.trajectory_type == TrajectoryType.CLEAN_TOOL:
        task_success = decision.action in (SafetyAction.EXECUTE, SafetyAction.ASK_CONFIRMATION)
    else:
        task_success = decision.action == SafetyAction.EXECUTE

    return replace(
        record,
        agent_safety_action=decision.action,
        confidence=decision.confidence,
        task_success=task_success,
        attack_success=attack_success,
        attribution_score=_span_iou(decision.attribution_span, record.injection_span),
        metadata={**record.metadata, "llm_rl_raw_decision": decision.raw_text, "llm_rl_valid_json": decision.valid_json},
    )


def build_safety_prompt(record: TrajectoryRecord, tokenizer: Any) -> str:
    messages = trajectory_to_safety_sft_example(record)["messages"][:-1]
    return _format_messages(messages, tokenizer, add_generation_prompt=True)


def parse_generated_decision(text: str) -> GeneratedDecision:
    parsed = _extract_json_object(text)
    if parsed is None:
        return GeneratedDecision(SafetyAction.REFUSE, 0.0, None, text, valid_json=False)
    action, valid = _parse_safety_action(parsed.get("action"))
    confidence = _clamp_float(parsed.get("confidence", 0.5), 0.0, 1.0)
    span = _tuple_or_none(parsed.get("attribution_span"))
    return GeneratedDecision(action, confidence, span, text, valid_json=valid)


def _parse_safety_action(value: Any) -> tuple[SafetyAction, bool]:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_GENERATED_ACTIONS:
        return SafetyAction(normalized), True
    if normalized in BYPASS_ACTION_ALIASES:
        return SafetyAction.REFUSE, False
    return SafetyAction.REFUSE, False


def _sample_decision(model: Any, tokenizer: Any, record: TrajectoryRecord, config: LoraRLConfig, device: Any) -> GeneratedDecision:
    torch, _, _ = _import_training_deps()
    prompt = build_safety_prompt(record, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_prompt_length).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    completion_ids = output_ids[inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return parse_generated_decision(text)


def _policy_gradient_loss(
    model: Any,
    tokenizer: Any,
    record: TrajectoryRecord,
    completion: str,
    advantage: float,
    config: LoraRLConfig,
    device: Any,
) -> Any:
    torch, _, _ = _import_training_deps()
    if not completion.strip():
        completion = json.dumps({"action": "execute", "confidence": 0.0, "attribution_span": None})
    prompt = build_safety_prompt(record, tokenizer)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        prompt + completion + (tokenizer.eos_token or ""),
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length + config.max_new_tokens,
    ).to(device)
    labels = full["input_ids"].clone()
    labels[:, : min(len(prompt_ids), labels.shape[-1])] = -100
    outputs = model(**full, labels=labels)
    nll = outputs.loss
    return float(advantage) * nll


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _span_iou(pred: tuple[int, int] | None, gold: tuple[int, int] | None) -> float:
    if pred is None or gold is None:
        return 0.0
    left = max(pred[0], gold[0])
    right = min(pred[1], gold[1])
    intersection = max(0, right - left)
    union = max(pred[1], gold[1]) - min(pred[0], gold[0])
    return intersection / union if union > 0 else 0.0


def _tuple_or_none(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))


def _write_progress(
    config: LoraRLConfig,
    stats: dict[str, Any],
    reward_total: float,
    advantage_total: float,
    loss_total: float,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = max(1, int(stats["samples"]))
    progress = {
        **stats,
        "mean_reward": reward_total / samples,
        "mean_advantage": advantage_total / samples,
        "mean_loss": loss_total / samples,
        "valid_json_rate": int(stats["valid_json"]) / samples,
    }
    (output_dir / "rl_progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _advantage_baseline(rewards: list[float], config: LoraRLConfig) -> float:
    if config.advantage_baseline == "zero":
        return 0.0
    if config.advantage_baseline == "batch_mean":
        return sum(rewards) / len(rewards)
    raise ValueError(f"Unknown advantage baseline: {config.advantage_baseline}")


def parse_args() -> LoraRLConfig:
    parser = argparse.ArgumentParser(description="Reward-train EvoGuard safety judge LoRA from tri-rollouts.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--rollout-jsonl", default=LoraRLConfig.rollout_jsonl)
    parser.add_argument("--sft-adapter-path")
    parser.add_argument("--output-dir", default=LoraRLConfig.output_dir)
    parser.add_argument("--max-prompt-length", type=int, default=LoraRLConfig.max_prompt_length)
    parser.add_argument("--max-new-tokens", type=int, default=LoraRLConfig.max_new_tokens)
    parser.add_argument("--per-device-train-batch-size", type=int, default=LoraRLConfig.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=LoraRLConfig.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=LoraRLConfig.learning_rate)
    parser.add_argument("--num-train-epochs", type=float, default=LoraRLConfig.num_train_epochs)
    parser.add_argument("--temperature", type=float, default=LoraRLConfig.temperature)
    parser.add_argument("--top-p", type=float, default=LoraRLConfig.top_p)
    parser.add_argument("--advantage-baseline", choices=("zero", "batch_mean"), default=LoraRLConfig.advantage_baseline)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=LoraRLConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=LoraRLConfig.lora_alpha)
    parser.add_argument("--target-modules", default=LoraRLConfig.target_modules)
    args = parser.parse_args()
    return LoraRLConfig(
        model_name_or_path=args.model_name_or_path,
        rollout_jsonl=args.rollout_jsonl,
        sft_adapter_path=args.sft_adapter_path,
        output_dir=args.output_dir,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        temperature=args.temperature,
        top_p=args.top_p,
        advantage_baseline=args.advantage_baseline,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=not args.no_bf16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
