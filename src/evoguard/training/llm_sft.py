"""Optional Hugging Face/PEFT SFT backend for safety-judge LLMs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoraSFTConfig:
    model_name_or_path: str
    train_jsonl: str = "data/processed/safety_sft_train.jsonl"
    output_dir: str = "outputs/checkpoints/llm_safety_lora"
    max_length: int = 1024
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 7
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    use_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def train_lora_sft(config: LoraSFTConfig) -> dict[str, Any]:
    """Train a LoRA safety judge. Imports heavy dependencies lazily."""

    torch, transformers, peft = _import_training_deps()
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.use_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if config.bf16 else None,
        quantization_config=quantization_config,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if config.use_4bit:
        model = peft.prepare_model_for_kbit_training(model)

    lora_config = peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[module.strip() for module in config.target_modules.split(",") if module.strip()],
    )
    model = peft.get_peft_model(model, lora_config)

    dataset = SafetySFTDataset(Path(config.train_jsonl), tokenizer, max_length=config.max_length)
    training_args = transformers.TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        bf16=config.bf16,
        fp16=config.fp16,
        seed=config.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = dict(result.metrics)
    metrics["num_examples"] = len(dataset)
    metrics["output_dir"] = config.output_dir
    metrics["config"] = asdict(config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


class SafetySFTDataset:
    def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
        self.examples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.examples.append(_encode_chat_example(row["messages"], tokenizer, max_length))
        if not self.examples:
            raise ValueError(f"No SFT examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class CausalLMCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch, _, _ = _import_training_deps()
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _encode_chat_example(messages: list[dict[str, str]], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    prompt_messages = messages[:-1]
    assistant_message = messages[-1]
    prompt = _format_messages(prompt_messages, tokenizer, add_generation_prompt=True)
    full_text = prompt + assistant_message["content"] + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = list(full["input_ids"])
    attention_mask = list(full["attention_mask"])
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _format_messages(messages: list[dict[str, str]], tokenizer: Any, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    text = ""
    for message in messages:
        text += f"<|{message['role']}|>\n{message['content']}\n"
    if add_generation_prompt:
        text += "<|assistant|>\n"
    return text


def _import_training_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        import peft
    except Exception as exc:
        raise RuntimeError(
            "LoRA SFT requires torch, transformers, and peft. "
            "Install requirements-a100.txt in a clean Python environment."
        ) from exc
    return torch, transformers, peft


def parse_args() -> LoraSFTConfig:
    parser = argparse.ArgumentParser(description="Train EvoGuard safety judge with LoRA SFT.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-jsonl", default=LoraSFTConfig.train_jsonl)
    parser.add_argument("--output-dir", default=LoraSFTConfig.output_dir)
    parser.add_argument("--max-length", type=int, default=LoraSFTConfig.max_length)
    parser.add_argument("--per-device-train-batch-size", type=int, default=LoraSFTConfig.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=LoraSFTConfig.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=LoraSFTConfig.learning_rate)
    parser.add_argument("--num-train-epochs", type=float, default=LoraSFTConfig.num_train_epochs)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=LoraSFTConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=LoraSFTConfig.lora_alpha)
    parser.add_argument("--target-modules", default=LoraSFTConfig.target_modules)
    args = parser.parse_args()
    return LoraSFTConfig(
        model_name_or_path=args.model_name_or_path,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        use_4bit=args.use_4bit,
        bf16=not args.no_bf16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
