# A100 Training Guide

This guide covers the current trainable EvoGuard backends.

## 1. Local Neural Safety Head

This backend is dependency-light and trains a small PyTorch MLP safety classifier.

```bash
python scripts/train_neural_safety_head.py \
  --rounds 3 \
  --epochs 40 \
  --hidden-dim 64 \
  --batch-size 32 \
  --device auto \
  --checkpoint-path outputs/checkpoints/neural_safety_head.pt
```

Use `--device cuda` on a GPU machine, or `--device cpu` for CPU-only debugging.

## 2. Prepare Safety SFT Data

The LoRA backend consumes chat-style JSONL examples.

```bash
python scripts/prepare_safety_sft_data.py
```

Default output:

```text
data/processed/safety_sft_train.jsonl
```

## 3. LoRA Safety Judge on A100

Create a clean environment and install optional training dependencies:

```bash
pip install -r requirements-a100.txt
```

Single A100 example:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct \
GPU_IDS=0 \
OUTPUT_DIR=outputs/checkpoints/llm_safety_lora \
scripts/training/run_a100_lora_sft.sh
```

Two A100s:

```bash
MODEL_NAME_OR_PATH=/path/to/local/instruct-model \
GPU_IDS=0,1 \
PER_DEVICE_BATCH=2 \
GRAD_ACCUM=8 \
EPOCHS=3 \
scripts/training/run_a100_lora_sft.sh
```

For larger models, enable 4-bit loading:

```bash
MODEL_NAME_OR_PATH=/path/to/local/14b-or-larger-model \
GPU_IDS=0 \
USE_4BIT=1 \
scripts/training/run_a100_lora_sft.sh
```

The LoRA adapter and tokenizer are saved under `OUTPUT_DIR`. Training metrics are written to:

```text
outputs/checkpoints/llm_safety_lora/train_metrics.json
```

## 4. Verified Server Conda Command

On this server, the `LLaMaFactory` conda environment already has compatible
`torch`, `transformers`, `peft`, and `accelerate`. A Qwen2.5-1.5B local model is
available at:

```text
/mnt/sata1/beihang_toolsafe/models/Qwen2.5-1.5B-Instruct
```

Smoke run on GPU 3:

```bash
GPU_ID=3 \
EPOCHS=0.03 \
PER_DEVICE_BATCH=1 \
GRAD_ACCUM=1 \
OUTPUT_DIR=outputs/checkpoints/llm_safety_lora_qwen15b_smoke \
scripts/training/run_server_qwen15b_lora.sh
```

Longer run:

```bash
GPU_ID=3 \
EPOCHS=3 \
PER_DEVICE_BATCH=2 \
GRAD_ACCUM=8 \
OUTPUT_DIR=outputs/checkpoints/llm_safety_lora_qwen15b_full \
scripts/training/run_server_qwen15b_lora.sh
```
