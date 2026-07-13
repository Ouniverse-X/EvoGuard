#!/usr/bin/env bash
set -euo pipefail

# Server-local LoRA runner verified with the LLaMaFactory conda environment.
#
# Smoke:
#   GPU_ID=3 EPOCHS=0.03 PER_DEVICE_BATCH=1 GRAD_ACCUM=1 scripts/training/run_server_qwen15b_lora.sh
#
# Longer run:
#   GPU_ID=3 EPOCHS=3 PER_DEVICE_BATCH=2 GRAD_ACCUM=8 scripts/training/run_server_qwen15b_lora.sh

CONDA_PYTHON="${CONDA_PYTHON:-/home/beihang/anaconda3/envs/LLaMaFactory/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mnt/sata1/beihang_toolsafe/models/Qwen2.5-1.5B-Instruct}"
GPU_ID="${GPU_ID:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/llm_safety_lora_qwen15b}"
TRAIN_JSONL="${TRAIN_JSONL:-data/processed/safety_sft_train.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-0.03}"
LR="${LR:-2e-4}"

"${CONDA_PYTHON}" scripts/prepare_safety_sft_data.py

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA_PYTHON}" scripts/train_llm_safety_lora.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --per-device-train-batch-size "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --num-train-epochs "${EPOCHS}" \
  --learning-rate "${LR}"
