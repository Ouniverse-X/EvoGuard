#!/usr/bin/env bash
set -euo pipefail

# Example:
#   MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct GPU_IDS=0 scripts/training/run_a100_lora_sft.sh
#   MODEL_NAME_OR_PATH=/path/to/local/model GPU_IDS=0,1 scripts/training/run_a100_lora_sft.sh

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a HF model id or local model path}"
GPU_IDS="${GPU_IDS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/llm_safety_lora}"
TRAIN_JSONL="${TRAIN_JSONL:-data/processed/safety_sft_train.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
USE_4BIT="${USE_4BIT:-0}"

python scripts/prepare_safety_sft_data.py

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
COMMON_ARGS=(
  scripts/train_llm_safety_lora.py
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-length "${MAX_LENGTH}"
  --per-device-train-batch-size "${PER_DEVICE_BATCH}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --num-train-epochs "${EPOCHS}"
  --learning-rate "${LR}"
)

if [[ "${USE_4BIT}" == "1" ]]; then
  COMMON_ARGS+=(--use-4bit)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${COMMON_ARGS[@]}"
else
  python "${COMMON_ARGS[@]}"
fi
