#!/usr/bin/env bash
set -euo pipefail

# Reward-based LoRA stage over EvoGuard tri-trajectory rollouts.
#
# Smoke:
#   GPU_ID=3 EPOCHS=0.03 BATCH=1 GRAD_ACCUM=1 scripts/training/run_server_qwen15b_rl.sh
#
# Use an SFT warm-start adapter:
#   SFT_ADAPTER=outputs/checkpoints/llm_safety_lora_qwen15b_smoke GPU_ID=3 scripts/training/run_server_qwen15b_rl.sh

CONDA_PYTHON="${CONDA_PYTHON:-/home/beihang/anaconda3/envs/LLaMaFactory/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mnt/sata1/beihang_toolsafe/models/Qwen2.5-1.5B-Instruct}"
GPU_ID="${GPU_ID:-3}"
ROLLOUT_JSONL="${ROLLOUT_JSONL:-data/rollouts/train_tri_rollouts_round0.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/llm_safety_lora_qwen15b_rl}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-0.03}"
LR="${LR:-5e-5}"
TEMP="${TEMP:-0.7}"
ADV_BASELINE="${ADV_BASELINE:-zero}"

"${CONDA_PYTHON}" scripts/export_sample_data.py

ARGS=(
  scripts/train_llm_safety_rl.py
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --rollout-jsonl "${ROLLOUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --per-device-train-batch-size "${BATCH}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --num-train-epochs "${EPOCHS}"
  --learning-rate "${LR}"
  --temperature "${TEMP}"
  --advantage-baseline "${ADV_BASELINE}"
)

if [[ -n "${SFT_ADAPTER}" ]]; then
  ARGS+=(--sft-adapter-path "${SFT_ADAPTER}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA_PYTHON}" "${ARGS[@]}"
