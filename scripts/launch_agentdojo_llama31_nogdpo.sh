#!/usr/bin/env bash
# Launch wrapper for the GDPO ABLATION arm of the Llama-3.1-8B AgentDojo run
# (`evoguard_agentdojo_llama31_nogdpo_v1`, configs/agentdojo_llama31_nogdpo.yaml).
#
# Byte-identical to scripts/launch_agentdojo_llama31.sh except for the config it
# execs. That is deliberate: the two runs differ in exactly one knob
# (training.grpo_gdpo), so the ENV CONTRACT must not drift either. Read that
# file's header for why each variable is a silent-degradation hazard if unset;
# the short version:
#   * EVOGUARD_PRIMARY_VLLM      -- the runner's primary probe defaults to :8000,
#                                   but the defender is on :8010.
#   * EVOGUARD_JUDGE_LLM_BASE_URL / EVOGUARD_PROGRESS_LLM_BASE_URL -- unset, the
#                                   step-safety judge returns one constant
#                                   `unclear` and the group-relative advantages
#                                   DELETE the safety gradient.
#   * EVOGUARD_TRAINER_MIN_FREE_MIB -- read at IMPORT time by native_runner.py.
#
# Serving prerequisites (identical to the GDPO-ON arm):
#   :8010 GPU6  llama3.1-8b-it  defender (--enable-lora --max-lora-rank 64
#               --max-loras 4 + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
#   :8000 GPU0  qwen2.5-7b-it   attacker      :8003 GPU1  qwen2.5-7b-it executor
#   :8004 GPU2  qwen3.5-9b      judges        :8006 GPU5  qwen3.5-9b    progress
#   GPU7        EMPTY -- the colocate engine profiles WHOLE-CARD free memory
#
# Nine rounds do not fit in one process (the colocate engine leaks ~19 GiB of the
# trainer card per round), so drive this through the leg chainer rather than
# calling it directly:
#   python scripts/run_legged_experiment.py \
#       --config configs/agentdojo_llama31_nogdpo.yaml \
#       --launch scripts/launch_agentdojo_llama31_nogdpo.sh \
#       --target-rounds 9 --rounds-per-leg 2 --trainer-gpu 7
set -euo pipefail
cd /root/paddlejob/workspace/yangxiao/EvoGuard

export EVOGUARD_PY_BIN=/root/paddlejob/workspace/yangxiao/miniconda3/envs/evoguard/bin/python
export EVOGUARD_PREWARM_SEEDS_DIR=data/seeds_v3
export EVOGUARD_PRIMARY_VLLM=http://127.0.0.1:8010/v1
export EVOGUARD_JUDGE_LLM_BASE_URL=http://127.0.0.1:8004/v1
export EVOGUARD_PROGRESS_LLM_BASE_URL=http://127.0.0.1:8006/v1
export EVOGUARD_JUDGE_LLM_MODEL=qwen3.5-9b
export EVOGUARD_REWARD_JUDGE_WORKERS=32
export EVOGUARD_TRAINER_MIN_FREE_MIB=40960
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAINER_CUDA_VISIBLE_DEVICES=7

exec bash scripts/run_grpo_experiment.sh configs/agentdojo_llama31_nogdpo.yaml
