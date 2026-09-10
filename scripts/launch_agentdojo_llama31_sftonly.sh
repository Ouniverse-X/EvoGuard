#!/usr/bin/env bash
# Launch wrapper for the NO-RL ABLATION arm of the Llama-3.1-8B AgentDojo ladder
# (`evoguard_agentdojo_llama31_sftonly_v1`, configs/agentdojo_llama31_sftonly.yaml).
# Every round is LoRA-SFT warm-started from the previous round's adapter; no GRPO,
# and therefore no GDPO, no advantages, no reward function.
#
# The ENV CONTRACT below is byte-identical to the two GRPO arms' launchers
# (scripts/launch_agentdojo_llama31{,_nogdpo}.sh). That is deliberate: the three
# runs differ in exactly one config knob, so the environment must not drift
# either. Two of these variables are silent-degradation hazards even here, where
# no reward is computed -- the trajectory-level AttackJudge and the utility judge
# still run in the rollout/val phases and still read them:
#   * EVOGUARD_PRIMARY_VLLM      -- run_grpo_experiment.sh probes :8000 by
#                                   default, but the defender is on :8010.
#   * EVOGUARD_JUDGE_LLM_BASE_URL / EVOGUARD_PROGRESS_LLM_BASE_URL
#   * EVOGUARD_TRAINER_MIN_FREE_MIB -- read at IMPORT time by native_runner.py,
#                                   and `_wait_for_free_gpu_memory` is on the SFT
#                                   path, so this gates ALL NINE rounds here (on
#                                   the GRPO arms it only gated r0).
#
# Serving prerequisites (identical to the GRPO arms):
#   :8010 GPU6  llama3.1-8b-it  defender (--enable-lora --max-lora-rank 64
#               --max-loras 4 + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
#   :8000 GPU0  qwen2.5-7b-it   attacker      :8003 GPU1  qwen2.5-7b-it executor
#   :8004 GPU2  qwen3.5-9b      judges        :8006 GPU5  qwen3.5-9b    progress
#   GPU7        EMPTY for the WHOLE run -- see EVOGUARD_TRAINER_MIN_FREE_MIB above:
#               `bash scripts/gpu_keeper.sh stop -g 7`, resume only after exit.
#
# Unlike the GRPO arms this needs NO leg chaining: the 2-rounds-per-process cap
# there is the colocate vLLM engine leaking the trainer card, and SFT never builds
# an engine. Run it directly, all nine rounds in one process:
#   nohup bash scripts/launch_agentdojo_llama31_sftonly.sh > /dev/null 2>&1 &
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

exec bash scripts/run_grpo_experiment.sh configs/agentdojo_llama31_sftonly.yaml
