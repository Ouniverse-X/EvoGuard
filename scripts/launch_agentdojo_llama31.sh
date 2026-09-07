#!/usr/bin/env bash
# Launch wrapper for the Llama-3.1-8B AgentDojo co-evolution run.
#
# Everything below is the ENV CONTRACT that configs/agentdojo_llama31_grpo.yaml's
# header documents but that scripts/run_grpo_experiment.sh does NOT export itself.
# It lives in a file rather than a one-line `nohup env A=1 B=2 ...` because three
# of these are silent-degradation hazards:
#
#   * EVOGUARD_PRIMARY_VLLM -- the runner's primary probe (and its
#     /load_lora_adapter route check) defaults to :8000. Unset, it would clear a
#     QWEN server and never test whether the LLAMA defender on :8010 can take a
#     hot-loaded adapter, so a broken hot-load chain would only surface at r1.
#   * EVOGUARD_JUDGE_LLM_BASE_URL / EVOGUARD_PROGRESS_LLM_BASE_URL -- if unset,
#     the step-safety judge falls back to a single `unclear` constant across the
#     whole attacked arm, and because advantages are group-relative that DELETES
#     the safety gradient instead of biasing it. The round would train progress
#     only, and the only symptom is a once-per-process WARNING.
#   * EVOGUARD_TRAINER_MIN_FREE_MIB -- read at IMPORT time by
#     training/native_runner.py, so it must be in the environment before python
#     starts, not set later from python.
#
# The two judges are split across :8004 (GPU2) and :8006 (GPU5), both the real
# Qwen3.5-9B, so the 32 reward-judge workers fan out over two engines instead of
# queueing on one -- a replica behind the SAME url would be dead weight, since
# each judge builder holds one client on one base_url.
#
# Serving prerequisites (see the config header for the full GPU table):
#   :8010 GPU6  Meta-Llama-3.1-8B-Instruct, served name llama3.1-8b-it, launched
#               with EVOGUARD_VLLM_EXTRA_ARGS="--enable-lora --max-lora-rank 64
#               --max-loras 4" and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
#   :8000 GPU0  qwen2.5-7b-it  attacker      :8003 GPU1  qwen2.5-7b-it  executor
#   :8004 GPU2  qwen3.5-9b     judges        :8006 GPU5  qwen3.5-9b     progress
#   GPU7        empty -- the colocate engine profiles WHOLE-CARD free memory
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

exec bash scripts/run_grpo_experiment.sh configs/agentdojo_llama31_grpo.yaml
