#!/usr/bin/env bash
# Launch wrapper for the ATTACKER ABLATION arm of the Llama-3.1-8B AgentDojo ladder
# (`evoguard_agentdojo_llama31_vendored_v1`, configs/agentdojo_llama31_vendored.yaml).
# The attack generator does not participate at all: `attacker.search_method:
# vendored` replays the <INFORMATION> injections already shipped in
# data/toolsafe/agentdojo-tragjnew/train, frozen (byte-identical every round), with
# no attacker LLM call and a no-op evolve(). Everything else is the GDPO-ON arm:
# GDPO true, advantage-shaping lambda 1.5, R = r_safety + r_progress - p_drift,
# K=1, the same base model and LoRA geometry, the same judges.
#
# The ENV CONTRACT below is byte-identical to the other three arms' launchers
# (scripts/launch_agentdojo_llama31{,_nogdpo,_sftonly}.sh). That is deliberate: the
# four runs differ in exactly one config knob each, so the environment must not
# drift either. Read launch_agentdojo_llama31.sh's header for why each variable is a
# silent-degradation hazard; the short version:
#   * EVOGUARD_PRIMARY_VLLM      -- the runner's primary probe defaults to :8000,
#                                   but the defender is on :8010.
#   * EVOGUARD_JUDGE_LLM_BASE_URL / EVOGUARD_PROGRESS_LLM_BASE_URL -- unset, the
#                                   step-safety judge returns one constant
#                                   `unclear` and the group-relative advantages
#                                   DELETE the safety gradient.
#   * EVOGUARD_TRAINER_MIN_FREE_MIB -- read at IMPORT time by native_runner.py.
#   * EVOGUARD_PREWARM_SEEDS_DIR -- kept for contract parity but INERT here: seed
#                                   prewarming feeds the attacker's initial
#                                   population, and this arm never seeds one.
#
# Serving prerequisites (identical to the other arms):
#   :8010 GPU6  llama3.1-8b-it  defender (--enable-lora --max-lora-rank 64
#               --max-loras 4 + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
#   :8000 GPU0  qwen2.5-7b-it   attacker      :8003 GPU1  qwen2.5-7b-it executor
#   :8004 GPU2  qwen3.5-9b      judges        :8006 GPU5  qwen3.5-9b    progress
#   GPU7        EMPTY -- the colocate engine profiles WHOLE-CARD free memory
# :8000 is still required even though no attack is generated: the attacker LLM
# client is constructed at setup regardless of the search backend.
#
# This is a GRPO arm, so the colocate cap applies (the engine leaks ~19 GiB of the
# trainer card per round, ~2 rounds per process). Drive it through the leg chainer,
# not directly:
#   python scripts/run_legged_experiment.py \
#       --config configs/agentdojo_llama31_vendored.yaml \
#       --launch scripts/launch_agentdojo_llama31_vendored.sh \
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

exec bash scripts/run_grpo_experiment.sh configs/agentdojo_llama31_vendored.yaml
