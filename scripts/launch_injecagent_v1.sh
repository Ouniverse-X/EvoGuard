#!/usr/bin/env bash
# Launch wrapper for the InjecAgent co-evolution run.
#
# Everything below is the ENV CONTRACT that configs/injecagent_grpo.yaml's
# header documents but that scripts/run_grpo_experiment.sh does NOT export
# itself. It lives in a file rather than a one-line `nohup env A=1 B=2 ...`
# because two of these are silent-degradation hazards:
#
#   * EVOGUARD_JUDGE_LLM_BASE_URL / EVOGUARD_PROGRESS_LLM_BASE_URL -- if unset,
#     the step-safety judge falls back to a single `unclear` constant across the
#     whole attacked arm, and because advantages are group-relative that DELETES
#     the safety gradient instead of biasing it. The round would train progress
#     only, and the only symptom is a once-per-process WARNING.
#   * EVOGUARD_TRAINER_MIN_FREE_MIB -- read at IMPORT time by
#     training/native_runner.py, so it must be in the environment before python
#     starts, not set later from python.
#
# The two judges are split across :8004 and :8006 (both the real Qwen3.5-9B on
# GPU5 / GPU6) so the 32 reward-judge workers fan out over two engines instead
# of queueing on one.
set -euo pipefail
cd /root/paddlejob/workspace/yangxiao/EvoGuard

export EVOGUARD_PY_BIN=/root/paddlejob/workspace/yangxiao/miniconda3/envs/evoguard/bin/python
export EVOGUARD_PREWARM_SEEDS_DIR=data/seeds_v3
export EVOGUARD_JUDGE_LLM_BASE_URL=http://127.0.0.1:8004/v1
export EVOGUARD_PROGRESS_LLM_BASE_URL=http://127.0.0.1:8006/v1
export EVOGUARD_JUDGE_LLM_MODEL=qwen3.5-9b
export EVOGUARD_REWARD_JUDGE_WORKERS=32
export EVOGUARD_TRAINER_MIN_FREE_MIB=40960
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAINER_CUDA_VISIBLE_DEVICES=7

exec bash scripts/run_grpo_experiment.sh configs/injecagent_grpo.yaml
