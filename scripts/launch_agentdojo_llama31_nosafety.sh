#!/usr/bin/env bash
# Launch wrapper for the SAFETY-TERM ABLATION arm of the Llama-3.1-8B AgentDojo
# ladder (`evoguard_agentdojo_llama31_nosafety_v1`,
# configs/agentdojo_llama31_nosafety.yaml).
#
# ONE knob differs from the GDPO-ON reference arm: training.grpo_disable_safety_term
# is true, so the trained reward is
#
#       R = r_progress - p_drift          (was: r_safety + r_progress - p_drift)
#
# GDPO stays ON, advantage-shaping lambda stays 1.5, K stays 1, same base model,
# same LoRA geometry, same dataset, same max_turns, MCTS attacker unchanged.
#
# The ENV CONTRACT below is byte-identical to the other four arms' launchers
# (scripts/launch_agentdojo_llama31{,_nogdpo,_sftonly,_vendored}.sh) -- the runs
# differ in exactly one config knob each, so the environment must not drift either.
#
# EVOGUARD_JUDGE_LLM_BASE_URL IS STILL EXPORTED, deliberately, even though this arm
# never calls the step-safety judge. Two reasons: (i) contract parity, and (ii) the
# progress judge FALLS BACK to this variable when its own is unset, so dropping it
# would silently couple the two ablations. The runner skips CONSTRUCTING the safety
# judge under the flag, so exporting it costs nothing here.
#
# EVOGUARD_PROGRESS_LLM_BASE_URL is now LOAD-BEARING IN A WAY IT WAS NOT BEFORE.
# r_progress is the ONLY surviving source of signal, so a progress-judge outage
# does not degrade this arm, it EMPTIES it: structural progress can only tell
# waste from neutral and can never award advance/+1.20. Before trusting any round,
# confirm the log shows `progress judge active` with no WARNING naming that
# variable.
#
# Other silent-degradation hazards (same as the sibling arms):
#   * EVOGUARD_PRIMARY_VLLM         -- the runner's primary probe defaults to
#                                      :8000, but the defender is on :8010.
#   * EVOGUARD_TRAINER_MIN_FREE_MIB -- read at IMPORT time by native_runner.py.
#   * EVOGUARD_PREWARM_SEEDS_DIR    -- feeds the MCTS attacker's initial
#                                      population; live in this arm (unlike the
#                                      vendored one).
#
# WHAT TO VERIFY IN THE FIRST GRPO ROUND, before spending five hours:
#   1. `ABLATION grpo_disable_safety_term=True` appears once per GRPO round.
#   2. The round's safety_source_tally is {"disabled:disabled": N}. A tally
#      containing `fallback:` means the flag did NOT reach the reward and the run
#      is measuring a judge outage instead of the ablation -- stop and fix it.
#   3. `GDPO fired on N generation batches` still appears (GDPO is ON here).
#   4. `progress judge active` with no WARNING -- see above.
#
# Serving prerequisites (identical to the other arms):
#   :8010 GPU6  llama3.1-8b-it  defender (--enable-lora --max-lora-rank 64
#               --max-loras 4 + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1)
#   :8000 GPU0  qwen2.5-7b-it   attacker      :8003 GPU1  qwen2.5-7b-it executor
#   :8004 GPU2  qwen3.5-9b      judges        :8006 GPU5  qwen3.5-9b    progress
#   GPU7        EMPTY -- the colocate engine profiles WHOLE-CARD free memory, so
#               pause any keeper ballast on it first (`gpu_keeper.sh pause -g 7`
#               BLOCKS until the ballast is really gone).
#
# This is a GRPO arm, so the colocate cap applies (~19 GiB of the trainer card
# leaks per round, ~2 rounds per process). Leg 0 is launched by hand with
# max_rounds: 2 because run_legged_experiment.py cannot bootstrap a fresh
# experiment (latest_trained_round() reads <exp>/latest_adapter_dir.txt). Then:
#   python scripts/run_legged_experiment.py \
#       --config configs/agentdojo_llama31_nosafety.yaml \
#       --launch scripts/launch_agentdojo_llama31_nosafety.sh \
#       --target-rounds 9 --rounds-per-leg 2 --trainer-gpu 7
# (add --wait-pid <leg0 pid> to start the chainer while leg 0 is still running).
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
exec bash scripts/run_grpo_experiment.sh configs/agentdojo_llama31_nosafety.yaml
