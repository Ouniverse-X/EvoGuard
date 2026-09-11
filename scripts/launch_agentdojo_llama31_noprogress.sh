#!/usr/bin/env bash
# Launch wrapper for the PROGRESS-TERM ABLATION arm of the Llama-3.1-8B AgentDojo
# ladder (`evoguard_agentdojo_llama31_noprogress_v1`,
# configs/agentdojo_llama31_noprogress.yaml).
#
# ONE knob differs from the GDPO-ON reference arm:
# training.grpo_disable_progress_term is true, so the trained reward is
#
#       R = r_safety - p_drift            (was: r_safety + r_progress - p_drift)
#
# GDPO stays ON, advantage-shaping lambda stays 1.5, K stays 1, same base model,
# same LoRA geometry, same dataset, same max_turns, MCTS attacker unchanged.
#
# The ENV CONTRACT below is byte-identical to the other five arms' launchers
# (scripts/launch_agentdojo_llama31{,_nogdpo,_sftonly,_vendored,_nosafety}.sh) --
# the runs differ in exactly one config knob each, so the environment must not
# drift either.
#
# THE JUDGE ROLES ARE THE REVERSE OF ARM 5 (nosafety):
#   * EVOGUARD_JUDGE_LLM_BASE_URL is now LOAD-BEARING. r_safety is the ONLY
#     surviving source of signal, and a safety-judge outage does not degrade this
#     arm, it EMPTIES it: the `unclear` fallback is one constant across the
#     attacked arm and advantages are group-relative, so the round would train on
#     -p_drift alone. Confirm NO "SAFETY term is DEAD" / judge-outage WARNING
#     naming that variable appears before trusting any round.
#   * EVOGUARD_PROGRESS_LLM_BASE_URL is now VESTIGIAL but STILL EXPORTED, for
#     contract parity with the sibling launchers. The runner skips CONSTRUCTING
#     the progress judge under the flag, so exporting it costs nothing, and
#     `progress judge active` will NOT appear in this arm's log -- that absence is
#     expected here and is NOT the check to run (see item 2 below).
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
#   1. `ABLATION grpo_disable_progress_term=True` appears once per GRPO round.
#   2. The round's progress_source_tally is {"disabled": N}. A tally containing
#      `structural` / `judge` / `fallback` means the flag did NOT reach the reward
#      and the run is measuring a progress-judge outage instead of the ablation --
#      stop and fix it. `used_progress_fallback` is NOT a valid substitute check:
#      it is True for healthy structurally-settled steps too.
#   3. `GDPO fired on N generation batches` still appears (GDPO is ON here).
#   4. safety_source_tally still shows real `judge:`/`structural:`/`clean:` keys
#      and n_safety_fallback stays small -- see the judge-role note above.
#
# READ BU/UA, NOT ASR. Two consequences of this ablation are foreseen and
# deliberately uncompensated: the CLEAN arm now trains on -p_drift alone (its
# r_safety is the per-prompt constant R_SAFETY_CLEAN_SERVED, i.e. zero gradient
# under group-relative advantages), and "do nothing" scores the structural safety
# `held` (+2.00) at p_drift 0.0, which is the arm's MAXIMUM -- so stalling is
# optimal and ASR is expected at 0.0 for the WRONG reason. The readable channels
# are BU, UA and blocked_unfinished_rate.
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
#       --config configs/agentdojo_llama31_noprogress.yaml \
#       --launch scripts/launch_agentdojo_llama31_noprogress.sh \
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
exec bash scripts/run_grpo_experiment.sh configs/agentdojo_llama31_noprogress.yaml
