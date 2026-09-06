#!/usr/bin/env bash
# Full-trajectory-replay evaluation on the held-out test split.
#
# This is the NON-stepwise evaluator: evoguard/eval/vendored_replay.py drives the
# real Controller + ToolEnv, so every step executes tools and the loop runs to
# max_turns or a final answer. Contrast with:
#   * eval/stepwise_eval.py        -> ONE agent.decide() per record, no env, no loop
#   * bench_base/.../eval_full_rollout.py -> loops decide() but appends observation=""
# Both of those are unsuitable for judging termination / utility behaviour.
#
# Usage: scripts/run_replay_heldout.sh <adapter_name|none> [tag] [concurrency]
#
# Adapters must already be hot-registered on the defender vLLM:
#   scripts/register_vllm_lora.sh <name> <adapter_weights_dir> 8000
#
# Dataset selection is overridable so this stays the single replay entry point:
#   EVOGUARD_REPLAY_CONFIG      config YAML (default: agentdojo)
#   EVOGUARD_REPLAY_DATASET_DIR leaf split dir the ATTACKS are read from
#   EVOGUARD_REPLAY_SPLIT       metadata["split"] filter on env TASKS; set to
#                               "" to disable. Must be "" for ASB-OPI and
#                               InjecAgent, whose split unit is the attack
#                               instance / attacker case rather than the task --
#                               filtering tasks there would discard most of the
#                               split's attacks, and InjecAgent declares no
#                               task-level split at all (see their READMEs).
# ASB-OPI test split:
#   EVOGUARD_REPLAY_CONFIG=configs/asb_opi_grpo.yaml \
#   EVOGUARD_REPLAY_DATASET_DIR=data/ASB/splits/test \
#   EVOGUARD_REPLAY_SPLIT= scripts/run_replay_heldout.sh none asb_test
# InjecAgent test split:
#   EVOGUARD_REPLAY_CONFIG=configs/injecagent_grpo.yaml \
#   EVOGUARD_REPLAY_DATASET_DIR=data/InjecAgent/splits/test \
#   EVOGUARD_REPLAY_SPLIT= scripts/run_replay_heldout.sh none injecagent_test

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADAPTER="${1:?usage: $0 <adapter_name|none> [tag] [concurrency]}"
TAG="${2:-$(echo "$ADAPTER" | tr -c '[:alnum:]_' '_')}"
CONC="${3:-8}"
CONFIG="${EVOGUARD_REPLAY_CONFIG:-configs/agentdojo_full_last_traj.yaml}"
DATASET_DIR="${EVOGUARD_REPLAY_DATASET_DIR:-data/toolsafe/agentdojo-tragjnew/test}"
# ${VAR-default}, not ${VAR:-default}: an explicitly empty value must survive.
SPLIT="${EVOGUARD_REPLAY_SPLIT-test}"
PY="${EVOGUARD_PYTHON:-/root/yangxiao/envs/train/bin/python}"
OUT="rounds/replay_test_${TAG}"

SPLIT_ARGS=()
if [[ -n "$SPLIT" ]]; then
    SPLIT_ARGS=(--split "$SPLIT")
fi

mkdir -p "$OUT"
# ${arr[@]+"${arr[@]}"}, not "${arr[@]}": bash < 4.4 (this box ships 4.2) treats an
# empty array expansion as an unbound variable under `set -u`, which made the
# documented EVOGUARD_REPLAY_SPLIT= (ASB-OPI / InjecAgent) invocation unrunnable.
PYTHONPATH="$PWD" "$PY" -m evoguard.eval.vendored_replay \
    --config "$CONFIG" \
    --dataset-dir "$DATASET_DIR" \
    ${SPLIT_ARGS[@]+"${SPLIT_ARGS[@]}"} \
    --lora-adapter "$ADAPTER" \
    --concurrency "$CONC" \
    --output-dir "$OUT" 2>&1 | tee "$OUT/replay.log"
