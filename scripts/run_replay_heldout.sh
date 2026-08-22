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

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADAPTER="${1:?usage: $0 <adapter_name|none> [tag] [concurrency]}"
TAG="${2:-$(echo "$ADAPTER" | tr -c '[:alnum:]_' '_')}"
CONC="${3:-8}"
CONFIG="${EVOGUARD_REPLAY_CONFIG:-configs/agentdojo_full_last_traj.yaml}"
PY="${EVOGUARD_PYTHON:-/root/yangxiao/envs/train/bin/python}"
OUT="rounds/replay_test_${TAG}"

mkdir -p "$OUT"
PYTHONPATH="$PWD" "$PY" -m evoguard.eval.vendored_replay \
    --config "$CONFIG" \
    --dataset-dir data/toolsafe/agentdojo-tragjnew/test \
    --split test \
    --lora-adapter "$ADAPTER" \
    --concurrency "$CONC" \
    --output-dir "$OUT" 2>&1 | tee "$OUT/replay.log"
