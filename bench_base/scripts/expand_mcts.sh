#!/usr/bin/env bash
# Expansion run: MCTS evolution targeting d3/d4 enrichment with fresh seeds.
# Uses higher generation count and broader task pool to maximize d3/d4 yield.
# Runs sequentially per (suite, seed) sharing single vLLM endpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO"

PY="${PY:-/ssd1/conda_envs/evoguard/bin/python}"
export MCTS_DEFENDER_URL="${MCTS_DEFENDER_URL:-http://127.0.0.1:8000/v1}"
export MCTS_JUDGE_URL="${MCTS_JUDGE_URL:-http://127.0.0.1:8002/v1}"
export MCTS_DEFENDER_MODEL="${MCTS_DEFENDER_MODEL:-qwen2.5-7b-it}"
export MCTS_JUDGE_MODEL="${MCTS_JUDGE_MODEL:-gemma-2-9b-it-fallback-judge}"
export MCTS_GENERATIONS="${MCTS_GENERATIONS:-12}"
export MCTS_MAX_TASKS="${MCTS_MAX_TASKS:-12}"
export MCTS_LOOKAHEAD="${MCTS_LOOKAHEAD:-8}"
export MCTS_MIN_DELTA_ACCEPT="${MCTS_MIN_DELTA_ACCEPT:-2}"
export MCTS_BENCH_ROOT="${MCTS_BENCH_ROOT:-$REPO/bench_base}"

LOG_DIR="$REPO/bench_base/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
ORCHESTRATION_LOG="$LOG_DIR/expansion_${TS}.log"
echo "[expand] start ts=$TS GENS=$MCTS_GENERATIONS TASKS=$MCTS_MAX_TASKS LOOK=$MCTS_LOOKAHEAD" \
    | tee "$ORCHESTRATION_LOG"

run_one () {
    local suite="$1"; local seed="$2"
    local tag="${suite}_seed${seed}_exp_${TS}"
    local logf="$LOG_DIR/run_${tag}.log"
    echo "[expand] >>> suite=$suite seed=$seed -> $logf" | tee -a "$ORCHESTRATION_LOG"
    export MCTS_SUITE="$suite"
    export MCTS_RNG_SEED="$seed"
    "$PY" "${SCRIPT_DIR}/mcts_evolve.py" >"$logf" 2>&1 \
        || { echo "[expand] FAILED suite=$suite seed=$seed (see $logf)" | tee -a "$ORCHESTRATION_LOG"; return 1; }
    echo "[expand] done suite=$suite seed=$seed" | tee -a "$ORCHESTRATION_LOG"
    tail -n 15 "$logf" >>"$ORCHESTRATION_LOG"
}

# Fresh seeds specifically targeting domains where d3/d4 have been thinnest.
# Higher generation count (12) + larger task pool (12) + lookahead 8 for deeper search.
COMBINATIONS=(
    "workspace 1001"
    "workspace 1103"
    "workspace 1207"
    "banking   1301"
    "banking   1409"
    "slack     1501"
    "slack     1607"
    "travel    1701"
    "travel    1803"
)

for entry in "${COMBINATIONS[@]}"; do
    set -- $entry
    suite="$1"; seed="$2"
    run_one "$suite" "$seed" || echo "[expand] continuing past failure..." | tee -a "$ORCHESTRATION_LOG"
done

echo "[expand] ALL DONE $(date +%Y%m%d_%H%M%S)" | tee -a "$ORCHESTRATION_LOG"

# Quick stats
echo "" >>"$ORCHESTRATION_LOG"
echo "--- _synthetic file sizes after expansion ---" >>"$ORCHESTRATION_LOG"
wc -l "$MCTS_BENCH_ROOT/scenarios/_synthetic/bucket_d"*.jsonl 2>/dev/null >>"$ORCHESTRATION_LOG"
