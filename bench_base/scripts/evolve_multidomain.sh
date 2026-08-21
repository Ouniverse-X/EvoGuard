#!/usr/bin/env bash
# Orchestrate Multi-Domain MCTS Evolution Round 2 against BASE MODEL defender.
#
# Runs mcts_evolve.py once per (suite,seed) combination SEQUENTIALLY,
# sharing the single Qwen :8000 endpoint. Each invocation writes its own tagged
# JSONL outputs under $MCTS_BENCH_ROOT/scenarios/_synthetic/. Logs are split per
# suite-seed pair under benches/logs/.
#
# Environment overrides (defaults shown below):
#   MCTS_DEFENDER_URL=http://127.0.0.1:8000/v1
#   MCTS_JUDGE_URL=http://127.0.0.1:8002/v1
#   MCTS_GENERATIONS=8        MCTS_MAX_TASKS=10       MCTS_LOOKAHEAD=6
#   MCTS_MIN_DELTA_ACCEPT=2   MCTS_BENCH_ROOT=<repo>/bench_base
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="${PY:-/ssd1/conda_envs/evoguard/bin/python}"
export MCTS_DEFENDER_URL="${MCTS_DEFENDER_URL:-http://127.0.0.1:8000/v1}"
export MCTS_JUDGE_URL="${MCTS_JUDGE_URL:-http://127.0.0.1:8002/v1}"
export MCTS_DEFENDER_MODEL="${MCTS_DEFENDER_MODEL:-qwen2.5-7b-it}"
export MCTS_JUDGE_MODEL="${MCTS_JUDGE_MODEL:-llama3-8b-judge}"
export MCTS_GENERATIONS="${MCTS_GENERATIONS:-8}"
export MCTS_MAX_TASKS="${MCTS_MAX_TASKS:-10}"
export MCTS_LOOKAHEAD="${MCTS_LOOKAHEAD:-6}"
export MCTS_MIN_DELTA_ACCEPT="${MCTS_MIN_DELTA_ACCEPT:-2}"
export MCTS_BENCH_ROOT="${MCTS_BENCH_ROOT:-$REPO/bench_base}"

LOG_DIR="$REPO/bench_base/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
ORCHESTRATION_LOG="$LOG_DIR/orchestration_${TS}.log"
echo "[orch] start ts=$TS cfg: GENS=$MCTS_GENERATIONS TASKS=$MCTS_MAX_TASKS LOOK=$MCTS_LOOKAHEAD MIN_D=$MCTS_MIN_DELTA_ACCEPT" \
    | tee "$ORCHESTRATION_LOG"

run_one () {
    local suite="$1"; local seed="$2"
    local tag="${suite}_seed${seed}_${TS}"
    local logf="$LOG_DIR/run_${tag}.log"
    echo "[orch] >>> suite=$suite seed=$seed -> $logf"
    export MCTS_SUITE="$suite"
    export MCTS_RNG_SEED="$seed"
    "$PY" "${SCRIPT_DIR}/mcts_evolve.py" >"$logf" 2>&1 \
        || { echo "[orch] FAILED suite=$suite seed=$seed (see $logf)" | tee -a "$ORCHESTRATION_LOG"; return 1; }
    echo "[orch] done suite=$suite seed=$seed tail:" >>"$ORCHESTRATION_LOG"
    tail -n 25 "$logf" >>"$ORCHESTRATION_LOG"
}

# Suite x Seed matrix chosen to maximize diversity & coverage of remaining buckets
COMBINATIONS=(
    "workspace 371"
    "workspace 509"
    "banking    601"
    "banking    701"
    "slack      809"
    "travel     907"
)

for entry in "${COMBINATIONS[@]}"; do
    set -- $entry
    suite="$1"; seed="$2"
    run_one "$suite" "$seed" || echo "[orch] continuing past failure..." | tee -a "$ORCHESTRATION_LOG"
done

echo "[orch] ALL DONE $(date +%Y%m%d_%H%M%S)" | tee -a "$ORCHESTRATION_LOG"

# Aggregate quick stats across all newly written summaries
echo "" >>"$ORCHESTRATION_LOG"
echo "--- Summary aggregation ---" >>"$ORCHESTRATION_LOG"
for s in $(ls "$MCTS_BENCH_ROOT/scenarios/_synthetic/"_mcts_run_summary_seed*.json 2>/dev/null); do
    cat "$s" >>"$ORCHESTRATION_LOG"
    echo "" >>"$ORCHESTRATION_LOG"
done
