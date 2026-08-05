#!/usr/bin/env bash
# Auto-retry launcher: scans GPUs in tight loop and fires the sequence-probe
# with `--pre-reserve-gb` to atomically claim territory before model load.
#
# Tier-1 update: env var MAX_THOUGHT_TOKENS overrides the new default 128-cap
# when GPU RAM budget requires it. Pass e.g. `MAX_THOUGHT_TOKENS=96 bash ...`
set -u
SCENARIOS_JSON="${1:-rounds/_preliminary/20260803_alertness_v3/all_scenarios.json}"
OUTPUT_DIR="${2:-rounds/_preliminary/20260803_alertness_v3/seq_probe}"
LIMIT_FLAG=""
if [ -n "${LIMIT_SCENARIOS:-}" ]; then
    LIMIT_FLAG="--limit-scenarios $LIMIT_SCENARIOS"
fi
RESERVE_GB="${RESERVE_GB:-12}"
MAXTT_FLAG=""
if [ -n "${MAX_THOUGHT_TOKENS:-}" ]; then
    MAXTT_FLAG="--max-thought-tokens ${MAX_THOUGHT_TOKENS}"
fi

MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2}"

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    # Pick first idle-ish card right now.
    target_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' '{ gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 500) print $1 }' \
        | head -1)
    if [ -z "$target_gpu" ]; then
        echo "[launcher] attempt $attempt/$MAX_ATTEMPTS no idle GPU; sleeping ${SLEEP_BETWEEN}s"
        sleep "$SLEEP_BETWEEN"
        continue
    fi
    echo "[launcher] attempt $attempt firing on GPU#$target_gpu max_thought_tokens=${MAX_THOUGHT_TOKENS:-<default 128>}"
    CUDA_VISIBLE_DEVICES=$target_gpu \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /ssd1/conda_envs/evoguard/bin/python -m preliminary.probe \
        --scenarios-json "$SCENARIOS_JSON" \
        --output-dir "$OUTPUT_DIR" \
        --cuda-visible-devices "$target_gpu" \
        --pre-reserve-gb "$RESERVE_GB" \
        $LIMIT_FLAG \
        $MAXTT_FLAG 2>&1 | tail -25
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        echo "[launcher] SUCCESS rc=$rc"
        exit 0
    fi
    echo "[launcher] failure rc=$rc; retrying in ${SLEEP_BETWEEN}s"
    sleep "$SLEEP_BETWEEN"
done
echo "[launcher] exhausted $MAX_ATTEMPTS attempts."
exit 99
