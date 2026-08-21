#!/usr/bin/env bash
# Launch stepwise evaluation of a trained defender LoRA adapter against the
# full vendored AgentDojo trajectory test set (data/toolsafe/agentdojo-tragj).
#
# Pre-reqs:
#   * Primary vLLM (:8000) running with --enable-lora + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
#   * Secondary judge vLLM (:8002) running
#   * Trained adapter already registered under name matching defense.llm.lora_adapter in YAML.
#
set -euo pipefail

CONFIG="${1:-configs/agentdojo_full_v4grpo_v2_test.yaml}"
PY=/ssd1/conda_envs/evoguard/bin/python

echo "[preflight] config: $CONFIG"
echo "[preflight] python: $PY ($($PY -V 2>&1))"

curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null \
    && echo "[preflight] OK primary vLLM :8000 reachable" \
    || { echo "[FAIL] primary vLLM :8000 not responding"; exit 1; }

curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null \
    && echo "[preflight] OK secondary vLLM :8002 reachable" \
    || { echo "[FAIL] secondary vLLM :8002 not responding"; exit 1; }

ADAPTER=$("$PY" -c "
import sys; sys.path.insert(0,'.')
from evoguard.config import ExperimentConfig
cfg = ExperimentConfig.from_file('$CONFIG')
print(cfg.defense.llm.lora_adapter or 'none')
")
echo "[preflight] lora_adapter: $ADAPTER"

if [ "$ADAPTER" != "none" ]; then
    curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null | "$PY" -c "
import json,sys
ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]
target='$ADAPTER'
assert target in ids, f'adapter {target!r} not registered on :8000'
print(f'[preflight] OK adapter {target} registered')
"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="rounds/stepwise_eval/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_${TS}.log"

echo "
[launching]
   pid        : (see below)
   log file      : $LOG

Tail logs:
   tail -F \"$LOG\" | tee /dev/stderr 2>/dev/null
" >&2

nohup "$PY" -m evoguard.eval.stepwise_eval \
    --config "$CONFIG" \
    >"$LOG" 2>&1 &

PID=$!
echo $PID
echo ""
echo "[launched]"
echo "   pid         : $PID"
echo "   started_at  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "   log         : $LOG"
echo ""
echo "Stop run cleanly:"
echo "   kill $PID && pkill -P $PID 2>/dev/null || true"
