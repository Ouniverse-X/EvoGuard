#!/usr/bin/env bash
# Launch a HELPER vLLM OpenAI-compatible server for the non-defender LLM roles
# (tool_executor / judge / utility_judge).
#
# Unlike the primary server this one needs no LoRA support -- the helper roles
# always run on stock base weights -- so it can be started multiple times on
# different ports/GPUs to spread rollout load. PID/log filenames are derived
# from the port so concurrent instances never clobber each other:
#
#   EVOGUARD_VLLM_PORT=8002 EVOGUARD_VLLM_GPU=2,3 scripts/start_vllm_secondary.sh
#   EVOGUARD_VLLM_PORT=8003 EVOGUARD_VLLM_GPU=5,6 scripts/start_vllm_secondary.sh
#
# Tensor-parallel size is derived from how many devices EVOGUARD_VLLM_GPU lists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

MODEL_PATH="${EVOGUARD_VLLM_MODEL:-/root/paddlejob/workspace/yangxiao/models/Qwen2.5-7B-Instruct}"
PORT="${EVOGUARD_VLLM_PORT:-8002}"
GPU_ID="${EVOGUARD_VLLM_GPU:-2,3}"
SERVED_NAME="${EVOGUARD_VLLM_NAME:-qwen2.5-7b-it}"
MEM_UTIL="${EVOGUARD_VLLM_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${EVOGUARD_VLLM_MAXLEN:-16384}"

_TP_FROM_GPUS=$(awk -F',' '{n=0; for(i=1;i<=NF;i++) if ($i != "") n++; print n}' <<<"$GPU_ID")
TP_SIZE="${EVOGUARD_VLLM_TP:-$_TP_FROM_GPUS}"

# Word-split passthrough for per-model engine flags this wrapper does not model.
# Needed by Qwen3.5's hybrid linear-attention layers: flashinfer 0.6.6's
# JIT-compiled sm90a gated-delta-rule prefill kernel aborts on the FIRST forward
# pass on this box --
#   gdn_prefill_launcher.cu:60: delta rule kernel does not support this device
#   major version: 0
# -- while the server still reports /v1/models healthy, so the failure only shows
# up as a dead process on the first real request. vLLM's own Triton/FLA path is
# unaffected:
#   EVOGUARD_VLLM_EXTRA_ARGS="--gdn-prefill-backend triton"
EXTRA_ARGS=()
if [[ -n "${EVOGUARD_VLLM_EXTRA_ARGS:-}" ]]; then
    read -ra EXTRA_ARGS <<<"$EVOGUARD_VLLM_EXTRA_ARGS"
fi

PYBIN="${EVOGUARD_JUDGE_PYBIN:-${EVOGUARD_VLLM_PYBIN:-/root/paddlejob/workspace/yangxiao/miniconda3/envs/evoguard/bin/python}}"
if [[ ! -x "$PYBIN" ]]; then
    echo "[FATAL] python interpreter not executable: $PYBIN" >&2
    exit 127
fi

mkdir -p rounds
PID_FILE="rounds/vllm_helper_${PORT}.pid"
LOG_FILE="rounds/vllm_helper_${PORT}.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    EXISTING_PID=$(cat "$PID_FILE")
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[SKIP] healthy helper vLLM already listening on :$PORT (pid=$EXISTING_PID)"
        exit 0
    fi
    echo "[WARN] stale PID file but port unresponsive -> killing old proc & relaunching" >&2
    kill -KILL "$EXISTING_PID" 2>/dev/null || true
fi

cat <<EOF
Launching helper vLLM server:
   role            : TOOL_EXECUTOR / JUDGE / UTILITY_JUDGE backend (no LoRA)
   model           : $MODEL_PATH   (served_name=$SERVED_NAME)
   pybin           : $PYBIN
   gpu             : cuda:$GPU_ID (tensor_parallel_size=$TP_SIZE)
   mem_util        : $MEM_UTIL
   max_model_len   : $MAX_MODEL_LEN
   port            : $PORT
   log file        : $LOG_FILE
EOF
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "   extra_args      : ${EXTRA_ARGS[*]}"

# Engine selection: probe for the knob instead of assuming it exists. vllm 0.8.x
# needed VLLM_USE_V1=0 for parity with the primary server (see
# scripts/start_vllm.sh); vllm >= ~0.11 removed V0 and the knob with it, so on
# 0.19.1 the variable is not in vllm.envs.environment_variables at all.
if "$PYBIN" -c "import sys, vllm.envs as e; sys.exit(0 if 'VLLM_USE_V1' in getattr(e, 'environment_variables', {}) else 1)" >/dev/null 2>&1; then
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"
else
    unset VLLM_USE_V1 2>/dev/null || true
fi

# With tensor_parallel_size>1 vllm stands up a torch.distributed TCPStore whose
# port comes from get_open_port(). Two instances launched close together can pick
# the same one and the second dies with
# "server socket has failed to listen ... EADDRINUSE". VLLM_PORT pins the search
# base, so derive a per-instance base from the API port to keep the ranges apart.
export VLLM_PORT="${VLLM_PORT:-$((20000 + (PORT - 8000) * 100))}"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
nohup "$PYBIN" -m vllm.entrypoints.openai.api_server \
    --model             "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port              "$PORT" \
    --host              127.0.0.1 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len     "$MAX_MODEL_LEN" \
    --trust-remote-code \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    >"$LOG_FILE" 2>&1 &
SEC_PID=$!
echo "$SEC_PID" >"$PID_FILE"
disown "$SEC_PID" || true

echo ""
echo "Launched helper vLLM pid=$SEC_PID on :$PORT; waiting up to 10 minutes for readiness..."
HEALTH_URL="http://127.0.0.1:${PORT}/v1/models"

for i in $(seq 1 120); do
    if ! kill -0 "$SEC_PID" 2>/dev/null; then
        echo "[FATAL] helper vLLM exited early. Last 40 lines of log:" >&2
        tail -n 40 "$LOG_FILE" >&2 || true
        rm -f "$PID_FILE"
        exit 1
    fi
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
        echo "[OK] ready after ~$((i*5))s at $HEALTH_URL"
        tail -n 4 "$LOG_FILE" || true
        exit 0
    fi
    sleep 5
done

echo "[TIMEOUT] not ready within 600s." >&2
tail -n 30 "$LOG_FILE" >&2 || true
exit 2
