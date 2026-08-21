#!/usr/bin/env bash
# Launch the PRIMARY vLLM OpenAI-compatible server: the LoRA-enabled defender
# backend that evoguard points `defense.llm.base_url` at.
#
# The server is launched in the background via nohup; logs go to
# rounds/vllm.log and the PID is recorded under rounds/vllm.pid so callers can
# stop it cleanly later (`scripts/stop_vllm.sh`).
#
# Interpreter: /root/yangxiao/envs/vllm085 (vllm 0.8.5.post1 + torch 2.6.0+cu124).
# It is deliberately SEPARATE from the training venv: vllm 0.8.5 pins
# transformers ~4.51 while the native TRL trainers run on transformers 5.x.
# Keeping two venvs lets each side hold its own pins without conflict.
#
# GPU_ID accepts a comma-separated list; tensor-parallel size is derived from
# how many devices are listed (e.g. EVOGUARD_VLLM_GPU="0,1" -> --tensor-parallel-size 2).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

MODEL_PATH="${EVOGUARD_VLLM_MODEL:-/root/yangxiao/models/downloads/qwen2.5-7b-instruct}"
PORT="${EVOGUARD_VLLM_PORT:-8000}"
GPU_ID="${EVOGUARD_VLLM_GPU:-0,1}"
SERVED_NAME="${EVOGUARD_VLLM_NAME:-qwen2.5-7b-it}"
MEM_UTIL="${EVOGUARD_VLLM_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${EVOGUARD_VLLM_MAXLEN:-16384}"

# Tensor-parallel size = number of devices listed in GPU_ID unless overridden.
_TP_FROM_GPUS=$(awk -F',' '{n=0; for(i=1;i<=NF;i++) if ($i != "") n++; print n}' <<<"$GPU_ID")
TP_SIZE="${EVOGUARD_VLLM_TP:-$_TP_FROM_GPUS}"

# Dynamic-LoRA support flags. When ENABLE_LORA != "0" we pass ``--enable-lora``
# along with rank/capacity hints so newly-trained adapters can be hot-loaded
# mid-experiment via POST /v1/load_lora_adapter instead of restarting the server
# every co-evolution round. Pre-loaded adapter paths can be supplied as
# EVOGUARD_VLLM_LORA_MODULES="<name1>=<path1>,<name2>=<path2>,...".
ENABLE_LORA="${EVOGUARD_VLLM_ENABLE_LORA:-1}"
MAX_LORA_RANK="${EVOGUARD_VLLM_MAX_LORA_RANK:-64}"
MAX_LORAS="${EVOGUARD_VLLM_MAX_LORAS:-4}"
LORA_MODULES_CSV="${EVOGUARD_VLLM_LORA_MODULES:-}"

# Resolve which python interpreter launches the vLLM server.
#
# Priority order (highest first):
#   1. EVOGUARD_VLLM_PYBIN explicit override (e.g. /opt/conda/envs/foo/bin/python)
#   2. EVOGUARD_PY_BIN exported by scripts/setup_evoguard_env.sh's activation hook
#   3. The dedicated vllm-serving venv built for this box.
if [[ -n "${EVOGUARD_VLLM_PYBIN:-}" && -x "$EVOGUARD_VLLM_PYBIN" ]]; then
    VLLM_PY="$EVOGUARD_VLLM_PYBIN"
elif [[ -n "${EVOGUARD_PY_BIN:-}" && -x "$EVOGUARD_PY_BIN" ]]; then
    VLLM_PY="$EVOGUARD_PY_BIN"
else
    VLLM_PY="/root/yangxiao/envs/vllm085/bin/python"
fi
echo "   python_bin   : $VLLM_PY"

mkdir -p rounds
LOG_FILE="rounds/vllm.log"
PID_FILE="rounds/vllm.pid"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "vLLM already running as PID $(cat $PID_FILE); not relaunching." >&2
    exit 0
fi

echo "Launching vLLM server:"
echo "   model         : $MODEL_PATH"
echo "   served_name   : $SERVED_NAME"
echo "   port          : $PORT"
echo "   gpu           : cuda:$GPU_ID (tensor_parallel_size=$TP_SIZE)"
echo "   mem_util      : $MEM_UTIL"
echo "   max_model_len : $MAX_MODEL_LEN"
if [[ "$ENABLE_LORA" != "0" ]]; then
    echo "   enable_lora    : yes (max_rank=$MAX_LORA_RANK, max_loras=$MAX_LORAS)"
    [[ -n "$LORA_MODULES_CSV" ]] && echo "   preloaded_loras: $LORA_MODULES_CSV"
fi
echo "   log file      : $LOG_FILE"

# Build the dynamic-LoRA argument block. We always pass --enable-lora when
# ENABLE_LORA != 0 so callers can register freshly-trained adapters at runtime
# via POST /v1/load_lora_adapter without restarting the server between rounds.
LORA_ARGS=()
if [[ "$ENABLE_LORA" != "0" ]]; then
    LORA_ARGS+=(
        --enable-lora
        --max-lora-rank "$MAX_LORA_RANK"
        --max-loras     "$MAX_LORAS"
    )
    # Expand "<name1>=<path1>,<name2>=<path2>" into repeated "--lora-modules name path".
    if [[ -n "$LORA_MODULES_CSV" ]]; then
        IFS=',' read -ra _pairs <<<"$LORA_MODULES_CSV"
        for kv in "${_pairs[@]}"; do
            name="${kv%%=*}"
            path="${kv#*=}"
            if [[ -z "$name" || -z "$path" ]]; then
                continue
            fi
            LORA_ARGS+=(--lora-modules "$name" "$path")
        done
    fi
fi

# vLLM 0.8.x gates the /v1/load_lora_adapter + /v1/unload_lora_adapter HTTP
# routes behind env var VLLM_ALLOW_RUNTIME_LORA_UPDATING (see
# vllm/entrypoints/openai/api_server.py:783). Passing --enable-lora alone is NOT
# enough -- without this env flag the routes never register and our pipeline's
# per-round hot-load call (scripts/register_vllm_lora.sh) returns HTTP 404.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-1}"

# Force LEGACY engine path. The experimental V1 engine in vllm 0.8.x does NOT
# fully initialize lora_manager when --enable-lora is passed, causing every
# POST /v1/load_lora_adapter call to fail with
#   AttributeError: 'GPUModelRunner' object has no attribute 'lora_manager'
# The legacy engine handles dynamic LoRA hot-loading correctly end-to-end as
# verified on 2026-07-21 with a probe server launched under evoguard python3.12.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model             "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port              "$PORT" \
    --host              127.0.0.1 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len     "$MAX_MODEL_LEN" \
    --trust-remote-code \
    "${LORA_ARGS[@]}" \
    >"$LOG_FILE" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" >"$PID_FILE"
disown "$VLLM_PID" || true

echo ""
echo "Launched vLLM pid=$VLLM_PID."
echo "Waiting up to 10 minutes for readiness..."

HEALTH_URL="http://127.0.0.1:${PORT}/v1/models"
for i in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[FATAL] vLLM process exited early. Last 30 lines of log:" >&2
        tail -n 30 "$LOG_FILE" >&2 || true
        rm -f "$PID_FILE"
        exit 1
    fi
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
        echo "[OK] vLLM ready after ~$((i*5))s at $HEALTH_URL"
        echo "(tail of startup log:)"
        tail -n 4 "$LOG_FILE" || true
        exit 0
    fi
    sleep 5
done

echo "[TIMEOUT] vLLM did not become healthy within 600s." >&2
echo "--- last 40 lines of log ---" >&2
tail -n 40 "$LOG_FILE" >&2 || true
exit 2
