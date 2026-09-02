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

MODEL_PATH="${EVOGUARD_VLLM_MODEL:-/root/yangxiao/models/Qwen2.5-7B-Instruct}"
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

# Word-split passthrough for per-model engine flags this wrapper does not model,
# e.g. EVOGUARD_VLLM_EXTRA_ARGS="--gdn-prefill-backend triton" for Qwen3.5's
# hybrid linear-attention layers (flashinfer's sm90a gated-delta-rule prefill
# kernel aborts on the first forward pass on this driver; see
# scripts/start_vllm_secondary.sh for the full trace).
EXTRA_ARGS=()
if [[ -n "${EVOGUARD_VLLM_EXTRA_ARGS:-}" ]]; then
    read -ra EXTRA_ARGS <<<"$EVOGUARD_VLLM_EXTRA_ARGS"
fi

# Resolve which python interpreter launches the vLLM server.
#
# Priority order (highest first):
#   1. EVOGUARD_VLLM_PYBIN explicit override (e.g. /opt/conda/envs/foo/bin/python)
#   2. EVOGUARD_PY_BIN exported by scripts/setup_evoguard_env.sh's activation hook
#   3. The conda env that also runs the trainer. `evoguard2` (2026-09-02) holds
#      vllm 0.19.1 + torch 2.10.0+cu128 + transformers 4.57.6 + trl 0.19.0 +
#      peft 0.19.0: one env serves AND trains. Why this exact set:
#        * vllm >= 0.20 ships a CUDA-13 wheel stack (nvidia-*-cu13) which needs
#          driver >= 580; this box is 550.127.08 / CUDA 12.4, so 0.19.1 is the
#          newest usable release. It registers Qwen3_5ForConditionalGeneration,
#          so Qwen3.5-9B serves here and Qwen2.5 keeps its SupportsLoRA path.
#        * torch 2.10 is what fixes the r0 crash: peft 0.19's UPCAST_DTYPES
#          references torch.float8_e8m0fnu (torch >= 2.7), which the previous
#          env's torch 2.6.0 did not define, so get_peft_model() died on the
#          very first SFT.
#        * trl stays at 0.19.0 on purpose -- _DeltaShapedGRPOTrainer overrides
#          GRPOTrainer._generate_and_score_completions (GDPO + trajectory pooling
#          + Delta shaping all hang off it), so trl 1.x is a code migration, not
#          an env bump.
#      The predecessor `evoguard` env (vllm 0.8.5 + torch 2.6) is left intact as
#      the rollback target.
if [[ -n "${EVOGUARD_VLLM_PYBIN:-}" && -x "$EVOGUARD_VLLM_PYBIN" ]]; then
    VLLM_PY="$EVOGUARD_VLLM_PYBIN"
elif [[ -n "${EVOGUARD_PY_BIN:-}" && -x "$EVOGUARD_PY_BIN" ]]; then
    VLLM_PY="$EVOGUARD_PY_BIN"
else
    VLLM_PY="/root/miniconda3/envs/evoguard2/bin/python"
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

# vLLM gates the /v1/load_lora_adapter + /v1/unload_lora_adapter HTTP routes
# behind env var VLLM_ALLOW_RUNTIME_LORA_UPDATING (0.8.x:
# entrypoints/openai/api_server.py; 0.19.x: entrypoints/serve/lora/api_router.py).
# Passing --enable-lora alone is NOT enough -- without this env flag the routes
# never register and our per-round hot-load call
# (scripts/register_vllm_lora.sh) returns HTTP 404.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-1}"

# Engine selection, decided by ASKING the interpreter rather than by assumption.
#
# On vllm 0.8.x the legacy V0 engine was mandatory for dynamic LoRA: V1 did not
# finish initialising lora_manager under --enable-lora, so every
# POST /v1/load_lora_adapter died with
#   AttributeError: 'GPUModelRunner' object has no attribute 'lora_manager'
# V0 also sidestepped V1's memory profiling, which refuses to start at a low
# --gpu-memory-utilization when another process already holds part of the card.
#
# vllm >= ~0.11 REMOVED V0 and with it the VLLM_USE_V1 knob (it is absent from
# vllm.envs.environment_variables on 0.19.1). Exporting it there is at best dead
# weight and at worst confusing when reading a log, and V1 LoRA hot-load has
# worked since 0.9. So probe for the knob and only set it when it exists.
if "$VLLM_PY" -c "import sys, vllm.envs as e; sys.exit(0 if 'VLLM_USE_V1' in getattr(e, 'environment_variables', {}) else 1)" >/dev/null 2>&1; then
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"
    echo "   engine       : V0 (VLLM_USE_V1=$VLLM_USE_V1; required for LoRA hot-load on vllm 0.8.x)"
else
    unset VLLM_USE_V1 2>/dev/null || true
    echo "   engine       : V1 only (this vLLM has no VLLM_USE_V1 knob)"
fi

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
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
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
