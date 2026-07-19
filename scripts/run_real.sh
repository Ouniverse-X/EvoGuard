#!/usr/bin/env bash
# Launch the real-model EvoGuard experiment under nohup, with QianFan
# credentials injected from a non-versioned secrets file.
#
# Usage:
#   scripts/run_real.sh configs/real_qianfan.yaml
#
# Secrets file (default: ~/.evoguard_qianfan.env) is sourced if present and must
# export EVOGUARD_QIANFAN_APPID + EVOGUARD_QIANFAN_TOKEN. If absent we fall back
# to inline literals matching docs/todo.md so out-of-the-box runs still work.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CONFIG="${1:-configs/real_qianfan.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "error: config not found: $CONFIG" >&2; exit 2
fi

SECRETS_FILE="${EVOGUARD_SECRETS_FILE:-$HOME/.evoguard_qianfan.env}"
if [[ -f "$SECRETS_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$SECRETS_FILE"; set +a
    echo "loaded credentials from $SECRETS_FILE"
else
    # No secrets file present — require env vars to be set explicitly.
    : "${EVOGUARD_QIANFAN_APPID:?set EVOGUARD_QIANFAN_APPID or create $SECRETS_FILE with export EVOGUARD_QIANFAN_APPID=app-<your-appid>}"
    : "${EVOGUARD_QIANFAN_TOKEN:?set EVOGUARD_QIANFAN_TOKEN or create $SECRETS_FILE with export EVOGUARD_QIANFAN_TOKEN=bce-v3/<your-bearer-token>}"
    export EVOGUARD_QIANFAN_APPID EVOGUARD_QIANFAN_TOKEN
fi

# Resolve python interpreter inside evoguard conda env.
# The conda env lives on /ssd1 not /root/miniforge3 -- mamba list shows its real home.
PYTHON_BIN="/ssd1/conda_envs/evoguard/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(/root/miniforge3/bin/mamba run -n evoguard which python 2>/dev/null || true)"
fi
if [[ -z "$PYTHON_BIN" ]] || [[ ! -x "$PYTHON_BIN" ]]; then
    echo "error: cannot locate 'evoguard' env's python interpreter" >&2
    exit 3
fi

# Derive a per-run log filename from config name to keep parallel launches distinguishable.
EXP_NAME=$(basename "$CONFIG")
EXP_NAME="${EXP_NAME%.yaml}"
EXP_NAME="${EXP_NAME%.yml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="rounds/${EXP_NAME}/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "config        : $CONFIG"
echo "python interp : $PYTHON_BIN"
echo "exp name      : $EXP_NAME"
echo "log file      : $RUN_LOG"
echo ""

export PYTHONUNBUFFERED=1
# Forward CUDA visibility just in case training flips dry_run=False later.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

nohup "$PYTHON_BIN" -m evoguard.run --config "$CONFIG" >"$RUN_LOG" 2>&1 &
PID=$!
disown "$PID" || true
echo "[launched] pid=$PID ; tail with:"
echo "   tail -F \"$RUN_LOG\""
