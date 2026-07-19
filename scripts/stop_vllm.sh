#!/usr/bin/env bash
# Stop the vLLM server started by scripts/start_vllm.sh (if still running).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PID_FILE="rounds/vllm.pid"
if [[ ! -f "$PID_FILE" ]]; then
    echo "no PID file at $PID_FILE; nothing to stop"
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "stopping vLLM pid=$PID ..."
    # SIGTERM first so it can flush LoRA caches / close sockets cleanly.
    kill -TERM "$PID"
    for i in $(seq 1 20); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "[OK] stopped after ${i}s"
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
    done
    echo "process didn't die on TERM, escalating to KILL" >&2
    kill -KILL "$PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "done."
