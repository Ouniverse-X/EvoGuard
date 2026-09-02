#!/usr/bin/env bash
# Stop the vLLM server started by scripts/start_vllm.sh (if still running).
#
# Usage: stop_vllm.sh [helper_port ...]
#   With no arguments, only the PRIMARY server (rounds/vllm.pid) is stopped --
#   the historical behaviour every caller relies on. Each extra argument names a
#   helper started by scripts/start_vllm_secondary.sh, whose pid file is derived
#   from its port (rounds/vllm_helper_<port>.pid). Needed when a helper occupies a
#   card that is about to be handed back to gpu_keeper: the keeper cannot take a
#   GPU that still has an 0.85-utilisation engine sitting on it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

stop_pidfile() {
    local pid_file="$1" label="$2"
    if [[ ! -f "$pid_file" ]]; then
        echo "no PID file at $pid_file; nothing to stop for $label"
        return 0
    fi
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        echo "stopping $label pid=$pid ..."
        # SIGTERM first so it can flush LoRA caches / close sockets cleanly.
        kill -TERM "$pid"
        for i in $(seq 1 20); do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[OK] $label stopped after ${i}s"
                rm -f "$pid_file"
                return 0
            fi
            sleep 1
        done
        echo "$label didn't die on TERM, escalating to KILL" >&2
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
    echo "done ($label)."
}

stop_pidfile "rounds/vllm.pid" "primary vLLM"
for port in "$@"; do
    stop_pidfile "rounds/vllm_helper_${port}.pid" "helper vLLM :${port}"
done
