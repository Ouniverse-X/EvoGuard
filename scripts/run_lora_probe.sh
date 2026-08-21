#!/usr/bin/env bash
# Thin wrapper that launches evoguard.training.probes.run_lora_layer_probe.
#
# Defaults to CUDA_VISIBLE_DEVICES=3 (per plan.md §"五、脚本运行约定") so the probe
# runs on a dedicated single GPU while the main inference vLLM server stays put on
# its own card. Override via --override-gpu <id> or by exporting the env var first.
#
# Usage:
#   bash scripts/run_lora_probe.sh configs/agentdojo_full_local.yaml [--dry-run]
#       [--override-gpu 3] [--max-pairs 80] [--stratified-per-domain 20]
#
# All CLI flags after the config path are forwarded to the python entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [extra flags forwarded to python entry point]" >&2
    exit 1
fi

CONFIG_PATH="$1"; shift

PY_BIN="${EVOGUARD_PY_BIN:-/ssd1/conda_envs/evoguard/bin/python}"
if [[ ! -x "$PY_BIN" ]]; then
    echo "[run_lora_probe] python bin not found at $PY_BIN; set EVOGUARD_PY_BIN env var." >&2
    exit 2
fi

export EVOGUARD_VLLM_GPU="${EVOGUARD_VLLM_GPU:-6}"   # main inference service stays on card #6.
mkdir -p rounds

echo "Launching LoRA layer probe:"
echo "   config      : $CONFIG_PATH"
echo "   py_bin      : $PY_BIN"
echo "   extra_flags : $*"

exec "$PY_BIN" -m evoguard.training.probes "$CONFIG_PATH" "$@"
