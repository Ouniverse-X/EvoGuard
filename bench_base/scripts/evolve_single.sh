#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYBIN="${PYTHON_BIN:-/ssd1/conda_envs/evoguard/bin/python}"

LOG_DIR="${REPO_ROOT}/rounds/_mcts_evolve/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/run_${TS}.log"

cd "${REPO_ROOT}"
export MCTS_MAX_TASKS="${MCTS_MAX_TASKS:-8}"
export MCTS_GENERATIONS="${MCTS_GENERATIONS:-6}"
export MCTS_RNG_SEED="${MCTS_RNG_SEED:-42}"

echo "[launch] mcts-evolve starting at ${TS}" >&2
echo "[launch] config: tasks=${MCTS_MAX_TASKS} gens=${MCTS_GENERATIONS} seed=${MCTS_RNG_SEED}" >&2
echo "[launch] log -> ${LOG}" >&2

nohup "${PYBIN}" -u "${SCRIPT_DIR}/mcts_evolve.py" >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" > "${REPO_ROOT}/rounds/_mcts_evolve/mcts_evolve.pid"
echo "[launch] PID ${PID}"
echo "[launch] tail: tail -f ${LOG}"
