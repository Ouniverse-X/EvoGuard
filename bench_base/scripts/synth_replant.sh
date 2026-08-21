#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYBIN="${PYTHON_BIN:-/ssd1/conda_envs/evoguard/bin/python}"

LOG_DIR="${REPO_ROOT}/rounds/_synth_replant/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/run_${TS}.log"

cd "${REPO_ROOT}"
echo "[launch] synth-replant starting at ${TS}, log -> ${LOG}" >&2
nohup "${PYBIN}" -u "${SCRIPT_DIR}/synth_replant.py" >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" > "${REPO_ROOT}/rounds/_synth_replant/synth_replant.pid"
echo "[launch] PID ${PID}"
echo "[launch] tail: tail -f ${LOG}"
