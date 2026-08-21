#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYBIN="${PYTHON_BIN:-/ssd1/conda_envs/evoguard/bin/python}"

LOG_DIR="${REPO_ROOT}/rounds/_synth_immediate/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/run_${TS}.log"

cd "${REPO_ROOT}"
export SYNTH_TARGETS_PER_SOURCE="${SYNTH_TARGETS_PER_SOURCE:-6}"
export SYNTH_MUTATIONS_PER_PAIR="${SYNTH_MUTATIONS_PER_PAIR:-2}"
export SYNTH_POSITIONS_PER_MUTATION="${SYNTH_POSITIONS_PER_MUTATION:-2}"
export SYNTH_MIN_DELTA_ACCEPT="${SYNTH_MIN_DELTA_ACCEPT:-0}"
export SYNTH_MAX_DELTA_ACCEPT="${SYNTH_MAX_DELTA_ACCEPT:-1}"
export SYNTH_PER_BUCKET_QUOTA="${SYNTH_PER_BUCKET_QUOTA:-40}"

echo "[launch] synth-immediate starting at ${TS}" >&2
echo "[launch] config: targets/source=${SYNTH_TARGETS_PER_SOURCE} mutations/pair=${SYNTH_MUTATIONS_PER_PAIR} positions/mut=${SYNTH_POSITIONS_PER_MUTATION} delta_window=[${SYNTH_MIN_DELTA_ACCEPT},${SYNTH_MAX_DELTA_ACCEPT}] quota=${SYNTH_PER_BUCKET_QUOTA}" >&2
echo "[launch] log -> ${LOG}" >&2

nohup "${PYBIN}" -u "${SCRIPT_DIR}/synth_immediate.py" >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" > "${REPO_ROOT}/rounds/_synth_immediate/synth_immediate.pid"
echo "[launch] PID ${PID}"
echo "[launch] tail: tail -f ${LOG}"
