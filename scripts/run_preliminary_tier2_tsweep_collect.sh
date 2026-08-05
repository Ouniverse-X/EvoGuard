#!/usr/bin/env bash
# Tier-2 temperature-sweep deconfounder rollout collection: re-collects workspace-only
# rollouts across four sampling-T strata {0.3,0.5,0.7,0.9} via patched harness.
# Each attacked_rollout gets tagged with its own 'collection_temperature' field
# flowing through probe→analyzer unchanged enabling downstream per-stratum analyses.
#
# Total budget per scenario = 24 sampled rollouts split evenly =6/T-stratum/scenario.
# Targeting up to 120 eligible scenarios (subject to ≥3 success+≥3 fail filter).
set -u

PYTHON=/ssd1/conda_envs/evoguard/bin/python
CONFIG=configs/agentdojo_full_local.yaml
BASE=rounds/_preliminary

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="${BASE}/20260805_tier2_tsweep_workspace"
mkdir -p "${OUTDIR}"
LOG_PATH="${OUTDIR}/logs/run_${TS}.log"
mkdir -p "$(dirname "${LOG_PATH}")"

echo "[tier2-tsweep] launching T-sweep collector into ${OUTDIR}" >&2
echo "[tier2-tsweep] log path: ${LOG_PATH}" >&2

CUDA_VISIBLE_DEVICES="" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PYTHON -m preliminary.collect \
    --config "${CONFIG}" \
    --output-dir "${OUTDIR}" \
    --model-path /ssd1/models/qwen2.5-7b-it \
    --cuda-visible-devices "" \
    --suites workspace \
    --target-scenarios 120 \
    --n-samples 24 \
    --sampling-temperature "0.3,0.5,0.7,0.9" \
    --skip-measure >"${LOG_PATH}" 2>&1
rc=$?

if [ -f "${OUTDIR}/all_scenarios.json" ]; then
    n_valid=$($PYTHON -c "import json; print(len(json.load(open('${OUTDIR}/all_scenarios.json'))))")
else
    n_valid="MISSING"
fi
echo "[tier2-tsweep] DONE rc=${rc} n_scenarios=${n_valid} at $(date)"
if [ ! -f "${OUTDIR}/logs/TIER2_TSWEEP_DONE" ]; then
    touch "${OUTDIR}/logs/TIER2_TSWEEP_DONE"
fi
