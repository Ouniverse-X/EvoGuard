#!/usr/bin/env bash
# Tier-1 multi-domain rollout collection: serially loops over banking→slack→travel,
# invoking patched collect for each suite with --skip-measure so the
# collection phase only hits primary vLLM :8000 (no local model load needed).
#
# Output dirs land under rounds/_preliminary/20260805_tier1_{banking,slack,travel}/.
# Each suite call: target_scenarios=50, n_samples=12, sampling_temperature=0.7.
set -u

PYTHON=/ssd1/conda_envs/evoguard/bin/python
CONFIG=configs/agentdojo_full_local.yaml
BASE=rounds/_preliminary

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE}/20260805_tier1_multidomain_collect/logs"
mkdir -p "$LOG_DIR"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator_${TS}.log"

echo "[tier1-orchestrator] start at $(date)" | tee -a "$ORCHESTRATOR_LOG"

for SUITE in banking slack travel; do
    OUTDIR="${BASE}/20260805_tier1_${SUITE}"
    mkdir -p "$OUTDIR"
    echo "" | tee -a "$ORCHESTRATOR_LOG"
    echo "============================================================" | tee -a "$ORCHESTRATOR_LOG"
    echo "[tier1] collecting suite=${SUITE} → ${OUTDIR}" | tee -a "$ORCHESTRATOR_LOG"
    echo "  cmd: collect --suites ${SUITE} \\"
    echo "        --output-dir ${OUTDIR} --target-scenarios 50 \\"
    echo "        --n-samples 12 --sampling-temperature 0.7 --skip-measure" | tee -a "$ORCHESTRATOR_LOG"
    echo "============================================================" | tee -a "$ORCHESTRATOR_LOG"

    CUDA_VISIBLE_DEVICES="" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PYTHON -m preliminary.collect \
        --config "${CONFIG}" \
        --output-dir "${OUTDIR}" \
        --model-path /ssd1/models/qwen2.5-7b-it \
        --cuda-visible-devices "" \
        --suites ${SUITE} \
        --target-scenarios 50 \
        --n-samples 12 \
        --sampling-temperature 0.7 \
        --skip-measure >>"$ORCHESTRATOR_LOG" 2>&1
    rc=$?
    if [ -f "${OUTDIR}/all_scenarios.json" ]; then
        n_valid=$($PYTHON -c "import json; print(len(json.load(open('${OUTDIR}/all_scenarios.json'))))")
    else
        n_valid="MISSING"
    fi
    echo "[tier1] suite=${SUITE} done rc=${rc} n_scenarios=${n_valid}" | tee -a "$ORCHESTRATOR_LOG"
done

# Emit completion marker for downstream gate.
touch "${LOG_DIR}/TIER1_COLLECT_DONE"
echo ""
echo "[tier1-orchestrator] ALL DONE at $(date); marker file written to:" | tee -a "$ORCHESTRATOR_LOG"
echo "  ${LOG_DIR}/TIER1_COLLECT_DONE" | tee -a "$ORCHESTRATOR_LOG"
