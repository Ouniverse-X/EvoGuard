#!/usr/bin/env bash
# Gemma-2-9B-IT: Tier-1 (rollout + probe + analyze) then Tier-2 T-sweep
# Uses GPU 3,5 for vLLM (TP=2) and GPU 6 for probe
set -euo pipefail
cd /ssd1/yx/yangxiao26/EvoGuard

MODEL_PATH="/ssd1/yx/models/gemma-2-9b-it"
VLLM_PORT=8003
VLLM_GPUS="3,4,5,6"
PROBE_GPU="7"
TIER1_DIR="rounds/_preliminary/20260811_tier1_gemma"
TIER2_DIR="rounds/_preliminary/20260811_tier2_tsweep_gemma"
CONFIG="configs/preliminary_tier1_gemma.yaml"
PYTHON="/ssd1/conda_envs/evoguard/bin/python"

export PYTHONPATH=/ssd1/yx/yangxiao26/EvoGuard

echo "[$(date)] Starting Gemma Tier-1 + Tier-2 pipeline"

# ===== Phase 1: Start vLLM for Gemma (TP=2) =====
echo "[$(date)] Starting vLLM: ${MODEL_PATH} on GPUs ${VLLM_GPUS}, port ${VLLM_PORT}"
CUDA_VISIBLE_DEVICES=${VLLM_GPUS} $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name gemma-2-9b-it \
    --port ${VLLM_PORT} \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager \
    --chat-template configs/gemma2_chat_template.jinja \
    &> "${TIER1_DIR}/vllm.log" &
VLLM_PID=$!
echo "[$(date)] vLLM PID=${VLLM_PID}"

# Wait for vLLM to become healthy
mkdir -p "${TIER1_DIR}"
for i in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "[$(date)] vLLM healthy after ${i}s"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM died during startup. Check ${TIER1_DIR}/vllm.log"
        exit 1
    fi
    sleep 1
done
if ! curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[$(date)] ERROR: vLLM did not become healthy in 240s"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# ===== Phase 2: Tier-1 Rollout Collection =====
echo "[$(date)] === TIER-1: Rollout collection ==="
$PYTHON preliminary/collect.py \
    --config "${CONFIG}" \
    --output-dir "${TIER1_DIR}" \
    --model-path "${MODEL_PATH}" \
    --cuda-visible-devices "${PROBE_GPU}" \
    --suites banking slack travel workspace \
    --n-samples 15 \
    --sampling-temperature 0.7 \
    --target-scenarios 120 \
    --min-per-bucket 3 \
    --skip-measure \
    2>&1 || echo "[$(date)] collect.py exited (expected metrics.csv error for --skip-measure)"

# Verify scenarios file exists
if [ ! -f "${TIER1_DIR}/all_scenarios.json" ]; then
    echo "[$(date)] ERROR: all_scenarios.json not found. Rollout failed."
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi
N_SCENARIOS=$(python3 -c "import json; print(len(json.load(open('${TIER1_DIR}/all_scenarios.json'))))")
echo "[$(date)] Tier-1 collected ${N_SCENARIOS} scenarios"

# ===== Phase 3: Tier-1 Probe =====
echo "[$(date)] === TIER-1: Probe ==="
# Kill vLLM to free GPU memory for probe
kill ${VLLM_PID} 2>/dev/null || true
sleep 5

$PYTHON preliminary/probe.py \
    --scenarios-json "${TIER1_DIR}/all_scenarios.json" \
    --output-dir "${TIER1_DIR}/seq_probe" \
    --model-path "${MODEL_PATH}" \
    --cuda-visible-devices "${PROBE_GPU}" \
    --max-thought-tokens 128

# ===== Phase 4: Tier-1 Analyze =====
echo "[$(date)] === TIER-1: Analyze ==="
$PYTHON preliminary/analyze.py --probe-dir "${TIER1_DIR}/seq_probe"

# Plot
$PYTHON scripts/plot_tier1_plan_a_bars.py \
    --results "${TIER1_DIR}/seq_probe/as_vs_af_sequence_results.json" \
    --output-dir "${TIER1_DIR}"

echo "[$(date)] === TIER-1 COMPLETE ==="

# ===== Phase 5: Tier-2 T-Sweep Rollout =====
echo "[$(date)] === TIER-2: Starting vLLM again for T-sweep rollout ==="
mkdir -p "${TIER2_DIR}"
CUDA_VISIBLE_DEVICES=${VLLM_GPUS} $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name gemma-2-9b-it \
    --port ${VLLM_PORT} \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager \
    --chat-template configs/gemma2_chat_template.jinja \
    &> "${TIER2_DIR}/vllm.log" &
VLLM_PID=$!
echo "[$(date)] vLLM PID=${VLLM_PID}"

for i in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "[$(date)] vLLM healthy after ${i}s"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM died during startup. Check ${TIER2_DIR}/vllm.log"
        exit 1
    fi
    sleep 1
done
if ! curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[$(date)] ERROR: vLLM did not become healthy in 240s"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

echo "[$(date)] === TIER-2: T-sweep rollout (T=0.3,0.5,0.7,0.9) ==="
$PYTHON preliminary/collect.py \
    --config "${CONFIG}" \
    --output-dir "${TIER2_DIR}" \
    --model-path "${MODEL_PATH}" \
    --cuda-visible-devices "${PROBE_GPU}" \
    --suites banking slack travel workspace \
    --n-samples 24 \
    --sampling-temperature "0.3,0.5,0.7,0.9" \
    --target-scenarios 120 \
    --min-per-bucket 3 \
    --skip-measure \
    2>&1 || echo "[$(date)] collect.py exited (expected metrics.csv error for --skip-measure)"

if [ ! -f "${TIER2_DIR}/all_scenarios.json" ]; then
    echo "[$(date)] ERROR: Tier-2 all_scenarios.json not found."
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi
N2=$(python3 -c "import json; print(len(json.load(open('${TIER2_DIR}/all_scenarios.json'))))")
echo "[$(date)] Tier-2 collected ${N2} scenarios"

# ===== Phase 6: Tier-2 Probe =====
echo "[$(date)] === TIER-2: Probe ==="
kill ${VLLM_PID} 2>/dev/null || true
sleep 5

$PYTHON preliminary/probe.py \
    --scenarios-json "${TIER2_DIR}/all_scenarios.json" \
    --output-dir "${TIER2_DIR}/seq_probe" \
    --model-path "${MODEL_PATH}" \
    --cuda-visible-devices "${PROBE_GPU}" \
    --max-thought-tokens 128

# ===== Phase 7: Tier-2 Analyze =====
echo "[$(date)] === TIER-2: Analyze ==="
$PYTHON preliminary/analyze.py --probe-dir "${TIER2_DIR}/seq_probe"

# Plot
$PYTHON scripts/plot_tier1_plan_a_bars.py \
    --results "${TIER2_DIR}/seq_probe/as_vs_af_sequence_results.json" \
    --output-dir "${TIER2_DIR}"

echo "[$(date)] === ALL DONE: Gemma Tier-1 + Tier-2 complete ==="
echo "  Tier-1: ${TIER1_DIR}"
echo "  Tier-2: ${TIER2_DIR}"
