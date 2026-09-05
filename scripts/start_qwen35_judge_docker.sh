#!/usr/bin/env bash
# Serve Qwen3.5-9B as the reward-path judge (:8004) via Docker.
#
# Why Docker and not a conda env on this box: the host is CentOS 7
# (glibc 2.17, kernel 3.10). vllm >= ~0.9 and torch >= 2.7 ship
# manylinux_2_28 wheels only (glibc >= 2.28), so no bare-metal env on this
# machine can load Qwen3_5ForConditionalGeneration -- pip falls back to a
# doomed source build. The vllm/vllm-openai image (v0.24.0, already pulled)
# has both the runtime and the architecture registered.
#
#   EVOGUARD_JUDGE_GPU=5 scripts/start_qwen35_judge_docker.sh
#
# Point the reward-path judges at it with:
#   export EVOGUARD_JUDGE_LLM_BASE_URL=http://127.0.0.1:8004/v1
#   export EVOGUARD_PROGRESS_LLM_BASE_URL=http://127.0.0.1:8004/v1
#   export EVOGUARD_JUDGE_LLM_MODEL=qwen3.5-9b

set -euo pipefail

MODEL_DIR="${EVOGUARD_QWEN35_DIR:-/ssd1/yx/models/Qwen3.5-9B}"
GPU="${EVOGUARD_JUDGE_GPU:-5}"
PORT="${EVOGUARD_JUDGE_PORT:-8004}"
NAME="${EVOGUARD_JUDGE_CONTAINER:-qwen35-judge}"
IMAGE="${EVOGUARD_JUDGE_IMAGE:-docker.m.daocloud.io/vllm/vllm-openai:latest}"

if docker ps -q -f "name=^${NAME}$" | grep -q . \
        && curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[SKIP] healthy ${NAME} already serving on :${PORT}"
    exit 0
fi
docker rm -f "$NAME" 2>/dev/null || true

# --gdn-prefill-backend triton: Qwen3.5's gated-delta-rule prefill; the
# flashinfer sm90a kernel aborts on some GPUs while the server reports
# healthy (see CLAUDE.md). Verified working on this box's A100s.
docker run -d --name "$NAME" --gpus "\"device=${GPU}\"" \
    -v "${MODEL_DIR}:/models/Qwen3.5-9B" \
    -p "127.0.0.1:${PORT}:8000" \
    --restart unless-stopped \
    "$IMAGE" \
    --model /models/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --gdn-prefill-backend triton

echo "launched ${NAME} on GPU ${GPU}; waiting for readiness (model load ~4-5 min)..."
for i in $(seq 1 60); do
    if ! docker ps -q -f "name=^${NAME}$" | grep -q .; then
        echo "[FATAL] container exited. Last log lines:" >&2
        docker logs "$NAME" 2>&1 | tail -30 >&2 || true
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        # Health is not enough: the gdn kernel failure only fires on the first
        # forward pass. Send one real completion before declaring victory.
        OUT=$(curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Reply with exactly one word: ok"}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
        if grep -q '"content"' <<<"$OUT"; then
            echo "[OK] ready and answering on :${PORT}"
            exit 0
        fi
        echo "[FATAL] health OK but first completion failed: $OUT" >&2
        exit 1
    fi
    sleep 10
done
echo "[TIMEOUT] not ready within 600s" >&2
docker logs "$NAME" 2>&1 | tail -20 >&2 || true
exit 2
