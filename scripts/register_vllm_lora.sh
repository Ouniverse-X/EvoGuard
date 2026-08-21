#!/usr/bin/env bash
# Register a freshly-trained LoRA adapter against a running vLLM server that
# was launched with ``--enable-lora`` (see scripts/start_vllm.sh).
#
# Usage:
#   scripts/register_vllm_lora.sh <lora_name> <lora_path> [port]
#
# After this returns successfully, downstream code can request the new adapter
# by passing it as the OpenAI client's ``model`` parameter — vLLM will route
# the call through the registered LoRA weights overlaid on top of the base
# served model.
#
# Idempotent: re-registering an existing name updates its path.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <lora_name> <lora_path> [port]" >&2
    exit 64
fi

LORA_NAME="$1"
LORA_PATH="$2"
PORT="${3:-${EVOGUARD_VLLM_PORT:-8000}}"

URL="http://127.0.0.1:${PORT}/v1/load_lora_adapter"

# Quick liveness check so we fail fast with a clear message instead of curl's
# cryptic connection-refused stderr.
if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[FATAL] no vLLM server reachable at port ${PORT}."  >&2
    echo "        start one first: bash scripts/start_vllm.sh" >&2
    exit 2
fi

PAYLOAD=$(printf '{"lora_name": "%s", "lora_path": "%s"}' "$LORA_NAME" "$LORA_PATH")

# Retry loop: vLLM sometimes returns HTTP 400/500 on first attempt immediately
# after SFT.save_pretrained() completes due to (a) filesystem flush race where
# adapter_model.safetensors isn't fully visible to the server process yet, or
# (b) transient GPU memory pressure while previous LoRA slot is being freed.
# Three attempts with exponential backoff covers both cases empirically.
MAX_ATTEMPTS=3
ATTEMPT=0
HTTP_CODE="000"
BODY=""
RESP_FILE="/tmp/vllm_register_resp.$$"
trap 'rm -f "$RESP_FILE" 2>/dev/null || true' EXIT

echo "Registering LoRA '$LORA_NAME' -> $LORA_PATH at $URL"

while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
    ATTEMPT=$((ATTEMPT+1))
    sleep 0.5   # small pre-call settle delay; harmless on retries as backstop.
    echo "  [attempt ${ATTEMPT}/${MAX_ATTEMPTS}] POST $URL ..."
    HTTP_CODE=$(
        curl -sS -o "$RESP_FILE" -w '%{http_code}' \
             -X POST "$URL" \
             -H 'Content-Type: application/json' \
             --data "$PAYLOAD" \
             --max-time 30 \
             2>&1 || true
    )
    BODY="$(cat "$RESP_FILE" 2>/dev/null || true)"

    case "$HTTP_CODE" in
        200|201)
            echo "[OK] registered ($HTTP_CODE) after ${ATTEMPT} attempt(s)"
            [[ -n "$BODY" ]] && echo "     response: $BODY"
            exit 0
            ;;
        *)
            echo "  [attempt ${ATTEMPT}] got HTTP=${HTTP_CODE} body=${BODY}" >&2
            # Backoff before next retry (skip backoff after final attempt).
            if [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; then
                BACKOFF=$((2 ** (ATTEMPT-1)))    # 1s, 2s, ...
                echo "  backing off ${BACKOFF}s before next attempt..." >&2
                sleep "$BACKOFF"
            fi
            ;;
    esac
done

# All attempts exhausted -- emit structured failure diagnostics for log mining.
echo "[FAIL] HTTP $HTTP_CODE from $URL after $MAX_ATTEMPTS attempts" >&2
[[ -n "$BODY" ]] && echo "       last_body: $BODY" >&2
echo "" >&2
echo "Diagnostic hints:" >&2
echo "  * Check rounds/vllm.log for server-side errors at $(date '+%H:%M:%S')" >&2
echo "  * Verify adapter dir contents are non-empty & readable by vLLM process user:" >&2
ls -la "$LORA_PATH"/adapter_config.json "$LORA_PATH"/adapter_model.safetensors >&2 2>&1 | sed 's/^/      /'
if grep -qi 'enable.lora\|not.*found\|unsupported\|already.*exist\|duplicate' <<<"$BODY"; then
    echo "  * Body suggests config/path/conflict issue. Inspect message above." >&2
fi
exit 1
