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

echo "Registering LoRA '$LORA_NAME' -> $LORA_PATH at $URL"
HTTP_CODE=$(
    curl -sS -o /tmp/vllm_register_resp.$$ -w '%{http_code}' \
         -X POST "$URL" \
         -H 'Content-Type: application/json' \
         --data "$PAYLOAD"
)
BODY="$(cat /tmp/vllm_register_resp.$$ 2>/dev/null || true)"
rm -f /tmp/vllm_register_resp.$$

case "$HTTP_CODE" in
    200|201)
        echo "[OK] registered ($HTTP_CODE)"
        [[ -n "$BODY" ]] && echo "     response: $BODY"
        exit 0
        ;;
    *)
        echo "[FAIL] HTTP $HTTP_CODE from $URL" >&2
        [[ -n "$BODY" ]] && echo "       body: $BODY" >&2
        # Common cause: server started without --enable-lora.
        if grep -qi 'enable.lora\|not.*found\|unsupported' <<<"$BODY"; then
            echo ""                                              >&2
            echo "Hint: ensure scripts/start_vllm.sh set EVOGUARD_VLLM_ENABLE_LORA=1." >&2
            echo "      The current default already enables it; check rounds/vllm.log for startup errors." >&2
        fi
        exit 1
        ;;
esac
