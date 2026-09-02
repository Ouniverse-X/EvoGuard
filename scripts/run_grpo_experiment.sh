#!/usr/bin/env bash
# Launch EvoGuard's full AgentDojo co-evolutionary experiment WITH native-GRPO
# defender reinforcement learning (spec §4.2):
#
#   r0              -> cold-start LoRA SFT (same as all prior baselines)
#   >=r1            -> incremental on-policy GRPO warm-started from previous adapter
#                      writing new weights back to <exp>/latest_adapter_dir.txt marker file
#                      so subsequent rounds auto-pick them up AND scripts/register_vllm_lora.sh
#                      hot-registers them onto the running :8000 vLLM instance mid-experiment.
#
# Usage:
#   bash scripts/run_grpo_experiment.sh [config_path]
#
# Default config path: configs/agentdojo_full_grpo.yaml (= v3_glmatk baseline + GRPO toggle).
# Pass an alternative YAML as $1 to override.
#
# Pre-flight expectations enforced below before nohup fires:
#   * primary vLLM serving qwen2.5-7b-it must already listen on http://127.0.0.1:8000/v1
#     with VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 set at launch time (--enable-lora alone insufficient!)
#     See scripts/start_vllm.sh for canonical launcher satisfying this constraint.
#   * secondary judge/tool-executor vLLM listening on :8002 (any served-model-name works;
#     current default is llama3-8b-judge until upgraded env supports true qwen3.6 loading).
#   * Trainer picks its own CUDA_VISIBLE_DEVICES="1" by default (override via env var
#     TRAINER_CUDA_VISIBLE_DEVICES=<gpu_id> if you need different placement).
#   * QianFan creds for attacker LLM fall back to inline literals matching sh/qianfan_run.sh
#     unless ~/.evoguard_qianfan.env exists exporting EVOGUARD_QIANFAN_APPID/TOKEN explicitly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CONFIG="${1:-configs/agentdojo_full_grpo.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG" >&2
    exit 2
fi
echo "[preflight] using config: $CONFIG"

# ---- Resolve python interpreter ---- #
PYTHON_BIN="${EVOGUARD_PY_BIN:-/root/yangxiao/envs/train/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[error] python interpreter not executable: $PYTHON_BIN" >&2
    exit 3
fi
echo "[preflight] python interp: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"

# ---- Probe required vLLM endpoints are up BEFORE firing long-running job ---- #
probe_endpoint() {
    local url="$1"
    local label="$2"
    local resp_code
    resp_code=$(curl -fsS -o /dev/null -w '%{http_code}' \
                   --max-time 5 "$url/models" 2>/dev/null || echo "000")
    if [[ "$resp_code" != "200" ]]; then
        echo "[error] ${label} not reachable at ${url} (HTTP=${resp_code}). "
        echo "        Start it first:"
        echo "          bash scripts/start_vllm.sh               # defender @ :8000 w/--enable-lora + VLLM_ALLOW_RUNTIME_LORA_UPDATING=1"
        echo "          EVOGUARD_VLLM_PORT=8002 EVOGUARD_VLLM_GPU=2,3 bash scripts/start_vllm_secondary.sh   # tool_executor"
        echo "          EVOGUARD_VLLM_PORT=8003 EVOGUARD_VLLM_GPU=5,6 bash scripts/start_vllm_secondary.sh   # judge + utility_judge"
        exit 4
    fi
    echo "[preflight] OK  ${label} reachable (${url}/models HTTP 200)"
}

# Helper endpoints are probed only if the config actually references them, so
# single-endpoint configs keep working unchanged. The port list is DERIVED from the
# config rather than hard-coded: it used to be a literal `8002 8003`, which meant a
# config pointing its judges anywhere else (v10 puts them on :8004) had its judge
# endpoint silently unprobed, so a dead judge only surfaced mid-round.
PRIMARY_VLLM_URL="${EVOGUARD_PRIMARY_VLLM:-http://127.0.0.1:8000/v1}"
probe_endpoint "$PRIMARY_VLLM_URL"   "primary vLLM (defense)"
PRIMARY_PORT="$(printf '%s' "$PRIMARY_VLLM_URL" | sed -n 's#.*:\([0-9]\+\)/v1.*#\1#p')"
HELPER_PORTS="$(sed -n 's#.*127\.0\.0\.1:\([0-9]\+\)/v1.*#\1#p' "$CONFIG" | sort -u)"
for helper_port in $HELPER_PORTS; do
    [[ "$helper_port" == "$PRIMARY_PORT" ]] && continue
    probe_endpoint "http://127.0.0.1:${helper_port}/v1" "helper vLLM :${helper_port}"
done

# Verify VLLM_ALLOW_RUNTIME_LORA_UPDATING was set when primary launched --
# otherwise POST /v1/load_lora_adapter returns *unstructured* 404 ("{\"detail\":\"Not Found\"}")
# every round breaking hot-load chain post-r0.
#
# Distinguish two flavours of HTTP-404 reply here:
#   - Route genuinely absent: short {"detail":"Not Found"} JSON envelope.
#   - Route present + valid request: application-level NotFoundError mentioning
#     'add_lora' / 'No adapter found' because we deliberately pointed lora_path=/nonexistent.
# Only the FIRST case is fatal; second confirms hot-load plumbing is wired correctly.
LORA_PROBE_BODY=$(curl -sS --max-time 5 "${PRIMARY_VLLM_URL}/load_lora_adapter" \
    -X POST -H 'Content-Type: application/json' \
    -d '{"lora_name":"__preflight_probe","lora_path":"/nonexistent"}' \
    2>/dev/null || true)
if printf '%s' "$LORA_PROBE_BODY" | grep -q '"detail":"Not Found"'; then
    echo "[error] primary vLLM has NOT registered /load_lora_adapter route."
    echo "        Cause: missing VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 at launch time."
    echo "        Fix: stop & restart via scripts/start_vllm.sh which exports this var automatically."
    exit 5
elif printf '%s' "$LORA_PROBE_BODY" | grep -qE '(add_lora|No adapter found|NotFoundError)'; then
    echo "[preflight] OK  /load_lora_adapter route registered (application-level NotFound expected on bogus probe)"
else
    echo "[warn] unexpected probe response -- continuing anyway:"
    printf '%s\n' "$LORA_PROBE_BODY" | head -c 300
fi

# ---- Inject QianFan creds for attacker backend ---- #
SECRETS_FILE="${EVOGUARD_SECRETS_FILE:-$HOME/.evoguard_qianfan.env}"
if [[ -f "$SECRETS_FILE" ]]; then
    # shellcheck disable=SC1090
    set +u; source "$SECRETS_FILE"; set -u
    export EVOGUARD_QIANFAN_APPID EVOGUARD_QIANFAN_TOKEN
    echo "[preflight] loaded qianfan creds from $SECRETS_FILE"
else
    # No inline literals here on purpose: this file is committed, and a bearer
    # token pushed to a git remote is a published token. Same contract as
    # scripts/run_real.sh -- supply the two vars, or drop them into a secrets file.
    : "${EVOGUARD_QIANFAN_APPID:?set EVOGUARD_QIANFAN_APPID or create $SECRETS_FILE exporting EVOGUARD_QIANFAN_APPID=app-<id>}"
    : "${EVOGUARD_QIANFAN_TOKEN:?set EVOGUARD_QIANFAN_TOKEN or create $SECRETS_FILE exporting EVOGUARD_QIANFAN_TOKEN=bce-v3/<your-bearer-token>}"
    export EVOGUARD_QIANFAN_APPID EVOGUARD_QIANFAN_TOKEN
    echo "[preflight] no secrets file at $SECRETS_FILE; using ambient env appid=$EVOGUARD_QIANFAN_APPID"
fi

# ---- Derive log filename + ensure exp dir structure ready ---- #
EXP_NAME=$(basename "$CONFIG")
EXP_NAME="${EXP_NAME%.yaml}"; EXP_NAME="${EXP_NAME%.yml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="rounds/${EXP_NAME}/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"

# Ambient CUDA visibility for any non-trainer subprocess path. Point it at the
# spare GPU so nothing can accidentally allocate on a serving GPU; the trainer
# overrides this from TrainingConfig.cuda_visible_devices inside Python.
export CUDA_VISIBLE_DEVICES="${TRAINER_CUDA_VISIBLE_DEVICES:-7}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# The GRPO trainer holds several near-identical large logits tensors live at once
# (policy / reference / old logprobs). With the default caching allocator that
# churn leaves multi-GiB reserved-but-unallocated holes -- the r3 OOM of run
# 20260816_233727 died with 3.87 GiB stranded that way. expandable_segments lets
# the allocator grow existing segments instead of fragmenting new ones.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Attacker-gateway pacing. The QianFan appid's quota measured 60 requests/min on
# 2026-08-17 (45 and 60 rpm sustained 75s -> zero 429s; 90 rpm -> exactly 60
# through, 37 rejected). QianFanClient paces sends process-wide below this value;
# without it the rollout fan-out offers hundreds of calls/min, retries exhaust,
# and rollout.py drops whole tasks from the round. 55 leaves a ~10% margin.
export EVOGUARD_QIANFAN_MAX_RPM="${EVOGUARD_QIANFAN_MAX_RPM:-55}"

cat <<EOF
[preflight]
  config        : $CONFIG
  python interp : $PYTHON_BIN
  exp name      : $EXP_NAME
  log file      : $RUN_LOG
  CUDA_VISIBLE  : $CUDA_VISIBLE_DEVICES  (for non-trainer subprocess only; trainer pins own device via yaml)
  qianfan_appid : $EVOGUARD_QIANFAN_APPID
  qianfan_rpm   : $EVOGUARD_QIANFAN_MAX_RPM  (process-wide send pacing; measured quota is 60/min)

[launching]
nohup "$PYTHON_BIN" -m evoguard.run --config "$CONFIG" >"$RUN_LOG" 2>&1 &
EOF

nohup "$PYTHON_BIN" -m evoguard.run --config "$CONFIG" >"$RUN_LOG" 2>&1 &
PID=$!
disown "$PID" || true

sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
    echo ""
    echo "[error] process exited immediately! Inspect first lines of log:"
    head -40 "$RUN_LOG" || true
    exit 6
fi

echo "[launched]"
echo "   pid        : $PID"
echo "   started_at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "   log        : $RUN_LOG"
echo ""
echo "Tail logs with:"
echo "   tail -F \"$RUN_LOG\" | tee /dev/stderr 2>/dev/null"
echo ""
echo "Stop run cleanly with:"
echo "   kill $PID && pkill -P $PID 2>/dev/null || true"
