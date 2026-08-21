#!/usr/bin/env bash
# Watcher script: waits for the LoRA layer-selection probe to finish, validates
# the refreshed artifact JSON, then launches the main v7_integrated experiment
# under nohup so it survives shell exit. Emits status lines into a sidecar log.

set -uo pipefail   # NOTE: NOT -e -- we want graceful handling of validation failures rather than hard aborts.

REPO_ROOT=/ssd1/yx/yangxiao26/EvoGuard
cd "$REPO_ROOT"

PID_FILE=/tmp/ducc_phaseA_prober.pid
PHASE_A_PID="$(cat "$PID_FILE" 2>/dev/null || echo '')"
ARTIFACT_PATH="/ssd1/yx/yangxiao26/EvoGuard/rounds/evoguard_agentdojo_full_local/probe_results/targets.json"
CONFIG_PATH="configs/agentdojo_full_v7_integrated.yaml"
EXP_NAME="evoguard_agentdojo_full_v7_integrated"
LOG_DIR="$REPO_ROOT/rounds/$EXP_NAME/logs"
mkdir -p "$LOG_DIR"
WATCHER_LOG="$LOG_DIR/watcher_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$WATCHER_LOG" >&2
}

log "[watcher] started."
log "[watcher] monitoring phase-A probe pid=$PHASE_A_PID until exit..."
log "[watcher] expecting fresh artifact at $ARTIFACT_PATH once done."

if [[ -z "${PHASE_A_PID:-}" ]]; then
    log "[watcher] ERROR: missing pid file at $PID_FILE; cannot proceed."
    exit 2
fi

WAIT_MINUTES=0
while kill -0 "$PHASE_A_PID" 2>/dev/null; do
    sleep 60
    WAIT_MINUTES=$((WAIT_MINUTES + 1))
    if (( WAIT_MINUTES % 5 == 0 )); then
        log "[watcher] still waiting for probe to finish (${WAIT_MINUTES}m elapsed)."
    fi
    # Safety valve: if probe runs >90 minutes something is wrong.
    if (( WAIT_MINUTES >= 120 )); then
        log "[watcher] ABORTING after ${WAIT_MINUTES}m of waiting (>2h threshold). Investigate manually."
        exit 3
    fi
done
log "[watcher] phase-A probe process exited after ${WAIT_MINUTES}m."

sleep 3   # give filesystem time to flush final writes from probe runner before we validate.

log "[watcher] validating refreshed artifact JSON ..."
VALIDATION_OUT=$(/ssd1/conda_envs/evoguard/bin/python -c "
import json, sys
try:
    d = json.load(open('$ARTIFACT_PATH'))
except Exception as e:
    print('FAIL_JSON_LOAD', e); sys.exit(1)
sv = str(d.get('schema_version',''))
mods = d.get('recommended_target_modules') or []
sel = d.get('selected_blocks_sorted_desc') or []
npairs = int(d.get('n_pairs_used') or -1)
print(f'schema_version={sv} n_pairs={npairs} n_blocks_selected={len(sel)} n_modules={len(mods)}')
print('selected_blocks_sorted_desc[:10]:', sel[:10])
if sv != '1' or not mods or npairs <= 0:
    print('INVALID'); sys.exit(2)
else:
    print('OK_VALID_ARTIFACT')
")
RV=$?
log "[watcher] validation output:"
echo "$VALIDATION_OUT" | sed 's/^/[artifact-validate] /' >> "$WATCHER_LOG"
if [[ $RV -ne 0 ]]; then
    log "[watcher] FATAL: artifact invalid or unreadable; will NOT auto-launch main experiment to avoid wasting GPU cycles on stale/default target modules."
    log "[watcher] inspect $ARTIFACT_PATH manually then rerun main launcher yourself if appropriate."
    exit 4
fi

TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_${TS}.log"

log "[watcher] launching MAIN EXPERIMENT under nohup ..."
log "[watcher]   config : $CONFIG_PATH"
log "[watcher]   exp_dir: $REPO_ROOT/rounds/$EXP_NAME/"
log "[watcher]   log    : $MAIN_LOG"

SECRETS_FILE="${EVOGUARD_SECRETS_FILE:-$HOME/.evoguard_qianfan.env}"
if [[ -f "$SECRETS_FILE" ]]; then
    # shellcheck disable=SC1090
    set +u; source "$SECRETS_FILE"; set -u
fi
: "${EVOGUARD_QIANFAN_APPID:?set EVOGUARD_QIANFAN_APPID or create $SECRETS_FILE}"
: "${EVOGUARD_QIANFAN_TOKEN:?set EVOGUARD_QIANFAN_TOKEN or create $SECRETS_FILE}"
export EVOGUARD_QIANFAN_APPID EVOGUARD_QIANFAN_TOKEN
export EVOGUARD_VLLM_GPU="${EVOGUARD_VLLM_GPU:-6}"

setsid /ssd1/conda_envs/evoguard/bin/python -m evoguard.run \
        --config "$CONFIG_PATH" \
        >"$MAIN_LOG" 2>&1 &
MAIN_PID=$!
disown || true

echo "$MAIN_PID" > /tmp/ducc_phaseB_main.pid
log "[watcher] MAIN EXPERIMENT launched successfully under setsid+nohup semantics."
log "[watcher]   main_pid=$MAIN_PID"
log "[watcher] tail command for live progress:"
log "[watcher]     tail -f $MAIN_LOG"
exit 0
