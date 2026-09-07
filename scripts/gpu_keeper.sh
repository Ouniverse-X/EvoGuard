#!/usr/bin/env bash
# Occupy otherwise-idle GPUs on this box, and get out of the way on demand.
#
# Why this exists: idle H800s on a shared node get claimed by whoever notices
# them first, and a run that needs 2-3 cards a few hours from now has no way to
# reserve them. The keeper holds the ballast AND keeps utilisation high, so the
# cards read as genuinely in use, then releases within ~2 s of being asked.
#
#   bash scripts/gpu_keeper.sh start            # every idle card (see auto-detect)
#   bash scripts/gpu_keeper.sh start -g 5,6
#   bash scripts/gpu_keeper.sh pause -g 3       # BEFORE starting anything on 3
#   bash scripts/gpu_keeper.sh resume -g 3      # after that thing has exited
#   bash scripts/gpu_keeper.sh status
#   bash scripts/gpu_keeper.sh stop [-g 5,6]
#
# `pause` BLOCKS until the card is actually free (or 60 s), because the two
# things that take cards here -- a vLLM server and the GRPO colocate engine --
# both profile WHOLE-CARD free memory inside their constructor. They fail
# outright rather than wait, so "asked it to release" is not good enough; the
# release has to have landed. Auto-backoff on foreign memory is the safety net
# for a tenant nobody announced, not a substitute for pause.
#
# Knobs (env):
#   EVOGUARD_KEEPER_TARGET_MIB  default 71680 (~70 GiB of an 80 GiB H800)
#   EVOGUARD_KEEPER_UTIL        default 0.90  duty cycle of the fp16 matmul
#   EVOGUARD_KEEPER_EXCLUDE     default 7     never auto-detected; the trainer
#                                             card must stay empty while
#                                             grpo_use_vllm_colocate is on
#   EVOGUARD_KEEPER_PYBIN       default the evoguard conda interpreter
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

FLAG_DIR="$REPO_ROOT/rounds/gpu_keeper"
TARGET_MIB="${EVOGUARD_KEEPER_TARGET_MIB:-71680}"
UTIL="${EVOGUARD_KEEPER_UTIL:-0.90}"
EXCLUDE="${EVOGUARD_KEEPER_EXCLUDE:-7}"
PYBIN="${EVOGUARD_KEEPER_PYBIN:-/root/paddlejob/workspace/yangxiao/miniconda3/envs/evoguard/bin/python}"
IDLE_MIB=2048            # a card using less than this is considered free

mkdir -p "$FLAG_DIR"

CMD="${1:-status}"; shift || true
GPUS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpus) GPUS="${2:-}"; shift 2 ;;
        *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
    esac
done

_csv() { tr ',' ' ' <<<"$1"; }

# Auto-detect = every card whose CURRENT memory.used is below IDLE_MIB, minus
# EVOGUARD_KEEPER_EXCLUDE. Deliberately a snapshot: a card serving vLLM reads
# ~75 GiB and is skipped, and a card the keeper itself already holds also reads
# high -- so `start` without -g never double-launches on a card it owns.
_detect_idle() {
    local excl=" $(_csv "$EXCLUDE") "
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | while IFS=',' read -r idx used; do
            idx="${idx// /}"; used="${used// /}"
            [[ " $excl " == *" $idx "* ]] && continue
            (( used < IDLE_MIB )) && printf '%s ' "$idx"
        done
}

_worker_pid() {
    local f="$FLAG_DIR/worker_$1.pid"
    [[ -f "$f" ]] || return 1
    local p; p="$(cat "$f" 2>/dev/null || true)"
    [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && { echo "$p"; return 0; }
    return 1
}

_held_mib() {
    local f="$FLAG_DIR/state_$1.json"
    [[ -f "$f" ]] || { echo 0; return; }
    sed -n 's/.*"held_mib": *\([0-9]*\).*/\1/p' "$f" | head -1 | grep -E '^[0-9]+$' || echo 0
}

do_start() {
    local list="$1"
    [[ -z "${list// /}" ]] && { echo "[keeper] no idle GPU to take"; return 0; }
    if [[ ! -x "$PYBIN" ]]; then
        echo "[error] interpreter not executable: $PYBIN" >&2; exit 3
    fi
    # The login profile puts /home/opt/nvidia_lib on LD_LIBRARY_PATH, whose raw
    # libcuda breaks every cuDNN handle in this container (see
    # scripts/run_grpo_experiment.sh). The keeper only needs cuBLAS, but it is
    # stripped anyway so a failure here can never be blamed on that path.
    local ld_clean="" p
    IFS=':' read -ra _parts <<<"${LD_LIBRARY_PATH:-}"
    for p in ${_parts[@]+"${_parts[@]}"}; do
        [[ -z "$p" || "${p%/}" == "/home/opt/nvidia_lib" ]] && continue
        ld_clean="${ld_clean:+$ld_clean:}$p"
    done
    for g in $list; do
        if _worker_pid "$g" >/dev/null; then
            echo "[keeper] gpu$g already held by pid $(_worker_pid "$g")"; continue
        fi
        rm -f "$FLAG_DIR/pause_$g.flag"
        CUDA_VISIBLE_DEVICES="$g" LD_LIBRARY_PATH="$ld_clean" \
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        nohup "$PYBIN" "$SCRIPT_DIR/gpu_keeper_worker.py" \
            --gpu "$g" --target-mib "$TARGET_MIB" --util "$UTIL" \
            --flag-dir "$FLAG_DIR" \
            >"$FLAG_DIR/worker_$g.log" 2>&1 &
        local pid=$!
        echo "$pid" >"$FLAG_DIR/worker_$g.pid"
        disown "$pid" 2>/dev/null || true
        echo "[keeper] gpu$g -> pid $pid (target ${TARGET_MIB}MiB, util $UTIL)"
    done
}

do_pause() {
    local list="$1"
    if [[ -z "${list// /}" ]]; then
        touch "$FLAG_DIR/pause_all.flag"
        list="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | tr '\n' ' ')"
        echo "[keeper] paused ALL cards"
    else
        for g in $list; do touch "$FLAG_DIR/pause_$g.flag"; echo "[keeper] paused gpu$g"; done
    fi
    # Block until the hold is really gone -- see the header. 60 s is ~30 poll
    # intervals; a worker that has not released by then is wedged, not slow.
    for _ in $(seq 1 60); do
        local pending=0
        for g in $list; do
            _worker_pid "$g" >/dev/null || continue
            (( $(_held_mib "$g") > 0 )) && pending=1
        done
        (( pending == 0 )) && { echo "[keeper] release confirmed"; return 0; }
        sleep 1
    done
    echo "[keeper][WARN] some workers still hold memory after 60s:" >&2
    do_status >&2
    return 1
}

do_resume() {
    local list="$1"
    if [[ -z "${list// /}" ]]; then
        rm -f "$FLAG_DIR"/pause_*.flag
        echo "[keeper] resumed all"
    else
        rm -f "$FLAG_DIR/pause_all.flag"
        for g in $list; do rm -f "$FLAG_DIR/pause_$g.flag"; echo "[keeper] resumed gpu$g"; done
    fi
}

do_stop() {
    local list="$1"
    [[ -z "${list// /}" ]] && list="$(ls "$FLAG_DIR"/worker_*.pid 2>/dev/null \
        | sed -n 's#.*worker_\([0-9]*\)\.pid#\1#p' | tr '\n' ' ')"
    for g in $list; do
        local pid
        if pid="$(_worker_pid "$g")"; then
            kill -TERM "$pid" 2>/dev/null || true
            for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
            kill -KILL "$pid" 2>/dev/null || true
            echo "[keeper] stopped gpu$g (pid $pid)"
        else
            echo "[keeper] gpu$g not running"
        fi
        rm -f "$FLAG_DIR/worker_$g.pid" "$FLAG_DIR/state_$g.json"
    done
}

do_status() {
    printf '%-5s %-8s %-10s %-10s %-10s %s\n' GPU PID STATE HELD_MiB FOREIGN CARD_USED_MiB
    while IFS=',' read -r idx used; do
        idx="${idx// /}"; used="${used// /}"
        local pid state held foreign
        pid="$(_worker_pid "$idx" || echo '-')"
        state='-'; held=0; foreign=0
        if [[ -f "$FLAG_DIR/state_$idx.json" ]]; then
            state="$(sed -n 's/.*"state": *"\([a-z]*\)".*/\1/p' "$FLAG_DIR/state_$idx.json" | head -1)"
            held="$(_held_mib "$idx")"
            foreign="$(sed -n 's/.*"foreign_mib": *\([0-9]*\).*/\1/p' "$FLAG_DIR/state_$idx.json" | head -1)"
        fi
        [[ -f "$FLAG_DIR/pause_all.flag" || -f "$FLAG_DIR/pause_$idx.flag" ]] && state="${state}/PAUSED"
        printf '%-5s %-8s %-10s %-10s %-10s %s\n' "$idx" "$pid" "$state" "$held" "$foreign" "$used"
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
}

case "$CMD" in
    start)  [[ -z "$GPUS" ]] && GPUS="$(_detect_idle)"; do_start "$(_csv "$GPUS")" ;;
    pause)  do_pause "$(_csv "$GPUS")" ;;
    resume) do_resume "$(_csv "$GPUS")" ;;
    stop)   do_stop "$(_csv "$GPUS")" ;;
    status) do_status ;;
    *) echo "usage: gpu_keeper.sh {start|pause|resume|stop|status} [-g 5,6]" >&2; exit 2 ;;
esac
