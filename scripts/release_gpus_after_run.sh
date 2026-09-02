#!/usr/bin/env bash
# Release the two GPUs borrowed for run 20260902_185437 back to gpu_keeper the
# moment the experiment process exits.
#
# GPU 2 -> defender vLLM :8000 (pid in rounds/vllm.pid)
# GPU 3 -> native SFT/GRPO trainer (inside the experiment process)
# Both were taken by `gpu_keeper.sh pause -g 2,3`, so nothing re-acquires them
# until something calls `resume`. This watcher is that something: it waits on the
# experiment pid, stops the vLLM server it no longer needs, and resumes the
# keeper so node utilisation does not sit at zero afterwards.
#
# Idempotent and non-destructive: it only signals the vLLM server we started
# ourselves (via scripts/stop_vllm.sh, TERM then KILL after 20s) and calls
# `gpu_keeper.sh resume`, which merely removes the pause flag.
set -uo pipefail

EXP_PID="${1:?usage: release_gpus_after_run.sh <experiment_pid>}"
GPUS="${2:-2,3}"
REPO_ROOT="/root/yangxiao/EvoGuard"
LOG="$REPO_ROOT/rounds/gpu_release_watch.log"

log() { printf '%s %s\n' "[$(date '+%F %T')]" "$*" >>"$LOG"; }

log "watching experiment pid=$EXP_PID; will release GPUs $GPUS when it exits"
while kill -0 "$EXP_PID" 2>/dev/null; do
    sleep 60
done
log "experiment pid=$EXP_PID has exited"

bash "$REPO_ROOT/scripts/stop_vllm.sh" >>"$LOG" 2>&1 || log "stop_vllm.sh returned nonzero"
sleep 10
/root/gpu_keeper.sh resume -g "$GPUS" >>"$LOG" 2>&1 || log "gpu_keeper resume returned nonzero"
log "released GPUs $GPUS back to gpu_keeper"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >>"$LOG" 2>&1 || true
