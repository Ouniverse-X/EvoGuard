#!/usr/bin/env bash
# Release the GPUs borrowed for a run back to gpu_keeper the moment the
# experiment process exits.
#
# Usage: release_gpus_after_run.sh <experiment_pid> [gpu_list] [helper_port ...]
#
# Current layout (run 20260902_233537, configs/agentdojo_universal_seeded_v10.yaml):
#   GPU 2 -> defender vLLM :8000     (pid in rounds/vllm.pid)
#   GPU 3 -> native SFT/GRPO trainer (inside the experiment process)
#   GPU 4 -> helper vLLM :8004       (pid in rounds/vllm_helper_8004.pid)
# All three were taken by `gpu_keeper.sh pause -g 2,3,4`, so nothing re-acquires
# them until something calls `resume`. This watcher is that something: it waits on
# the experiment pid, stops the vLLM servers it no longer needs, and resumes the
# keeper so node utilisation does not sit at zero afterwards.
#
# Any helper ports listed are passed to stop_vllm.sh -- a helper left running at
# gpu_memory_utilization 0.85 would block the keeper from taking its card back,
# which is the exact failure this script exists to prevent.
#
# Idempotent and non-destructive: it only signals vLLM servers we started
# ourselves (via scripts/stop_vllm.sh, TERM then KILL after 20s) and calls
# `gpu_keeper.sh resume`, which merely removes the pause flag.
set -uo pipefail

EXP_PID="${1:?usage: release_gpus_after_run.sh <experiment_pid> [gpu_list] [helper_port ...]}"
GPUS="${2:-2,3,4}"
shift 2 2>/dev/null || shift $#
HELPER_PORTS=("$@")
REPO_ROOT="/root/yangxiao/EvoGuard"
LOG="$REPO_ROOT/rounds/gpu_release_watch.log"

log() { printf '%s %s\n' "[$(date '+%F %T')]" "$*" >>"$LOG"; }

log "watching experiment pid=$EXP_PID; will release GPUs $GPUS when it exits (helpers to stop: ${HELPER_PORTS[*]:-none})"
while kill -0 "$EXP_PID" 2>/dev/null; do
    sleep 60
done
log "experiment pid=$EXP_PID has exited"

bash "$REPO_ROOT/scripts/stop_vllm.sh" ${HELPER_PORTS[@]+"${HELPER_PORTS[@]}"} >>"$LOG" 2>&1 \
    || log "stop_vllm.sh returned nonzero"
sleep 10
/root/gpu_keeper.sh resume -g "$GPUS" >>"$LOG" 2>&1 || log "gpu_keeper resume returned nonzero"
log "released GPUs $GPUS back to gpu_keeper"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >>"$LOG" 2>&1 || true
