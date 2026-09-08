#!/usr/bin/env bash
# Hand the cards a finished experiment was using back to gpu_keeper, so node
# utilisation does not sit at zero between runs.
#
# Usage: keeper_takeover_after_run.sh <experiment_pid> [explicit_gpu_csv]
#        (default explicit list: 7 -- the trainer card)
#
# Why this is NOT scripts/release_gpus_after_run.sh:
#   * That script calls `gpu_keeper.sh resume`, which only DELETES pause flags.
#     It cannot put a worker on a card the keeper never held, and the trainer
#     card is exactly that case -- EVOGUARD_KEEPER_EXCLUDE=7 means the keeper has
#     never auto-detected GPU 7, so there is no paused worker to resume. This
#     script calls `start`.
#   * That script also runs stop_vllm.sh. Here the serving endpoints must SURVIVE
#     the training run: the held-out test replay still needs the defender
#     (:8010), the tool executor (:8003) and both judges (:8004/:8006). Their
#     cards are handed over by a later, manual `gpu_keeper.sh start` once the
#     replay is done -- automating that would race the replay.
#
# The explicit list is passed with -g, which bypasses _detect_idle and therefore
# EVOGUARD_KEEPER_EXCLUDE as well; that is the point. A second, argument-less
# `start` then sweeps up anything else that fell idle. It cannot double-launch:
# `do_start` skips a card whose worker pid is alive, and a card the keeper
# already holds reads ~70 GiB so _detect_idle skips it too.
#
# Idempotent and non-destructive: it only ever ADDS keeper workers. Nothing here
# kills a process, and a card that turns out to be occupied merely makes the
# worker's ballast acquire stop early (it fills in 1 GiB chunks until OOM).
set -uo pipefail

EXP_PID="${1:?usage: keeper_takeover_after_run.sh <experiment_pid> [gpu_csv]}"
EXPLICIT_GPUS="${2:-7}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEEPER="$REPO_ROOT/scripts/gpu_keeper.sh"
LOG="$REPO_ROOT/rounds/keeper_takeover.log"

log() { printf '%s %s\n' "[$(date '+%F %T')]" "$*" >>"$LOG"; }

log "watching experiment pid=$EXP_PID; on exit will keeper-start gpu(s) $EXPLICIT_GPUS then auto-detect"
while kill -0 "$EXP_PID" 2>/dev/null; do
    sleep 60
done
log "experiment pid=$EXP_PID has exited"

# The trainer's CUDA context and the colocate engine's allocations are torn down
# by process exit, but the driver needs a moment to account for it -- starting a
# 70 GiB ballast against a card nvidia-smi still reports as full would just OOM
# early and hold a fraction of the target.
sleep 30

bash "$KEEPER" start -g "$EXPLICIT_GPUS" >>"$LOG" 2>&1 || log "keeper start -g $EXPLICIT_GPUS returned nonzero"
sleep 20
bash "$KEEPER" start >>"$LOG" 2>&1 || log "keeper auto-detect start returned nonzero"
sleep 20
bash "$KEEPER" status >>"$LOG" 2>&1 || true
log "handover done; serving endpoints left running for the held-out test replay"
