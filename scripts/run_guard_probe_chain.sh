#!/usr/bin/env bash
# Run the four TS-Guard / PIGuard cells of the camouflage probe, sequentially.
#
# Each cell is a full replay (evoguard/eval/vendored_replay.py) over all four
# suites: 48 attacked scenarios + 43 clean tasks. The pairs are run
# back-to-back so an endpoint hiccup cannot land on only one arm of a
# comparison.
#
# Preconditions -- verify BOTH capability controls first, they are cheap and a
# fail-open arm silently reports the undefended base ASR:
#   python scripts/probe_piguard_detection.py     # exits 3 if any forward failed
#   python scripts/probe_tsguard_defense.py       # exits 3 if the guard never fired
#
# Endpoints: defender :8000, tool executor :8003, judge :8004, TS-Guard :8009.
# PIGuard needs no endpoint (in-process on CPU).
#
# TS-Guard is served on GPU7 with `--gpu-memory-utilization 0.27 --max-num-seqs
# 16 --max-model-len 8192`. Both bounds are forced, not chosen: 0.27 is the most
# of the card's total that fits in the ~23 GiB GPU7 has left, and at vLLM's
# default max_num_seqs the sampler warm-up (1024 dummy requests) OOMs *after*
# KV-cache allocation and CUDA-graph capture have already succeeded, so the
# engine dies looking healthy.
#
#   EVOGUARD_VLLM_MODEL=/root/paddlejob/workspace/yangxiao/models/TS-Guard \
#   EVOGUARD_VLLM_NAME=ts-guard EVOGUARD_VLLM_GPU=7 EVOGUARD_VLLM_PORT=8009 \
#   EVOGUARD_VLLM_MEM_UTIL=0.27 EVOGUARD_VLLM_MAXLEN=8192 \
#   EVOGUARD_VLLM_EXTRA_ARGS="--max-num-seqs 16" \
#     bash scripts/start_vllm_secondary.sh
#
# Then summarize both pairs against the base/secalign/struq arms:
#   python scripts/summarize_latent_vs_stock.py \
#       tsguard=rounds/replay_test_adjstock_tsguard:rounds/replay_test_adjlatent_tsguard \
#       piguard=rounds/replay_test_adjstock_piguard:rounds/replay_test_adjlatent_piguard

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONC="${1:-8}"

for arm in tsguard piguard; do
    for ds in stock latent; do
        tag="adj${ds}_${arm}"
        echo "=================================================================="
        echo "[cell] $tag  (concurrency=$CONC)"
        echo "=================================================================="
        /bin/rm -rf "rounds/replay_test_${tag}"
        EVOGUARD_REPLAY_CONFIG="configs/agentdojo_${ds}_${arm}.yaml" \
        EVOGUARD_REPLAY_DATASET_DIR="data/agentdojo_${ds}" \
        EVOGUARD_REPLAY_SPLIT= \
            bash scripts/run_replay_heldout.sh none "$tag" "$CONC"
    done
done

echo "all four cells done"
