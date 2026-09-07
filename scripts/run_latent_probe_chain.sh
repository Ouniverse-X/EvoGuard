#!/usr/bin/env bash
# Run one chain of camouflage-probe replay cells sequentially.
#
# Six cells (3 arms x 2 datasets) share the tool executor (:8003) and the judge
# (:8004), so they are NOT all launched at once; the base/ShieldAgent arms also
# share the defender (:8000) with each other. Split into two chains -- one per
# defender endpoint -- and run the chains in parallel:
#
#   scripts/run_latent_probe_chain.sh a    # base + shieldagent  (defender :8000)
#   scripts/run_latent_probe_chain.sh b    # secalign            (defender :8008)
#
# `EVOGUARD_REPLAY_SPLIT=` (empty) is mandatory for every cell: neither probe
# dataset declares metadata["split"], so a task filter would match nothing.
# Output dirs are wiped first because records.jsonl is APPENDED to.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHAIN="${1:?usage: $0 <a|b>}"
CONC="${2:-6}"

run_cell() {
    local cfg="$1" data="$2" adapter="$3" tag="$4"
    echo "=========== CELL $tag ($(date +%H:%M:%S)) ==========="
    /bin/rm -rf "rounds/replay_test_${tag}"
    EVOGUARD_REPLAY_CONFIG="$cfg" \
    EVOGUARD_REPLAY_DATASET_DIR="$data" \
    EVOGUARD_REPLAY_SPLIT= \
        bash scripts/run_replay_heldout.sh "$adapter" "$tag" "$CONC" \
        | tail -n 60
    echo "=========== CELL $tag done ($(date +%H:%M:%S)) ==========="
}

case "$CHAIN" in
a)
    run_cell configs/agentdojo_stock_probe.yaml       data/agentdojo_stock  none adjstock_base
    run_cell configs/agentdojo_latent_probe.yaml      data/agentdojo_latent none adjlatent_base
    run_cell configs/agentdojo_stock_shieldagent.yaml data/agentdojo_stock  none adjstock_shieldagent
    run_cell configs/agentdojo_latent_shieldagent.yaml data/agentdojo_latent none adjlatent_shieldagent
    ;;
b)
    run_cell configs/agentdojo_stock_secalign.yaml  data/agentdojo_stock  metasecalign adjstock_secalign
    run_cell configs/agentdojo_latent_secalign.yaml data/agentdojo_latent metasecalign adjlatent_secalign
    ;;
*)
    echo "[FATAL] unknown chain $CHAIN (expected a|b)" >&2
    exit 2
    ;;
esac
