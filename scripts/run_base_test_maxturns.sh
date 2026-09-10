#!/usr/bin/env bash
# Base (no-adapter) held-out TEST replay on all three datasets at a CONFIGURABLE
# turn budget.
#
# Why this exists: each dataset's shipped config carries a dataset-specific
# `defense.max_turns` (ASB 6, InjecAgent 4, AgentDojo 8) sized for TRAINING cost,
# while the camouflage probe (`configs/agentdojo_{stock,latent}_probe.yaml`) runs
# 18. Base ASR is not comparable across those budgets: measured 2026-09-08, the
# agentdojo_stock base arm's 22 successes include 10 that needed >=10 steps, so an
# 8-turn cap truncates them. This script re-runs the three splits at ONE budget so
# the turn cap is held constant.
#
# It never edits the shipped configs -- `defense.max_turns` is a load-bearing
# training knob (see the InjecAgent config header). Patched copies land in
# rounds/replay_configs/ and are what the run actually reads.
#
# Usage: scripts/run_base_test_maxturns.sh [max_turns] [tag_suffix] [concurrency]
#   scripts/run_base_test_maxturns.sh 18 mt18 8

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TURNS="${1:-18}"
SUFFIX="${2:-mt${TURNS}}"
CONC="${3:-8}"
PY="${EVOGUARD_PYTHON:-/root/paddlejob/workspace/yangxiao/miniconda3/envs/evoguard/bin/python}"

CFGDIR="rounds/replay_configs"
mkdir -p "$CFGDIR"

# name|source config|dataset dir|split value ("" = pass --split "" i.e. no task filter,
# "test" = task-level filter; only agentdojo_split declares metadata["split"])
run_one() {
    local name="$1" src="$2" ddir="$3" split="$4"
    local cfg="$CFGDIR/$(basename "${src%.yaml}")_${SUFFIX}.yaml"
    # Only the single `  max_turns:` line under `defense` exists in all three
    # configs (verified); anchor on two-space indent so an env/attacker knob of the
    # same name could never be hit.
    sed -E "s/^(  max_turns:)[[:space:]]*[0-9]+/\1 ${TURNS}/" "$src" > "$cfg"
    local got
    got="$(grep -E '^  max_turns:' "$cfg" | head -1)"
    if [[ "$got" != "  max_turns: ${TURNS}" ]]; then
        echo "FATAL: max_turns patch failed for $src (got '$got')" >&2
        exit 2
    fi
    echo "===== ${name}: ${cfg} (${got// /}) dataset=${ddir} split='${split}' ====="
    EVOGUARD_PYTHON="$PY" \
    EVOGUARD_REPLAY_CONFIG="$cfg" \
    EVOGUARD_REPLAY_DATASET_DIR="$ddir" \
    EVOGUARD_REPLAY_SPLIT="$split" \
    bash scripts/run_replay_heldout.sh none "${name}_base_${SUFFIX}" "$CONC"
}

run_one asb        configs/asb_opi_grpo.yaml            data/ASB/splits/test                    ""
run_one injecagent configs/injecagent_grpo.yaml         data/InjecAgent/splits/test             ""
run_one agentdojo  configs/agentdojo_sinkdivert_v11.yaml data/toolsafe/agentdojo-tragjnew/test  "test"

echo "ALL DONE -- summarise with:"
for n in asb injecagent agentdojo; do
    echo "  $PY scripts/summarize_replay.py rounds/replay_test_${n}_base_${SUFFIX}"
done
