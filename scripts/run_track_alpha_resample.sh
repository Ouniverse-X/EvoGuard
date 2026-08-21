#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
PYBIN="${PYTHON_BIN:-/ssd1/conda_envs/evoguard/bin/python}"

echo "[track-alpha] Step 1/3: mine (cap=110, workspace-only)..."
"$PYBIN" -m preliminary.cli --config "${REPO_ROOT}/configs/preliminary_entropy.yaml" --stage mine || {
  echo "[track-alpha] Miner invocation failed." >&2 ; exit 1 ; }

echo "[track-alpha] Step 2/3: migrate v1 corpus -> v2 scenarios/techniques/..."
"$PYBIN" -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from evoguard.process.bench_migrate import migrate
summary = migrate(src_dir='${REPO_ROOT}/bench', dst_dir='${REPO_ROOT}/bench',
                  canonical_aliases_out='${REPO_ROOT}/bench/techniques/aliases.jsonl')
print('migrate summary:', summary)
" || { echo "[track-alpha] Migrate failed." >&2 ; exit 1 ; }

echo "[track-alpha] Step 3/3: release-gate evaluation against refreshed bench/"
"${SCRIPT_DIR}/run_bench_release_gate.sh" --bench-root "${REPO_ROOT}/bench" --contrasts 4

cat <<'POSTRUN'

Track-α completed. Review diagnostics/power_calc.json ratios:
  • If imm/d1/d2 ratios ≥ 1.00 proceed to launch Track-β synthesizer filling d3/d4 deficits.
  • Otherwise investigate why natural pools smaller than predicted; expand mining.source_experiments list further.
POSTRUN
