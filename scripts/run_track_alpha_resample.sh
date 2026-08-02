#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
PYBIN="${PYTHON_BIN:-/ssd1/conda_envs/evoguard/bin/python}"

echo "[track-alpha] Re-running miner scoped to workspace-only with elevated cap..."
"$PYBIN" -m preliminary.cli --config "${REPO_ROOT}/configs/preliminary_entropy.yaml" --stage mining-only || {
  echo "[track-alpha] Miner invocation failed." >&2 ; exit 1 ; }

echo "[track-alpha] Running release-gate evaluation against refreshed bench/"
"${SCRIPT_DIR}/run_bench_release_gate.sh" --bench-root "${REPO_ROOT}/bench" --contrasts 4

cat <<'POSTRUN'

Track-α completed. Review diagnostics/power_calc.json ratios:
  • If imm/d1/d2 ratios ≥ 1.00 proceed to launch Track-β synthesizer filling d3/d4 deficits.
  • Otherwise investigate why natural pools smaller than predicted; expand mining.source_experiments list further.
POSTRUN
