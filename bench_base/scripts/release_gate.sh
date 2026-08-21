#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYBIN="${PYTHON_BIN:-/ssd1/conda_envs/evoguard/bin/python}"
exec "$PYBIN" -c '
import sys,argparse
sys.path.insert(0,"'"${REPO_ROOT}"'")
parser=argparse.ArgumentParser()
parser.add_argument("--bench-root",required=True)
parser.add_argument("--contrasts",type=int,default=4)
a=parser.parse_args()
from evoguard.process.bench_release_gate import evaluate_release_state
verdict=evaluate_release_state(bench_root=a.bench_root,planned_contrasts_count=a.contrasts)
import json
print(json.dumps(verdict,indent=2))
' "$@"
