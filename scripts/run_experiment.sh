#!/usr/bin/env bash
# Run a real EvoGuard experiment from a YAML/JSON config.
#
# Example:
#   scripts/run_experiment.sh configs/example.yaml
#
# The config drives everything: dataset selection, attacker GA, defender LLM,
# training dry-run vs. live, termination criteria, etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.yaml|json>" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
exec python -m evoguard.run --config "$@"
