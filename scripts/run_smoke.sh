#!/usr/bin/env bash
# Quick offline smoke run: no GPU, no model server, no network.
# Exercises the full pipeline end-to-end with the deterministic MockClient.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
exec python -m evoguard.run --smoke "$@"
