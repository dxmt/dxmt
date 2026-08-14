#!/usr/bin/env bash
# Launch DXMT VectorDB with GPU (CUDA) acceleration enabled.
#
# GPU stays an opt-in cargo feature -- this wrapper just sets up the CUDA 11
# runtime libs (via scripts/gpu-env.sh) and runs the server with `--features gpu`.
# Default builds/runs are unaffected.
#
# Usage:
#   scripts/run-gpu.sh                                  # simple mode, 1024 MB
#   scripts/run-gpu.sh --simple --memory-limit 2048
#   scripts/run-gpu.sh --host 0.0.0.0 --port 9999       # standard mode
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=gpu-env.sh
source "$REPO/scripts/gpu-env.sh"

export RUST_LOG="${RUST_LOG:-info}"
cd "$REPO"

if [ "$#" -eq 0 ]; then
  set -- --simple --memory-limit 1024
fi

echo "[run-gpu] launching: vectordb serve $*"
exec cargo run --features gpu --bin vectordb -- serve "$@"
