#!/usr/bin/env bash
# Launch DXMT VectorDB with GPU (CUDA) acceleration enabled.
#
# GPU stays an opt-in cargo feature -- this wrapper just sets up the CUDA 11
# runtime libs (via scripts/gpu-env.sh) and runs the server with `--features gpu`.
# Default builds/runs are unaffected.
#
# Usage:
#   scripts/run-gpu.sh                                  # defaults
#   scripts/run-gpu.sh --host 0.0.0.0 --port 9999
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=gpu-env.sh
source "$REPO/scripts/gpu-env.sh"

export RUST_LOG="${RUST_LOG:-info}"
cd "$REPO"

echo "[run-gpu] launching: vectordb serve $*"
exec cargo run --features gpu --bin vectordb -- serve "$@"
