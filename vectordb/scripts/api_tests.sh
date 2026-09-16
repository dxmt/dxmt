#!/usr/bin/env bash
# Run tests/api against a server we start ourselves, once per storage mode.
#
# Why the script owns the server and pytest does not: a session-scoped fixture that restarts the
# server mid-session would have to own process reaping, log capture, orphan cleanup and an
# escape hatch for "I already have a server", underneath a module-scoped table fixture. `trap`
# does all of that in five lines. It also keeps `conftest.py` reading nothing but `DXMT_URL` and
# `DXMT_STORAGE_MODE`, so a developer with a server already running still types
# `cd vectordb && pytest tests/api`.
#
# Usage:
#   scripts/api_tests.sh                 both modes, whole suite
#   scripts/api_tests.sh -m "not slow"   functional tests only
#   scripts/api_tests.sh -k filter -x    anything else is passed through to pytest
#
#   DXMT_MODES="memory"       one mode only
#   DXMT_SKIP_BUILD=1         the binary is already built (CI)
#   DXMT_FEATURES=text-bm25   build with these cargo features, and tell the tests the server has
#                             them: a test that needs one then fails where it would otherwise
#                             skip, so a build that lost the feature cannot pass. Naming
#                             api-swagger also starts the server with `--swagger`, which is what
#                             makes the OpenAPI examples checkable
#   PYTEST=...                a specific interpreter's pytest

# Deliberately not `set -e`: both modes must run, so the summary can say which one failed.
set -uo pipefail
cd "$(dirname "$0")/.."

MODES=${DXMT_MODES:-"memory disk"}
PYTEST=${PYTEST:-"python3 -m pytest"}
BIN=target/debug/vectordb
export DXMT_FEATURES=${DXMT_FEATURES:-}

# Compiling Swagger in is not the same as serving it -- `--swagger` is a separate runtime
# decision, and without the flag the binary carries the document and answers 404 for it. So the
# flag follows the feature, and `test_openapi_examples.py` turns its skip into a failure when
# `DXMT_FEATURES` names api-swagger: otherwise "the examples are checked" degrades into "the
# examples were not there to check" without anything going red.
SERVE_FLAGS=""
case ",$(echo "$DXMT_FEATURES" | tr ' ' ','), " in *,api-swagger,*) SERVE_FLAGS="--swagger" ;; esac

if [ "${DXMT_SKIP_BUILD:-0}" != "1" ]; then
  cargo build --bin vectordb ${DXMT_FEATURES:+--features "$DXMT_FEATURES"} || exit 1
fi
[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 1; }

free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

server_pid=""
wal_dir=""
cleanup() {
  # The server may have forked; kill the group. `setsid` below is what makes that possible, and
  # what a bare `kill $pid` would leave behind.
  [ -n "$server_pid" ] && kill -- "-$server_pid" 2>/dev/null
  [ -n "$wal_dir" ] && rm -rf "$wal_dir"
}
trap cleanup EXIT INT TERM

failed=0
summary=""

for mode in $MODES; do
  port=$(free_port)
  # A fresh WAL directory per mode is not optional. A table keeps the store it already has:
  # `open_record_store` finds an existing row file and keeps using it *even under*
  # `DB_RECORD_STORE=memory`, logging an info line as it does. Share a directory between the two
  # runs and let one table name collide, and the memory run silently exercises the disk path and
  # passes.
  wal_dir=$(mktemp -d)
  log=$(mktemp)

  # `DB_WAL_DIR` must be set. Without one, *both* stores fall back to memory whatever these
  # variables say, so "disk mode" would silently test memory and go green.
  #
  # `DB_INDEX_ALGORITHM` must be pinned. `create_table` ignores the caller's `index_type` and
  # lets `select_optimal_index` choose from `available_memory_gb`, so a CI runner and a laptop
  # can select different indexes and the recall invariant would be measuring different things.
  DB_RECORD_STORE="$mode" \
  DB_VECTOR_STORE="$mode" \
  DB_WAL_DIR="$wal_dir" \
  DB_INDEX_ALGORITHM=hnsw-pq \
  DB_SNAPSHOT_INTERVAL_SECS=0 \
  DB_EVICTION_INTERVAL_SECS=0 \
  RUST_LOG=${RUST_LOG:-info} \
    setsid "$BIN" serve --host 127.0.0.1 --port "$port" $SERVE_FLAGS > "$log" 2>&1 &
  server_pid=$!

  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS -m 2 -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null; then ready=1; break; fi
    kill -0 "$server_pid" 2>/dev/null || break   # it died; stop waiting on a corpse
    sleep 0.5
  done

  if [ "$ready" != "1" ]; then
    echo "### $mode: server never answered on port $port"
    tail -30 "$log"
    summary="$summary\n  $mode: FAIL (server did not start)"
    failed=1
  else
    echo "### $mode: server on port $port, wal $wal_dir"
    DXMT_URL="http://127.0.0.1:$port" DXMT_STORAGE_MODE="$mode" $PYTEST tests/api "$@"
    rc=$?
    if [ "$rc" = 0 ]; then
      summary="$summary\n  $mode: PASS"
    else
      summary="$summary\n  $mode: FAIL (pytest exit $rc)"
      # Only on failure, and only the lines that are not routine chatter: a debug server logs
      # an info line per table created, which on a failing run buries the pytest output that
      # says which test failed.
      echo "--- $mode: server warnings and errors ---"
      grep -iE "warn|error|panic" "$log" | tail -20
      failed=1
    fi
  fi

  kill -- "-$server_pid" 2>/dev/null
  wait "$server_pid" 2>/dev/null
  server_pid=""
  rm -rf "$wal_dir" "$log"
  wal_dir=""
done

printf "\nfunctional API tests:%b\n" "$summary"
exit $failed
