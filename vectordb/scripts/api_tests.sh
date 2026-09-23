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
#                             api-dashboard also passes `--dashboard`, which is now about
#                             strictness rather than enablement: the console is served by
#                             default, and asking for it by name turns a missing bundle from a
#                             warning into a refusal to start -- which is what a job that has
#                             just built the bundle wants
#   PYTEST=...                a specific interpreter's pytest
#   DXMT_SKIP_STATEFUL=1      leave out the stateful walk (tests/stateful) that runs after both
#                             modes

# Deliberately not `set -e`: both modes must run, so the summary can say which one failed.
set -uo pipefail
cd "$(dirname "$0")/.."

MODES=${DXMT_MODES:-"memory disk"}
PYTEST=${PYTEST:-"python3 -m pytest"}
BIN=target/debug/vectordb
export DXMT_FEATURES=${DXMT_FEATURES:-}

# Both the OpenAPI document and the console are served by default now, so these flags are no
# longer what turns them on. `--dashboard` is still worth passing when the caller named the
# feature: it makes a missing bundle fatal instead of a logged warning, which is the difference
# between "the console tests ran" and "the console tests quietly found nothing to test".
# `test_openapi_examples.py` still turns its skip into a failure when `DXMT_FEATURES` names
# api-swagger, for the same reason.
# Appending, not assigning: with two features named, the second used to silently replace the
# first, so `DXMT_FEATURES=api-swagger,api-dashboard` served only one of them.
SERVE_FLAGS=""
_features=",$(echo "$DXMT_FEATURES" | tr ' ' ','), "
case "$_features" in *,api-dashboard,*) SERVE_FLAGS="$SERVE_FLAGS --dashboard" ;; esac

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
  # `DXMT_RECORD_STORE=memory`, logging an info line as it does. Share a directory between the two
  # runs and let one table name collide, and the memory run silently exercises the disk path and
  # passes.
  wal_dir=$(mktemp -d)
  backup_root=$(mktemp -d)
  log=$(mktemp)

  # `DXMT_WAL_DIR` must be set. Without one, *both* stores fall back to memory whatever these
  # variables say, so "disk mode" would silently test memory and go green.
  #
  # `DXMT_INDEX_ALGORITHM` must be pinned. `create_table` ignores the caller's `index_type` and
  # lets `select_optimal_index` choose from `available_memory_gb`, so a CI runner and a laptop
  # can select different indexes and the recall invariant would be measuring different things.
  DXMT_RECORD_STORE="$mode" \
  DXMT_VECTOR_STORE="$mode" \
  DXMT_WAL_DIR="$wal_dir" \
  DXMT_BACKUP_ROOT="$backup_root" \
  DXMT_INDEX_ALGORITHM=hnsw-pq \
  DXMT_SNAPSHOT_INTERVAL_SECS=0 \
  DXMT_EVICTION_INTERVAL_SECS=0 \
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
    # `DXMT_WAL_DIR` is passed so a test can assert something about where the server keeps its
    # durable state -- `POST /backup` refusing a destination inside it, for one. Tests that need
    # it skip when it is absent, so attaching to someone else's server still works.
    #
    # `DXMT_BACKUP_ROOT` is the same value the server was started with, in the pytest process
    # this time: a backup may only be written inside it, so a test that takes one has to know
    # where that is. It used to work by accident -- the WAL lives under `/tmp`, so the default
    # root was `/tmp` and any `mkdtemp` landed inside it -- and an accident is not coverage.
    #
    # Server and tests used to spell these differently (`DB_*` / `DXMT_*`); the prefix rename
    # collapsed the pair into one name, which is what they always meant.
    DXMT_URL="http://127.0.0.1:$port" DXMT_STORAGE_MODE="$mode" DXMT_WAL_DIR="$wal_dir" \
    DXMT_BACKUP_ROOT="$backup_root" \
      $PYTEST tests/api "$@"
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

# The stateful walk starts, kills and restarts its own servers, so it runs once, after the
# per-mode loop, with the binary the loop used. Its seeds and length are the defaults in
# tests/stateful/test_walk.py; DXMT_STATEFUL_SEEDS and DXMT_STATEFUL_STEPS lengthen it locally.
if [ "${DXMT_SKIP_STATEFUL:-0}" != "1" ]; then
  echo "### stateful walk (writes, restarts and crashes against a model)"
  DXMT_BIN="$BIN" $PYTEST tests/stateful -q -p no:cacheprovider
  if [ $? = 0 ]; then
    summary="$summary\n  stateful walk: PASS"
  else
    summary="$summary\n  stateful walk: FAIL"
    failed=1
  fi
fi

# Data written by earlier builds, opened by this one (tests/compat). Seconds, and the only thing
# here that fails when a storage format changes in a way old data cannot survive.
echo "### storage compatibility (tests/compat/fixtures/*)"
DXMT_BIN="$BIN" $PYTEST tests/compat -q -p no:cacheprovider
if [ $? = 0 ]; then
  summary="$summary\n  storage compatibility: PASS"
else
  summary="$summary\n  storage compatibility: FAIL"
  failed=1
fi

printf "\nfunctional API tests:%b\n" "$summary"
exit $failed
