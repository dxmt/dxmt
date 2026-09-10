# Functional API tests

What the HTTP surface does, asserted against a running server. Not performance, not recall on a
benchmark dataset: the answers a caller gets for a table small enough that every expected id can
be written down.

## Running them

The entry point starts a server for you, once per storage mode, and reports which mode failed:

    cd vectordb
    scripts/api_tests.sh                  # both modes, whole suite
    scripts/api_tests.sh -m "not slow"    # functional tests only, seconds
    scripts/api_tests.sh -k filter -x     # anything else goes through to pytest

To attach to a server you are already running, invoke pytest from `vectordb/` -- the suite is a
package and uses relative imports, so it has to be `pytest tests/api`, not `cd tests/api`:

    cd vectordb && pytest tests/api
    DXMT_URL=http://host:9999 pytest tests/api

`make test-api` and `make test-api-quick` wrap the script.

## Why this exists next to the Rust tests

`tests/*.rs` exercise the storage layer in-process. They cannot catch a route that stopped being
mounted, a field the handler renames on the way out, or a filter that the HTTP layer forwards
differently from the way the storage layer reads it. `api_contract_test.rs` checks that every
route an SDK calls still answers, but not that it answers *correctly*.

## Shape

Borrowed from Qdrant's `tests/openapi` (Apache 2.0 — the shape, not the code):

- One fixture builds the same small, known table for every test, so assertions name exact ids
  rather than counts. A test that only checks `len(hits) == 3` passes when the wrong three come
  back.
- The suite runs twice, once with records and vectors in memory and once with them in row files.
  Those are different code paths here -- different readers, and `rerank_exact` sources vectors
  from a mapped file in one and from the record in the other -- and until now nothing ran the
  same functional assertions across both. Unset means *disk*, so the mode that had never been
  exercised over HTTP was memory.
