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
    DXMT_FEATURES=text-bm25 scripts/api_tests.sh   # full-text search built in; its tests must not skip
    DXMT_FEATURES=api-swagger scripts/api_tests.sh # also starts with --swagger, so the OpenAPI
                                                   # examples are executed rather than skipped

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

## `test_openapi_examples.py` — the Swagger page as a console

Every operation in the OpenAPI document that takes a body carries an example, and the examples
are one walkthrough: create a four-dimensional table called `demo`, put `doc-1`..`doc-4` in it,
search, filter, update, branch, delete. This file runs them in that order against the server and
fails on the first one it refuses.

The reason it exists: Swagger UI fills "Try it out" from the schema when there is no example,
and the schema-derived body here is `{"name": "string", "dimension": 0}`, which the server
rejects. Adding the examples turned the page from something you read into something you run;
this keeps it that way. It caught three wrong examples the day it was written — a field named
`destination` that the handler calls `path`, a `vector` the hybrid route only accepts as
`dense_vector`, and a `top_k` that full-text search spells `limit`.

It skips unless the server serves `/api/docs/openapi.json`, which needs `--features
api-swagger` *and* `--swagger`; `DXMT_FEATURES=api-swagger` arranges both and turns the skip
into a failure.

## `property/` — invariants over a generated corpus

The files above assert exact answers for a corpus of eight. `property/` asserts what must stay
true at 2,000 rows, where no expected id can be written down by hand: the count matches, every
id comes back and none twice, a page is the same page when asked for again, payloads round-trip,
a record is its own nearest neighbour, and the index returns the neighbours a brute-force scan
would. The invariant list is borrowed from Chroma's property suite (Apache 2.0 — the list, not
the code); the corpus is generated from a hash of the row number, so it is identical on every
run and a failure reproduces without a shrinker.

It is marked `slow`, so `-m "not slow"` skips it; CI runs it, because a recall guardrail that
only exists on a developer's machine guards nothing.

Recall is measured against a brute-force answer the test computes in plain Python — CI installs
`tests/requirements.txt` (pytest, httpx, jsonschema) and nothing else. It is a tripwire for wrong neighbours, not a measure of
quantization quality: at 2,000 rows the HNSW-PQ graph is still exact
(`DEFAULT_QUANTIZE_AT_ROWS` is 65,536), so recall sits near 1.0 and the threshold is 0.95.
Raising it to 0.99 would make it grade the graph's parameters instead. Quantization quality is
`make bench-recall`.

`DXMT_PROPERTY_N` and `DXMT_PROPERTY_DIM` shrink the corpus while iterating on a test; the
defaults are the sizes the thresholds were chosen for.
