# Changelog

Notable changes to DXMT VectorDB. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims at [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once there is
something to be semantic about.

**Nothing has been released yet.** `Cargo.toml` says `0.2.0`, no version has been published, and
the `v0.2.0-*` tags in this repository are measurement checkpoints from benchmarking runs, not
releases. Everything below is unreleased work on `main`.

## [Unreleased]

### Added

- **The JS SDK has an eslint configuration**, so `npm run lint` runs instead of failing with
  "looked for configuration files and found none". Seven source files, clean.

- **A LangGraph example, and the LangGraph and LlamaIndex suites now run.** `langgraph` and
  `llama-index-core` are installed in CI's live job, `dxmt-python-sdk/examples/langgraph_memory.py`
  is an agent whose long-term memory is a DXMT namespace, and the LlamaIndex integration tests
  embed locally instead of calling OpenAI -- they had skipped themselves everywhere, so that
  package's live path had never run at all.

- **A field the server does not have is now an error.** Every request body carries
  `deny_unknown_fields` and goes through `ApiJson` (`src/api/extract.rs`), so a misspelled or
  invented field is a 400 naming it and listing the ones that exist. serde's default -- ignore
  it -- had cost this API real behaviour three times: `mode` on an insert (ten names for one
  code path, removed from the server while every SDK went on sending it), `alpha` and `top_k`
  on a hybrid search (in this crate's own Swagger example, tuning nothing), `named_vectors` on
  a table that then built no named index and searched the default vector instead. None of them
  failed; they were accepted, dropped, and answered wrongly. `ApiJson` also replaces axum's
  rejection -- a 422 with a `text/plain` body -- with the normal error envelope, so a client
  reading `error.code` no longer chokes on the most ordinary mistake there is.
  `tests/unknown_fields_test.rs` checks every body endpoint, simple mode's three, a nested
  filter condition, and -- by reading the source -- that no handler has gone back to a plain
  `Json<T>` body or a request type without the attribute.

- **`Field`, a filter builder for Python.** `(Field("lang") == "en") & Field("year").between(2020,
  2024) & ~Field("tags").is_in(["spam"])`, accepted wherever a filter is. Python's `and`/`or`/`not`
  and chained comparisons raise `TypeError`, since each would silently keep one side.

- **Paging by cursor.** `records/list` returns `next_cursor`, and takes it back as `cursor`. An
  offset shifts under writes — the list is newest first, so each insert pushes the rest back and
  the next page repeats a record — where a cursor names a place in the order: every record
  present for a whole scroll is returned exactly once. The Python and JavaScript SDKs' `scroll`
  pages by it, and their `upload` loads any iterable in batches, in parallel, retrying what a
  retry can fix and reporting what it could not.

- **Every client can send the API key.** `DXMT_API_KEY` (22fc06a) left the Python SDK, the
  LangChain and LlamaIndex packages, and the Rust and Go clients with no way to send it — against
  a keyed server they could not be used at all. Each takes a key now (`api_key=`,
  `with_api_key`, `WithAPIKey`; the JavaScript SDK already had `apiKey`), and none of them
  prints it: pydantic stores keep it off their fields, and the Rust and Go clients redact it
  from `Debug` and `%v`. CI runs every client's live suite a second time against a server that
  requires a key, after proving the server refuses a request without one.

- **Filtering on array fields.** A condition on a field holding an array is satisfied when any
  element satisfies it, so `tags == "rag"` finds `tags: ["rag", "llm"]` — Qdrant's rule, and
  what Chroma's `$contains` and Weaviate's `ContainsAny` are for. It used to compare the whole
  array with `"rag"` and match nothing. `NotEqual` and `NotIn` exclude a record when any element
  matches; Qdrant's `except` on an array instead keeps it if any element does not match, which
  lets `["spam", "x"]` through an exclusion of `"spam"`.

- **An optional API key.** `DXMT_API_KEY` puts a shared secret in front of every route but the
  health probes, sent as `api-key` (Qdrant's header name) or `Authorization: Bearer`. Off by
  default, as it is in Qdrant, Chroma and Weaviate. One static key, no rotation and no accounts
  — a seatbelt for a port that outlived its firewall rule, not a substitute for TLS and a
  private network.

- **Durability.** A write-ahead log on the write path, replayed at startup: `kill -9` and
  restart preserves acknowledged writes. Per-table snapshots of records and index carry recovery
  past the log they cover, taken periodically, on shutdown, and after a rebuild; the log is
  compacted once a snapshot reads back. `HnswIndex` and `HnswPqIndex` gained a versioned on-disk
  format so a restart does not rebuild the graph from scratch.
- **Records and vectors on disk.** With a WAL directory configured, a table's records and full
  vectors live in row files beside it and only the graph, the PQ codes and the id map stay
  resident — about 0.5 KB per 768-dimension row, against 8.3 KB when everything was in the map.
  `DXMT_RECORD_STORE=memory` / `DXMT_VECTOR_STORE=memory` put a new table back in memory.
- **Backup and restore**, replacing two earlier paths that never worked, with a test that proves
  a backup restores rather than that it reported success.
- **Object storage as the durable home for snapshots and WAL chunks**, plus table eviction:
  a table nothing has queried is dropped from memory entirely and rehydrated on the next query.
  Still experimental; the local WAL remains the readiness boundary.
- **Partitioned IVF** whose partitions load from object storage on demand, for a single table
  larger than memory. Experimental.
- **An MCP server** over stdio (`vectordb mcp`), backed by the same durable storage as the HTTP
  server.
- **Operational probes.** `/live` for liveness and `/ready` for readiness, the latter returning
  503 while local durability is degraded; `/health/metrics` exposes the WAL frontier, last fsync
  and snapshot times, and snapshot failure counts.
- **A functional HTTP test suite** (`tests/api/`) that runs the same assertions against a server
  with records in memory and again with them in row files, plus property invariants over a
  generated corpus.
- `DXMT_HNSW_EF_CONSTRUCTION` to set the build effort of new tables; `search_mode` so a caller can
  ask for more search effort without a rebuild.

### Changed

- **The SDKs no longer send `mode` on an insert** (Python, JavaScript), and the parameter is
  gone from `insert_records`/`ainsert_records` and `InsertRecordsParams`. The server deleted the
  field; a call still passing it now fails loudly in the client rather than being ignored by the
  server. Nothing has been released, so this breaks no published API.

- **A filter comparing against an array or an object is refused** with `INVALID_INPUT`, from
  every endpoint. `tags == ["a", "b"]` used to return no records from `records/list`, a
  selective search or `search_mode: "exact"`, and the right ones from a broad or full-text
  search, depending on which path answered. `Equal`, `NotEqual`, `In` and `NotIn` take strings,
  numbers, booleans and null, which is what Qdrant's match accepts too.

- **A filtered page no longer costs the whole match.** `records/list` with a filter matching
  2.5M of 10M rows took 2.9 s a page, almost none of it filtering: every match became an owned
  id, was hashed back to its row to read a timestamp, and was freed, to return five. The index
  now keeps each record's `created_at` and chooses the page itself; 0.2 s. At 1M rows, 211 ms to
  13 ms.

- **The payload index rebuilds in half the time and memory.** The first filtered query after a
  restart rebuilds it, and at 10M rows that went from 40 s to 27 s end to end, with the server's
  peak memory from 8.7 GB to 7.2 GB and every answer unchanged. A value's records are held
  inline when there is one and in a roaring bitmap when there are many, where each distinct
  value used to get its own hash set; record ids are stored once rather than twice; and the
  index's maps use a keyed `ahash` instead of `SipHash`.

- **Every server-side environment variable is now `DXMT_*`.** `DB_` is not a namespace — it is
  the word "database", and `DB_HOST`, this server's bind address, is also what almost every
  compose file calls its Postgres. Qdrant namespaces its variables and so does Weaviate; this
  follows the convention while renaming is still free. The old `DB_*` name is still read, warns
  once at startup naming its replacement, and will be removed before the first release.

- **The default feature set is now the minimal supported surface**: `tokio-runtime` +
  `basic-storage`. Full-text search (`text-bm25`), Swagger, GraphQL, WebSocket, S3, LMDB,
  cluster and multimodal are opt-in. An interface that was not compiled in returns an explicit
  configuration error rather than mounting a route that does nothing.
- **Index building moved off the single core it was on.** Every node's edges sit behind their own
  lock, neighbour planning happens in parallel, queued insert requests merge into one index
  insert, and a bulk build makes one pass across all cores. Measured on Cohere 1M: total load
  time 7,478 s to 1,020 s.
- **HNSW-PQ is the default for a new table**: an exact graph below 65,536 rows, PQ codes above
  with an exact rerank from the row file. Accepted at 1M and 10M against exact HNSW within 0.2
  points of recall@100, at 1.9x the QPS and 4.66 GB resident against 66.9 GB.
- `ef_search` is read at search time and scales with table size; it used to be a hard-coded
  constant, so the configured value never reached a query.
- Snapshots, rebuilds and evictions no longer hold the table lock across their disk work, and a
  rebuild converges its catch-up outside the lock.
- jemalloc is the default allocator, with its background decay thread on.

### Fixed

- **The Go client can clear a payload.** `Update.Payload` is `map[string]any` with
  `omitempty`, which drops an empty map as readily as a nil one -- so "clear this payload"
  never reached the wire, and the server, which keeps what an update does not mention, left the
  old payload in place. A nil map now means "leave it alone" and an empty one clears it.

- **`import dxmt` is quiet again.** `metagpt_vector_store.DXMTVectorStore` declared `client` as
  a pydantic field, shadowing the abstract property `BasePydanticVectorStore` declares, and
  carried a v1-style `class Config`: two warnings printed on every import once llama-index was
  installed. It now uses `PrivateAttr` plus the property the interface asks for, and
  `ConfigDict`, which is what the standalone LlamaIndex package always did.

- **A LangGraph namespace round-trips.** A namespace is a table, joined with `_` and split on
  `_`, so any component containing one came back wrong: `("agent", "memory", "session_123")`
  listed as four components, a prefix search for it matched nothing, and `("a_b",)` and
  `("a", "b")` quietly shared one table. An underscore inside a component is now doubled. A
  namespace whose table name would exceed the server's 63 characters says so, rather than
  failing on a name the caller never chose; and a store whose `embedding_dims` disagrees with
  the table the namespace already has is refused by name -- a store with no `embed_fn` writes a
  1-element placeholder, so it used to make every later put fail with the server's bare
  "expected 384, got 1". Seven tests of this store had been failing behind a skip.

- **The Python SDK translates a MongoDB-style filter exactly or not at all.** Four shapes ran as
  a different query and returned the wrong records: `{"$or": [{"a": 1, "b": 2}, {"c": 3}]}`
  became `a or b or c`; a key beside `$and`/`$or` was dropped; a nested `$and` became a
  condition on a field named "$and"; a native `must` filter lost any other key. A range with a
  string bound (`{"date": {"$gte": "2024-01-01"}}`, in the README's own example) matched
  nothing, since ranges compare numbers. Each now raises `ValueError`, as does anything else
  without an exact form in `must`/`should`/`must_not`. The LangGraph store's query-less search
  filters on the server instead of re-reading every record with a second, disagreeing
  implementation of the filter.

- **A filter now returns the same records from every endpoint.** `records/list`, a selective
  filtered search, `search_mode: "exact"` and the k-NN graph take the payload index's answer
  as final, while other searches check each record; the two disagreed in five ways. `NotEqual`
  dropped records lacking the field or holding an array; `NotIn` missed arrays; `id` was
  answered from a payload field of that name rather than the record's id; numbers were keyed
  at six decimal places and saturated at ±9.2e12, so every microsecond timestamp was one value;
  and `5` and `5.0` were equal to one and not the other. On a 400-row table, 12 of 14 filters
  returned wrong records through both `records/list` and exact search.
- **`5` now equals `5.0` in a filter**, everywhere. JSON has one kind of number, and a
  JavaScript client cannot send `5.0` at all.

- **Filtered search returned almost nothing below 5% selectivity.** The graph walk put the
  exploration queue and the result set behind the same filter test, so a non-matching entry point
  left nothing to explore: at 2% selectivity, 97.65% of queries returned zero rows.
- **String filters silently returned short pages.** `Contains` and `EndsWith` estimated
  selectivity from a hard-coded constant, so the over-fetch was too small and the retry the
  comments promised did not exist. All twelve filter operators now return a full page.
- **Hybrid search ranked the dense side backwards**, feeding distances into a fusion that scores
  higher-is-better; the nearest vector sorted last.
- **Concurrent snapshots corrupted each other** through a shared staging directory, and an index
  write failure was folded into "this index type has no disk format" — so a snapshot reported
  success with the graph silently dropped.
- **Row files grew with the number of writes rather than with the amount of data.** Both the
  vector row file and the record row file are append-only: an update appends a new row and
  repoints the id, and the row it left behind kept its blocks. Nothing reclaimed them -- the
  only thing that claimed to was a comment describing a rewrite no code performed, and the
  sweep that rebuilds an index fires on a change of algorithm, which a table of constant size
  never has. Measured in a soak at 200 rows/s over a fixed 400,000-row key space: 5.2 GB of
  vector file holding 1.2 GB of live rows after 2.4 hours, growing at about 52 GB a day
  whatever the table's size. A new sweep (`DXMT_ROW_COMPACTION_INTERVAL_SECS`, default 600)
  releases the blocks behind dead rows. Row numbers are unchanged, so nothing above the file
  layer moves; the file's apparent length still grows, and its allocated size no longer does.
- **Quantizing a table stopped every insert in the process for the length of the training.**
  Training the PQ codebooks at 65,536 rows was taken out of the index write lock in an earlier
  change, which fixed what searches saw and left writers where they were: measured in a soak at
  200 rows/s on eight cores, acknowledged writes stopped dead for the 256 s the training ran,
  the WAL sequence did not move, and eight batches timed out at the client and were answered on
  arrival by `Failed to send insert response: insertion result was dropped`. Two things
  serialised them, and removing either alone was not enough. Training ran on the storage layer's
  single insertion task, so inserts queued behind the `await`; it now runs in a spawned blocking
  task. And it ran on rayon's global pool as one unyielding chunk per worker, which is the same
  pool `HnswIndex::plan_batch` submits to above `PARALLEL_PLAN_MIN_NODES`, so an insert could
  not be picked up until a worker finished its whole share of the subspaces; it now trains in a
  pool of its own, sized to half the cores.
- **An upsert grew the index for ever.** A graph node cannot move -- where it sits is decided
  by the vector it had when it was linked in -- so an update tombstones the old slot and
  appends a new one, and nothing gave the slot back. A table of constant size under steady
  upserts therefore grew one slot per write. Measured in a 17-hour soak at 200 rows/s over a
  400,000-row key space: 11.1 million slots holding 400,000 records, `index_size_bytes` 5.3 GB
  where the live rows need about 200 MB, resident memory climbing 8.4 GB a day with the row
  count flat for sixteen of those hours, snapshots grown from 1.2 s to 18.9 s, and 196 s to
  come back from `kill -9`. The re-evaluation sweep now compares live entries against slots and
  rebuilds when the graph carries twice the slots it needs; `rebuild_index` no longer counts
  such a graph as equivalent to the one it would build.
- **The two append-only sidecars grew for ever as well**, for the same reason and with no dead
  regions to release: the vector file's `(id, row)` list and the record file's append log take
  a line per write, and an upsert writes. 214 MB and 303 MB respectively over the same 17
  hours. Both are now rewritten by the reclamation sweep -- the row list keeping each id's
  latest row, the record log keeping the live rows and replacing each run of dead ones with a
  `gap` that advances the offset accumulator without creating a row, so the surviving rows
  still name their own bytes and the data file needs no rewrite. Each commits with a single
  `rename`.
- **A restored table could come back as the wrong index.** Recovery ignored the algorithm the
  snapshot recorded, rebuilding a 94,000-row brute-force table as HNSW before binding the port.
- `update_record` never wrote to the WAL, so updates did not survive a restart. Upserts left
  stale sparse postings, payload index entries and named vectors behind.
- Non-finite vectors are refused at the one path every writer goes through.
- Score ties are broken by id before any cut to `k`, so the same query returns the same page.
- A dropped table stays dropped; an evicted table is no longer erased by the next snapshot.

### Removed

- **25 files of leftover debugging from the LlamaIndex package's root**, plus three from its
  examples: seven progress-report markdown files (`FIXES_APPLIED.md`, `TEST_RESULTS_CN.md`,
  `RAGAS_QUICKFIX.md`, …) and sixteen ad-hoc `test_*.py` scripts that printed rather than
  asserted, hardcoded `/localdisk/...` paths from another machine, and were collected by
  nothing -- everything they probed is covered by `tests/`. The one analysis worth keeping
  moved to `design/`, which is not distributed.

- 51 source files that were never part of the build, five modules superseded by the code that is
  actually wired up, and every example that did not run.
- `api/grpc.rs`, which was never declared by `api/mod.rs`. There is no gRPC interface.

### Known limitations

See the "Known Limitations" section of [README.md](README.md). The short version: the graph is
built inside the write request, so ingest and queries contend and a rebuild is always a whole
table; object-storage mirroring is experimental; several index names fall back to brute force;
and the multimodal modules return placeholder vectors.
