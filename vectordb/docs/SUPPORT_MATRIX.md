# Support Matrix

**The answer to "can I use this?"** — separate from "does this module exist?" and from "is it
wired up?". A module being present in this repository is not, by itself, a promise that it is
safe to use: this tree contains modules that compile, carry unit tests, and are reached by no
binary.

This is a single-node database. Distributed mode was designed, was never implemented, and is
not being worked on; the scaffolding for it is in `planning/`, not in the shipped crate.
Everything below describes one process.

Numbers quoted here were measured on one machine with the harness in this repository; the
README carries the method beside each headline figure.

## What the 1.0 compatibility promise covers

A breaking change to anything in this table needs a major version. Everything here is
single-node.

| Area | Supported path | Notes |
| --- | --- | --- |
| Server | `vectordb serve` | Axum HTTP server; routes live under `/api/v1`, probes at the root. One data model, several protocols (REST, gRPC, GraphQL, WebSocket, MCP) — the `serve --simple` mode that served a second model at `/collections/*` has been removed. |
| Index tuning | `hnsw` and `quantization` on create | Per table: `m`, `ef_construction`, `ef_search`, and `num_subvectors`, `num_centroids`, `quantize_at_rows`. Persisted, so rebuilds and restarts keep them; `quantize_at_rows` above the table's eventual size keeps a table exact. `GET /tables/{t}/stats` reports `quantized`, which is the regime the index is in and decides both its recall and its memory. |
| Tables | Create, list, alter, delete, branch | Table configuration is persisted in WAL and snapshots. |
| Records | Insert/upsert, get, update, delete (by id or filter), list, count | JSON payloads, versions, sparse vectors and named vectors are retained. |
| Concurrent insert and search | Supported | Verified by `StreamingPerformanceCase`, which ingests and searches at once: recall 0.508 -> 0.993 across the four ingest stages at 1,777 -> 3,301 concurrent QPS. Until `0ed359b` the self-quantization at 65,536 rows trained its codebooks inside the index write lock, freezing every read and write on the table for the duration -- 56 s on the shipped 192-subspace default, and a 47.7 s concurrent p99 in that one window. Training now runs with no lock held. |
| Capacity | Bounded by ingest rate, not by memory, on a 121 GB machine | `CapacityDim960` loaded 27.8M x 960-dim vectors on 13,506 MB resident (0.49 KB a row) and was still going when the two-hour window closed; `CapacityDim128` 4.8M x 128-dim on 2,413 MB. Neither reached the case's terminus, which is exhausting memory: at 0.5 KB a row that needs more than 200M rows. |
| Dense search | HNSW-PQ (the default), HNSW, brute force | A new table is HNSW-PQ: an exact graph below 65,536 rows, PQ codes above with an exact rerank of 5k candidates from the row file. Accepted at 1M and 10M against exact HNSW (recall@100 within 0.2 points, 1.9x the QPS, 4.66 GB resident against 66.9 GB at 10M). `DXMT_MEMORY_OPTIMIZATION=false` restores the size-based selector (brute force / HNSW). `ef_search` doubles per decade of rows above a million, read at search time (`e0e65fb`): a fixed budget loses recall as the graph grows, and segmented engines get that growth for free from their segment count. **Full standard-case coverage, one binary, 2026-09-09**: 768d 1M 0.9806/6,310 QPS, 10M 0.9793/3,682; 1536d 50K 0.9962/5,577, 500K 0.9897/4,489, 5M 0.9816/4,281; 1024d 1M 0.9488/5,900, 10M 0.9640/3,741. The 1024d figures are lower because Bioasq needs more traversal budget than Cohere at the same scale, not because the codes lose more -- the accurate mode reaches 0.9786 on the same codes. The rule is calibrated on Cohere; a different corpus may want a different `ef_search`. A head-to-head comparison against other engines on the same machine exists in the internal measurement notes and is deliberately not quoted here: a competitive number is worth publishing only beside the full method and the tuning used on both sides. |
| Filtering | Payload equality, range, string operations, `IsEmpty`/`IsNull`, on top-level and nested fields (`meta.lang`) | Payload indexes are an optimization; `Record::matches_filter` remains the source of truth, and the index is only allowed to answer where the two provably agree. **Numbers are a sorted array per field, not a set per distinct value** (since `14034a4`, 2026-09-10): a range condition is two binary searches and a contiguous slice, and a numeric value costs 16 bytes in the index. Until then a field with a value per row -- the benchmark filters on the primary key -- was ten million single-element sets: 10M rows at 1% selectivity answered at 108.8 QPS with a 236 ms serial p99 against 3,682 QPS unfiltered, and the one integer field cost 2.5 GB resident (7,149 MB against 4,659 MB with no payload). Those are the *before* figures; the 10M case has not been re-measured since, so do not quote an after. Strings, booleans and nulls stay in inverted sets, which is the right shape for them. **The cost that remains is in the middle of the selectivity range** (measured at 1M rows, 2026-09-26, `tests/scale_filters_test.rs`): a filter matching under ~5% of the table is scored exactly over the matches, one matching nearly everything is a plain graph search, and one matching 12-50% is 160-800 ms against ~2 ms unfiltered, because the planner has only a post-filtered graph walk to offer there. **Every payload field is indexed unless the table says otherwise**, which is what makes a filter fast with nothing declared; `payload_index` on `POST /tables` (`only` / `except`), `PUT`/`DELETE /tables/{t}/payload_index/{field}` and `GET /tables/{t}/payload_index` are how a field that nobody filters on stops paying for it. A filter naming an excluded field is **refused with a 400**, not answered: the index is the only thing that answers a filter here, so the alternative is an empty page that looks like a result. |
| Persistence | Local WAL, local snapshots, backup directory, row files | The server uses `new_with_wal`; `DXMT_WAL_DIR` controls the location. Records (`wal/records/<table>.rec` + `.log`) and full vectors (`wal/vectors/<table>.f32`, or `.f16` with `DXMT_VECTOR_F16=1`) are row files beside the WAL; a snapshot of such a table carries the index and no records, so a 10M snapshot takes 7 s rather than 108 s. A backup must include them. |
| Recovery | Snapshot load followed by WAL replay | A corrupt snapshot is a startup error after the covered WAL has been compacted. |
| Durability failure | Mutations return HTTP 503 and the process becomes read-only until restart | The failed mutation is not acknowledged; it may have reached memory and is not guaranteed to survive recovery. `snapshot` and `backup` also refuse in this state — deliberately, so an unacknowledged in-memory write cannot become a new durable frontier — so a backup has to be taken from the on-disk WAL and snapshot directory instead. |
| Text search | Tantivy/BM25 via `/search/text` | In the default build since 2026-10-02 (`text-bm25`); a table opts in with `enable_full_text_search` and `text_fields`. A text field with `"analyzer": "chinese"` is indexed and queried through jieba word segmentation, lowercased, mixed-script text included; the default analyzer treats a run of CJK characters as one word, so it finds a Chinese sentence only by the whole sentence. The analyzer is a property of the table's index, chosen at create. |
| Hybrid search | Vector/text endpoint `/search/hybrid` | Sparse-vector fusion is outside the promise. |
| Client surface | Python, JavaScript, Go and Rust clients | Every route each one calls is listed in `tests/api_contract_test.rs` and served by the router, and all four run against a live server in CI. The LangChain, LlamaIndex and LangGraph adapters go through the Python client and have live suites of their own. |
| Authentication | `DXMT_API_KEY`, optional | Unset means open, which is what every comparable database ships. When set, every route but `/live`, `/ready` and `/health` requires `api-key` or `Authorization: Bearer`, compared in constant time. Every client can send it. |
| Request validation | Unknown fields and parameters are refused | A body or query string carrying a field the server does not have is a 400 naming it, rather than being accepted and ignored — which is how `mode`, `alpha` and `named_vectors` each spent a period being sent, dropped, and answered wrongly. |
| Per-table export/import | `tables/{t}/export`, `tables/import` | One table to a file and back, on this server or another. Records keep their ids, payloads, versions and `created_at`; the graph is not in the file and the import rebuilds it. Both paths are bounded by `DXMT_BACKUP_ROOT` — the path comes off the wire and this server has no authentication, so an unbounded export is a write primitive and an unbounded import a read one. |
| Diversity reranking | `mmr` on `search` | Server-side MMR over the candidates' own vectors, so no client-side re-embedding. Diversity is bounded by `fetch_k`. |
| Facets / value counts | `POST /tables/{t}/facets` | How many records hold each value of a payload field, optionally under a filter. Computed from the payload index, which is already a value-to-records map, so it does not page the table. **The answer carries `covered` and `matched` and a caller must read them**: an index-backed facet loses what the index does not hold as a *count of zero* rather than as an absence, so a record with no value for the field -- or one whose value is an object, which the index records as presence only -- is in no bucket at all. Their difference is what the facet cannot describe, and `hits` alone will show a corpus that is not the corpus. A field the table's `payload_index` excludes is refused rather than counted as zero, for the same reason a filter on one is. `covered` costs a transient set of record ordinals, because a record with an array value belongs in several buckets and summing counts would double it. The page is the `limit` most common values (default 20, capped at 1,000), count descending then value ascending so two identical calls answer identically; `distinct_values` says how many there were. **Measured at 1M rows, 2026-09-26**: 37 ms on an 8-value field, 71 ms on a 2,000-value one. Under a filter matching 6,290 records it is still 13 ms and 23 ms -- **the walk is over the field's postings, not over the matched rows**, so a facet inside a narrow filter costs what the whole field costs, not what the selection costs. Budget by the field's size, not by the filter's. |
| Geographic filtering | `GeoRadius` and `GeoBox` on any payload field holding `{"lat": …, "lon": …}` | The first filter whose value is an object, which is why it has an index half of its own: a bounding box of latitudes intersected with one of longitudes is not a circle, and the search path **takes the index's answer** rather than re-checking it, so a superset would be wrong records returned as right ones. The geo half stores both coordinates together, uses the latitude order to bound the walk, and runs `GeoPoint::distance_m` -- the same function `Record::matches_filter` calls, on the same `f64` bits -- in place. A box whose west edge is east of its east edge crosses the antimeridian. A coordinate written as a two-element array is **not** a point: GeoJSON says `[lon, lat]` and every map UI says `[lat, lon]`, and guessing puts a record on the wrong continent with nothing about the answer looking wrong. Geometry that is not on the earth is a 400, not an empty page. Costs 24 bytes a point on top of the two numeric entries the same value already makes -- `shop.lat` range filters keep working -- and `payload_index`'s `except` is how to decline it. `tests/geo_filter_test.rs` checks the index against the matcher on 60 random circles over a random corpus. **Measured at 1M rows, 2026-09-26** (`tests/scale_filters_test.rs`, 16C/62G): a radius costs what its latitude band costs and nothing more -- 0.09 ms at 1 km (60 matches), 1.2 ms at 10 km (6,290), 13.4 ms at 100 km (125,000), 32.7 ms at 1,000 km (375,000). Memory for one geographic field at 1M rows is **56 MB**: 24 MB of points plus 16 MB each for the `lat` and `lon` numeric halves that keep `field.lat` range filters working. |
| Server-side embedding | A table names a model; records and searches carry `text` | `embedding` on `POST /tables` names an OpenAI-compatible endpoint (`base_url` reaches OpenAI, Jina, Voyage, text-embeddings-inference, Infinity, ollama, vLLM, llama.cpp). A record with `text` and no vector is embedded on the way in and the text is filed in the payload, so a hit carries the chunk it came from; a search with `text` is embedded with the *table's* model, never the caller's choice. The API key is never in a request body or a snapshot -- `api_key_env` names an environment variable of the server process. **A provider that will not answer fails the request**: no zero vector, no hashed stand-in, because a wrong vector stores cleanly and ranks wrongly forever. `model` cannot be changed by `PATCH` on a table that holds rows; the endpoint can. `dimension` may be omitted when `embedding` is set: the server asks the model once, at create, and a create whose provider is unreachable fails rather than inventing a width. The MCP tools (`create_table`, `insert_vector`, `vector_search`) take text as well, which is the protocol whose caller certainly has no embedding model. `tests/server_embedding_test.rs` (18 cases) and `tests/mcp_stdio_test.rs` run a real OpenAI-protocol server on loopback and cover the reorder, 429, 401, wrong-width, unreachable, width-inference and restart paths. |
| Search paging | `offset` on `search` | Pages tile the ranking. Costs `offset + top_k` at the index, so deep paging is not flat; `records/list` with a cursor is. |

## What each client covers

Six clients ship from this repository, and they are not equivalent. Python and JavaScript
wrap most of the HTTP surface -- every route in the table below, plus aliases, payload-index
policy, facets, grouped search, export/import and DXQL -- and leave the operational routes
(snapshot, backup, branch, named vectors, schema evolution) to a raw request; Rust and Go are
deliberately small clients over the core, kept honest by live tests in CI rather than by
feature parity. Picking one of the latter for a
feature it does not wrap means one `rawRequest`-shaped call of your own, not a missing server
feature -- every capability below is in the HTTP API.

| | Python | JavaScript | Rust | Go | LangChain | LlamaIndex |
| --- | --- | --- | --- | --- | --- | --- |
| Tables: create, list, get, drop | yes | yes | yes | yes | via Python | via Python |
| Records: insert, get, update, delete | yes | yes | yes | yes | yes | yes |
| Search (filters, modes, `offset`, `mmr`) | yes | yes | search + filter | search + filter | search + MMR | search |
| `count`, cursor `scroll`, bulk `upload` | yes | yes | — | — | — | — |
| Sparse search, grouped search | yes | yes | — | — | — | — |
| Payload index management | yes | yes | — | — | — | — |
| Aliases, per-table export/import | yes | yes | — | — | — | — |
| Full-text and hybrid search | yes | yes | — | — | — | — |
| DXQL (`POST /query`) | yes | — | — | — | — | — |

| Table aliases | `aliases` | Another name for a table, usable anywhere a table name is. Moving one is the zero-downtime half of a rebuild. Stored in `{DXMT_WAL_DIR}/aliases`, so a backup that copies the WAL directory carries them. |
| Sparse search | `search/sparse` | Dot product over the WAND-pruned sparse index, with filters and a score floor. Default build, no `text-bm25` needed. The index is derived from the records and rebuilt on demand, so the first sparse query after a restart pays for the build. |
| Grouped search | `search/groups` | Groups a window of nearest neighbours by a payload field, widening it until enough groups appear (cap 10,000 candidates). `candidates_fetched` and `ungrouped` are in the answer because grouping over a window cannot see a group whose every member is further away than the window reached. |
| Counting | `records/count` | Exact, with or without a filter, computed from the index that answers the filter rather than by paging the matches. |
| Paging | `records/list` by cursor | `next_cursor` goes back as `cursor`, and a record present for a whole scroll is returned exactly once however the table is written to meanwhile. An offset is still accepted, and still shifts under concurrent inserts. |
| Ground truth | `search_mode: "exact"` | Scores every live row, so recall is measurable on your own data rather than only against a published dataset. 300 ms for 1M x 960 with the row file in page cache (2026-10-02); a cold read of the file is the real cost. Refused above `DXMT_EXACT_SCAN_MAX` rows (default 10,000,000 since 2026-10-02; it was 1,000,000, which refused exactly the size where a ground truth is most wanted) and for named vector spaces. |
| API reference | `GET /api/docs` | Swagger UI and the OpenAPI document, served by default (`--no-swagger` turns it off); `api-swagger` is a default feature. Every documented example is executed by `tests/api/test_openapi_examples.py`, so the document cannot drift from the server. |
| Web console | `GET /dashboard` | Read-only, served by default when the bundle was compiled in (`make dashboard-build`); `--no-dashboard` turns it off, and `--dashboard` makes a missing bundle a refusal to start rather than a warning. Behind `DXMT_API_KEY` like every other route. |
| Monitoring | `GET /metrics` | Prometheus text exposition at the root, in the default build: `dxmt_up`, `dxmt_tables`, `dxmt_records`, `dxmt_ongoing_operations{kind=…}`, `dxmt_operations_total`, the WAL/snapshot frontier and the last-sync and last-snapshot timestamps. `tests/prometheus_metrics_test.rs` parses every sample, so a value a scraper would reject fails the build. |
| Operations | `/health`, `/live`, `/ready`, `/health/metrics` | `/live` is liveness; `/ready` is authoritative readiness and returns 503 when local durability is degraded. |

## Experimental or Not Yet Supported

These modules may compile or have unit tests, and are **not** covered by the 1.0 promise.
They can change or be removed in a minor release:

| Area | Current status | Required before promotion |
| --- | --- | --- |
| HNSW-PQ below 65,536 rows | Exact graph; the codebooks are not trained until then | Nothing — it is the same graph HNSW builds. Listed so nobody expects a 10,000-row table to be smaller than under HNSW. |
| Object storage | Snapshot and WAL mirroring can be enabled | Recovery, failure injection and multi-process safety tests. |
| Partitioned IVF | Implemented behind the adaptive path | Long-running maintenance and remote-segment recovery tests. |
| DiskANN / ScaNN / QuantizedIVF | Not selectable as stable indexes | Complete index implementations and round-trip tests. |
| EnhancedDiskANN | Reachable only with `DXMT_MEMORY_OPTIMIZATION=false`: the size-based selector picks it above a million rows once a notional in-memory HNSW would exceed half the machine's RAM (~18M rows at 768d on a 121 GB box, ~2M on a 16 GB one). 748 lines, a real search, **no delete** (`NOT_IMPLEMENTED`) and **no test of its own**. Until 2026-09-07 the periodic sweep routed *every* table here at that size, whatever it was created with. | Delete support and a round-trip test before anything is routed here by default. |
| LMDB / RocksDB | **Not in the crate.** In `planning/src/storage/` since 2026-09-11 | A documented factory path and complete search semantics. |
| Cluster / consensus | **Not in the crate.** Scaffolding, in `planning/src/` | Replication, split-brain and recovery semantics. |
| Multimodal / GPU / PyO3 / the legacy embedding clients | **Not in the crate.** Moved to `planning/src/` on 2026-09-27 — this row previously read "reachable from the server but return placeholder vectors", and that is what was fixed. `gpu` and `python` did not compile; `multimodal` compiled and would have answered with made-up vectors | Real model backends and reproducible integration tests. |
| GraphQL / WebSocket / MCP | Optional interfaces | Contract tests and versioned compatibility guarantees. |

Unsupported operations must return an explicit `NOT_IMPLEMENTED` error. They must not silently
fall back to brute-force search or discard part of a query.

## Runtime Data Path

The supported server path is intentionally narrow:

```text
REST / simple API / MCP
              |
              v
  IntelligentIndexedStorage
      |       |       |
  records   indexes   payload/text indexes
      |
      +--> local WAL --> snapshot manifest and segments
                              |
                              +--> optional object storage mirror
```

Records and indexes are resident in the process while a table is hot. WAL is the durability
boundary for acknowledged writes. Snapshots reduce replay time and are the only copy of data
covered by a compacted WAL, so the snapshot directory must be included in backups.

## Probe Contract

- `GET /live` (and `/health/live`) only proves that the process and HTTP runtime are responding.
- `GET /ready` (and `/health/ready`) proves that the local WAL directory is accessible and the
  storage instance has not entered the post-fsync-failure read-only state. It returns HTTP 503
  when not ready. The same readiness routes are mounted under `/api/v1` for SDKs.
- `GET /health` is a diagnostic health document and may report load/backpressure state; it is not
  a replacement for the readiness probe.
- `GET /health/metrics` includes operation counters plus WAL/snapshot frontier, last-sync and
  last-snapshot timestamps, snapshot failure diagnostics, and resident-table information. These
  metrics are process-local and are not a distributed monitoring contract.

## Configuration Contract

The documented persistence controls are:

```text
DXMT_WAL_DIR
DXMT_SNAPSHOT_INTERVAL_SECS
DXMT_EVICTION_INTERVAL_SECS
DXMT_REINDEX_INTERVAL_SECS
DXMT_PARTITION_THRESHOLD_MB
DXMT_COMPRESSION
```

`DXMT_STORAGE_TYPE` was removed on 2026-09-27. It was parsed but selected no implementation,
and this paragraph used to say it "must not be presented as a backend switch until it is
connected" -- removing it is how that was settled.

## Recall on published ground truth

Measured 2026-09-26 on one 16-core box, through HTTP, against the `neighbors` array each
`ann-benchmarks` dataset ships — a ground truth this project did not compute. 200 queries per
mode, `top_k` 10, shipped defaults otherwise. These are the first cross-dataset figures here
that are not scored against our own exact scan.

| dataset | rows × dim | metric | ingest | fast | normal | accurate | p50 accurate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnist-784 | 60K × 784 | euclidean | 1,766/s | 0.9645 | 1.0000 | 1.0000 | 6.0 ms |
| fashion-mnist-784 | 60K × 784 | euclidean | 1,869/s | 0.9820 | 1.0000 | 1.0000 | 5.0 ms |
| sift-128 | 1M × 128 | euclidean | 4,570/s | 0.9805 | 0.9965 | 0.9985 | 7.4 ms |
| gist-960 | 1M × 960 | euclidean | 811/s | 0.8700 | 0.9830 | 0.9955 | 14.4 ms |
| nytimes-256 | 290K × 256 | cosine | 1,685/s | 0.8400 | 0.9470 | 0.9770 | 12.3 ms |
| lastfm-64 | 292K × 65 | dot | 5,816/s | 0.8935 | 0.9130 | 0.9910 | 3.9 ms |
| glove-25 | 1.18M × 25 | cosine | 10,125/s | 0.7475 | 0.7450 | 0.9325 | 5.9 ms |
| glove-50 | 1.18M × 50 | cosine | 6,052/s | 0.7995 | 0.8445 | 0.9715 | 8.2 ms |
| glove-200 | 1.18M × 200 | cosine | 1,995/s | 0.7425 | 0.8620 | 0.9495 | 11.4 ms |

**Ingest rate tracks dimension, not rows -- and the 811/s at 960 was the client.** Re-measured
on 2026-10-02 (release build, 16 cores, the first 300K rows of GIST-960, batches of 1,000):

| client | rows/s | where the client's time went |
| --- | --- | --- |
| 1 client, Python stdlib `json` (every earlier figure) | 762 | 201 s encoding, 193 s waiting on the server |
| 1 client, orjson (what the Python SDK uses) | 972 | 39 s encoding, 270 s waiting |
| 4 clients, orjson | **1,507** | 16 s encoding, 180 s waiting -- the server's ceiling |

A 1,000 x 960 batch is a 20 MB JSON body that stdlib `json` takes 0.8 s to produce and the
server 0.12 s to parse and validate. The SDK's `upload(parallel=N)` with orjson installed is
the shape that reaches the ceiling. Inside the server, per-stage timers over a 1M load (one
`info` line per 250 groups) put **graph construction at 81%** of storage time, the WAL at 14%
(the vector is written to the row file and again, compressed, to the log), the row file at
3% and the record map at 2%; the server averaged 7 of 16 cores under one client. The next
gain is the planner's parallel efficiency, not the wire.

**The PQ code is sized by the table's width** (since 2026-10-02; `HnswPqConfig::default_subvectors`).
It was four dimensions per subspace at every width, the layout measured at 768, and at 25
dimensions that was a 100-byte vector in a 5-byte code -- the 0.745 in the table above is that
code, not the graph. Re-measured on 2026-10-02, 1.18M rows, recall@10 at `normal`, p50 in
brackets:

| corpus | 4 dims/subspace (old default) | 2 | 1 | no PQ |
| --- | --- | --- | --- | --- |
| glove-25 | 0.7595 (3.0 ms) | — | **1.0000** (3.7 ms) | 1.0000 (3.8 ms) |
| glove-100 | 0.8950 (4.6 ms) | **0.9235** (5.6 ms) | 0.9275 (6.3 ms) | 0.9210 (3.8 ms) |
| glove-200 | 0.8485 (5.5 ms) | **0.8775** (6.5 ms) | 0.8810 (5.3 ms) | — |
| gist-960 | **0.9780** (7.8 ms) | — | — | — |

The default is now one dimension per subspace below 64, two below 256, four from there (the
bold column). At 25 dimensions the code was the whole problem and the new layout is exact at
`normal`; at 100 and 200 it is worth three points and then matches the unquantized graph to
within the noise. **What remains there is the corpus, not the graph.** hnswlib built on the
same glove-100 rows, same machine, M=16 and `ef_construction` 200, queried at the budgets
`normal` and `accurate` use:

| glove-100, recall@10 | ef 512 (`normal`) | ef 2048 (`accurate`) |
| --- | --- | --- |
| hnswlib 0.8 | 0.9160 | 0.9745 |
| DXMT, unquantized | 0.9210 | 0.9770 |
| DXMT, PQ 2 dims/subspace | 0.9235 | 0.9760 |

Doubling `ef_construction`, doubling `m` or inserting in batches of 100 moved `normal` by at
most 1.6 points. So on a corpus like this the recall a budget buys is what HNSW buys, and a
recall *target* has to be met by spending more budget on that table than on an easy one --
calibrating `ef_search` per table against its own exact scan is the next change; until
then `accurate` is the mode for a hard corpus. The same run put hnswlib's build at 13,000
rows/s on 16 threads against about 4,300 here, which is the ingest gap stated in cores. A
table created before this change keeps the layout it was built with.

**A small fraction of records cannot be found by the graph on some corpora.** Searching with
a vector that is already stored returns it for **100% of a modern-embedding corpus** — 60,000
paragraphs through bge-m3 (1024d) and through OpenAI `text-embedding-3-large` (3072d), zero
missed in both — and for 98% of GIST-960. The 2% are points nothing else is near, which the
graph's neighbour selection leaves with no incoming edges; GIST is a 2008 image descriptor
full of near-duplicates and isolated points, and the shortfall has not been reproduced on
anything a retrieval application would store. `ef` and `top_k` do not help; they are a
budget and this is not. Ordinary retrieval is unaffected, because those points are nobody's
nearest neighbour (recall@10 was 0.9955 on GIST while 5% was unreachable at a million rows).
Deduplication and near-duplicate detection are the workloads that hit it, and
`search_mode: "exact"` is the deterministic answer for them.
`index_record_count` on `GET /tables/{t}/stats` distinguishes this from records the index
never received. The standard mitigation (`keepPrunedConnections`) was implemented, measured
on three corpora and removed — see `select_neighbors_heuristic` for the table.

**A standard dataset may not load as it ships.** GIST-960 contains ten all-zero descriptors
and nytimes-256 contains 239. They are legitimate under euclidean (the origin is a point) and
meaningless under cosine (no direction), and are accepted and refused accordingly. What is
worth knowing before a bulk load is that **a batch is all-or-nothing**: one refused record
fails the whole request, so a million-row load stops at the first bad row and the caller has
to find it. Filter them out, or load in small batches.

## Choosing an embedding model, which is a different question

The table above measures **the index**: how close the graph's answer is to the exact one, on
datasets whose right answers were published by somebody else. It says nothing about whether a
model is right for your documents, and that is the question that decides whether retrieval
works.

`examples/generative_benchmark.py` answers it without needing labels: an LLM writes, for each
passage, the query a reader would type that this passage answers, and the pair is the label.
Every query then runs through the graph **and** through `search_mode: "exact"`, so a miss is
attributed to the model or to the index rather than merely counted.

Five models, this repository's own documentation, 165 passages, the same 165 generated queries
every time -- only the embedding changes. Measured 2026-09-28:

| provider | model | dim | recall@1 | recall@10 | NDCG@10 | misses the index caused |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | text-embedding-3-large | 3072 | **0.6391** | 0.9231 | 0.7706 | 0 of 61 |
| Gemini | gemini-embedding-001 | 3072 | 0.6331 | **0.9349** | **0.7790** | 0 of 62 |
| OpenAI | text-embedding-ada-002 | 1536 | 0.5976 | 0.8698 | 0.7316 | 0 of 68 |
| Gemini | gemini-embedding-2 | 3072 | 0.5385 | 0.8876 | 0.7125 | 0 of 78 |
| OpenAI | text-embedding-3-small | 1536 | 0.5325 | 0.8698 | 0.7000 | 0 of 79 |

**Across five models and 348 misses, the index caused none of them.** `accurate` and `exact`
returned identical scores in all five runs. On this corpus the retrieval quality is decided
entirely by the embedding, and no amount of `ef_search` tuning would move it — which is worth
knowing before spending a week on the index.

**Two of the five are newer models that scored worse than the ones they replace.**
`text-embedding-3-small` loses to `ada-002` by 6.5 points of recall@1 at the same width, and
`gemini-embedding-2` loses to `gemini-embedding-001` by 9.5. Picking the newest model is a
reasonable default and is wrong here, twice, in different families. That is the argument for
measuring on your own corpus rather than reading a leaderboard.

These numbers are about *this* corpus — 165 chunks of one project's documentation, and
generated queries, which are only as good as the model that wrote them. Read them with
`--show` before trusting the ranking. What transfers is the method, not the order.

**Any OpenAI-compatible endpoint works.** The Gemini rows went through
`base_url: https://generativelanguage.googleapis.com/v1beta/openai` with no code change; the
key is named by `api_key_env` and never appears in a create body, a snapshot or an export.

## Promotion Rules

A feature can move into the supported table only when it has:

1. A documented API and configuration contract.
2. A clean restart and recovery test.
3. A corruption or dependency-failure test where applicable.
4. A deterministic CI entry point.
5. No silent fallback that changes the requested semantics.
