# DXMT VectorDB

A vector database written in Rust. HNSW, HNSW-PQ and product quantization with runtime SIMD
dispatch, hybrid retrieval and a query language behind a REST API, plus a zero-config
collection mode. Full-text search (Tantivy), GraphQL, WebSocket, Swagger and S3 are opt-in
build features — see [Feature flags](#feature-flags) — so a default build is the minimal
supported single-node server and nothing more.

> **Status: Alpha development, not production-ready.** Records are held in memory and made durable by a write-ahead log,
> replayed on startup — `kill -9` and restart preserves your tables. The log defaults to
> `~/.dxmt/store` — everything the server writes lives under one root, `DXMT_ROOT_DIR`, which
> defaults to `~/.dxmt`. Under the home directory rather than the working directory, so
> starting the server from somewhere else does not open a different, empty database. **Set
> `DXMT_ROOT_DIR` for anything real**: a home directory is usually on the root filesystem, this
> table can be tens of gigabytes, and it is often NFS, where `mmap` and `fsync` do not mean
> what a WAL needs them to mean.
>
> ```text
> ~/.dxmt/                DXMT_ROOT_DIR
> ├── store/              the database — back this up      (DXMT_WAL_DIR)
> ├── backups/            where POST /backup may write     (DXMT_BACKUP_ROOT)
> └── cache/              derived; deleting it costs time, not data  (DXMT_CACHE_DIR)
> ```
>
> The three have three different lifetimes, which is why they are three directories. Each can
> be moved on its own; the root moves all of them. All four are created and checked for
> writability at startup, so an unwritable `backups/` is a refusal to start rather than a
> surprise for whoever takes the first backup. When configured, snapshots, WAL chunks and
> partition segments are mirrored to object storage, but that mirror is still experimental and
> local WAL remains the readiness boundary. Distributed/cluster code exists but is not
> production-ready. See [Known Limitations](#known-limitations).

Anything this README does not describe is experimental, even when its module compiles or has
standalone tests.

## Quick Start

Requires a recent stable Rust toolchain.

```bash
# Build
cargo build --release

# Run
./target/release/vectordb serve --host 0.0.0.0 --port 9999
```

Verify it's up:

```bash
curl http://localhost:9999/health
# Readiness is the deployment probe; it returns 503 after a local WAL durability failure.
curl -f http://localhost:9999/ready
```

Two things are served without asking, because a database you cannot look at and whose API you
cannot read is harder to evaluate than one you can:

| | Endpoint | Turn off with |
| --- | --- | --- |
| API reference | `GET /api/docs`, spec at `/api/docs/openapi.json` | `--no-swagger` |
| Web console | `GET /dashboard` — read-only | `--no-dashboard` |

Both are read-only, and both require `DXMT_API_KEY` when one is set, like every other route.
The console needs its bundle compiled in (`make dashboard-build`, then rebuild); a binary built
without it says so once at startup and serves everything else. Passing `--dashboard` explicitly
turns that warning into a refusal to start, which is what you want in a deployment that is
supposed to have one.

GraphQL and WebSocket are not compiled into the default build, and are off unless asked for:

```bash
cargo build --release --features api-graphql,api-ws
./target/release/vectordb serve --graphql --websocket
```

| Flag          | Endpoint                                            |
| ------------- | --------------------------------------------------- |
| `--graphql`   | `POST /graphql`                                      |
| `--websocket` | `GET /ws` (upgrade)                                  |
| `--grpc-port` | gRPC on that port, alongside HTTP (`proto/dxmt.proto`) |

`GET /api/v1/tables/{name}/stats` reports `resident`: whether the table was in memory when
asked. A query against an evicted table rehydrates it transparently, so without this a caller
has no way to know the next one will pay for a fetch from object storage.

`GET /metrics` is the Prometheus scrape endpoint, at the root where every scrape config
expects it — the same numbers as below, in text exposition format, in the default build.

`GET /health/metrics` (also `/api/v1/health/metrics`) includes the local durability frontier:
`storage.wal_next_sequence`, `storage.last_wal_sync_at`, `storage.last_snapshot_at`, and
`storage.snapshot_failures`/`last_snapshot_error`, plus resident and total table counts. These
timestamps and counters reset on restart; the WAL and snapshot manifest remain authoritative.

## Counting the values of a field

```bash
curl -X POST localhost:9999/api/v1/tables/docs/facets \
  -d '{"key": "topic", "limit": 20}'
```

```json
{"hits": [{"value": "rust", "count": 412}, {"value": "go", "count": 87}],
 "distinct_values": 14, "covered": 499, "matched": 512}
```

Counted from the payload index, which is already a value-to-records map, so it does not page
the table. A `filter` narrows it, and `matched` then reports what the filter selected.

**`covered` against `matched` is not decoration.** A facet built from an index describes what
the index holds; a record with no value for the field, or one whose value is an object, is
missing from every bucket rather than having a bucket of its own. Here that is 13 records the
answer cannot describe — and a chart drawn from `hits` alone would show a corpus of 499 where
there are 512, with nothing to say so. A field the table's `payload_index` excludes is refused
outright, for the same reason a filter on one is.

## Filtering by place

Any payload field holding `{"lat": …, "lon": …}` can be filtered on:

```json
{"vector": [...], "top_k": 10, "filter": {"must": [
  {"key": "shop", "operation": {"type": "GeoRadius",
   "value": {"center": {"lat": 52.5163, "lon": 13.3777}, "radius_m": 1500}}}]}}
```

`GeoBox` takes `top_left` and `bottom_right` instead, and is the right shape for a map
viewport; a box whose west edge is east of its east edge crosses the antimeridian rather than
being empty. Both combine with every other condition, including under `must_not`.

Three things are deliberate:

* **A radius is a circle, not its bounding box.** The index stores the two coordinates together
  and runs the real great-circle test, because the search path takes the index's answer rather
  than re-checking it — a superset there is wrong records returned as right ones.
* **A two-element array is not a point.** GeoJSON writes `[lon, lat]` and every map UI writes
  `[lat, lon]`; anywhere both numbers are a plausible latitude the two cannot be told apart, so
  a record whose field is an array simply does not match. Write the object.
* **Geometry that is not on the earth is a 400.** A latitude of 452 is a swapped pair or a
  degrees/radians slip, and an empty page is a plausible answer to a real question — which
  makes it the worst possible answer to that one.

## Sending text instead of vectors

A table can name an embedding model, and then records and queries carry text:

```bash
export OPENAI_API_KEY=...            # read by the server; never sent in a request body

# No dimension: the server asks the model how wide its vectors are. 1536 for
# text-embedding-3-small, 1024 for bge-m3, 384 for all-MiniLM-L6-v2 -- the number
# whose cost, when you get it wrong, is a table that accepts no writes.
curl -X POST localhost:9999/api/v1/tables \
  -d '{"name": "docs", "embedding": {"model": "text-embedding-3-small"}}'

curl -X POST localhost:9999/api/v1/tables/docs/records \
  -d '{"records": [{"id": "1", "text": "rust borrows memory instead of collecting it"},
                   {"id": "2", "text": "go collects garbage instead of borrowing"}]}'

curl -X POST localhost:9999/api/v1/tables/docs/search \
  -d '{"text": "memory management", "top_k": 5}'
```

`base_url` points it anywhere that speaks the OpenAI `/embeddings` protocol, which is most
things — a local GPU box, for instance, with no key and no egress:

```json
{"model": "BAAI/bge-m3", "base_url": "http://10.0.0.7:8080/v1", "api_key_env": "TEI_KEY"}
```

Verified against text-embeddings-inference, Infinity, ollama (`/v1`), vLLM, LM Studio and
llama.cpp by protocol rather than by testing each: the request is
`POST {base_url}/embeddings` with `{"model": …, "input": [...]}` and a bearer token, and the
reply is read by `data[].index`, not by array position.

The details that decide whether this is usable:

| | |
| --- | --- |
| Where the key lives | `api_key_env` names an environment variable **of the server**. No key is in a create-table body, a snapshot, an export, or any response. |
| What is stored | The text is written into the payload (field `text` by default), so a hit carries the chunk it came from and you are not keeping a second copy of the corpus. |
| Provider down | The request fails — 503, nothing written. There is no zero-vector or hashed fallback: a wrong vector stores cleanly and ranks wrongly forever. |
| Wrong width | A model that answers at a width other than the table's is refused naming both, before anything is stored. |
| Mixing | A table with a model still accepts vectors you computed. A record that brings its own is not sent to the provider. |
| Changing it | `PATCH` can repoint `base_url`, `api_key_env`, `field`, `batch_size`. It cannot change `model` on a table that holds rows — those vectors came from the old one, and a query embedded by a different model is not a worse answer but an unrelated one. |

The MCP tools take text too, which is the protocol where it matters most: the caller is a
language model driving tools and has no way to turn a sentence into 1,536 floats, so
`insert_vector` and `vector_search` requiring a vector made them unusable by the only sort of
client that speaks MCP. `create_table` there takes `embedding` and infers the width.

```
create_table  {"name": "notes", "embedding": {"model": "text-embedding-3-small"}}
insert_vector {"table": "notes", "id": "1", "text": "..."}
vector_search {"table": "notes", "text": "...", "limit": 5}
```

Embedding client-side is still first-class and is what the LangChain and LlamaIndex packages
do; `langchain-dxmt/examples/embedding.py` picks the best model available without a key.

### gRPC

The data path also speaks gRPC, on its own port, when the server is built with the feature and
told to serve it:

```bash
cargo build --release --features api-grpc
./target/release/vectordb serve --grpc-port 9998        # HTTP stays on 9999
```

`proto/dxmt.proto` is the schema: search, insert, get, update, delete and the table lifecycle.
Hybrid search, full-text search, DXQL, snapshots and backups stay on REST, where an extra
millisecond of JSON does not matter and a second schema would only have to be kept in step.

**It does not make this database faster, and the measurement says so.** A 768-dimension search
request is 3,084 bytes as protobuf against 9,694 as JSON — 3.1x less on the wire — but
`src/bin/loadgen` and `src/bin/loadgen_grpc` put the two transports against the same server,
table and queries, and the throughput is the same:

| 100,000 x 768, same server, clients on their own cores | JSON | gRPC |
| --- | --- | --- |
| k=100, 16 connections | 1,369 QPS, p50 11.2 ms | 1,431 QPS, p50 10.6 ms |
| k=10, 48 connections | 1,735 QPS, p50 27.0 ms | 1,766 QPS, p50 26.7 ms |
| client CPU per query | 53–56 µs | 194–235 µs |

Between 2% and 5%, which is inside the run-to-run noise, and the gRPC client costs four times
the CPU per query. The reason is not subtle: the server spends milliseconds walking the graph
and microseconds parsing the request, so the format of the request cannot matter. A cheap JSON
client was already enough to saturate it.

What gRPC is still worth having for:

- **Bandwidth**, where it is not free. Three times less traffic is three times less traffic over
  a WAN, a metered link, or a shared NIC at high QPS. The measurements above are loopback.
- **A schema that cannot be misspelt.** A filter's operator is a string over REST, so `"Equalz"`
  is a runtime error; in `proto/dxmt.proto` it is a `oneof` and does not compile.
- **Being expected.** The comparable databases all offer it.

What it is *not* yet: streaming. Every method is unary, so the one thing gRPC does that REST
genuinely cannot — a server-streamed result set for a bulk export — is not implemented.

Both interfaces go through the same storage calls, including the one that decides what an
update leaves alone, so they cannot drift into meaning different things.

### MCP server

The database also speaks the Model Context Protocol over stdio, backed by the same durable
storage as the HTTP server:

```bash
./target/release/vectordb mcp --wal-dir ~/.dxmt/store
```

Point an MCP client at it — seven tools are exposed (`vector_search`, `insert_vector`,
`create_table`, `list_tables`, `get_table_info`, `delete_vectors`, `hybrid_search`):

```json
{
  "mcpServers": {
    "dxmt-vectordb": {
      "command": "/path/to/vectordb",
      "args": ["mcp", "--wal-dir", "/path/to/data/wal"]
    }
  }
}
```

## Architecture

One data model, reached through several protocols, over one storage implementation:

```
REST /api/v1/*  ·  gRPC  ·  GraphQL  ·  WebSocket  ·  MCP
                        ↓
             IntelligentIndexedStorage
               ├─ records: in-process HashMap      ← all data lives here
               ├─ store:   ~/.dxmt/store           ← durability; replayed on startup
               ├─ snapshot: ~/.dxmt/store/snapshot ← every 300s, on SIGTERM, after a rebuild
               └─ index:   HNSW / HNSW-PQ / BruteForce / DiskANN Enhanced
```

When object storage is configured, the write-ahead log and snapshots are also mirrored there. A
table nothing has queried for a while can be dropped from memory entirely — records and index —
leaving its snapshot segments as the recovery source. The next query rehydrates it
transparently. Without object storage, the durable source remains the local WAL and snapshot
directory. This behavior is tuned by `DXMT_EVICTION_INTERVAL_SECS`.

A table whose vectors exceed `DXMT_PARTITION_THRESHOLD_MB` (default 512) switches to a partitioned
IVF index
— on its next `index/rebuild`, or automatically at the next re-evaluation sweep: centroids stay in memory and each partition is an immutable segment
in object storage, fetched only when a query probes it. That bounds memory for a single table
larger than RAM, which table-level eviction cannot do.

This path is still experimental.

Index selection is re-checked on a timer (`DXMT_REINDEX_INTERVAL_SECS`, default 900) and the
index is rebuilt when a table's size no longer suits the algorithm it has:

| Dataset size | Algorithm         |
| ------------ | ----------------- |
| < 5K         | BruteForce        |
| 5K–100K      | HNSW              |
| 100K–1M      | HNSW-PQ           |
| ≥ 1M         | DiskANN Enhanced  |

One caveat: this table applies when `DXMT_MEMORY_OPTIMIZATION=false`; the default (`true`) takes
a separate path that yields HNSW below 100K and HNSW-PQ above it.

The same sweep also rebuilds a table whose graph has accumulated tombstones. An update cannot
move a node, so it deletes the old slot and appends a new one; a table of constant size under
steady upserts would otherwise grow one slot per write without bound. The rebuild is the
reclamation, and it waits until the graph carries about twice the slots it needs, so the cost
is one O(n) pass per as many writes as the table has rows.

`CREATE TABLE` still picks from a fixed estimate, because a new table has no rows to size
against — the periodic re-evaluation is what corrects it. `POST /api/v1/tables/{name}/index/rebuild`
forces it immediately. A rebuild builds the replacement before swapping it in, so queries keep
being served by the old index throughout.

Interfaces: REST (`src/api/handlers.rs`), gRPC, GraphQL, WebSocket and MCP — **several
protocols over one data model**, which is what every comparable database does; none of them
serves a second, differently shaped model beside its own. A `serve --simple` mode did exactly
that here, at `/collections/*`, and is gone: see the CHANGELOG. REST routes are served under
`/api/v1`; system probes are at the root (`/health`, `/live`, `/ready`, `/metrics`) and under
the SDK prefix (`/api/v1/health`, `/api/v1/health/ready`).

## Configuration

Configuration is read from the environment. Every variable this server reads is `DXMT_*`.
The `DB_*` names it used to use are still accepted and warn once at startup — `DB_` is not a
namespace, it is the word "database", and `DB_HOST` is both this server's bind address and
what nearly every compose file calls its Postgres. Copy `.env.example` and adjust:

```bash
DXMT_HOST=0.0.0.0
DXMT_PORT=9999
DXMT_MEMORY_OPTIMIZATION=true # index-selection path; see Architecture
DXMT_SNAPSHOT_INTERVAL_SECS=300 # periodic snapshot; 0 disables
DXMT_EVICTION_INTERVAL_SECS=600 # drop cold tables from memory; 0 disables
DXMT_REINDEX_INTERVAL_SECS=900  # re-check index choice against table size; 0 disables
DXMT_ROW_COMPACTION_INTERVAL_SECS=600 # release the disk behind dead row-file rows; 0 disables
DXMT_PARTITION_THRESHOLD_MB=512 # vector bytes per table above which it partitions
DXMT_MEMORY_CACHE_SIZE=512    # MB
DXMT_COMPRESSION=zstd         # zstd | lz4 | none
```

`DXMT_STORAGE_TYPE` is parsed but currently has no effect — the server constructs
`IntelligentIndexedStorage` unconditionally.

### Where the data goes

One variable moves everything, and each directory can still be moved on its own:

| variable | default | what lives there |
| --- | --- | --- |
| `DXMT_ROOT_DIR` | `~/.dxmt` | the root the rest hang off |
| `DXMT_WAL_DIR` | `$DXMT_ROOT_DIR/store` | WAL, snapshots and row files — **this is the database** |
| `DXMT_BACKUP_ROOT` | `$DXMT_ROOT_DIR/backups` | the only place `POST /backup` may write |
| `DXMT_CACHE_DIR` | `$DXMT_ROOT_DIR/cache` | regenerable; safe to delete while stopped |
| `DXMT_SSD_CACHE_DIR` | `$DXMT_CACHE_DIR/ssd` | the SSD tier |
| `DXMT_PROCESSES_DATA_DIR` | `$DXMT_ROOT_DIR/processes` | per-process scratch |

Setting `DXMT_WAL_DIR` alone moves the database and takes the backups with it, on the reasoning
that backups belong beside what they back up; naming `DXMT_ROOT_DIR` explicitly means it, and the
backups stay under the root. `DXMT_WAL_DIR=` — set but empty — disables persistence entirely: the
server runs from memory and loses everything on restart, which is occasionally what you want in
a test and never what you want otherwise.

At startup the server creates `store/` and `backups/` and **writes a probe file into each**,
refusing to serve if either fails. Creating a directory is not the check: `create_dir_all`
succeeds against a directory that already exists with `r-x` and every write after it fails. It
does not touch `cache/` — that is made by whatever fills it, so a server that never caches
leaves nothing behind.

With no home directory to find — `/nonexistent` under systemd's `DynamicUser=`, `/` in some
images — the root falls back to the working directory and the server says so on startup. Set
`DXMT_ROOT_DIR` there, or `cd` elsewhere and you have opened a different, empty database.

### Authentication

Off by default, like Qdrant, Chroma and Weaviate. Set `DXMT_API_KEY` and every route but the
health probes answers 401 without it:

```bash
DXMT_API_KEY=$(openssl rand -hex 32) ./target/release/vectordb serve
curl -H "api-key: $DXMT_API_KEY" localhost:9999/api/v1/tables
curl -H "Authorization: Bearer $DXMT_API_KEY" localhost:9999/api/v1/tables   # also accepted
```

`api-key` is Qdrant's header name, so a client already shaped for that needs no new code. The
key is compared in constant time and never appears in a log or an error body.

**`/health`, `/live` and `/ready` stay open.** A Kubernetes liveness probe cannot carry a
header, and a server that fails its own health check because authentication was switched on is
a worse outage than the one the key prevents. They report liveness and durability state, not
data. `/status` and `/health/metrics` are *not* exempt — those name tables.

**Know what this is.** One static key, no rotation, no accounts, no per-table scoping, and it
does nothing about anyone who can read the process environment. It is a seatbelt, not a vault:
it turns an accidentally-reachable port from "anyone who finds it owns the data" into "anyone
who finds it gets 401", which is the case that actually happens — a demo box that outlived its
firewall rule. Put TLS in front of it; sending a key over plain HTTP hands it to every hop on
the way, which is worse than sending none.

Object storage credentials come from the environment only — never commit them. They are read
into the configuration. Object-storage mirroring is opt-in and best effort:

```bash
export DXMT_OBJECT_STORAGE_PROVIDER=s3
export DXMT_S3_BUCKET=<your-bucket>
export DXMT_S3_ENDPOINT=<your-endpoint>
export DXMT_AWS_ACCESS_KEY_ID=<your-access-key>
export DXMT_AWS_SECRET_ACCESS_KEY=<your-secret-key>
```

### Feature Flags

Default: `tokio-runtime`, `basic-storage`, `jemalloc`, `api-swagger`, `api-dashboard`,
`server-embedding`.

Opt-in: `text-bm25`, `api-graphql`, `api-ws`, `api-grpc`, `s3-storage`, `benchmarks`,
`experimental-indexes`, `adaptive-index`, `test-failpoints`.

That is the whole list. Until 2026-09-27 it also advertised `gpu`, `cluster`, `multimodal`,
`embeddings`, `python`, `metrics` and `hf-datasets`: of those, two did not compile, three
gated no code at all, and `metrics` was never a feature of this crate. They are removed, and
the code behind them is in `planning/src/`.

`embeddings` is **not** what lets you send text — that is `server-embedding`, which is on by
default. `embeddings` compiles a directory of provider clients that no route reaches.

## Web console

A read-only dashboard is served from the binary itself — no separate process, no CDN, works on
an air-gapped host:

```bash
make dashboard-build                                     # needs Node 20+
cargo build --release --features api-dashboard
./target/release/vectordb serve --dashboard              # http://localhost:9999/dashboard
```

Off at both levels by default: not in the default feature set, and not served without
`--dashboard` even when compiled in. It adds about 1.5 MB to the binary.

**It is read-only, and the server is not.** The console issues only GETs and the search-shaped
POSTs; the port it is served on still accepts writes, snapshots and backups from anything that
can reach it. What the dashboard changes is discoverability — it turns an exposed port into a
browsable console — so bind it to a private interface, set `DXMT_API_KEY` (below), or put an
authenticating proxy in front. With a key set, the console asks for it once per tab and sends
it on every request; its static files stay reachable without one, because they carry no data.

## Client SDKs

| Directory                          | Package                                  |
| ---------------------------------- | ---------------------------------------- |
| `dxmt-python-sdk/`                 | Python client + framework integrations   |
| `dxmt-js-sdk/`                     | TypeScript/JavaScript client             |
| `langchain-dxmt/`                  | LangChain vector store                   |
| `llama-index-vector-stores-dxmt/`  | LlamaIndex vector store                  |

The Python SDK additionally ships integrations for LangGraph, Haystack, mem0, CrewAI, MetaGPT,
and AutoGen. See `dxmt-python-sdk/README.md`.

## Development

```bash
make build              # Debug build
make release            # Release build
make test               # Rust tests
cargo test --lib        # Unit tests only
make fmt && make lint   # rustfmt + clippy
make verify             # the full gate: fmt-check, clippy -D warnings, unit + all integration tests
make verify-core        # the same without the full integration suite, for iterating
```

The broader legacy scripts under `scripts/` are not part of the Alpha quality gate.

Note that a default `cargo test` does not run the feature-gated suites — the benchmark,
dataset, adaptive-index and LMDB tests compile to zero tests without their features. CI's
`optional-features` job compiles them; run them locally with, for example,
`cargo test --features benchmarks --test dataset_test`.

The examples in `examples/` that process PDFs do not ship with a corpus. Supply your own:

```bash
DEMO_PDF_PATH=/path/to/your.pdf python3 examples/pdf_rag_simple.py
```

## Measured performance

One machine — 32 vCPU, 121 GB, local NVMe — one binary per run, ingested over REST with
parallel writers. Cohere for the 1M and 10M rows, LAION for 100M.

| Corpus | Ingest | Resident | recall@100 | QPS | Serial p99 |
| --- | --- | --- | --- | --- | --- |
| 768d × 1M | 166 s (~6,000 rows/s) | 610 MB | 0.9798 | 6,465 | 3.6 ms |
| 768d × 10M | 1,396 s (~7,200 rows/s) | 4.68 GB | 0.9628 | 5,645 | 4.4 ms |
| 768d × 100M, fp16 vectors | 3.75 h (~7,400 rows/s) | 44.7 GB | 0.9735 | 4,886 | 5.7 ms |

Those are parallel writers on 32 vCPU. To see what your own machine does, over HTTP, with one
writer: `python3 scripts/http_ingest_rate.py --port 9999 --dim 768 --rows 200000`. It reports
the rate counting only the requests (a Python generator building 768 gaussians a row is itself
good for about 1,700 rows/s, and timing it with the server hides most of the answer), checks
recall against `search_mode: "exact"` on the table it just built, and waits for the PQ training
that lands minutes *after* the last batch. On a 16 vCPU box: 2,875 rows/s at 768 dimensions,
recall@10 1.000, search p50 1.6 ms, and the index falling from 848 MB to 132 MB when the
quantizer installs.

`SearchMode::Accurate` buys recall back without a rebuild: 0.9912 at 4,351 QPS on 1M, 0.9803 at
3,720 on 10M, 0.9825 at 3,546 on 100M.

Those recall figures are against datasets with published ground truth, which is not the dataset
you have. `search_mode: "exact"` scores every live row and skips the index entirely, so you can
run the same query twice and measure the difference on your own data:

```bash
curl -s localhost:9999/api/v1/tables/mine/search -d '{"vector": [...], "top_k": 10, "search_mode": "exact"}'
curl -s localhost:9999/api/v1/tables/mine/search -d '{"vector": [...], "top_k": 10}'
# recall@10 = |overlap| / 10
```

It costs a full pass over the table's vectors, so it is refused above `DXMT_EXACT_SCAN_MAX` rows
(default 1,000,000) rather than running for minutes; a filter narrows what it has to read.

Four things the table does not say on its own:

- **The client is inside the measurement.** Those QPS figures come from VectorDBBench's Python
  client, which costs about 70x more CPU per query than the server does. Driven instead by
  `src/bin/loadgen.rs` — one keep-alive connection per thread, request bodies serialised once at
  startup — the same 1M table on the same machine answers **8,822 QPS at 32 connections with a
  4.56 ms p99**, and a single connection sees a 3.07 ms p99. The Python figures are kept above
  because they are the ones comparable with published numbers.
- **`ef_search` now scales with table size.** A fixed traversal budget loses recall as the graph
  grows, so it doubles per decade of rows above a million. On the coverage run after that change:
  768d/1M 0.9806 at 6,310 QPS, 768d/10M 0.9793 at 3,682 QPS. Wider vectors on the same run:
  1536d/50K 0.9962, 1536d/500K 0.9897, 1536d/5M 0.9816; 1024d/1M 0.9488, 1024d/10M 0.9640 — the
  1024d figures are lower because that corpus needs more traversal than Cohere at the same
  scale, not because the codes lose more.
- **Reading and writing at once was measured too**, with periodic snapshots and the eviction
  sweep left on rather than turned off for the run: against a 2,000 rows/s ingest target the
  server sustained 1,987 rows/s while serving 1,161–1,819 QPS at concurrency 10, p95
  10.3–11.4 ms and p99 17.8–24.4 ms. Once ingest stops, p99 returns to 4.5 ms.
- **Memory is the reason 100M fits.** The graph walks on PQ codes with an exact rerank from the
  row file, so a row costs about 0.45 KB resident rather than the 8.3 KB it costs with vectors
  and records in the process.

These are single-machine numbers on two corpora. They say what this build did on that hardware,
not what it will do on yours, and nothing at all about other systems.

### Listing, filtering and restart

The table above is about vector search. These are the other things a table does, on the same
machine, measured across four orders of magnitude rather than at one size — because the costs
here are not all linear and a single figure would hide that.

Synthetic corpus, eight dimensions, three payload fields, one of them a distinct string per row.
The filter matches a quarter of the table, which is a deliberately awkward shape: a real filter
usually selects far less, and the cost of answering one follows what matches rather than how
many rows there are. Measured on 16 cores and 62 GB; an earlier version of this table came from
a 32-core machine, and its numbers are not comparable with these.

| Rows | Restart to serving | First filtered query | Filtered page after it | Unfiltered page |
| --- | --- | --- | --- | --- |
| 10,000 | 0.3 s | 24 ms | 0.9 ms | 1.2 ms |
| 100,000 | 0.3 s | 187 ms | 1.4 ms | 1.4 ms |
| 1,000,000 | 2.0 s | 2.3 s | 13 ms | 24 ms |
| 10,000,000 | 26 s | 23 s | 0.2 s | 209 ms |

A real corpus of 384-dimension sentence embeddings, filtering on a four-valued `topic`, after a
restart:

| Rows | First filtered query | Then | Page | Vector search | BM25 |
| --- | --- | --- | --- | --- | --- |
| 20,000 | 47 ms | 1.0 ms | 0.7 ms | 1.2 ms | 0.4 ms |
| 120,000 | 353 ms | 8.4 ms | 1.0 ms | 2.3 ms | 0.5 ms |

**The first filtered query is the one that is not like the others.** It rebuilds the payload
index by reading every record, because that index is derived from them and is not persisted.
At 10M rows on the machine above that is 23 s, down from 40 s, with the server's peak memory
7.1 GB rather than 8.7 GB: a value's records are held inline when there is one and in a roaring
bitmap when there are many, instead of a hash set per distinct value, and each record id is
stored once instead of twice. What is left is mostly decoding ten million records from disk and
keeping ten million distinct strings, which a field with a value per row genuinely has.

Every filtered page after it is chosen by the index, which keeps each record's `created_at`
beside it, so only the records on the page are looked up: 0.2 s at 10M for a filter matching
2.5M rows, where it was 2.9 s when each match was turned into an id and hashed back to a row.

At a hundred thousand rows none of this is visible. At ten million the first filtered query is
still most of half a minute, and a server that restarts often and filters immediately will feel
it.

## Known Limitations

These are real gaps, not roadmap aspirations. [docs/SUPPORT_MATRIX.md](docs/SUPPORT_MATRIX.md)
is the per-subsystem version of this list: what is part of the single-node contract, what is
experimental, and what a feature has to have before it moves between the two.

- **A backup may only be written inside `DXMT_BACKUP_ROOT`**, which defaults to the WAL
  directory's parent. `POST /backup` takes the destination from the request body, and this
  server has no authentication unless `DXMT_API_KEY` is set — so without a bound, anything that
  can reach the port can name any directory the process can write to and have a copy of the
  database put there. The root is
  checked after symlinks and `..` are resolved. Point it at the mount backups belong on.
- **A snapshot is the only copy of what it covers**, so back it up — see
  [docs/operations/backup-restore.md](docs/operations/backup-restore.md).
  Briefly: The WAL is compacted once a snapshot
  verifies, so the entries below that point exist nowhere else. The snapshot is read back before
  the log is trimmed, and an unreadable snapshot makes the server refuse to start rather than
  come up empty — but back up `data/wal/snapshot/` like the data it is.
- **One fsync per write batch.** Durability is paid for synchronously; there is no group
  commit yet.
- **A snapshot briefly delays writes**, on every table, while it compacts the log it covers.
  Measured on 2026-09-25 with four tables of 40,000 rows and writers running throughout: the
  snapshot took 0.32 s and the worst write waited 90 ms; at 80,000 rows a table, 1.00 s and
  313 ms. Searches are not affected. The cost is proportional to the log since the last
  snapshot, so `DXMT_SNAPSHOT_INTERVAL_SECS` trades this delay against how much the WAL holds.
- **A large batch delays searches on the table it is written to**, and only that table.
  Measured on 2026-09-25 with 5,000 rows of 768 dimensions going into a 12,000-row table while
  searches ran against it: the worst search waited 1.86 s, the median 5.9 ms; `GET /tables` and
  searches on other tables were unaffected throughout. Sending the same 5,000 rows as five
  batches of 1,000 takes the worst wait to 478 ms and costs 8% of the ingest rate, which is the
  trade to make if the table is serving reads while it loads. (A batch into a table too small
  for the parallel planner -- under 10,000 rows -- used to hold the lock for the *whole* batch,
  8.0 s of it; that path now takes the lock per 128 rows and the same measurement is 302 ms.)
- **`POST /api/v1/tables/{name}/records` caps a request at 10,000 records** and returns
  `Validation failed` above that.
- **`POST /api/v1/tables/{name}/records/list` omits vectors unless asked.** Pass
  `"with_vector": true` for them. The default is off because a page of 100 rows at 768
  dimensions is over a megabyte of JSON floats, and responses are not compressed.
- **The first filtered query on a restored table rebuilds the index it filters with.** The
  payload index is derived from the records and is not written anywhere, so a restart has to
  build it again — and it is built on demand rather than at startup, so a server that never
  filters never pays for it. That first query is proportional to the table: 187 ms at 100,000
  rows, 23 s at 10M. Every query after it is proportional to what matches. See the numbers
  above.
- **Object-storage mirroring is experimental.** Snapshots, WAL chunks and partition segments are
  uploaded when configured, and a fresh node can recover from them, but remote upload failures
  are logged without failing an already durable local write. Multi-process coordination and full
  failure-injection coverage are not yet part of the Alpha contract.
- **`POST /api/v1/tables/{name}/hybrid_search` is dense-only.** It rejects sparse/text fusion
  with `NOT_IMPLEMENTED` instead of silently discarding query components. For BM25 use
  `/search/text`; for real vector+text fusion use `/search/hybrid`.
- **`DiskANN`, `ScaNN` and `QuantizedIVF`** fall back to brute force in
  `create_index_instance`.
- **gRPC carries the data path only.** Search, insert and the record routes, plus the table
  lifecycle; hybrid search, full-text search, DXQL, snapshots and backups stay on REST. It is
  off by default and off without `--grpc-port`.
- **Nothing in `src/` is a stub you can reach by accident any more.** Multimodal (placeholder
  embeddings pending `ort` 2.0), the GPU/SIMD acceleration modules, the PyO3 bindings, the
  cluster and consensus scaffolding, and `embeddings/legacy.rs` — four overlapping provider
  abstractions nothing called — moved to `planning/src/` on 2026-09-27, 11,668 lines, with
  their cargo features. The embedding path that does run is `src/embeddings/service.rs`; see
  "Sending text instead of vectors".
- **The graph is built inside the write request.** There is no compaction or optimizer stage
  behind it: a table finishes ingest already holding its index, which is why a load is not
  followed by a minutes-long build step. The other side of that is contention — with ingest
  running at 2,000 rows/s against a million-row table, query p95 roughly doubles compared with
  the same table idle — and that when a rebuild is needed it rebuilds the whole table rather
  than merging a segment. Ingest throughput is also a parallel-writer number: one connection
  feeding one table is several times slower than eight.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

    Copyright 2026 上海点线面体信息科技有限公司 (Shanghai Dianxianmianti Information Technology Co., Ltd.)
