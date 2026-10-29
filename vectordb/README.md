# DXMT VectorDB

A single-node vector database written in Rust. HNSW, HNSW-PQ and product quantization with
runtime SIMD dispatch, payload filtering, full-text (BM25, with Chinese segmentation) and
hybrid retrieval, server-side embedding and a query language behind one REST API, with
Swagger and a read-only console served by default. GraphQL, WebSocket, gRPC and S3 are opt-in
build features — see [Feature flags](#feature-flags) — so a default build is the supported
single-node server and nothing more.

> **Status: 1.3**, under the compatibility promise 1.0 made. Durability is a write-ahead log,
> replayed on startup, plus periodic snapshots — `kill -9` and restart preserves your tables.
> Records and full vectors are **not** held in memory: with a WAL directory, which is the
> default, they live in row files beside it (`store/records/<table>.rec`,
> `store/vectors/<table>.f32`) and only the graph, the PQ codes and the id map stay resident —
> about 0.45 KB per 768-dimension row, measured at 100 million, against 8.3 KB when
> everything was in the map. That figure predates 1.2, which also keeps 4-bit fast-scan codes
> for filtered search beside the PQ codes (the same bytes again; `quantization.fast_scan:
> false` at creation leaves them out). `docs/architecture/disk-index.md` is how that was done and what
> it cost. (This paragraph said "records are held in memory" from 2026-08-14, when the WAL
> went on the write path, until 2026-09-27: true when written, and left behind by the two
> phases that moved the rows to disk.) The log defaults to
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
> be moved on its own; the root moves all of them. `store/` and `backups/` are created and
> checked for writability at startup, so an unwritable `backups/` is a refusal to start rather
> than a surprise for whoever takes the first backup; `cache/` is created only by whatever
> first writes to it. When configured, snapshots, WAL chunks and
> partition segments are mirrored to object storage, but that mirror is still experimental and
> local WAL remains the readiness boundary. There is no replication or clustering: this is
> one process on one machine. See [Known Limitations](#known-limitations).

Anything this README does not describe is experimental, even when its module compiles or has
standalone tests.

## Quick Start

Prebuilt Linux binaries, with the web console compiled in, are attached to each
[release](https://github.com/dxmt/dxmt/releases) from 1.4.0 on: `dxmt-vectordb-<version>-<x86_64|aarch64>-linux-gnu.tar.gz`,
built on glibc 2.28 (RHEL/Rocky 8, Debian 10, Ubuntu 20.04 or newer), each with a `.sha256` beside
it. Unpack and run `./vectordb serve`.

To build it yourself you need [rustup](https://rustup.rs). The Rust version is pinned in `rust-toolchain.toml` (1.99.0), and
rustup installs it the first time you run `cargo` in `vectordb/`. The web console is compiled
into the binary only if its bundle exists when you build: `make dashboard-build` (Node 20+)
first, or the server runs without `/dashboard` and says so at startup.

```bash
# Build
cargo build --release

# Run -- on this machine only; `--host 0.0.0.0` serves the network (set DXMT_API_KEY first)
./target/release/vectordb serve --port 9999
```

It prints where everything is -- the API, the console, the data directory -- and which
embedding model text goes through.

### Notes in, notes out: no configuration

With an embedding model the server can reach, text goes straight in and searches come back by
meaning. The quickest is a local [ollama](https://ollama.com) with a multilingual model, which
the server finds on its own:

```bash
ollama pull bge-m3                     # multilingual; an English-only model searches Chinese badly
./target/release/vectordb serve --host 127.0.0.1

# The `kb` table is made by its first write: semantic search plus Chinese-aware keywords.
curl -X POST localhost:9999/api/v1/tables/kb/records -H 'content-type: application/json' \
     -d '{"records":[{"id":"n1","text":"HNSW 是一种用于向量检索的图索引"},
                     {"id":"n2","text":"Python 的 GIL 限制了多线程并行"}]}'
curl -X POST localhost:9999/api/v1/tables/kb/search/hybrid -H 'content-type: application/json' \
     -d '{"text":"为什么 Python 跑不满多核","limit":3}'
```

A whole file is a document: it is split into overlapping chunks (500 characters, 50 shared)
on paragraph and sentence boundaries, each embedded and searchable on its own, and writing the
same id again replaces it. A hit's payload says which document and which chunk it is.

```bash
curl -X PUT localhost:9999/api/v1/tables/kb/documents/notes%2Frust.md \
     -H 'content-type: application/json' \
     -d "$(jq -n --rawfile text notes/rust.md '{text: $text, metadata: {path: "notes/rust.md"}}')"
curl localhost:9999/api/v1/tables/kb/documents/notes%2Frust.md          # its chunks, in order
curl -X DELETE localhost:9999/api/v1/tables/kb/documents/notes%2Frust.md
```

The MCP server (`vectordb mcp`) has the same as `add_document` and `delete_document`, and
`hybrid_search` takes text alone, so an assistant can file and find notes with no vectors.

Any OpenAI-compatible endpoint works instead: `DXMT_EMBEDDING_MODEL=text-embedding-3-small`
with `OPENAI_API_KEY` set, or `DXMT_EMBEDDING_BASE_URL` for another provider. A table created
with neither a `dimension` nor an `embedding` takes the server's model.

Verify it's up:

```bash
curl http://localhost:9999/health
# Readiness is the deployment probe; it returns 503 after a local WAL durability failure.
curl -f http://localhost:9999/ready
```

Create a table, put three vectors in it, and search — with a payload filter on the last one:

```bash
curl -X POST localhost:9999/api/v1/tables \
  -H 'Content-Type: application/json' \
  -d '{"name": "quickstart", "dimension": 4, "distance_metric": "cosine"}'

curl -X POST localhost:9999/api/v1/tables/quickstart/records \
  -H 'Content-Type: application/json' \
  -d '{"records": [
        {"id": "a", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"topic": "rust"}},
        {"id": "b", "vector": [0.4, 0.3, 0.2, 0.1], "payload": {"topic": "go"}},
        {"id": "c", "vector": [0.1, 0.2, 0.3, 0.5], "payload": {"topic": "rust"}}]}'

curl -X POST localhost:9999/api/v1/tables/quickstart/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 2}'
# {"success":true,"data":{"hits":[{"id":"a","score":1.0,...},{"id":"c","score":0.994,...}],...}}

curl -X POST localhost:9999/api/v1/tables/quickstart/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 2,
       "filter": {"must": [{"key": "topic", "operation": {"type": "Equal", "value": "go"}}]}}'
```

Every request body is JSON and needs the `Content-Type: application/json` header; without it
the server answers `400 INVALID_INPUT`. The client SDKs (see [Client SDKs](#client-sdks)) set it
for you.

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
  -H 'Content-Type: application/json' \
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
  -H 'Content-Type: application/json' \
  -d '{"name": "docs", "embedding": {"model": "text-embedding-3-small"}}'

curl -X POST localhost:9999/api/v1/tables/docs/records \
  -H 'Content-Type: application/json' \
  -d '{"records": [{"id": "1", "text": "rust borrows memory instead of collecting it"},
                   {"id": "2", "text": "go collects garbage instead of borrowing"}]}'

curl -X POST localhost:9999/api/v1/tables/docs/search \
  -H 'Content-Type: application/json' \
  -d '{"text": "memory management", "top_k": 5}'
```

`base_url` points it anywhere that speaks the OpenAI `/embeddings` protocol, which is most
things — a local GPU box, for instance, with no key and no egress:

```json
{"model": "BAAI/bge-m3", "base_url": "http://10.0.0.7:8080/v1", "api_key_env": "TEI_KEY"}
```

With `TEI_KEY` unset no key is sent and that works as written. If the box does need a key,
the server must be told both names first — `DXMT_ALLOWED_KEY_ENVS=TEI_KEY` and
`DXMT_ALLOWED_PROVIDER_HOSTS=10.0.0.7` — because a table's settings come from its callers
(see "Which key goes where" below).

Compatible by protocol with text-embeddings-inference, Infinity, ollama (`/v1`), vLLM, LM
Studio and llama.cpp; tested against the OpenAI-compatible fake provider in
`tests/fake_provider/`, which also answers out of order, rate-limits and returns the wrong
width on purpose. The request is
`POST {base_url}/embeddings` with `{"model": …, "input": [...]}` and a bearer token, and the
reply is read by `data[].index`, not by array position.

The details that decide whether this is usable:

| | |
| --- | --- |
| Where the key lives | `api_key_env` names an environment variable **of the server**. No key is in a create-table body, a snapshot, an export, or any response. |
| Which key goes where | A key is sent only if `api_key_env` is a known provider variable (`OPENAI_API_KEY`, `COHERE_API_KEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, `MISTRAL_API_KEY`, ...), the one named in `DXMT_EMBEDDING_API_KEY_ENV`, or listed in `DXMT_ALLOWED_KEY_ENVS`; and only to a known provider's API host, this machine, the host of `DXMT_EMBEDDING_BASE_URL`, or a host listed in `DXMT_ALLOWED_PROVIDER_HOSTS`. Link-local and cloud metadata addresses are refused outright. Otherwise any caller that can create a table could have the server send any of its environment variables to any URL. |
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
do; `clients/langchain-dxmt/examples/embedding.py` picks the best model available without a key.

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

Point an MCP client at it — nine tools are exposed (`vector_search`, `insert_vector`,
`create_table`, `list_tables`, `get_table_info`, `delete_vectors`, `add_document`,
`delete_document`, `hybrid_search`):

```json
{
  "mcpServers": {
    "dxmt-vectordb": {
      "command": "/path/to/vectordb",
      "args": ["mcp", "--wal-dir", "/path/to/dxmt/store"]
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
               ├─ records: row files in ~/.dxmt/store/{records,vectors}
               ├─ store:   ~/.dxmt/store           ← durability; replayed on startup
               ├─ snapshot: ~/.dxmt/store/snapshot ← every 300s, on SIGTERM, after a rebuild
               └─ index:   HNSW-PQ (default) / HNSW / BruteForce, resident
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

A new table is HNSW-PQ at every size: an exact HNSW graph until it holds 65,536 vectors,
after which it quantizes itself in place and searches the codes with an exact rerank — no
rebuild between the two. `DXMT_MEMORY_OPTIMIZATION=false` restores an older selector that
picks brute force below 5,000 rows and HNSW above, moving to HNSW-PQ, or to EnhancedDiskANN
at a million rows and up, only when HNSW would not fit the memory budget. Either way the
choice is re-checked on a timer (`DXMT_REINDEX_INTERVAL_SECS`, default 900).

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

Configuration is read from the environment. Every variable this server reads is `DXMT_*`,
except the provider key a table's embedding model names (`api_key_env`, e.g. `OPENAI_API_KEY`).
The `DB_*` names it used to use are still accepted and warn once at startup — `DB_` is not a
namespace, it is the word "database". `DB_HOST` and `DB_PORT` are the exception: no release
read them, and they are what nearly every compose file calls its Postgres, so they are ignored.
`serve` reads a `.env` in the directory it starts in; a variable set in the environment wins
over it, and a flag (`--host`, `--port`, `--grpc-port`) wins over both. Copy `.env.example`
and adjust:

```bash
DXMT_HOST=127.0.0.1             # the default; 0.0.0.0 to serve other machines
DXMT_PORT=9999
DXMT_MEMORY_OPTIMIZATION=true # index-selection path; see Architecture
DXMT_SNAPSHOT_INTERVAL_SECS=300 # periodic snapshot; 0 disables
DXMT_EMBEDDING_MODEL=           # the default model for text; unset, a local ollama's is used if found
DXMT_EMBEDDING_BASE_URL=        # its OpenAI-compatible endpoint (default https://api.openai.com/v1)
DXMT_EMBEDDING_API_KEY_ENV=     # the variable holding its key (default OPENAI_API_KEY)
DXMT_EMBEDDING_AUTODETECT=1     # 0: do not look for a local ollama
DXMT_ALLOWED_KEY_ENVS=          # comma-separated: more variables a table's api_key_env may name
DXMT_ALLOWED_PROVIDER_HOSTS=    # comma-separated: more hosts a provider key may be sent to
DXMT_KB_TABLE=kb                # the notes table made on first write; `off` disables it
DXMT_BACKUP_INTERVAL_SECS=0     # scheduled backups into DXMT_BACKUP_ROOT/auto-<time>; 0 (default) disables
DXMT_BACKUP_KEEP=7              # scheduled backups kept; older ones deleted, hand-taken ones never
DXMT_EVICTION_INTERVAL_SECS=600 # drop cold tables from memory; 0 disables
DXMT_REINDEX_INTERVAL_SECS=900  # re-check index choice against table size; 0 disables
DXMT_ROW_COMPACTION_INTERVAL_SECS=600 # release the disk behind dead row-file rows; 0 disables
DXMT_PARTITION_THRESHOLD_MB=512 # vector bytes per table above which it partitions
DXMT_MEMORY_CACHE_SIZE=512    # MB
```

Less often needed, and read the same way:

| variable | what it does |
| --- | --- |
| `DXMT_GRPC_PORT` | serve gRPC on this port too (same as `--grpc-port`; needs the `api-grpc` build) |
| `DXMT_WORKER_THREADS` | request worker threads (default: one per CPU core) |
| `DXMT_RECORD_STORE`, `DXMT_VECTOR_STORE` | `disk` (the default when there is a WAL directory) or `memory` |
| `DXMT_INDEX_ALGORITHM` | `hnsw-pq` (default) or `hnsw` for new tables |
| `DXMT_HNSW_EF_CONSTRUCTION` | graph build effort |
| `DXMT_RECALL_TARGET` | the recall a table calibrates its search budget to (default 0.95) |
| `DXMT_VECTOR_F16` | `1` keeps new tables' full vectors on disk as fp16 rather than f32 |
| `DXMT_MIN_FREE_MEMORY_MB` | refuse writes when free memory falls below this |

Through 1.3.2, `DXMT_HOST` and `DXMT_PORT` were listed here and read by nothing: `serve` bound
`--host` and `--port`, whose defaults are the same values. `DXMT_WORKER_THREADS` was parsed
and sized nothing. All three work now. `DXMT_COMPRESSION` is gone: it was parsed and
selected nothing, and the WAL and row files are not compressed. Logging is `RUST_LOG`
(`RUST_LOG=info`); there is no `DXMT_LOG_LEVEL`.

`DXMT_STORAGE_TYPE` is gone as of 2026-09-27. It was parsed into config and selected nothing —
the server constructs `IntelligentIndexedStorage` unconditionally — and a documented variable
that reads as a backend switch is worse than no variable. The 832-line `StorageFactory` it
would have driven had no caller either, and was removed with it.

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

### Object storage and read replicas

With `DXMT_OBJECT_STORAGE_PROVIDER` set (`s3` needs the `s3-storage` build; `filesystem`
writes under `DXMT_SSD_CACHE_DIR`, e.g. a network mount), every acknowledged write is also
uploaded as a WAL chunk and every snapshot is uploaded, so a machine with an empty data
directory recovers from the bucket alone. An upload that fails is retried, in order, by the
next write or the next start; `dxmt_wal_mirrored_next_sequence` on `/metrics` shows how far
the bucket trails `dxmt_wal_next_sequence`.

Google Cloud Storage works through its S3-compatible API: `DXMT_S3_ENDPOINT=https://storage.googleapis.com`,
`DXMT_S3_REGION=auto`, and an HMAC key pair in `DXMT_AWS_ACCESS_KEY_ID` / `DXMT_AWS_SECRET_ACCESS_KEY`.

A second server can follow that bucket as a **read replica** (experimental):

```bash
DXMT_OBJECT_STORAGE_PROVIDER=s3 DXMT_S3_BUCKET=my-bucket \
  ./target/release/vectordb serve --replica --port 9998
```

It recovers from the bucket, applies new WAL chunks every `DXMT_REPLICA_POLL_MS` (default
1,000), answers every read, and refuses every write with `403 READ_ONLY_REPLICA`. It never
writes the bucket. `dxmt_replica_applied_next_sequence` is how far it has applied; if it falls
behind by more than `DXMT_WAL_RETAIN_SECS` (default 600: chunks stay in the bucket that long
after a snapshot covers them), it reports not ready
(`dxmt_replica_stale 1`) and exits with status 3, so a process supervisor restarts it from the
newest snapshot (`DXMT_REPLICA_EXIT_ON_STALE=0` keeps it running, not ready). Aliases live in
a local file and are not followed.

A replica's data directory is a disposable cache, marked with a `REPLICA` file and emptied at
every start; it refuses a directory that holds a primary's data rather than empty it.

**Failover.** Stop the primary, then start the replica's server *without* `--replica` on the
same directory and bucket: it empties the cache, recovers from the bucket like any new
machine, and becomes the primary.

An old primary still running when another one starts is stopped, and how exactly depends on
the store. Where the store honours conditional writes -- AWS S3, MinIO, the `filesystem`
provider -- the WAL manifest is a lock: a new primary rewrites it with its epoch, and every
change the old one makes to the bucket is a write conditional on the version it last wrote,
so its next write is refused (503, not ready) and nothing it acknowledges is missing from the
new one. **GCS through its S3 API ignores conditions** (measured), and so does any store the
startup probe catches doing so; there each primary checks `wal/PRIMARY` every
`DXMT_PRIMARY_CHECK_MS` (default 1,000) instead, and writes the old one acknowledges inside
that interval are not on the new one. On those, stop the old primary first.

The old primary's directory is then out of date, and it refuses to start on it (it remembers
its claims in `PRIMARY`), so a supervisor restarting it cannot take the bucket back. To bring
that machine back as the primary, stop the current one and start it on an empty directory: it
recovers from the bucket.

### A cluster: many tables over many machines (experimental)

When the tables -- one per user, one per agent -- add up to more than one machine, run several
servers over one bucket and a router in front of them:

```bash
export DXMT_OBJECT_STORAGE_PROVIDER=s3 DXMT_S3_BUCKET=my-bucket   # + endpoint and keys

vectordb serve --shard a --advertise http://10.0.0.1:9999          # shard a's primary
vectordb serve --shard a --advertise http://10.0.0.2:9999 --replica
vectordb serve --shard b --advertise http://10.0.0.3:9999          # shard b's primary
vectordb router --host 0.0.0.0 --port 9990                         # clients use this one
```

Each shard is an ordinary store -- its WAL, snapshots and lock under `shards/<id>/` in the
bucket, a primary and any read replicas. Every table belongs to one shard, recorded in
`cluster/PLACEMENT`; every node writes who it is to `cluster/nodes/` every
`DXMT_HEARTBEAT_SECS` (2). The router serves the same `/api/v1` as a single server and forwards
each request to its table's shard: creating a table puts it on the shard holding the fewest,
listing asks every shard, and a read sent with `X-DXMT-Read: replica` goes to a replica (which
trails the primary; a 404 there is retried on the primary). Responses say where they were
served (`X-DXMT-Shard`, `X-DXMT-Node`); `GET /cluster/status` shows the whole cluster. Routers
keep nothing of their own, so run as many as you like.

There is no consensus service: the cluster's state is in the bucket and changed by
conditional writes, so a cluster needs a store that honours them -- AWS S3, MinIO, or the
`filesystem` provider; GCS's S3 API does not, and is refused.

**Failover is automatic.** A shard's primary holds a lease (`cluster/shards/<id>/LEASE`) and
renews it every heartbeat; it takes writes only until the term it last secured runs out. When
the primary stops renewing -- dead, or cut off from the bucket -- a replica waits for the term
plus `DXMT_LEASE_GRACE_SECS` (3), takes the lease by conditional write (one replica wins),
applies what is left in the bucket, and becomes the primary in place; the routers follow. With
the defaults that is about 15 seconds; `DXMT_NODE_TTL_SECS` shortens it. A node in a cluster
acknowledges a write only once the bucket has it, so nothing acknowledged is lost in a
failover. `DXMT_AUTO_FAILOVER=0` leaves failover to you: restart a replica's directory without
`--replica`.

**Tables move between shards.** `POST /cluster/tables/<t>/move {"to": "b"}` freezes the table
on its shard (writes answer 503 `TABLE_MOVING` with `Retry-After`; reads go on), carries it
through the bucket with the same export and import a single server has -- ids, versions,
`created_at`, payloads and vectors -- checks the copy's record count, points the placement at
it and drops the original; anything that fails before the switch is undone. Routers that have
not reread the placement yet are answered 503 or 404 by the old shard, reread, and retry on the
new one. `POST /cluster/rebalance` (`{"dry_run": true}` for the plan alone) moves tables, one at
a time, from the shard with the most to the one with the fewest until they differ by at most
one -- which is how a new shard gets its share.

Not yet: one table spread over several shards, and routes that span tables (aliases, DXQL,
backups), which the router answers with 501. Tables are created and dropped through the router; a node refuses them
directly (`DXMT_CLUSTER_SECRET`, when set, is the value the router proves itself with).

### Feature Flags

Default: `tokio-runtime`, `basic-storage`, `jemalloc`, `api-swagger`, `api-dashboard`,
`server-embedding`, `text-bm25`.

Opt-in: `api-graphql`, `api-ws`, `api-grpc`, `s3-storage`, `benchmarks`,
`experimental-indexes`, `adaptive-index`, `test-failpoints`.

`text-bm25` became a default on 2026-10-02. A build without it (`--no-default-features` plus
the rest) still compiles and still answers `/search/text`, and `/search/hybrid` with `text`,
with `501 FEATURE_DISABLED` rather than a dense-only page. A table's text field may declare
`"analyzer": "chinese"`, which indexes and queries that table with jieba word segmentation
(lowercased, mixed-script text included); the default analyzer sees a Chinese sentence as one
word and cannot find anything shorter than it.

That is the whole list. Until 2026-09-27 it also advertised `gpu`, `cluster`, `multimodal`,
`embeddings`, `python`, `metrics` and `hf-datasets`: of those, two did not compile, three
gated no code at all, and `metrics` was never a feature of this crate. They are removed, and
so is the code behind them.

`embeddings` is **not** what lets you send text — that is `server-embedding`, which is on by
default. `embeddings` compiles a directory of provider clients that no route reaches.

## Web console

A read-only dashboard is served from the binary itself — no separate process, no CDN, works on
an air-gapped host:

```bash
make dashboard-build                                     # needs Node 20+
cargo build --release
./target/release/vectordb serve                          # http://localhost:9999/dashboard
```

On by default: `api-dashboard` is a default feature, and the console is served whenever its
bundle was built into the binary. A checkout built without Node still compiles and serves
everything else, and says the console is missing. `--no-dashboard` turns it off; passing
`--dashboard` explicitly makes a missing bundle a refusal to start. It adds about 1.5 MB to the
binary.

**It is read-only, and the server is not.** The console issues only GETs and the search-shaped
POSTs; the port it is served on still accepts writes, snapshots and backups from anything that
can reach it. What the dashboard changes is discoverability — it turns an exposed port into a
browsable console — so bind it to a private interface, set `DXMT_API_KEY` (below), or put an
authenticating proxy in front. With a key set, the console asks for it once per tab and sends
it on every request; its static files stay reachable without one, because they carry no data.

## Client SDKs

All six live under `clients/`.

| Directory                                   | Package                                  |
| -------------------------------------------- | ---------------------------------------- |
| `clients/dxmt-python-sdk/`                   | Python client + framework integrations   |
| `clients/dxmt-js-sdk/`                        | TypeScript/JavaScript client             |
| `clients/dxmt-rust-sdk/`                      | Rust client — core routes only           |
| `clients/dxmt-go-sdk/`                        | Go client — the same core surface        |
| `clients/langchain-dxmt/`                     | LangChain vector store                   |
| `clients/llama-index-vector-stores-dxmt/`     | LlamaIndex vector store                  |

The Python SDK additionally ships integrations for LangGraph, Haystack, mem0, CrewAI and
MetaGPT. See `clients/dxmt-python-sdk/README.md`.

None of the packages is published to PyPI, npm or crates.io yet. Until the packages are
published, install from source (paths relative to `vectordb/`):

```bash
pip install ./clients/dxmt-python-sdk
pip install ./clients/langchain-dxmt                     # after the SDK above
pip install ./clients/llama-index-vector-stores-dxmt     # after the SDK above
cd clients/dxmt-js-sdk && npm install && npm run build   # then `npm install <path>` from your project
```

```toml
# Cargo.toml
dxmt-client = { git = "https://github.com/dxmt/dxmt", package = "dxmt-client" }
```

For Go, `go get github.com/dxmt/dxmt/vectordb/clients/dxmt-go-sdk@main` (the module lives in
this repository's subdirectory and needs no registry).

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

The broader legacy scripts under `scripts/` are not part of the quality gate.

Note that a default `cargo test` does not run the feature-gated suites — the benchmark,
dataset and adaptive-index tests compile to zero tests without their features. CI's
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
curl -s localhost:9999/api/v1/tables/mine/search -H 'Content-Type: application/json' -d '{"vector": [...], "top_k": 10, "search_mode": "exact"}'
curl -s localhost:9999/api/v1/tables/mine/search -H 'Content-Type: application/json' -d '{"vector": [...], "top_k": 10}'
# recall@10 = |overlap| / 10
```

It costs a full pass over the table's vectors: **300 ms for 1M × 960 with the row file in page
cache** (measured 2026-10-02, release build, 16 cores), on a thread pool of its own so other
queries do not wait; a cold read of the file is the real cost, about a second per gigabyte.
It is refused above `DXMT_EXACT_SCAN_MAX` rows (default 10,000,000) rather than running for
minutes on a table that large; a filter narrows what it has to read.

**The server runs that comparison itself and tunes the budget from it.** How much recall a
given `ef_search` buys is a property of the corpus -- glove-100 needs four times the budget
of a 768-dimension embedding corpus for the same recall@10, and hnswlib on the same rows
shows the same curve -- so a fixed default is either too little on a hard corpus or wasted on
an easy one. Every table has a `recall_target` (default 0.95; `PATCH /tables/{t}` with
`{"recall_target": 0.98}` changes it, `0` opts out). From 20,000 rows, and again whenever the
table doubles, the server makes queries from pairs of its own rows, scores them exactly, and
raises the `normal` budget by doublings until recall@10 meets the target; `accurate` is four
times whatever that is. `POST /tables/{t}/recall/calibrate` does it on demand, and
`GET /tables/{t}/stats` shows the budget in use, the recall measured, and the curve it came
from.

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

**The first filtered query is the one that is not like the others** — or was, until 1.2.0.
The payload index is derived from the records and is not persisted, so a restart has to read
every record to build it again. The "first filtered query" columns above are that build, paid
by whichever caller filtered first. Since 1.2.0 the server builds it in the background as
soon as recovery is done, decoding rows on every core under short read locks (26.8 s for the
payload index and 3.1 s for the node map on a 10M-row table), so a filtered query waits for
it only if it arrives before the warm-up finishes, or if a write reached the table first —
then it pays the same build itself.
At 10M rows on the machine above the build is 23 s, down from 40 s, with the server's peak memory
7.1 GB rather than 8.7 GB: a value's records are held inline when there is one and in a roaring
bitmap when there are many, instead of a hash set per distinct value, and each record id is
stored once instead of twice. What is left is mostly decoding ten million records from disk and
keeping ten million distinct strings, which a field with a value per row genuinely has.

Every filtered page after it is chosen by the index, which keeps each record's `created_at`
beside it, so only the records on the page are looked up: 0.2 s at 10M for a filter matching
2.5M rows, where it was 2.9 s when each match was turned into an id and hashed back to a row.

At a hundred thousand rows none of this is visible. At ten million the build is still most of
half a minute; it now runs in the background, but a server that restarts often and filters the
moment it is up will still catch it before it is done.

## Known Limitations

These are real gaps, not roadmap aspirations. [docs/SUPPORT_MATRIX.md](docs/SUPPORT_MATRIX.md)
is the per-subsystem version of this list: what is part of the single-node contract, what is
experimental, and what a feature has to have before it moves between the two.

- **A backup may only be written inside `DXMT_BACKUP_ROOT`**, which defaults to
  `$DXMT_ROOT_DIR/backups` (beside the store, when only `DXMT_WAL_DIR` was moved). `POST /backup` takes the destination from the request body, and this
  server has no authentication unless `DXMT_API_KEY` is set — so without a bound, anything that
  can reach the port can name any directory the process can write to and have a copy of the
  database put there. The root is
  checked after symlinks and `..` are resolved. Point it at the mount backups belong on.
- **A snapshot is the only copy of what it covers**, so back it up — see
  [docs/operations/backup-restore.md](docs/operations/backup-restore.md).
  Briefly: The WAL is compacted once a snapshot
  verifies, so the entries below that point exist nowhere else. The snapshot is read back before
  the log is trimmed, and an unreadable snapshot makes the server refuse to start rather than
  come up empty — but back up `snapshot/` under the WAL directory (`~/.dxmt/store/snapshot/`
  by default) like the data it is.
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
- **The payload and sparse indexes are rebuilt after a restart that did not shut down
  cleanly.** A clean shutdown (SIGTERM, Ctrl-C) writes them beside the store and the next
  start reads them back if the records are exactly as it left them -- 0.3 s against a 2.5 s
  rebuild at 1M rows -- and deletes the file either way. Otherwise the server builds them in the background
  once recovery is done (26.8 s at 10M rows, on every core), so the cost is CPU after startup
  rather than a slow first query — unless a filtered or sparse query arrives before the
  build finishes, in which case that query waits for it: up to 23 s at 10M. Writes during the
  build are caught up into it, and the build never holds the table lock, so other tables and
  unfiltered queries are not held up. Every query after it is proportional to what matches.
  See the numbers above.
- **Object-storage mirroring is experimental.** Snapshots, WAL chunks and partition segments are
  uploaded when configured, and a fresh node can recover from them, but remote upload failures
  are logged without failing an already durable local write. Multi-process coordination and full
  failure-injection coverage are outside the 1.0 compatibility promise.
- **`POST /api/v1/tables/{name}/hybrid_search` is dense-only.** It rejects sparse/text fusion
  with `NOT_IMPLEMENTED` instead of silently discarding query components. For BM25 use
  `/search/text`; for real vector+text fusion use `/search/hybrid`.
- **`DiskANN`, `ScaNN` and `QuantizedIVF` are not implemented.** A table that asks for one is
  refused with an error naming the alternative, rather than silently given brute force.
- **gRPC carries the data path only.** Search, insert and the record routes, plus the table
  lifecycle; hybrid search, full-text search, DXQL, snapshots and backups stay on REST. It is
  off by default and off without `--grpc-port`.
- **Nothing in `src/` is a stub you can reach by accident any more.** Multimodal (placeholder
  embeddings pending `ort` 2.0), the GPU/SIMD acceleration modules, the PyO3 bindings, the
  cluster and consensus scaffolding, and `embeddings/legacy.rs` — four overlapping provider
  abstractions nothing called — were taken out of the crate on 2026-09-27, 11,668 lines, with
  their cargo features. The embedding path that does run is `src/embeddings/service.rs`; see
  "Sending text instead of vectors".
- **A 1.1 binary cannot read a log written by 1.2.** 1.2 logs a reference to the row a
  vector is in (`UpsertRecordInRowFile`) instead of the vector. Take a snapshot
  (`POST /snapshot`) before downgrading; 1.2 reads every earlier log and index file.
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
