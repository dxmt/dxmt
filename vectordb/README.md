# DXMT VectorDB

A vector database written in Rust. HNSW, HNSW-PQ and product quantization with runtime SIMD
dispatch, hybrid retrieval and a query language behind a REST API, plus a zero-config
collection mode. Full-text search (Tantivy), GraphQL, WebSocket, Swagger and S3 are opt-in
build features — see [Feature flags](#feature-flags) — so a default build is the minimal
supported single-node server and nothing more.

> **Status: Alpha development, not production-ready.** Records are held in memory and made durable by a write-ahead log,
> replayed on startup — `kill -9` and restart preserves your tables. The log defaults to
> `./data/wal`; set `DB_WAL_DIR` to move it. When configured, snapshots, WAL chunks and
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

# Zero-config mode — collection API, auto-tuned
./target/release/vectordb serve --simple --memory-limit 1024

# Standard mode
./target/release/vectordb serve --host 0.0.0.0 --port 9999
```

Verify it's up:

```bash
curl http://localhost:9999/health
# Readiness is the deployment probe; it returns 503 after a local WAL durability failure.
curl -f http://localhost:9999/ready
```

Swagger, GraphQL and WebSocket are optional integrations and are not compiled into the default
single-node build. Build them explicitly when needed, then opt in per process:

```bash
cargo build --release --features api-swagger,api-graphql,api-ws
./target/release/vectordb serve --swagger --graphql --websocket
```

| Flag          | Endpoint                                            |
| ------------- | --------------------------------------------------- |
| `--swagger`   | `GET /api/docs`, spec at `/api/docs/openapi.json`    |
| `--graphql`   | `POST /graphql`                                      |
| `--websocket` | `GET /ws` (upgrade)                                  |
| `--grpc-port` | gRPC on that port, alongside HTTP (`proto/dxmt.proto`) |

`GET /api/v1/tables/{name}/stats` reports `resident`: whether the table was in memory when
asked. A query against an evicted table rehydrates it transparently, so without this a caller
has no way to know the next one will pay for a fetch from object storage.

`GET /health/metrics` (also `/api/v1/health/metrics`) includes the local durability frontier:
`storage.wal_next_sequence`, `storage.last_wal_sync_at`, `storage.last_snapshot_at`, and
`storage.snapshot_failures`/`last_snapshot_error`, plus resident and total table counts. These
timestamps and counters reset on restart; the WAL and snapshot manifest remain authoritative.

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

What it buys is the vector. A 768-dimension search request is 3,090 bytes as protobuf and
15,571 as JSON — **5.0x** — and the protobuf side needs no parsing beyond a copy, where every
float in the JSON is decimal text the sender formats and the receiver parses. Measured with
`cargo run --features api-grpc --example grpc_client`, which prints it.

Both interfaces go through the same storage calls, including the one that decides what an
update leaves alone, so they cannot drift into meaning different things.

### MCP server

The database also speaks the Model Context Protocol over stdio, backed by the same durable
storage as the HTTP server:

```bash
./target/release/vectordb mcp --wal-dir ./data/wal
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

Both server modes converge on one storage implementation, `IntelligentIndexedStorage`:

```
serve --simple  →  /collections/*   (zero-config collection API, at the root)
serve           →  /api/v1/*
                        ↓
             IntelligentIndexedStorage
               ├─ records: in-process HashMap      ← all data lives here
               ├─ wal:     ./data/wal              ← durability; replayed on startup
               ├─ snapshot: ./data/wal/snapshot    ← every 300s, on SIGTERM, after a rebuild
               └─ index:   HNSW / HNSW-PQ / BruteForce / DiskANN Enhanced
```

When object storage is configured, the write-ahead log and snapshots are also mirrored there. A
table nothing has queried for a while can be dropped from memory entirely — records and index —
leaving its snapshot segments as the recovery source. The next query rehydrates it
transparently. Without object storage, the durable source remains the local WAL and snapshot
directory. This behavior is tuned by `DB_EVICTION_INTERVAL_SECS`.

A table whose vectors exceed `DB_PARTITION_THRESHOLD_MB` (default 512) switches to a partitioned
IVF index
— on its next `index/rebuild`, or automatically at the next re-evaluation sweep: centroids stay in memory and each partition is an immutable segment
in object storage, fetched only when a query probes it. That bounds memory for a single table
larger than RAM, which table-level eviction cannot do.

This path is still experimental.

Index selection is re-checked on a timer (`DB_REINDEX_INTERVAL_SECS`, default 900) and the
index is rebuilt when a table's size no longer suits the algorithm it has:

| Dataset size | Algorithm         |
| ------------ | ----------------- |
| < 5K         | BruteForce        |
| 5K–100K      | HNSW              |
| 100K–1M      | HNSW-PQ           |
| ≥ 1M         | DiskANN Enhanced  |

One caveat: this table applies when `DB_MEMORY_OPTIMIZATION=false`; the default (`true`) takes
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

Interfaces: REST (`src/api/handlers.rs`), a zero-config simple mode
(`src/api/simple_core.rs`), GraphQL, and WebSocket. Standard-mode REST routes are served
under `/api/v1`; system probes are available at the root (`/health`, `/live`, `/ready`) and
under the SDK prefix (`/api/v1/health`, `/api/v1/health/ready`).

## Configuration

Configuration is read from the environment. Copy `.env.example` and adjust:

```bash
DB_HOST=0.0.0.0
DB_PORT=9999
DB_MEMORY_OPTIMIZATION=true # index-selection path; see Architecture
DB_WAL_DIR=./data/wal       # write-ahead log + snapshots; empty string disables persistence
DB_SNAPSHOT_INTERVAL_SECS=300 # periodic snapshot; 0 disables
DB_EVICTION_INTERVAL_SECS=600 # drop cold tables from memory; 0 disables
DB_REINDEX_INTERVAL_SECS=900  # re-check index choice against table size; 0 disables
DB_ROW_COMPACTION_INTERVAL_SECS=600 # release the disk behind dead row-file rows; 0 disables
DB_PARTITION_THRESHOLD_MB=512 # vector bytes per table above which it partitions
DB_MEMORY_CACHE_SIZE=512    # MB
DB_COMPRESSION=zstd         # zstd | lz4 | none
```

`DB_STORAGE_TYPE` is parsed but currently has no effect — the server constructs
`IntelligentIndexedStorage` unconditionally.

Object storage credentials come from the environment only — never commit them. They are read
into the configuration. Object-storage mirroring is opt-in and best effort:

```bash
export DB_OBJECT_STORAGE_PROVIDER=s3
export DB_S3_BUCKET=<your-bucket>
export DB_S3_ENDPOINT=<your-endpoint>
export DB_AWS_ACCESS_KEY_ID=<your-access-key>
export DB_AWS_SECRET_ACCESS_KEY=<your-secret-key>
```

### Feature Flags

Default: `tokio-runtime`, `basic-storage`.

Opt-in: `text-bm25`, `api-swagger`, `api-graphql`, `api-ws`, `api-grpc`, `s3-storage`, `gpu` (CUDA),
`cluster`, `multimodal`, `embeddings`, `python` (PyO3 bindings), `metrics`, `hf-datasets`.

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

`SearchMode::Accurate` buys recall back without a rebuild: 0.9912 at 4,351 QPS on 1M, 0.9803 at
3,720 on 10M, 0.9825 at 3,546 on 100M.

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

## Known Limitations

These are real gaps, not roadmap aspirations:

- **A snapshot is the only copy of what it covers**, so back it up — see
  [docs/operations/backup-restore.md](docs/operations/backup-restore.md).
  Briefly: The WAL is compacted once a snapshot
  verifies, so the entries below that point exist nowhere else. The snapshot is read back before
  the log is trimmed, and an unreadable snapshot makes the server refuse to start rather than
  come up empty — but back up `data/wal/snapshot/` like the data it is.
- **One fsync per write batch.** Durability is paid for synchronously; there is no group
  commit yet.
- **`POST /api/v1/tables/{name}/records` caps a request at 10,000 records** and returns
  `Validation failed` above that.
- **Object-storage mirroring is experimental.** Snapshots, WAL chunks and partition segments are
  uploaded when configured, and a fresh node can recover from them, but remote upload failures
  are logged without failing an already durable local write. Multi-process coordination and full
  failure-injection coverage are not yet part of the Alpha contract.
- **`POST /api/v1/tables/{name}/hybrid_search` is dense-only.** It rejects sparse/text fusion
  with `NOT_IMPLEMENTED` instead of silently discarding query components. For BM25 use
  `/search/text`; for real vector+text fusion use `/search/hybrid`.
- **Two implementations behind one name.** `search/quantization/product_quantizer.rs` is a
  52-line stand-in whose `encode_batch` returns zeroed codes; the working one is
  `product_quantization.rs`. Only the latter is reachable from the live path.
- **`DiskANN`, `ScaNN` and `QuantizedIVF`** fall back to brute force in
  `create_index_instance`.
- **Multimodal** (`src/multimodal/`) — returns placeholder embeddings, pending a stable
  `ort` 2.0 release.
- **gRPC carries the data path only.** Search, insert and the record routes, plus the table
  lifecycle; hybrid search, full-text search, DXQL, snapshots and backups stay on REST. It is
  off by default and off without `--grpc-port`.
- **Stub modules**, present but not functional: `cluster`, `consensus`, and the Rust-side
  `embeddings`.
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
