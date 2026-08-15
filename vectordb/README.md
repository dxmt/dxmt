# DXMT VectorDB

A vector database written in Rust. HNSW, HNSW-PQ and product quantization with runtime SIMD
dispatch, full-text search via Tantivy, hybrid retrieval, and a query language — behind REST,
GraphQL and WebSocket, plus a Chroma-compatible zero-config mode.

> **Status: pre-1.0.** Records are held in memory and made durable by a write-ahead log,
> replayed on startup — `kill -9` and restart preserves your tables. The log defaults to
> `./data/wal`; set `DB_WAL_DIR` to move it. The tiered object-storage backend exists in the
> tree but is still not on the write path, so "object storage as the source of truth" is not
> yet true. Distributed/cluster code exists but is not production-ready. See
> [Known Limitations](#known-limitations) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Quick Start

Requires a recent stable Rust toolchain.

```bash
# Build
cargo build --release

# Zero-config mode — Chroma-compatible API, auto-tuned
./target/release/vectordb serve --simple --memory-limit 1024

# Standard mode
./target/release/vectordb serve --host 0.0.0.0 --port 9999
```

Verify it's up:

```bash
curl http://localhost:9999/health
```

Swagger, GraphQL and WebSocket are compiled in by default but not served unless asked for —
each one widens the exposed surface, so opt in per process:

```bash
./target/release/vectordb serve --swagger --graphql --websocket
```

| Flag          | Endpoint                                            |
| ------------- | --------------------------------------------------- |
| `--swagger`   | `GET /api/docs`, spec at `/api/docs/openapi.json`    |
| `--graphql`   | `POST /graphql`                                      |
| `--websocket` | `GET /ws` (upgrade)                                  |

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
serve --simple  →  /collections/*   (Chroma-compatible, at the root)
serve           →  /api/v1/*
                        ↓
             IntelligentIndexedStorage
               ├─ records: in-process HashMap      ← all data lives here
               ├─ wal:     ./data/wal              ← durability; replayed on startup
               ├─ snapshot: ./data/wal/snapshot    ← every 300s, on SIGTERM, after a rebuild
               └─ index:   HNSW / HNSW-PQ / BruteForce / DiskANN Enhanced
```

Both the write-ahead log and the snapshots are uploaded to object storage — the log as immutable
chunks, so a write acknowledged between two snapshots survives losing the machine. and a table nothing has queried for a while is
dropped from memory entirely — records and index — leaving only its segments in the object
store. The next query on it rehydrates transparently, so memory tracks the working set rather
than the whole dataset. Tuned by `DB_EVICTION_INTERVAL_SECS`.

A table past `DB_PARTITION_THRESHOLD` rows (default 100,000) switches to a partitioned IVF index
on its next `index/rebuild`: centroids stay in memory and each partition is an immutable segment
in object storage, fetched only when a query probes it. That bounds memory for a single table
larger than RAM, which table-level eviction cannot do.

See [docs/ROADMAP.md](docs/ROADMAP.md) §W3 for what remains.

Index selection runs once, when the table is created:

| Dataset size | Algorithm         |
| ------------ | ----------------- |
| < 5K         | BruteForce        |
| 5K–100K      | HNSW              |
| 100K–1M      | HNSW-PQ           |
| ≥ 1M         | DiskANN Enhanced  |

Two caveats. This table applies when `DB_MEMORY_OPTIMIZATION=false`; the default (`true`)
takes a separate path that yields HNSW below 100K and HNSW-PQ above it. And the size fed to
the selector at `CREATE TABLE` time is a fixed estimate, not the eventual row count — call
`POST /api/v1/tables/{name}/index/rebuild` to re-select against the real count.

Interfaces: REST (`src/api/handlers.rs`), a Chroma-compatible simple mode
(`src/api/simple_core.rs`), GraphQL, and WebSocket. Standard-mode REST routes are served
under `/api/v1`; only `/health` and `/status` sit at the root.

## Configuration

Configuration is read from the environment. Copy `.env.example` and adjust:

```bash
DB_HOST=0.0.0.0
DB_PORT=9999
DB_MEMORY_OPTIMIZATION=true # index-selection path; see Architecture
DB_WAL_DIR=./data/wal       # write-ahead log + snapshots; empty string disables persistence
DB_SNAPSHOT_INTERVAL_SECS=300 # periodic snapshot; 0 disables
DB_EVICTION_INTERVAL_SECS=600 # drop cold tables from memory; 0 disables
DB_MEMORY_CACHE_SIZE=512    # MB
DB_COMPRESSION=zstd         # zstd | lz4 | none
```

`DB_STORAGE_TYPE` is parsed but currently has no effect — the server constructs
`IntelligentIndexedStorage` unconditionally.

Object storage credentials come from the environment only — never commit them. They are read
into the configuration but, as above, the object-storage path is not yet on the write path:

```bash
export DB_OBJECT_STORAGE_PROVIDER=s3
export DB_S3_BUCKET=<your-bucket>
export DB_S3_ENDPOINT=<your-endpoint>
export DB_AWS_ACCESS_KEY_ID=<your-access-key>
export DB_AWS_SECRET_ACCESS_KEY=<your-secret-key>
```

### Feature Flags

Default: `tokio-runtime`, `basic-storage`, `s3-storage`, `text-bm25`, `storage-lmdb`,
`api-swagger`, `api-graphql`.

Opt-in: `storage-rocksdb`, `gpu` (CUDA), `cluster`, `multimodal`, `python` (PyO3 bindings),
`metrics`, `hf-datasets`.

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
make test               # Rust tests + Python API script
cargo test --lib        # Unit tests only
make fmt && make lint   # rustfmt + clippy
make verify             # fmt + lint + test
```

Note that `make test` and `make bench` guard their Python steps with `|| true`, so read the
output rather than relying on the exit code.

The examples in `examples/` that process PDFs do not ship with a corpus. Supply your own:

```bash
DEMO_PDF_PATH=/path/to/your.pdf python3 examples/pdf_rag_simple.py
```

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
- **Object-storage tier not on the write path.** `IntelligentIndexedStorage` holds a
  `TieredStorage` in an `#[allow(dead_code)]` field and never reads or writes through it.
- **`POST /api/v1/tables/{name}/hybrid_search` is dense-only.** It accepts `alpha`/`beta`/
  `gamma` weights and a sparse vector, then silently ignores the sparse and text components
  (`src/api/handlers.rs:961-966`). For BM25 use `/search/text`; for real fusion use
  `/search/hybrid`.
- **Two implementations behind one name.** `search/quantization/product_quantizer.rs` is a
  52-line stand-in whose `encode_batch` returns zeroed codes; the working one is
  `product_quantization.rs`. Only the latter is reachable from the live path.
- **`DiskANN`, `ScaNN` and `QuantizedIVF`** fall back to brute force in
  `create_index_instance`.
- **Multimodal** (`src/multimodal/`) — returns placeholder embeddings, pending a stable
  `ort` 2.0 release.
- **No gRPC.** It is not built; `api/grpc.rs` was never declared by `api/mod.rs` and has been
  removed.
- **Stub modules**, present but not functional: `cluster`, `consensus`, and the Rust-side
  `embeddings`.
- **CI does not run.** The workflows live under `vectordb/.github/`, which GitHub Actions
  does not read — and `make verify` fails on 154 pre-existing clippy errors, so it cannot serve
  as a gate either. Use `cargo test --lib`.

See [docs/ROADMAP.md](docs/ROADMAP.md) for what is being done about these and in what order.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

    Copyright 2026 上海点线面体信息科技有限公司 (Shanghai Dianxianmianti Information Technology Co., Ltd.)
