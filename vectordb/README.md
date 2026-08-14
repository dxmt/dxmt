# DXMT VectorDB

A lightweight vector database written in Rust, with object storage as the source of truth and
memory/SSD as cache tiers. A single binary runs from embedded use up to a single-node server.

> **Status: pre-1.0, single-node.** The current focus is polishing the core search, storage, and
> API layers. Distributed/cluster code exists in the tree but is not production-ready — see
> [Known Limitations](#known-limitations) before depending on anything outside single-node use.

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

With the default features, Swagger UI is available for browsing the REST API.

## Architecture

Storage is tiered, with object storage as the durable source of truth:

```
Memory (hot)  →  LRU cache
     ↓
SSD (warm)    →  filesystem cache
     ↓
S3/MinIO/GCS (cold)  →  source of truth
```

Index selection adapts at runtime to dataset size:

| Dataset size | Algorithm         |
| ------------ | ----------------- |
| < 5K         | BruteForce        |
| 5K–100K      | HNSW              |
| 100K–1M      | HNSW-PQ           |
| ≥ 1M         | DiskANN Enhanced  |

Interfaces: REST (`src/api/handlers.rs`), a Chroma-compatible simple mode
(`src/api/simple_core.rs`), GraphQL, gRPC, and WebSocket.

## Configuration

Configuration is read from the environment. Copy `.env.example` and adjust:

```bash
DB_HOST=0.0.0.0
DB_PORT=9999
DB_STORAGE_TYPE=tiered      # memory | indexed | object | tiered | lmdb
DB_MEMORY_CACHE_SIZE=512    # MB
DB_COMPRESSION=zstd         # zstd | lz4 | none
```

Object storage credentials come from the environment only — never commit them:

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

- **DXQL executor** (`src/dxql/executor.rs`) — the parser works, execution is stubbed and
  returns empty results.
- **LMDB / RocksDB backends** — do not yet fully implement the `Storage` trait.
- **Hybrid search** — the sparse-vector component is not implemented.
- **Multimodal** (`src/multimodal/`) — returns placeholder embeddings, pending a stable
  `ort` 2.0 release.
- **Stub modules**, present but not functional: `cluster`, `consensus`, `distributed`,
  `monitoring`, and the Rust-side `embeddings` and `integrations`.

See `docs/TODO_PRIORITY.md` for the tracked list.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

    Copyright 2026 上海点线面体信息科技有限公司 (Shanghai Dianxianmianti Information Technology Co., Ltd.)
