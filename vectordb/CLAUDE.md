# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DXMT VectorDB is a lightweight vector database written in Rust. The **current development focus is single-node polish** — getting the core search, storage, and API layers production-ready before expanding distributed features.

**Architecture Philosophy:**
- Object Storage (S3/MinIO) as source of truth; Memory/SSD are cache layers
- Runtime adaptive index selection based on dataset size and workload
- Single binary runs from embedded to single-node server modes
- Distributed/cluster features exist but are not the current priority

## Build and Test Commands

```bash
make build          # Debug build
make release        # Release build (optimized)
make dev            # fmt + cargo watch -x run (live reload)
make run            # Run debug server (port 9999)
make run-release    # Run release server

# Testing
make test           # All tests (Rust + Python integration)
make test-unit      # Unit tests only
make test-integ     # Integration tests only
cargo test --lib    # Library unit tests only
cargo test --test wal_persistence_test       # Single integration test file
cargo test test_concurrent_writes            # Single test function
cargo test -- --nocapture                    # With stdout output

# WAL tests specifically
bash scripts/run_wal_tests.sh

# Quality
make fmt && make lint   # Format + clippy
make verify             # fmt + lint + test
make audit              # Security audit

# Benchmarks — the only registered bench target
cargo bench --bench filter_performance_bench --features text-bm25
```

All commands run from `vectordb/` — there is no `Cargo.toml` at the repository root.

Integration tests live in `tests/*.rs` (Rust) and `tests/*.py` (Python, require a running
server). `make test`, `make test-integ`, and `make bench` each append a `python3 scripts/...`
step whose script no longer exists in the tree; the steps are `|| true`-guarded and no-op
silently, so these targets are effectively plain `cargo test` / `cargo bench`.

### Running the Server

```bash
# Simple mode — zero-config, Chroma-compatible API
cargo run -- serve --simple --memory-limit 1024

# Standard mode
cargo run -- serve --host 0.0.0.0 --port 9999
```

## Architecture

### Storage Layer

Tiered architecture where S3 is the source of truth:

```
Memory (Hot Tier)  → 512MB LRU cache, <1ms latency
    ↓
SSD (Warm Tier)    → 10GB filesystem cache, <50ms latency
    ↓
S3 (Cold Tier)     → Unlimited capacity, <500ms latency
```

Key files:
- `src/storage/object/tiered.rs` — TieredStorage implementation
- `src/storage/tier_promotion.rs` — Access pattern tracking (`hotness_score`)
- `src/storage/intelligent_indexed_storage.rs` — `IntelligentIndexSelector` + channel-based insertion
- `src/storage/factory.rs` — StorageFactory; entry point for creating backends
- `src/storage/lmdb/` — LMDB backend (feature: `storage-lmdb`, enabled by default)
- `src/storage/rocksdb/` — RocksDB backend (feature: `storage-rocksdb`, opt-in)

### Search / Index Layer

Runtime adaptive index selection thresholds:

| Dataset size | Algorithm | Notes |
|---|---|---|
| < 5K | BruteForce | 100% recall |
| 5K–100K | HNSW | High recall, fast |
| 100K–1M | HNSW-PQ | Memory-efficient |
| ≥ 1M | DiskANN Enhanced | SSD-based |

Key files:
- `src/search/hnsw.rs`, `src/search/hnsw_pq.rs`, `src/search/diskann_enhanced.rs`
- `src/search/brute_force.rs`, `src/search/ivf_pq.rs`
- `src/search/full_text.rs` — Tantivy-based full-text search (feature: `text-bm25`)
- `src/search/hybrid_search.rs` — Vector + text reranking
- `src/search/quantization/` — PQ, scalar, binary quantization
- `src/adaptive_index/` — Runtime index selection logic (distinct from `IntelligentIndexSelector`)

### WAL (Write-Ahead Log)

Two implementations — prefer async for new write paths:

1. **Sync** (`src/wal.rs`) — bincode serialization, optional zstd compression, 256KB batch flush
2. **Async** (`src/async_wal.rs`) — background writer via mpsc channel, non-blocking inserts

Tests: `tests/wal_persistence_test.rs`, plus `scripts/run_wal_tests.sh`

### API Layers

1. **Simple mode** (`src/api/simple_core.rs`) — Chroma-compatible, zero-config, auto-embedding
2. **Standard REST** (`src/api/handlers.rs`) — full table/record/search API
3. **GraphQL** (`src/api/graphql.rs`, feature: `api-graphql`)
4. **gRPC** (`src/api/grpc.rs`)
5. **WebSocket** (`src/api/websocket.rs`)
6. **Swagger** — available with feature `api-swagger` (enabled by default)

Full-text search endpoints (requires `text-bm25` feature):
- `POST /tables/:name/search/text` — BM25 text search
- `POST /tables/:name/search/hybrid` — vector + text hybrid

### Key Implementation Patterns

**Interior mutability for shared indexes:**
```rust
// src/search/index_wrapper.rs
pub struct IndexWrapper(pub Arc<RwLock<Box<dyn VectorIndex>>>);
```

**Channel-based insertion** (eliminates lock contention):
```rust
// src/storage/intelligent_indexed_storage.rs
struct InsertRequest {
    table_name: String,
    records: Vec<CreateRecordRequest>,
    response_tx: oneshot::Sender<Result<Vec<Record>>>,
}
```
A background task processes inserts sequentially. All write paths should go through this channel, not direct storage mutation.

## Configuration

```bash
DB_HOST=0.0.0.0
DB_PORT=9999
DB_STORAGE_TYPE=tiered      # memory | indexed | object | tiered
DB_MEMORY_CACHE_SIZE=512    # MB
DB_COMPRESSION=zstd         # zstd | lz4 | none
DB_MEMORY_OPTIMIZATION=true # Phase 4 hybrid engine vs Phase 3

# S3
DB_OBJECT_STORAGE_PROVIDER=s3
DB_S3_BUCKET=dxmt-vectordb
DB_S3_REGION=us-east-1
```

See `.env.example` for the full list including GPU, cluster, and performance tuning options.

### Feature Flags

```toml
default = ["tokio-runtime", "basic-storage", "s3-storage", "text-bm25", "storage-lmdb", "api-swagger", "api-graphql"]

storage-lmdb      # LMDB backend (default on)
storage-rocksdb   # RocksDB backend (opt-in)
s3-storage        # S3/MinIO object storage
text-bm25         # Full-text search via Tantivy
api-graphql       # GraphQL API
api-swagger       # Swagger/OpenAPI UI
gpu               # CUDA/GPU acceleration (opt-in)
cluster           # Consul-based distributed mode (not current focus)
multimodal        # ONNX-based embeddings (blocked on ort 2.0 stable)
python            # PyO3 Python bindings
```

## Known Issues and TODOs

See `docs/TODO_PRIORITY.md` for the full tracking list. Current blockers for single-node production readiness:

1. **DXQL — two executors, only one real.** `Executor` in `src/dxql/executor.rs` is a
   storage-free shell that returns empty results for every plan type by design. Real execution
   lives in `StorageAwareExecutor` (`src/dxql/executor_impl.rs`), re-exported from
   `executor.rs`. Wire new call sites to `StorageAwareExecutor`.
2. **Hybrid Search** — sparse vector component not yet implemented
3. **Multimodal ONNX** — `src/multimodal/` returns placeholder embeddings pending ort 2.0
4. **ScaNN** (`src/search/scann.rs`) — API mismatch with `BruteForceIndex`; its tests are
   commented out
5. **RaBitQ** (`src/search/quantization/rabitq.rs`) — scalar fallback only, no SIMD path

Several modules are minimal stubs and should not be relied upon: `cluster`, `distributed`, `embeddings` (Rust-side), `monitoring`, `consensus`.

## Python SDK Integrations

The Python SDK lives in `dxmt-python-sdk/dxmt/`. All integrations use the HTTP client (`dxmt/client.py`) and implement the target framework's storage/memory interface.

| File | Framework | Interface |
|---|---|---|
| `langchain_vectorstore.py` | LangChain / Bisheng | VectorStore |
| `langgraph_store.py` | LangGraph | BaseStore |
| `haystack.py` + `haystack_retrievers.py` | Haystack | DocumentStore + Retrievers |
| `mem0.py` | mem0 | VectorStoreBase |
| `crewai_tool.py` + `crewai_storage.py` | CrewAI | Tool + Storage |
| `metagpt_vector_store.py` + `metagpt_retriever.py` | MetaGPT | VectorStore |

AutoGen integration: see `docs/autogen_integration.md` and `examples/autogen_dxmt_*.py`.

Examples: `examples/` directory. Tests: `dxmt-python-sdk/tests/` and `tests/*.py`.

## Module Organization

```
src/
├── api/              # REST, GraphQL, gRPC, WebSocket, Swagger
├── search/           # Index algorithms + quantization + full-text + hybrid
│   ├── quantization/ # PQ, scalar, binary
│   ├── fastscan/     # FastScan SIMD-accelerated scan
│   └── bm25/         # Pure-Rust BM25 (supplementary to Tantivy)
├── storage/          # Tiered storage, backends, factory
│   ├── lmdb/         # LMDB backend
│   ├── rocksdb/      # RocksDB backend
│   ├── object/       # S3/MinIO tiered storage
│   └── hybrid_engine/# Hybrid storage engine (Phase 4)
├── acceleration/     # GPU (CUDA) and SIMD acceleration
├── adaptive_index/   # Runtime index selection (higher-level than IntelligentIndexSelector)
├── index_management/ # Phase 3 index lifecycle management
├── types/            # Core types: Record, Table, Vector, search requests
├── config/           # Configuration loading and validation
├── auth/             # Authentication
├── schema/           # Schema versioning
├── worker_pool/      # Optimized async worker pool
├── memory_pool/      # Vector memory allocation
├── precomputed_query/# Search optimization via precomputed results
├── mcp/              # Model Context Protocol support (P0 priority)
├── dxql/             # DXMT query language (parser done; executor stubbed)
├── vectorql/         # Alternative SQL-like query language
├── embeddings/       # Embedding utilities (stub)
├── multimodal/       # CLIP/Whisper (stub, pending ort 2.0)
├── integrations/     # Rust-side LangChain/LlamaIndex bindings (stub)
├── compression/      # Compression utilities
├── crypto/           # Cryptography
├── observability/    # Metrics and tracing
├── plugins/          # Plugin system
├── platform/         # Platform abstractions
├── lifecycle/        # VM/node lifecycle management
├── multi_tenant/     # Multi-tenancy
├── cluster/          # Distributed coordination (stub)
├── consensus/        # Raft consensus (stub)
└── utils/            # Shared utilities
```

## Demo Data

`pdf/` is gitignored — the PDF RAG examples ship without a corpus. Supply your own:

```bash
DEMO_PDF_PATH=/path/to/your.pdf python3 examples/pdf_rag_simple.py
```

## Repository Hygiene

This repo is being prepared for open source. Two standing rules:

- **Never hardcode credentials**, not even in `scripts/` throwaways. Read them from the
  environment (`DB_AWS_ACCESS_KEY_ID`, `OPENAI_API_KEY`, …). A live Stripe key and a GCS
  HMAC pair had to be scrubbed from this tree already.
- **No progress-report markdown.** Files named `*_SUMMARY.md`, `*_REPORT.md`,
  `*_COMPLETE.md`, `*_STATUS.md` were work logs that accumulated to 89 files and were
  deleted. Put durable knowledge in `docs/` or this file; put session findings in the
  session, not on disk.
