# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DXMT VectorDB is a lightweight vector database written in Rust. The **current development focus is single-node polish** — getting the core search, storage, and API layers production-ready before expanding distributed features.

**Read `docs/ROADMAP.md` before planning any change.** It records what is actually wired up,
what exists in the tree but has no caller, and which names in this codebase do not match their
implementations. The intended architecture and the shipped architecture differ in three places
that matter, and the differences are easy to miss from the module names alone.

**Intended architecture** (not all of it is reached by the running server — see the roadmap):
- Object Storage (S3/MinIO) as source of truth; Memory/SSD are cache layers
- Runtime adaptive index selection based on dataset size and workload
- Single binary runs from embedded to single-node server modes
- Distributed/cluster features exist but are not the current priority

**What the server actually does today:** both `serve` and `serve --simple` construct
`IntelligentIndexedStorage`, which holds every record in an in-process `HashMap`. Standard mode
opens it through `new_with_wal` (`src/api/server.rs`), so writes are logged to `./data/wal`
(`DB_WAL_DIR` overrides) and replayed at startup — a restart no longer loses data. Its
`TieredStorage` field is still `#[allow(dead_code)]`: durability comes from the WAL, not from
object storage. Index algorithm is chosen once at `CREATE TABLE` from a fixed size estimate,
not from the eventual row count.

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

Tiered architecture where S3 is the source of truth — implemented, but **not reached by the
live write path**. `IntelligentIndexedStorage` builds a `TieredStorage` in its constructor and
never calls it (`self.storage` appears zero times in that file):

```
Memory (Hot Tier)  → 512MB LRU cache, <1ms latency
    ↓
SSD (Warm Tier)    → 10GB filesystem cache, <50ms latency
    ↓
S3 (Cold Tier)     → Unlimited capacity, <500ms latency
```

`DB_STORAGE_TYPE` is parsed into config but selects nothing; `StorageFactory` has no caller.

Key files:
- `src/storage/object/tiered.rs` — TieredStorage implementation
- `src/storage/tier_promotion.rs` — Access pattern tracking (`hotness_score`)
- `src/storage/intelligent_indexed_storage.rs` — `IntelligentIndexSelector` + channel-based insertion
- `src/storage/factory.rs` — StorageFactory; **no caller**, the server does not use it
- `src/storage/lmdb/` — LMDB backend (feature: `storage-lmdb`, enabled by default); reachable
  only through `storage/hybrid_engine/`, which itself has no caller
- The standalone `src/storage/rocksdb/` backend was never declared by `storage/mod.rs` and has
  been removed; the `storage-rocksdb` feature now gates only `hybrid_engine`'s own RocksDB path

### Search / Index Layer

Index selection thresholds (`src/storage/index_selector.rs:108`):

| Dataset size | Algorithm | Notes |
|---|---|---|
| < 5K | BruteForce | 100% recall |
| 5K–100K | HNSW | High recall, fast |
| 100K–1M | HNSW-PQ | Memory-efficient |
| ≥ 1M | DiskANN Enhanced | SSD-based |

Three things this table does not tell you:

1. It only applies when `DB_MEMORY_OPTIMIZATION=false`. The default is **true**, which routes
   to `select_memory_optimized_index` — HNSW below 100K, HNSW-PQ above, with two branches
   carrying `// … not yet available` fallback comments.
2. Selection happens once at `CREATE TABLE`, against a hardcoded `estimated_size = 10_000`
   (`intelligent_indexed_storage.rs:474`), not the eventual row count. Automatic
   re-evaluation is commented out at `:1608`. Only `POST /tables/{name}/index/rebuild`
   re-selects against reality.
3. `DiskANN`, `ScaNN` and `QuantizedIVF` fall back to `BruteForceIndex` in
   `create_index_instance`. Of the enum's seven variants, four are real.

Key files:
- `src/search/hnsw.rs`, `src/search/hnsw_pq.rs`, `src/search/diskann_enhanced.rs`
- `src/search/brute_force.rs`, `src/search/ivf_pq.rs`
- `src/search/full_text.rs` — Tantivy-based full-text search (feature: `text-bm25`)
- `src/search/hybrid_search.rs` — Vector + text reranking
- `src/search/quantization/` — PQ, scalar, binary, RaBitQ. Note `product_quantization.rs` is
  the working product quantizer; `product_quantizer.rs` is a 52-line stand-in that returns
  zeroed codes, used only by the unreachable `fastscan/`
- `src/adaptive_index/` — Runtime index selection logic, distinct from `IntelligentIndexSelector`
  and **not wired**: its only referent is `index_management/`, which itself has no caller

### WAL (Write-Ahead Log)

1. **Sync** (`src/wal.rs`) — MessagePack, optional zstd, **on the live write path**.
   `IntelligentIndexedStorage::new_with_wal` opens it; `wal_append` writes and fsyncs after each
   batch, and the constructor replays the log before returning.
2. **Async** (`src/async_wal.rs`) — background writer via mpsc channel; still no callers, and no
   replay. Worth adopting when the per-batch fsync becomes the bottleneck (roadmap §3.3).

Writes are logged *after* being applied and *before* the caller is acknowledged. That is not
literal write-ahead ordering, but the in-memory state is rebuilt from the log anyway, so it
still guarantees no acknowledged write is lost — and it avoids logging operations that the
insertion processor then rejects.

`src/storage/atomic_ops.rs` also uses `wal.rs`, on a separate path reachable only from the
`cluster/shard_router.rs` stub. Do not confuse the two.

Tests: `tests/wal_recovery_test.rs` (7 crash-recovery tests), `tests/wal_persistence_test.rs`,
plus `scripts/run_wal_tests.sh`

### API Layers

1. **Simple mode** (`src/api/simple_core.rs`) — Chroma-compatible, zero-config, auto-embedding.
   Routes sit at the **root** (`/collections/*`), not under `/api/v1`.
2. **Standard REST** (`src/api/handlers.rs`) — full table/record/search API, mounted under
   `/api/v1`. Only `/health`, `/health/metrics` and `/status` are at the root. The Python SDK
   sets `base_url` to `{host}:{port}/api/v1`, so it talks to standard mode, **not** simple mode.
3. **GraphQL** (`src/api/graphql.rs`, feature: `api-graphql`) — `POST /graphql`
4. **WebSocket** (`src/api/websocket.rs`) — `GET /ws`
5. **Swagger** (feature: `api-swagger`) — `GET /api/docs`, spec at `/api/docs/openapi.json`

The `api-swagger` / `api-graphql` cargo features control **compilation only**. Serving them is
a separate, runtime decision: `create_app()` used to hardcode
`create_app_with_features(false, false, false)`, so all three were compiled into the binary and
unreachable — verified by curl returning 404. `main.rs` now passes `--swagger` / `--graphql` /
`--websocket` through. They stay off by default; each widens the exposed surface.

There is no gRPC. `api/grpc.rs` was never declared by `api/mod.rs`, so it never compiled;
it has been removed. `api/{batch_operations,streaming_search,cross_modal_batch,rate_limiter}.rs`
compile but `server.rs` mounts no routes for them.

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

See `docs/ROADMAP.md` for the ordered plan. Current blockers for single-node production
readiness, in dependency order:

1. **Startup replays the entire log** — recovery re-inserts every record, so boot time scales
   with the dataset. No checkpoint or log compaction yet.
2. **No index persistence** — only `src/search/ivf.rs` implements `SerializableIndex`; HNSW
   and HNSW-PQ have no save/load, so even with §2.1 a restart means a full index rebuild.
   Roadmap §2.2.
3. **Tiered storage not wired** — roadmap §2.3.
4. **DXQL — two executors, only one real.** `Executor` in `src/dxql/executor.rs` is a
   storage-free shell that returns empty results for every plan type by design. Real execution
   lives in `StorageAwareExecutor` (`src/dxql/executor_impl.rs`), re-exported from
   `executor.rs`, and that is what `POST /query` uses. Wire new call sites to it.
5. **Hybrid Search** — `POST /tables/{name}/hybrid_search` silently drops the sparse and text
   components and returns dense-only results while still accepting fusion weights
   (`api/handlers.rs:961-966`).
6. **Multimodal ONNX** — `src/multimodal/` returns placeholder embeddings pending ort 2.0
7. **ScaNN** (`src/search/scann.rs`) — API mismatch with `BruteForceIndex`; its tests are
   commented out
8. **RaBitQ** (`src/search/quantization/rabitq.rs`) — scalar fallback only, no SIMD path

Minimal stubs, do not rely on them: `cluster`, `consensus`, `embeddings` (Rust-side).

Compiled, but with no caller from any binary — do not assume these run:
`index_management/` + `adaptive_index/`, `async_wal.rs`, `multi_tenant/`, `worker_pool/`,
`memory_pool.rs`. Each is classified in `docs/ROADMAP.md` §3.

`mcp/` is wired as of W4: `vectordb mcp` serves the Model Context Protocol over stdio against
the same WAL-backed storage as `serve`. Note that stdout is the JSON-RPC channel there, so
`main.rs` routes tracing to stderr for that subcommand — never `println!` on an MCP path.

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

Marked **[unwired]** where the module compiles but no binary can reach it — see
`docs/ROADMAP.md` §3 before building on one.

```
src/
├── api/              # REST (/api/v1), simple mode (/collections), GraphQL, WebSocket, Swagger
├── search/           # Index algorithms + quantization + full-text + hybrid
│   ├── quantization/ # PQ, scalar, binary, RaBitQ
│   ├── fastscan/     # FastScan SIMD-accelerated scan
│   └── bm25/         # Pure-Rust BM25 (supplementary to Tantivy)
├── storage/          # The live write path plus several backends that are not on it
│   ├── lmdb/         # LMDB backend (feature: storage-lmdb)
│   ├── object/       # S3/MinIO tiered storage — implemented, not wired
│   └── hybrid_engine/# LMDB + RocksDB hybrid engine
├── acceleration/     # GPU (CUDA) and SIMD acceleration
├── types/            # Core types: Record, Table, Vector, search requests
├── config/           # Configuration loading and validation
├── dxql/             # Query language behind POST /query (see Known Issues #4)
├── benchmarks/       # Datasets, recall/nDCG/MAP metrics — used by the `benchmark` bin
├── embeddings/       # Embedding utilities (stub)
├── multimodal/       # CLIP/Whisper (placeholder embeddings, pending ort 2.0)
├── wal.rs            # Write-ahead log — works, not on the write path
├── async_wal.rs      # [unwired] Async WAL; prefer this when doing roadmap §2.1
├── adaptive_index/   # [unwired] Runtime index selection, only used by index_management
├── index_management/ # [unwired] Index lifecycle, migration, transaction log
├── mcp/              # [unwired] Model Context Protocol server — no subcommand starts it
├── multi_tenant/     # [unwired] Quotas, usage limits, shard/query routing
├── worker_pool/      # [unwired] Async worker pool
├── memory_pool.rs    # [unwired] Vector allocation pool
├── precomputed_query.rs # Search optimization via precomputed results
├── cluster/          # Distributed coordination (stub)
├── consensus/        # Raft consensus (stub)
└── utils/            # Shared utilities
```

Removed in commits `ff53809` and `25400ce`: ten top-level modules that `lib.rs` never declared
(`auth`, `crypto`, `observability`, `integration`, `platform`, `lifecycle`, `plugins`,
`compression`, `metrics.rs`, `tracing.rs`) plus `vectorql`, `integrations`, `schema`,
`monitoring`, `distributed`, `index`, and 21 orphan files. None of it was ever a build input.
Design ideas worth reviving are catalogued in `docs/ROADMAP.md` §4.

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
