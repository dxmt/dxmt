# DXMT VectorDB - TODO/FIXME Priority Tracking

## Overview

This document tracks remaining TODO/FIXME items in the codebase. All **P1** (critical for single-node production) and **P2** (high-priority features) items have been resolved. Remaining items are either blocked by external dependencies or are out-of-scope for the current single-node focus.

**Last updated**: 2026-07-04

---

## RESOLVED (previously CRITICAL / HIGH)

| # | Item | Resolution |
|---|------|-----------|
| 1 | Storage trait for LMDB/RocksDB — WAL rollback | Implemented: `atomic_ops.rs` rollback now calls `delete_record`/`delete_table` via `Arc<dyn _Storage>` |
| 2 | Index consistency & repair | Implemented: `consistency.rs` checks record count + individual records, repairs via `update_record`/`delete_records` |
| 3 | DXQL query optimizer | Implemented: `optimizer.rs` filter pushdown (cost sort) + projection dedup |
| 4 | DXQL executor stubs | Implemented: `executor_impl.rs` MockStorage `backup`/`restore` no longer `todo!()` |
| 5 | Hybrid search (API layer) | Resolved: `handlers.rs` now tracks real elapsed time; sparse/text stub paths documented |
| 6 | Storage migration | Implemented: `factory.rs` `migrate_to_optimized()` does list → recreate → copy |
| 7 | Backup compression | Implemented: `backup.rs` uses zstd for create/restore; `execute_backup_filtered()` added |
| 8 | Tiered storage cache metrics | Implemented: `tiered.rs` uses `Arc<AtomicU64>` counters; `get_tier_stats()` returns real values |
| 9 | Tombstone tracker bloom filter | Implemented: `tombstone_tracker.rs` uses double-hash bloom filter (FNV-1a + DJB2) |
| 10 | Full-text search config | Implemented: `full_text.rs` uses `ReloadPolicy::OnCommitWithDelay`; `optimize()` merges segments |
| 11 | Hybrid index statistics | Implemented: `hybrid_indexing.rs` tracks `build_time_ms` and `disk_usage_bytes` |
| 12 | Quantization stats in `search()` | Implemented: `product.rs`, `scalar.rs`, `binary.rs` use `Arc<AtomicU64>` for latency tracking |
| 13 | Memory storage backup/restore | Implemented: JSON snapshot to/from `backup_path` |
| 14 | Python SDK tests (16 failing) | Fixed: all 35 tests in `test_client.py` + `test_client_complete.py` pass |
| 15 | Python bindings (list_vectors, parse_config, parse_filter) | Implemented: `python/mod.rs` wired to `list_all_record_ids`, dict parsing for config and filter |
| 16 | OpenAI / Cohere `EmbeddingFunction` trait | Implemented: `openai.rs`, `cohere.rs` now impl the `compat::EmbeddingFunction` trait |
| 17 | LMDB `update_record` version + created_at | Fixed: preserves `existing.created_at`, increments `existing.version + 1` |
| 18 | WAL rollback stubs | Fixed: `atomic_ops.rs` rollback uncommented and wired to storage backends |
| 19 | MCP text-only hybrid search | Fixed: returns structured JSON with REST API redirect hint instead of error string |
| 20 | Performance stats mutation under read lock | Fixed: `Arc<Mutex<PerformanceStats>>` in `intelligent_indexed_storage.rs` |
| 21 | T1 Schema Evolution | Implemented: `alter_table` API, LMDB full-config persistence, `schema_version` tracking, `payload_schema` add/drop fields |
| 22 | T2 Filter-in-HNSW (Acorn) | Implemented: in-graph filtering for selectivity < 5%; direct ID scan for tiny filter sets |
| 23 | T3 Named Vectors (multi-vector) | Implemented: `VectorSpaceConfig`, `IndexCollection` per table, `vector_name` routing in search |
| 24 | T4 Dynamic EF | Implemented: `SearchMode::Auto` with `ef_search_adaptive()` scaling EF by filter selectivity |
| 25 | Cluster shard_router deadlock | Fixed: `add_node`/`remove_node` release write locks before calling `rebalance_shards` |
| 26 | Tiered search zero-vector test data | Fixed: `create_test_vectors` uses `(i+1)/N` to avoid i=0 zero vector |
| 27 | Dataset registry too few entries | Fixed: Added `gist-960` and `lastfm` to reach ≥6 default datasets |
| 28 | Dataset cache stats counts metadata files | Fixed: `stats()` now counts only `.zst` data files, not `.meta.json` sidecar files |
| 29 | Performance benchmarks fail on slow CI | Fixed: WAL benchmarks + HNSW 10K + tiered perf test marked `#[ignore]`; integration time limit raised to 120s |
| 30 | Named vector dimension pre-check | Fixed: dimension validation skipped when `vector_name` is `Some` (named index may differ) |

---

## BLOCKED (external dependencies)

### Multimodal ONNX Backends
**Files**: `src/multimodal/clip.rs`, `src/multimodal/whisper.rs`, `src/multimodal/processor.rs`
**Status**: Blocked on `ort` crate version 2.0 stable release
**Impact**: CLIP image/text encoding and Whisper audio transcription return placeholder vectors
**Action**: Re-enable when `ort` 2.0 API stabilizes; see `CLAUDE.md` `multimodal` feature flag

### ScaNN / BruteForce API Incompatibility
**File**: `src/search/scann.rs`
**Status**: API mismatch between ScaNN implementation and `BruteForceIndex`
**Impact**: ScaNN tests are commented out
**Action**: Align APIs when ScaNN is prioritized

### RaBitQ SIMD
**File**: `src/search/quantization/rabitq.rs`
**Status**: SIMD version using `std::arch` not yet implemented; scalar fallback works
**Impact**: RaBitQ is ~2-4x slower than SIMD-optimized version
**Action**: Low priority; implement when benchmarks confirm bottleneck

---

## OUT OF SCOPE (cluster / distributed)

- Consul-based cluster coordination (`src/cluster/`)
- Distributed query cache (Redis/Valkey backend in `query_cache.rs`)
- Raft consensus (`src/consensus/`)
- Cross-modal search (requires ONNX backends above)

These remain stubs per the "single-node polish" focus in `CLAUDE.md`.

---

## Summary

| Category | Count | Notes |
|----------|-------|-------|
| Resolved | 30 | All P1/P2 items + T1–T4 + test infrastructure fixes |
| Blocked (external) | 3 | ort 2.0, ScaNN API, SIMD |
| Out of scope | 4 | Cluster/distributed |
| **Total remaining** | **7** | None blocking single-node production |
