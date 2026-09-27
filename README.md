# DXMT VectorDB

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

A vector database written in Rust: HNSW, HNSW-PQ and product quantization with runtime SIMD
dispatch, hybrid retrieval, and a query language behind a REST API. Full-text search, GraphQL,
WebSocket and S3 are opt-in build features — a default build is the minimal supported
single-node server and nothing more.

> **Status: Alpha, not production-ready.** A write-ahead log plus periodic snapshots is the
> durability story — `kill -9` and restart preserves your tables. Records and full vectors
> live in row files beside the log; what stays resident is the graph, the quantization codes
> and the id map, about 0.45 KB per 768-dimension row measured at 100 million. Read
> [Known Limitations](vectordb/README.md#known-limitations) before depending on this for real
> data.

## Repository layout

Everything lives under **[`vectordb/`](vectordb/)** — the Rust crate `dxmt-vectordb` and its
client packages for Python, JavaScript, Rust, Go, LangChain and LlamaIndex. There is no
`Cargo.toml` at the repository root; every `cargo` and `make` command runs from `vectordb/`.

The full project README — features, architecture, storage model, feature flags, and the
measured limitations above — lives at **[`vectordb/README.md`](vectordb/README.md)**. This
file is a landing page, not a second copy of it.

## Quick start

```bash
git clone https://github.com/dxmt/dxmt.git
cd dxmt/vectordb
cargo build --release
./target/release/vectordb serve --host 0.0.0.0 --port 9999
```

Then see [`vectordb/README.md`](vectordb/README.md#quick-start) for the API reference, the web
console, and the client SDKs.

## Where things are

| | |
| --- | --- |
| What this is, features, storage model, limitations | [`vectordb/README.md`](vectordb/README.md) |
| What each client SDK actually covers | [`vectordb/docs/SUPPORT_MATRIX.md`](vectordb/docs/SUPPORT_MATRIX.md) |
| Building, testing, opening a pull request | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Reporting a vulnerability | [`SECURITY.md`](SECURITY.md) |
| Community expectations | [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](vectordb/NOTICE).
