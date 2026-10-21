# DXMT VectorDB

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

A vector database written in Rust: HNSW, HNSW-PQ and product quantization with runtime SIMD
dispatch, BM25 full-text and hybrid retrieval, and a query language behind a REST API, with a
web console and an OpenAPI reference served by default. GraphQL, WebSocket, gRPC and S3 are
opt-in build features. It is a single-node server.

> **Status: 1.3**, under the compatibility promise 1.0 made. A write-ahead log plus periodic snapshots is the durability story —
> `kill -9` and restart preserves your tables. Records and full vectors live in row files
> beside the log; what stays resident is the graph, the quantization codes and the id map,
> about 0.45 KB per 768-dimension row measured at 100 million.
>
> The compatibility promise covers the HTTP API under `/api/v1` and the six clients' public
> surfaces. It does **not** cover anything
> [`docs/SUPPORT_MATRIX.md`](vectordb/docs/SUPPORT_MATRIX.md) marks experimental, or the
> cargo features that are off by default. That matrix answers "can I use this", which is a
> different question from "does this exist", and
> [Known Limitations](vectordb/README.md#known-limitations) is worth reading before you
> depend on this for real data.

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
./target/release/vectordb serve          # http://127.0.0.1:9999
```

It listens on this machine only. To serve other machines, add `--host 0.0.0.0` and set
`DXMT_API_KEY` first: without a key every route is open.

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
