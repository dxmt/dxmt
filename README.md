# DXMT - Data / eXperience / Memory / Thinking

### Memory for agents, owned by you.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/dxmt/dxmt/actions/workflows/ci.yml/badge.svg)](https://github.com/dxmt/dxmt/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dxmt/dxmt)](https://github.com/dxmt/dxmt/releases)

**The open-source memory database for personal AI agents.** One persistent, portable memory of
you and your work, shared by every agent you use -- Claude, Cursor, ChatGPT, your own code --
kept on your machine.

```bash
ollama pull bge-m3                       # a local, multilingual embedding model
claude mcp add dxmt -- vectordb mcp      # or the same block in Claude Desktop / Cursor
```

Your agent can now `remember` facts, preferences, events and instructions, `recall` them by
meaning and keyword in any later conversation, let a new fact **supersede** an old one (kept as
history), and `forget` on request. Every other agent on the machine sees the same memory.

**D**ata → e**X**perience → **M**emory → **T**hinking: DXMT stores what you and your agents
produce, remembers what is worth keeping, and gives it back when it matters -- and, underneath,
it is a real database: its own write-ahead log and snapshots, HNSW/HNSW-PQ vector search, BM25
full-text with Chinese segmentation, hybrid retrieval, filters and backups, in one Rust binary;
object-storage replication, read replicas with automatic failover and a cluster mode when one
machine is not enough.

> **Status: 1.4.** The compatibility promise covers the HTTP API under `/api/v1` and the
> clients' public surfaces; it does **not** cover anything
> [`docs/SUPPORT_MATRIX.md`](vectordb/docs/SUPPORT_MATRIX.md) marks experimental. That matrix
> answers "can I use this", and [Known Limitations](vectordb/README.md#known-limitations) is
> worth reading before you depend on DXMT for real data.

## Repository layout

Everything lives under **[`vectordb/`](vectordb/)** — the Rust crate `dxmt-vectordb` and its
client packages for Python, JavaScript, Rust, Go, LangChain and LlamaIndex. There is no
`Cargo.toml` at the repository root; every `cargo` and `make` command runs from `vectordb/`.

The full project README — features, architecture, storage model, feature flags, and the
measured limitations above — lives at **[`vectordb/README.md`](vectordb/README.md)**. This
file is a landing page, not a second copy of it.

## Quick start

Prebuilt binaries, web console included, are on [Releases](https://github.com/dxmt/dxmt/releases):
Linux x86_64 and aarch64 (glibc 2.28+: RHEL/Rocky 8, Debian 10, Ubuntu 20.04 or newer), and,
from 1.5, macOS on Apple silicon and Intel (`*-apple-darwin.tar.gz`). On Linux:

```bash
v=1.4.0; arch=$(uname -m)   # x86_64 or aarch64
curl -LO https://github.com/dxmt/dxmt/releases/download/v$v/dxmt-vectordb-$v-$arch-linux-gnu.tar.gz
tar xzf dxmt-vectordb-$v-$arch-linux-gnu.tar.gz
./dxmt-vectordb-$v-$arch-linux-gnu/vectordb serve          # http://127.0.0.1:9999
```

Or, from 1.5, the container image (linux/amd64 and linux/arm64). The data is in the volume;
`OLLAMA_HOST` lets it find an ollama on the host for memory and text search:

```bash
docker run -d --name dxmt -p 127.0.0.1:9999:9999 -v dxmt:/data \
  --add-host=host.docker.internal:host-gateway -e OLLAMA_HOST=host.docker.internal:11434 \
  ghcr.io/dxmt/dxmt:latest
```

Or from source (Rust via [rustup](https://rustup.rs); add Node 20 and `make dashboard-build`
first if you want the web console in the binary):

```bash
git clone https://github.com/dxmt/dxmt.git
cd dxmt/vectordb
cargo build --release
./target/release/vectordb serve
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
