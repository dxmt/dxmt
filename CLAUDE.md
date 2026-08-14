# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Everything lives under `vectordb/` — the Rust crate `dxmt-vectordb` and its four client SDK
packages. The repo root holds only `LICENSE` and this file.

**There is no root `Cargo.toml`.** Every `cargo` and `make` command must run from `vectordb/`:

```bash
cd vectordb && cargo build
```

`vectordb/CLAUDE.md` is the detailed engineering guide — storage tiers, adaptive index
selection, WAL, API layers, feature flags, module map, known stubs. Read it before touching
the Rust source. This file covers only repo-level facts that are not in it.

| Path | Contents |
| --- | --- |
| `vectordb/src/` | The `dxmt-vectordb` crate (bins: `vectordb`, `benchmark`) |
| `vectordb/dxmt-python-sdk/` | Python client + LangGraph/Haystack/mem0/CrewAI/MetaGPT integrations |
| `vectordb/dxmt-js-sdk/` | TypeScript client (tsup + vitest) |
| `vectordb/langchain-dxmt/` | LangChain vector store package |
| `vectordb/llama-index-vector-stores-dxmt/` | LlamaIndex vector store package |

Each SDK is an independently packaged project with its own tests and dependency manifest —
changing the Rust HTTP API means updating the SDKs separately; nothing cross-validates them.

## Per-package commands

```bash
# Rust crate
cd vectordb && make verify              # fmt-check + clippy --all-targets -D warnings + test
cd vectordb && cargo test --lib         # unit tests only, the fastest useful signal

# Python SDK — pytest.ini sets asyncio_mode=auto and --strict-markers
cd vectordb/dxmt-python-sdk && pytest
cd vectordb/dxmt-python-sdk && pytest -m "not integration"   # skip tests needing a live server

# JS SDK
cd vectordb/dxmt-js-sdk && npm test     # vitest run

# LangChain / LlamaIndex packages
cd vectordb/langchain-dxmt && pytest tests/unit_tests
```

Tests marked `integration` (Python) and most SDK suites expect a server already listening on
`localhost:9999`. Start **standard** mode — `cargo run -- serve` — not `--simple`. The SDKs set
`base_url` to `{host}:{port}/api/v1`, whereas simple mode serves `/collections/*` at the root
and has no `/api/v1` at all, so nothing will connect.

## Traps worth knowing before you trust a green result

- **Some Makefile targets still call scripts that are gone.** `test-integ`, `bench`,
  `test-cluster` and the release targets append `python3 scripts/...` steps for files that left
  the tree, guarded with `|| true` so they no-op silently. `test` and `verify` were cleaned up
  in W5; the rest still advertise work they do not do.
- **CI does not run.** The workflows in `vectordb/.github/workflows/` invoke `cargo` at the
  checkout root with no `working-directory:`, and GitHub Actions only reads `.github/` at the
  true repo root — so they neither trigger nor would pass now that the crate is nested.
- **`make verify` passes as of W5** and is now a usable gate: `fmt-check` + `clippy
  --all-targets -D warnings` + `test`. It used to fail on 154 clippy errors, and its first step
  was `cargo fmt`, which writes rather than checks — a gate that edits your tree is not a gate.
  Note `cargo fmt` is not idempotent on this repo; run it twice if `fmt-check` still complains.
- **Half the module names do not describe what runs.** `vectordb/docs/ROADMAP.md` is the
  authority on which code the server actually reaches; read it before trusting a module name,
  a doc claim, or a passing test as evidence that a feature works. The headline: records live
  in a process-local `HashMap`, made durable by a WAL plus periodic snapshots (W1, W2), while
  the tiered object-storage backend is implemented but still not on the write path.
- **Active work is tracked in `vectordb/docs/ROADMAP.md` §7**, as self-contained items W1–W7
  with acceptance commands. Pick up from the first unchecked box; tick it and append the commit
  SHA in a follow-up commit (a commit cannot carry its own hash). One commit per item, and paste
  real command output into the message rather than claiming it should work.
- **Two forward-looking lists, different axes.** `vectordb/TODO.md` is the competitive
  algorithm roadmap (Chinese, P0/P1/P2, benchmarked against Chroma/Qdrant/LanceDB/Weaviate).
  `vectordb/docs/ROADMAP.md` is the architecture/wiring plan. `docs/TODO_PRIORITY.md` was
  deleted — several of its "implemented" entries pointed at files that never compiled.

## Standing rules

- **Never hardcode credentials**, not even in throwaway scripts — read from the environment
  (`DB_AWS_ACCESS_KEY_ID`, `OPENAI_API_KEY`, …). Live keys have already had to be scrubbed
  from this tree once.
- **No progress-report markdown.** Files like `*_SUMMARY.md`, `*_REPORT.md`, `*_COMPLETE.md`,
  `*_STATUS.md` accumulated to 89 files and were deleted. Durable knowledge goes in `docs/` or
  a CLAUDE.md; session findings stay in the session.
