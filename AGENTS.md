# AGENTS.md

Instructions for AI coding agents (Codex, Cursor, Copilot, Claude Code and others) working in
this repository. The full guide is [CLAUDE.md](CLAUDE.md) at the root and
[vectordb/CLAUDE.md](vectordb/CLAUDE.md) for the Rust server; read both before changing code.
This file is the short version.

- **Everything lives under `vectordb/`.** There is no root `Cargo.toml`: run every `cargo` and
  `make` command from `vectordb/`.
- **The gate is `cd vectordb && make verify`** (fmt-check, clippy with `-D warnings`, unit and
  integration tests). `make verify-core` skips the full integration suite while iterating.
  `cargo fmt` is not idempotent here; run it twice.
- **Each client under `vectordb/clients/` is its own package** with its own tests. Changing the
  HTTP API means updating them, and adding a route an SDK calls means adding it to
  `CONTRACTS` in `vectordb/tests/interfaces/api_contract_test.rs`.
- **Never hardcode credentials**, not even in a throwaway script; read them from the environment.
- **No progress-report markdown** (`*_SUMMARY.md`, `*_REPORT.md`, ...). Durable knowledge goes in
  `vectordb/docs/` or a CLAUDE.md.
- **A test file behind `#![cfg(feature = "...")]` compiles to nothing** without the feature, and
  a test file not listed as a `[[test]]` in `vectordb/Cargo.toml` never runs. Check the count of
  tests that ran, not the colour.
- **Keep `vectordb/docs/SUPPORT_MATRIX.md` true.** A feature goes into its supported table only
  after its "Promotion Rules" are met.
