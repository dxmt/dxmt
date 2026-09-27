# Contributing

Thanks for taking the time. This file covers the things that are specific to this repository;
it assumes you already know how to open a pull request. Participation is covered by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Layout, and the one thing that surprises everyone

Everything lives under `vectordb/`: the Rust crate `dxmt-vectordb`, its six client packages
under `vectordb/clients/`, and its Docker files under `vectordb/docker/`. **There is no
`Cargo.toml` at the repository root.** Every `cargo` and `make` command runs from `vectordb/`:

```bash
cd vectordb && cargo build
```

| Path | Contents |
| --- | --- |
| `vectordb/src/` | The `dxmt-vectordb` crate (bins: `vectordb`, `benchmark`) |
| `vectordb/clients/dxmt-python-sdk/` | Python client and the framework integrations built on it |
| `vectordb/clients/dxmt-js-sdk/` | TypeScript client |
| `vectordb/clients/dxmt-rust-sdk/` | Rust client — the core routes only; `tests/live.rs` runs in CI |
| `vectordb/clients/dxmt-go-sdk/` | Go client — the same core surface; `live_test.go` runs in CI |
| `vectordb/clients/langchain-dxmt/` | LangChain vector store |
| `vectordb/clients/llama-index-vector-stores-dxmt/` | LlamaIndex vector store |

Each client package is independently packaged, with its own tests and dependency manifest, so a
change to the Rust HTTP API means updating them separately. `vectordb/tests/api_contract_test.rs`
is the link between them: it lists every method/path pair the SDKs issue and fails if the router
stops serving one. Add to its `CONTRACTS` when an SDK starts calling something new.

## Before you open a pull request

```bash
cd vectordb
make verify        # fmt-check + clippy -D warnings + unit + recovery/contract + cargo test --tests
```

`make verify-core` is the same thing without the full integration suite, for iterating. Both are
pure cargo and run on a checkout with no Python.

Two things worth knowing:

- `cargo fmt` is not idempotent on this tree. If `fmt-check` still complains after you ran
  `cargo fmt`, run it again.
- The default feature set is deliberately the minimal supported surface: `tokio-runtime` +
  `basic-storage`. Full-text search, Swagger, GraphQL, WebSocket and S3 are opt-in. A default
  build has no full-text search and returns an explicit configuration error for `--swagger`,
  `--graphql` and `--websocket`. If your change touches code behind a feature flag, build it
  with that flag — CI has a job that does, because conditional compilation rots unseen.

## The test layers

| Layer | Where | How to run |
| --- | --- | --- |
| Unit | inline `#[cfg(test)]` | `cd vectordb && cargo test --lib` |
| Integration, in-process | `vectordb/tests/*.rs` | `cd vectordb && cargo test --tests` |
| Functional, over HTTP | `vectordb/tests/api/` (pytest) | `cd vectordb && scripts/api_tests.sh` |
| Stateful walk | `vectordb/tests/stateful/` | run by `scripts/api_tests.sh` |
| Storage compatibility | `vectordb/tests/compat/` | run by `scripts/api_tests.sh` |
| Client packages | each package's `tests/` | see below |

`scripts/api_tests.sh` starts a server itself, **twice** — once with records and vectors in
memory, once with them in row files — and reports which mode failed. Those are different code
paths, which is the whole reason it runs the same assertions across both. It needs `python3`
with `pytest` and `httpx`.

After both modes it runs two more suites that start their own servers. The **stateful walk**
drives a server through random writes, updates, deletes, snapshots, clean restarts and `kill -9`,
checking everything it answers against a model of what it should hold (`tests/semantics.py`);
a failure prints the seed and step that reproduce it, and `DXMT_STATEFUL_SEEDS=1-40
DXMT_STATEFUL_STEPS=400` makes it a soak. **Storage compatibility** opens data written by earlier
builds (`tests/compat/fixtures/`) and checks nothing stored was lost.

To attach to a server you already have, run `pytest tests/api` from `vectordb/`. It is a package
with relative imports, so it has to be that and not `cd tests/api`.

The client packages:

```bash
cd vectordb/clients/dxmt-python-sdk              && pytest -m "not integration"
cd vectordb/clients/dxmt-js-sdk                  && npm test
cd vectordb/clients/dxmt-rust-sdk && DXMT_URL=http://127.0.0.1:9999 cargo test --test live
cd vectordb/clients/dxmt-go-sdk   && DXMT_URL=http://127.0.0.1:9999 go test ./...
cd vectordb/clients/langchain-dxmt               && pytest
cd vectordb/clients/llama-index-vector-stores-dxmt && pytest
```

Tests that need a live server skip themselves when there is none, and take `DXMT_HOST` /
`DXMT_PORT` when there is one. A skip is not a pass: if you are changing an adapter, start a
server and watch them actually run.

Start the server in **standard** mode for anything the SDKs touch:

```bash
cd vectordb && cargo run -- serve          # /api/v1/*, port 9999
```

There is one server mode. `serve --simple`, which served a second data model at
`/collections/*` that the SDKs could not talk to, has been removed.

## What CI runs

`.github/workflows/ci.yml`, from the repository root with `working-directory: vectordb`:

- **verify** — fmt-check, clippy `--all-targets -D warnings`, `cargo test --lib`, `cargo test --tests`
- **minimal-core** — compiles the default (minimal) feature set
- **optional-features** — compiles and tests everything that is opt-in, including the
  feature-gated test targets that the default build compiles to nothing
- **build** — release build
- **api-text** — the functional HTTP suite with full-text search built in
- **sdk-live** — starts the server binary and runs the Python, JavaScript, Rust, Go, LangChain
  and LlamaIndex packages against it, then again against a server that requires an API key

`nightly.yml` measures recall against exact ground truth and runs the filter-selectivity
benchmark.

## Before a release

Write the release's storage fixture, from the commit being released, and commit it:

```bash
cd vectordb && cargo build --bin vectordb && python3 scripts/make_compat_fixture.py <version>
```

It writes `tests/compat/fixtures/<version>/` — a small data directory covering every persisted
format, and the model of what it holds. Fixtures are never regenerated or removed: every later
build has to open every one of them. If a change makes an old fixture fail, the change needs a
migration, not a new fixture.

## House rules

A few conventions this repository holds to, which reviewers will ask about:

- **Name the whole suite, never a subset.** The integration step is `cargo test --tests`. It was
  briefly a hand-picked list of four targets, which silently took 24 of the 28 test binaries out
  of the gate while everything still looked green. Feature-gating a test file has the same
  effect: a `#![cfg(feature = "…")]` on a test CI does not enable is a file that compiles and
  asserts nothing.
- **In `src/` means a binary can reach it; in `examples/` means a build touches it.** Code that
  nothing can reach does not belong in either.
- **No credentials, ever, not even in a throwaway script.** Read them from the environment
  (`DXMT_AWS_ACCESS_KEY_ID`, `OPENAI_API_KEY`, …).
- **Paste real command output into the commit message.** Not "this should work".
- **No progress-report markdown.** Files like `*_SUMMARY.md`, `*_REPORT.md`, `*_COMPLETE.md`
  accumulated to 89 of them once and were deleted. Durable knowledge goes in `docs/` or a
  README; everything else belongs in the pull request description.

## Reporting a bug

Use the issue templates. The single most useful thing you can include is the commit you built
from and the feature flags you built with — most confusing behaviour reported so far has turned
out to be a feature that was not compiled in.

For anything security-related, do not open an issue. See [SECURITY.md](SECURITY.md).

## Licensing

This project is Apache-2.0. Contributions are accepted under the same license — by opening a
pull request you agree that your contribution is licensed under Apache-2.0, as section 5 of the
license describes. There is no separate CLA.
