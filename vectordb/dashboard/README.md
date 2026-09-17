# The DXMT VectorDB web console

A read-only browser console, served by the server binary itself at `/dashboard`.

```bash
cd vectordb
make dashboard-build                                  # npm ci && vite build -> dashboard/dist
cargo build --release --features api-dashboard
./target/release/vectordb serve --dashboard           # http://localhost:9999/dashboard
```

For front-end work, run Vite against a server you already have — no Rust rebuild in the loop:

```bash
cd vectordb && cargo run -- serve      # port 9999
make dashboard-dev                     # Vite on :3000, proxying /api to :9999
```

The dev server proxies rather than pointing at an absolute origin, so the request paths in
development are the same relative paths as in production and the two cannot drift. (Pointing it
at `http://127.0.0.1:9999` would also work, because the server's CORS allow-list contains
`localhost:3000` exactly — but only on that port, and only for the methods that list happens to
include.)

## Read-only, enforced rather than promised

The server has no authentication: anything that can reach the port can already rewrite every
table. That is the reason this console must not be able to, and the reason it is not left to
review discipline:

- `src/api/client.ts` exports `get` and `post` and nothing else. No `put`, `patch` or `delete`
  function exists in the tree.
- `post`'s path parameter is a **closed union** of four read-only routes, so a write call is a
  compile error. (Verified by writing one and watching `tsc` refuse it.)
- An eslint rule forbids bare `fetch(` outside that module, so nothing can route around it.
- `dashboard-routes.json` lists every route the app issues, and two Rust tests read it: one
  asserts each is under `CONTRACTS` in `tests/api_contract_test.rs`, the other asserts none of
  them writes. Both run in the ordinary `cargo test --tests`, with no Node and no bundle.

`POST /api/v1/query` is deliberately absent. DXQL executes INSERT, UPDATE, DELETE and DROP
TABLE, so a query console would make "read-only" false.

## Why not the JavaScript SDK

The same argument `tests/api/helpers/client.py` makes about the tests. The SDK renames `payload`
to `metadata` and unwraps the response envelope; a console whose entire job is showing what the
server returned must not show the SDK's opinion of what the server meant. It is also released on
its own cadence while this ships inside the binary it must match — and it has no `search_mode`,
which is the one field the search and quality pages exist to vary.

## Dependencies

Four at runtime: `react`, `react-dom`, `react-router-dom`, `@tanstack/react-query`. For
comparison, Qdrant's console has fifty.

No component library — about 700 lines of CSS covers the dozen primitives this needs, and MUI
would be a forty-fold multiplier on that inside a binary. No Monaco (2 MB, and a console that can
only issue GETs is a crippled console). No charting library; the scatter plot is canvas.

`react-query` is here for correctness rather than polish: `staleTime` is what stops a tab switch
from re-issuing `records/list`, which the server answers by scanning and sorting the entire
table regardless of `limit`.

`npm run build` fails above a 4 MB bundle, because the bundle is compiled into the binary.

## Layout

| path | what |
| --- | --- |
| `src/api/types.ts` | the wire, with the server's own names — `tookms`, `payload_json`, and all |
| `src/api/envelope.ts` | the five failure shapes, classified |
| `src/api/client.ts` | the only module that calls `fetch` |
| `src/api/normalize.ts` | the two record shapes reconciled; vectors held outside React state |
| `src/api/capabilities.ts` | does this build have `search_mode: "exact"`, does it have BM25 |
| `src/quality/recall.ts` | the recall method, and why the query point is dropped by id |
| `test/live.test.ts` | the wire assertions, against a running server (`DXMT_LIVE_URL`) |
