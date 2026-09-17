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
| `src/lib/filter.ts` | the filter model, and the SDK's Mongo shorthand detected by name |
| `src/pages/TableRecords.tsx` | the record browser, designed around a listing that scans the table |
| `test/live.test.ts` | the wire assertions, against a running server (`DXMT_LIVE_URL`) |

## Pages

| page | answers | notes |
| --- | --- | --- |
| Overview | is it healthy, how much would a restart replay | the replay figure is derived; no endpoint states it |
| Tables | what exists, how big, which index | sorting and filtering are client-side because the endpoint offers neither |
| Table detail | schema, footprint, bytes per record | shows all three record counts when they disagree |
| Records | what is actually stored | see below |
| Record | one record in full, with its vector | `GET /records/{id}` always has the vector |
| Search | why did this come back, and what is the index costing | see below |
| Index quality | how much recall is the index losing, on this data | see below |

### The search inspector

The same query under every mode, side by side — the page `search_mode: "exact"` was added for.
On a 100,000-row table, one query:

    mode        server ms    overlap with exact
    fast                3    60%
    normal              2    80%
    accurate            6    90%
    auto                5    90%
    exact              18    (ground truth)

Two things it deliberately does not do. It does not run the modes **in parallel** — concurrent
searches contend for the index and the page cache, and each `tookms` would then be partly a
measurement of the others. A test makes the comparison fail by name if that changes. And it does
not pre-empt the server's validation: a wrong-length vector gets a warning and a *send it
anyway* checkbox, because seeing the real `DIMENSION_MISMATCH` is sometimes the point.

On a server without the exact mode — detected by a probe that reads the plain-text 422 the axum
extractor answers with — the page says so and stops calling anything recall. `accurate` is 4x
ef and about 0.98 itself, so agreement with it is an upper bound, blind to whatever every mode
misses.

### Index quality

The method is Qdrant's (Apache 2.0 — the method, not the code): existing records as queries,
each run twice, recall as the overlap. Twenty sample points on the 100,000-row demo table, which
costs twenty exhaustive passes plus eighty indexed searches:

    mode        recall@10    @1   score ratio    ms   vs truth   histogram
    fast      0.650±0.136  0.75        0.9975    <1        10×   ····▃▄▅▆▇··
    normal    0.840±0.110  0.95        0.9993     1        10×   ·······▆▇▆▅
    accurate  0.955±0.076  1.00        0.9999     4       2.5×   ········▄▄█
    auto      0.955±0.076  1.00        0.9999     3       3.3×   ········▄▄█

The score ratio is why recall alone is not enough: `fast` misses 35% of the true neighbours and
still returns 99.75% of the summed similarity, so what it substituted was almost as close. The
histogram is why the mean is not enough: `fast`'s recall is spread from 0.4 to 0.9 where
`accurate`'s is bunched at the top, and a mean cannot tell "everyone misses one" from "a few are
catastrophic". The worst five points are listed by id and link to the record.

Three things it refuses to do. It will not start while an index is rebuilding — every query on
the server is queued behind that, so the numbers would be about the rebuild. It will not run
four separate comparisons: the truth for a point does not depend on which mode it is compared
against, so it is fetched once and reused, N exhaustive passes rather than 4N (a test counts the
requests and fails if that regresses). And it will not call the result recall when there was no
ground truth.

Stopping a run keeps what it measured and says the figures are partial. The sample is drawn by a
seeded PRNG with the seed in the URL, so a run is reproducible by sharing the link.

### What shapes the record browser

`POST /records/list` collects every matching record, sorts the whole set by `created_at`, and
then slices. `limit` and `offset` shape the response, not the work — measured at **121 ms for a
page of 25 from a 100,000-row table**, whether or not vectors are included. So:

- nothing is fetched automatically above 50,000 rows; the page says why and offers a button;
- page sizes stop at 100, and there is no infinite scroll, no prefetch and no polling;
- `staleTime: Infinity`, so returning to the tab is not another table scan;
- vectors are off by default behind a toggle that states the cost (5.7 KB → 67 KB for 25 rows
  at 128 dimensions; roughly 400 KB at 768).

A vector never appears in a cell as numbers. The cell is a canvas sparkline of the first
components plus the dimension and the norm, drawn from the `Float32Array` map in `normalize.ts`
— React never holds the floats.
