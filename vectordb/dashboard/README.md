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
| Console | vary a request and send it again, with what it cost | read-only, see below |
| Visualize | does this embedding have structure, and can the picture be trusted | see below |

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

### Visualize

Two layouts, and the choice is the point.

**Neighbour graph** (default). The server computes the true nearest neighbours of the sample and
sends the graph; this lays it out with a force simulation. No vector crosses the wire, so the
sample size stops being decided by how much JSON a tab can parse. Measured against the same
thousand records of a 384-dimension table:

    layout   fetch        wire   compute   neighbourhoods kept
    graph     53 ms   0.36 MB    154 ms                   27%
    pca      370 ms   8.45 MB    158 ms                   12%

Twenty-three times less data, seven times faster to fetch, and it keeps more than twice as much
of the neighbourhood structure — because it optimises for that directly, where PCA optimises for
variance and only incidentally for what a reader of a scatter plot is asking.

On `persist_demo`, whose vectors are uniformly random, **both score 2%**. That is the important
control: a layout that "improved" on structureless data would be inventing it.

**PCA** stays, because it answers a different question (which two directions carry the variance)
and because it needs no server support. Its ceiling is the wire — about 1,300 points at 768
dimensions; the graph's is the server's node cap, 4,000 at any dimension, since the computation
is quadratic. A 4,000-node graph is still smaller than 1,000 vectors.

PCA on a high-dimensional embedding is a weak summary, measured on the two demo tables:

    persist_demo   1000 x 128   PC1 1.4%  PC2 1.4%  together  2.8%   -> the page warns
    papers          300 x   8   PC1 15.4% PC2 16.0% together 31.4%   -> it does not

2.8% is what isotropic noise in 128 dimensions looks like: the scatter means nothing, and saying
so is the page's job. Five things do that work, and none is decoration:

1. the explained-variance ratio **and the neighbourhood-preservation rate** are in the header,
   not a tooltip;
2. a caption fires when either one is poor, and states that distance on the plot is not distance
   in the embedding;
3. nothing is drawn beyond the points — no hulls, no density, no groupings, and the word
   "cluster" appears nowhere;
4. **clicking a point runs a real search** and rings its true neighbours where they actually
   landed. If they scatter, the reader watches the projection fail. Neighbours that fell outside
   the sample are counted and stated, because an unstated absence reads as evidence;
5. the sample is described: how many of how many, drawn from how many insertion-time windows,
   and why the ceiling is where it is.

The budget is a pair because there are two costs. `n x d` bounds the wire (JSON floats, ~13
bytes a dimension, uncompressed), and the number of `records/list` calls bounds the server
(each is a full table sort). That gives ~4,000 points at 128 dimensions, ~1,300 at 768, ~650 at
1,536. Since `records/list` is `created_at` descending, offset 0 would be a sample of last
Tuesday — so the requests stride across evenly spaced windows, and the page says that is a
cluster sample over insertion time rather than a uniform one.

The projection runs in a worker on a transferred flat `Float32Array` (6 MB moved, not 24 MB
copied as boxed numbers), centres once instead of inside both inner loops of all 64 iterations,
and carries a `jobId` so a run started two slider adjustments ago cannot land on the current
one. 110 ms for 1,000 x 128.

The renderer is 2-D canvas, not WebGL: the sampling budget caps this at four thousand points,
where batching by colour is under a millisecond a frame.

#### Two measures, because they come apart

Explained variance says how much of the *variance* the two components carry. It says nothing
about whether the points near each other on screen are near each other in the embedding — which
is the only question a reader of a scatter plot is asking. So the page measures both: for fifty
sampled points, the overlap between the k nearest in the plot and the k nearest in full
dimensions, in the table's own metric.

They disagree, and that is the point:

    persist_demo   variance  2.8%   neighbourhoods kept   2%
    papers         variance 31.4%   neighbourhoods kept  17%

`papers` clears any variance threshold and still scrambles five neighbours in six. A
variance-only warning would have let that plot through without a word, which is why the caption
fires on either measure.

It is deliberately not called *trustworthiness*: that names a specific rank-weighted measure,
and this is the plain overlap. Using the word would claim a property that was not computed. The
measurement is taken on the **uncentred** vectors, because PCA centres its input in place —
which leaves Euclidean distances alone and changes every cosine.

## When it breaks

A render crash shows the crash, scoped to the page so the navigation survives and the reader can
go somewhere else without reloading. A console is opened *because* something is already wrong; a
blank document with the message only in the browser's own console is the least useful thing it
could do at the moment it matters most.

Keyboard: a skip link, `:focus-visible` rings that appear for Tab and not for a mouse click, and
an `aria-label` on the scatter canvas carrying the same two figures the header shows — a canvas
is opaque to a screen reader, and stating them in text is also the honest summary for anyone
reading quickly.

## Console

A scratchpad for requests, beside the Swagger page rather than instead of it. Swagger is
per-operation — a form per route, a modal at a time, the body reset whenever you navigate — which
is right for reading what an API does and wrong for the loop this is for: vary one field, send it
again, compare against what came back a minute ago. So free text, a history that persists, and
the timing and size of every response.

    # Ctrl+Enter sends.
    POST /api/v1/tables/articles/search
    { "vector": [0.1, 0.2], "top_k": 5, "search_mode": "exact" }

**It sends nothing that changes data**, and that is enforced rather than intended. `client.ts`
makes a write a compile error, which is worth everything for code we wrote and nothing at all for
a string someone types into a box — so the console has a second gate, in `lib/console.ts`, and
`sendRead` accepts only a value that gate produced.

That last claim was false when first written. `Parsed` was a plain interface, so
`{ method: 'POST', path: '/api/v1/tables', body: {} }` satisfied it and walked straight around
the check; the probe that was supposed to fail compiled cleanly. It carries a private brand now,
and the same probe is rejected: *Property '[checked]' is missing in type … but required in type
'Parsed'*.

A refused request names the Swagger page, because "I need to create a table" is a real need and
deserves an answer rather than a wall. No Monaco: two megabytes for highlighting two lines of
JSON, compiled into a server binary whose entire console is 115 KB.
