# The disk index: how a table larger than memory is served

This is the design the storage engine actually implements, and the measurements that decided
each part of it. It exists because a dozen doc comments in `src/` cite it: when the code says
"phase 3", "route C" or "section 5.5.1", this is the document it means.

**The problem it solves.** With the graph, the full vectors and the record map all resident,
a 768-dimension row cost about **8.3 KB of process memory**. A 128 GB machine therefore held
about 15 million rows, and the target was 100 million. Today the same row costs about
**0.45 KB** and 100 million of them are served from one process.

Everything below was measured on this codebase with the benchmark harness in this repository.
Unless a row says otherwise the corpus is a 768-dimension embedding dataset and the metric is
`recall@100` — see [Why recall@100](#2-why-recall100) for why that number and not `recall@10`.

---

## 1. The choice: three routes, and why C

Every candidate shares one idea — **walk the graph on a compressed copy, then rerank the
survivors against the full vectors**. They differ in what lives where.

| | A: partitioned (IVF-PQ) | B: graph on disk (DiskANN/Vamana) | **C: graph resident, codes resident, full vectors on disk** |
| --- | --- | --- | --- |
| What the walk reads | centroids + PQ codes | fixed-size `[vector][neighbours]` records on disk | resident graph + resident PQ codes |
| Random disk reads per query | 0, plus R for the rerank | ~20 pages, rerank rides along | R, for the rerank |

**A was rejected on arithmetic, not taste.** At equal recall (0.98) a partitioned index scans
5–10% of the data, which is **6–10× the CPU** of the graph when comparing PQ codes and 25–40×
when comparing exact vectors, at 1M rows; at 10M it is 22–92×. Partitioning earns its keep when
a lookup is one object-storage round trip at ~100 ms. On NVMe at ~100 µs with the codes already
resident, it only spends CPU. (The measurement covers plain IVF. SPANN-style structures were
not tested; the decisive argument against A is the memory arithmetic in §3, not this table.)

**B is not a replacement for C, it is the road after it.** Putting the graph on disk exists to
get past "the graph itself does not fit". A compact graph plus PQ192 codes for 100M rows is
~35 GB, which fits in 128 GB, so there is no reason to pay 20 disk reads per query and a
Vamana rebuild for it. B becomes the right answer somewhere above 500M rows.

**C, with two ideas borrowed from B:**

1. **The rerank rides along; it is not a second round.** The vectors a rerank reads are laid
   out by row and fetched as one batch with prefetch, not as R independent random reads.
2. **Quantization exists to remove R̄ from the I/O equation, not to save memory.** That makes
   resident PQ codes a hard requirement — codes are never paged out.

### The residency budget, 100M rows × 768 dimensions, M=16

| Resident | Bytes/row | Total |
| --- | --- | --- |
| level-0 adjacency (flat `[u32; 32]`) | 128 | 12.8 GB |
| upper-level adjacency (~1/16 of nodes) | ~8 | 0.8 GB |
| PQ192 codes | 192 | 19.2 GB |
| id ↔ node number | ~20 | 2.0 GB |
| level, tombstone | 2 | 0.2 GB |
| **total** | **~350** | **~35 GB** |
| on disk: full vectors | 3,072 | 307 GB |

This table was *computed*, before the compact graph existed. The measured figure at 100M is
**447 bytes/row**; the extra ~100 comes from the id map, because 100 million string ids are
1–2 GB of bytes however they are stored. Both numbers are in §4.

---

## 2. Why recall@100

Published vector-search benchmarks report `recall@100`. Earlier probes in this project used
`recall@10` with `ef ≤ 512`, and **those are not the same quantity** — a `recall@10` figure
cannot be compared with a published `recall@100` one. Everything here is `recall@100`, and the
server's `accurate` mode (`ef` = 1024) is the setting that corresponds to it.

Three facts fell out of aligning the two, and each one shaped the defaults.

**Code length decides how wide the rerank pool has to be.** 1M rows, full code scan plus exact
rerank of R candidates:

| codes | R=200 | R=500 | R=1000 | R=2000 |
| --- | --- | --- | --- | --- |
| PQ48 | 0.6793 | 0.8377 | 0.9131 | 0.9578 |
| PQ96 | 0.8605 | 0.9677 | 0.9908 | 0.9979 |
| PQ192 | 0.9908 | 0.9999 | 1.0000 | 1.0000 |

**PQ192 plus a 500-candidate rerank matches an exact graph at the same `ef`.** 1M rows:

| ef | exact HNSW | PQ192 walk + R=200 | + R=500 |
| --- | --- | --- | --- |
| 512 | 0.9785 | 0.9717 | 0.9775 |
| 1024 | 0.9895 | 0.9816 | **0.9894** |
| 2048 | 0.9944 | 0.9857 | **0.9945** |

At 10M the gap is 0.06–0.07 of a point (0.9795 against 0.9801 at `ef` = 1024). PQ96 needs
R≈1000 to reach the same place, which means twice the `ef` and 2–5× the disk reads. **PQ192 is
the default**, at 96 bytes/row more than PQ96 — 9.6 GB at 100M.

**Quantization is close to free; the traversal budget is what binds.** On one 10M table the
unquantized exact graph at `ef` = 512 scores 0.9652 and the quantized graph with a 500-candidate
rerank scores 0.9628 — 0.24 of a point. The ceiling is `ef`, not code precision. §5 is what
follows from that.

---

## 3. The layout

Three things are resident and two are on disk. With a WAL directory configured (the default),
a table's files sit beside it:

```
{DXMT_WAL_DIR}/
  vectors/<table>.f32     full vectors, fixed-size rows, append-only  (disk)
  records/<table>.rec     records, variable length, MessagePack        (disk)
  fulltext/<table_id>/    Tantivy index, when text-bm25 is built       (disk)
  snapshot/               table metadata + index files                 (disk)
```

resident: the HNSW graph (~218 bytes/row), the PQ codes (192 bytes/row at PQ192), and a
`RowStore` holding the only copy of each id plus a tombstone bitmap and the record offsets
(~45 bytes/row).

**A row number is a record's identity inside its table, and it is never reused.** An update
appends a new vector row and a new record row and repoints the id; a delete sets a tombstone.
Dead rows are compacted at the next rebuild. This is what lets the record map — which used to
be a `HashMap<String, Arc<Record>>` next to a second `HashMap<String, u64>`, each holding its
own copy of the id string — disappear from memory entirely.

**Reads take one of two paths, chosen at run time.** A rerank needs ~500 rows at once. When the
pages are warm, `mmap` with a zero-copy borrow of the mapped slice is fastest. When they are
cold, `mmap` is the wrong tool: page faults are serial, and a cold `mmap` rerank measured
p99 = 80 ms against 7.5 ms for the same reads issued in parallel through a `pread` pool. The
engine watches `getrusage` for major faults on a few probe rows and switches; `DXMT_VECTOR_IO`
pins one path for measurement.

**Half-precision rows are optional.** `DXMT_VECTOR_F16=1` stores 2 bytes per dimension instead
of 4. Recall moves by 0.03–0.08 of a point, which is inside the noise, and throughput by 3–5%,
which is the decode. It halves the vector file — 307 GB to 154 GB at 100M, doubling what the
page cache can cover. It is not the default because a caller who reads a vector back and
expects it to 1e-6 should not silently get f16.

**Environment variables.** `DXMT_RECORD_STORE=memory` and `DXMT_VECTOR_STORE=memory` put a
*new* table back in memory; a table that already has row files keeps them whatever the
environment says. Without a WAL directory — embedded use, and most tests — everything is in
memory. (These were `DB_*` when the phases below were run. The `DB_` spelling is still accepted
and warns once; see `src/config/env.rs` for why it was not a namespace.)

---

## 4. What each phase cost and bought

Each phase had its acceptance criteria written down **before** it was measured.

### Phase 0 — make `HnswPqIndex` mean what it says

`HnswIndex::quantize(pq)` encodes nodes to PQ codes and drops the node vectors; the walk uses
an ADC table. Reranking moved down into the storage layer as
`VectorIndex::rerank_candidates(k)` — `max(5k, 100)` — so all four search paths, filtered
included, share it.

The first run was a **false pass**: streaming writes had built an exact graph, and
`rebuild_index` judged the replacement equivalent and skipped the rebuild that would have
quantized it, so memory was identical to plain HNSW. A table now quantizes itself in place once
it passes `DEFAULT_QUANTIZE_AT_ROWS` (65,536).

| 1M rows | HNSW | HNSW-PQ192 + rerank 500 |
| --- | --- | --- |
| resident | 8,295 MB (8.3 KB/row) | **4,506 MB (4.5 KB/row)** |
| accurate recall / QPS | 0.9915 / 1,534 | 0.9922 / 2,173 |

### Phase 1 — a compact graph and a SIMD ADC kernel

The old node layout cost 616–785 bytes. `hnsw-v4` is a flat `[u32; 32]` adjacency array plus a
count byte, 1,024 striped locks instead of one per node, and an id byte-arena with offsets:
**218 bytes/row**, counted in the containers rather than inferred from RSS.

The ADC kernel's cost turned out not to be the table lookups but **192 serial floating-point
adds**. Eight independent accumulators plus an AVX2 gather inverted the relationship:

| ef | exact walk | quantized walk | ratio |
| --- | --- | --- | --- |
| 512 | 4.47 ms | 3.11 ms | 0.70 |
| 1024 | 8.40 ms | 5.19 ms | **0.62** |
| 2048 | 15.32 ms | 8.97 ms | 0.59 |

**The quantized walk is 1.4–1.7× faster than the exact one**, at equal recall, because a node
carries 192 bytes instead of a 3 KB vector and the traversal's cache footprint is 1/16.
The target had been "no more than 1.2× slower".

Two things were found on the way and are worth knowing. Quantizing 1M rows took 442 s because
the encode was serial — parallelising it took it to 22 s. And plain HNSW gained 1.55 KB/row
from the new layout, far more than the graph's own 474 bytes, because the old layout gave every
node its own 3 KB heap allocation.

### Phase 2 — full vectors on disk

A fixed-size row file plus a `.rows` manifest, mmap for reads, `fdatasync` after append, and a
refusal to open a file shorter than its manifest. The vector is written and synced *before* the
WAL entry and before the caller is acknowledged; a snapshot records the row count as a
watermark.

1M rows: resident **4,166 → 1,269 MB (1.27 KB/row)**, and with the zero-copy borrow in place
throughput came back to 97% of the in-memory build. On NVMe, a genuinely cold rerank (after
`drop_caches`) measured p50/p99/max = **6.2 / 7.5 / 11.6 ms**.

**The 10M run failed its recall target — 0.9267 against 0.979 — and the cause was one line.**
Not the build method: exact, decoded and streaming builds all scored 0.978–0.980. The storage
layer was constructing HNSW-PQ graphs with `ml = 1/ln 2` (1.44) where HNSW's own formula is
`1/ln m` (0.36), so about three quarters of nodes had an upper level instead of one sixteenth
and the greedy descent crossed far too many. Fixing it moved five numbers at once: load
2,037 → 1,421 s, resident 11.4 → 10.3 GB, normal recall 0.9097 → 0.9629, accurate
0.9267 → **0.9799**, matching the probe's 0.9800 exactly.

An unexpected result closed the phase: **vectors on disk were 7% faster than vectors in
memory** (3,937 against 3,687 QPS). The in-memory rerank fetched each candidate record and
cloned a 3 KB vector; the disk path borrows the mapped slice.

### Phase 3 — the record table on disk

After phase 2, of 1,262 MB resident at 1M rows, roughly 800 MB was the record side and almost
all of it was structure: a `Record` rounding to 256 bytes, an `Arc` header, two hash maps each
with its own copy of the id string, and a third copy inside the graph. The `RowStore` replaced
all of it with one id arena, one tombstone bitmap and one offset table.

| | 1M | 10M |
| --- | --- | --- |
| resident | 0.59 GB against 6.7 GB (**9%**) | 4.7 GB against 66.9 GB (**7%**) |
| accurate recall@100 | 0.9913 / 0.9908 | 0.9796 / 0.9812 |
| accurate QPS | 3,799 / 2,009 (**1.9×**) | 3,309 / 1,728 (**1.9×**) |
| snapshot | ~1 s | 7 s against 108–144 s |
| genuinely cold p99 | 7–9 ms | — |

Snapshots stopped serialising records, which is where the 108 s went.

**A 2.4 GB allocation nobody had looked for** showed up at 10M: the search's visited set was a
`u32` stamp array sized by node count, 40 MB per thread at 10M. At 100M that would have been
400 MB per thread — over 13 GB. It is an open-addressing hash set now, a few MB per thread
regardless of graph size.

**A filtered search on a disk-backed table took three attempts to fix**, and the first two
targeted the wrong thing. A 1%-selectivity filter at 1M rows ran at 43 QPS, p99 480 ms. The
first fix cut the rerank from 30,000 row reads to ~300 and changed nothing measurable: the cost
was not the rerank but **the walk that had been asked for 30,000 candidates**, maintaining a
30,000-entry heap. The real fix is `VectorIndex::scan_ids` — score a given set of ids directly,
via ADC on a quantized index, and skip the graph entirely when selectivity is under 5% and the
match set is at most 100,000 rows. **Recall 1.000, 1,293 QPS, p99 19.5 ms.** Recall is exactly 1
because that path is exhaustive over the match set.

### 100M rows

768 dimensions, L2, f16 rows, on a 32 vCPU / 121 GB cloud instance with local NVMe.

| | measured |
| --- | --- |
| ingest | 13,500 s (3.75 h, 7,407 rows/s) |
| resident | **44.7 GB — 447 bytes/row**; peak 56.4 GB |
| default mode | recall@100 0.9735, 4,886 QPS, serial p99 5.7 ms |
| accurate mode | **0.9825**, 3,546 QPS, p99 8.8 ms |
| warm p99 | 10.7 ms |
| cold p99, first 1,000 queries after `drop_caches` | 282 ms |
| snapshot | 54.3 s, writing a 36.4 GB index file |
| on disk | 153.6 GB vectors (f16) + 8.6 GB records |

**The cold tail is specific to this size** and is not a defect to be fixed so much as a fact to
plan around: the vector file is 154 GB against ~75 GB of available page cache, so the cache can
never hold more than half of it. Steady state is the 10.7 ms warm figure.

---

## 5. Two defects these runs found

Both are fixed. They are here because the *shape* of each is worth recognising again.

### `ef` was a constant while the graph it walked was not

A default-mode search scored 0.9798 at 1M and 0.9628 at 10M. Two explanations were checked and
**both were wrong**, which is the reason this is written down:

- *"`ef` does not scale with `k`."* It did, invisibly: `search_reranked` raises `ef` to the
  rerank pool width, so every `k` = 100 query was already walking at `ef` = 500, not the 256 the
  constant said. Changing the constant to `max(configured, 5k)` computed the same number, and
  the commit was reverted.
- *"The probe and the server disagree."* They did not. The claimed 1.6-point gap came from
  reading a **1M** row out of a table and comparing it against a **10M** server figure. Line by
  line, probe and server match to three decimals at both sizes.

The actual cause: **each 10× in rows needs roughly 2× the traversal budget** to hold the same
recall. `ef_search` now doubles per decade of rows, read at *search* time rather than at create
time — a table's size when it was created is an estimate, its size when it is searched is a
fact — capped at 8×. `DEFAULT_EF_SEARCH` also went from 256 to 512, which changed nothing that
runs: the rerank pool was already forcing 500, so the constant had simply been lying.

The work point was written down before the run and hit to within 0.0006 of recall and 0.03% of
QPS: 10M default mode went 0.9628 → **0.9797**, landing exactly on what `accurate` scored
before the change.

### A filter that could not match anything took 4,600× longer than no filter at all

On one 5M × 1536 table, same query vector:

| | time | hits |
| --- | --- | --- |
| no filter | 3.6 ms | 100 |
| filter on a field no record has | **16.6 s** | 0 |
| filter on a field every record has | 5.7 ms | 10 |

It held the table's read lock for 4.6 s while doing it, and the lock is write-preferring, so
every request behind it queued: 0.061 QPS at concurrency 1.

The cause was not the estimate, it was **the post-filter escalation loop having no bound**. On
a miss it refetched 8× as many candidates, three times, and the last attempt set the candidate
count to the whole table — which `search_reranked` then multiplied by the rerank factor. On a
5M-row table the final attempt asked the graph for 25 million results with `ef` = 25 million,
then exactly reranked 25 million candidates. At `ef` = n the graph contributes nothing; the walk
degenerates into a full table scan carrying an n-sized heap, which is *worse* than brute force.

Fixing it took three changes together, and the first two alone moved 120 ms to 54 ms:
clamp any candidate count to the index's own row count; test the escalation against the value
*after* the rerank factor is applied; and enumerate by the predicate instead of by the graph
when the amplified value would already cover the table. The condition that triggers that branch
is exactly the condition that makes the match set small, so the exhaustive path is cheap
precisely when it is used.

**What makes this worth its own section is not that filtering was slow. It is that filtering
had no bound when it failed.** `tests/filters/filter_escalation_test.rs` reproduces it in 20,000 rows.

### A related correction, kept because the correction is the lesson

A low-selectivity filter case was recorded at 26.8 QPS / p99 2,500 ms and written into the
roadmap as this engine's weakest point. **That was wrong.** The row came from a stale binary
that was no longer on the machine, started by a script that — alone among the benchmark
scripts — left the background snapshot and eviction sweeps on. Two variables differed at once.
Separating them gave 2,192 QPS / 11.1 ms on the current binary with sweeps off, and 2,197 QPS /
12.0 ms **with sweeps on**: the background sweeps cost nothing, and the 82× was entirely the
binary.

The lesson is about the results table, not about filtering. A results file that mixes rows from
several binaries and several environments, with none of that in the row name, makes "compare
against the baseline" an unreliable operation.

---

## 6. Where the time goes, and three optimisations that are not worth doing

Sweeping code length from PQ192 down to PQ48 on one 1M table, `ef` = 512, single-threaded:

| codes | table | ms | recall |
| --- | --- | --- | --- |
| PQ192 | 192 KB | 2.077 | 0.9779 |
| PQ96 | 96 KB | 1.488 | 0.9528 |
| PQ48 | 48 KB | 1.089 | 0.8470 |

Three points fit a line: **0.76 ms fixed (37%), 1.32 ms of distance computation (63%)**. Quarter
the code length and the time falls to 0.52, not 0.25, because the fixed part — adjacency
access, the candidate heap, the visited set — does not depend on code length.

That one number rules out three things without further experiments:

- **Wider SIMD gathers** halve the distance half at best: ~25–30% off the walk, ~20% off a
  query.
- **int8 scalar quantization for the walk** cannot beat 0.76 ms however fast its distances are —
  an upper bound of 2.7× — and costs 768 bytes/row against 192, which is **4× the residency**,
  76.8 GB against 19.2 GB at 100M. Whatever a microbenchmark says, it cannot be the default. It
  was never run, because the conclusion does not depend on the result.
- **Sharding a single query across segments** is not parallelism over one HNSW walk, which is
  inherently serial; it is several independent indexes plus a merge, multiplying total CPU per
  query by N and touching build, insert routing, deletes, snapshots, recovery, filtering and the
  row files.

The 37% is the target for the next round: no extra memory, no recall traded away, and no
conflict with fitting 100M rows into 128 GB.

---

## 7. One index cost 250 bytes per row to store an i64

The weakest point in a filter matrix was a 1% filter over 10M rows, 108.8 QPS / p99 236 ms. The
obvious reading — cost is linear in the match set — could not be distinguished from the truth
by the benchmark, because the match set was always 1% of the table, so the two grew together.
The client's payload was `{"id_num": <primary key>}`, which on a 10M table gives the field
**10 million distinct values**, and the range implementation walked every distinct value of the
field on every query. At ~15 ns per random access that predicts 150 ms at 10M, against 236 ms
measured.

Memory said the same thing: 4,659 MB without the payload, 7,149 MB with one integer field —
**2.5 GB, 54% of the index, for one i64 per row.** The structure was a
`HashMap<ValueKey, HashSet<RecordId>>`: ten million single-element hash sets, each holding a
`String`. It is a sorted array of 12-byte entries now, about 364 MB — one seventh.

---

## Reading this against the code

| this document | in the tree |
| --- | --- |
| route C, the resident half | `src/search/hnsw_pq.rs` |
| the compact graph, `hnsw-v4` | `src/search/hnsw.rs` |
| phase 2, the vector row file | `src/storage/vector_file.rs` |
| phase 3, the record row file | `src/storage/record_store.rs` |
| rerank, `ef` scaling, the filter paths | `src/storage/intelligent_indexed_storage.rs` |
| `rerank_candidates`, `scan_ids` | `src/search/index_traits.rs` |
| the escalation defect, reproduced | `tests/filters/filter_escalation_test.rs` |
| a filter on a field no record has | `tests/filters/filter_absent_field_test.rs` |

`docs/SUPPORT_MATRIX.md` is the separate question of which of this is safe to depend on.
Recall figures measured against published ground truth, on nine `ann-benchmarks` datasets this
project did not compute the answers for, are in the same file.
