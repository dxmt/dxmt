"""The same answers, from a table whose graph walks on codes instead of vectors.

**This is a deployment shape the rest of this suite never reaches.** An `HnswPqIndex` is an
exact HNSW graph until it crosses `quantize_at_rows`, which defaults to 65,536; every other
table in `tests/api/` holds eight records. So the codes, the ADC walk and the rerank — the
path a production table over 65,536 rows is on for its whole life — are compiled, shipped, and
not covered here at all. `GET /tables/{t}/stats` says `quantized: false` for every one of them.

The way in is `quantization.quantize_at_rows` at creation, which this table sets to 256.
`num_centroids` comes down with it: the default is 256 centroids per subspace, calibrated for
768-dimension corpora, and k-means with 256 centroids over 2,000 vectors puts most centroids on
one point each — codes that are a lossless id, which would make every assertion below pass for
the wrong reason. Four subspaces of 16 centroids is four bytes for a 16-dimension vector, a
deliberately coarse code with measurable loss in it.

`scripts/api_tests.sh` runs this in both storage modes like everything else here. A third
whole-suite mode was the obvious alternative and is not viable: the shared corpus is eight
records, and no threshold makes eight records train a codebook worth testing.

What is asserted is the part that has to survive quantization, and nothing else:

* the table really did quantize — checked first, because every other assertion here is
  vacuous if it did not, and a silent fallback to the exact graph is exactly how this file
  would rot into a slower copy of `test_search.py`
* an exact search is still exact. `search_mode: "exact"` never touches the graph, so
  quantization must not move it by even a float
* recall against that exact answer clears a floor
* a stored vector finds itself
* filters, updates and deletes answer the same as anywhere else, because the rerank reads the
  row file while the walk reads codes, and the two disagreeing is the failure this catches
"""

import math
import uuid

import pytest

from .helpers.client import delete, get, post, put

DIM = 16
ROWS = 2000
QUANTIZE_AT = 256
TOP_K = 10
QUERIES = 50
#: Floor for recall of the graph against the exact scan, over `QUERIES` queries at `TOP_K`.
#:
#: Measured before it was chosen, because a threshold nobody measured is either flaky or
#: vacuous. Six builds of this table on one machine: 0.9180, 0.9240, 0.9620, 0.9740, 0.9780,
#: 0.9840 — mean 0.9567, sigma 0.0261. The spread is the *build*, not the measurement: the
#: codebooks train on a random sample and HNSW assigns levels randomly, so each build is a
#: different index. 0.85 sits 2.6 sigma below the lowest of the six.
#:
#: `ROWS` is what makes this mean anything. The rerank pool is `max(5k, 100)` = 100 rows, so
#: at 600 rows it covers a sixth of the table and recall was 1.0000 every time — the pool was
#: answering, and the codes could have been anything. At 2,000 it is 5% and the codes start to
#: show; at 5,000 it is 2% and the same corpus scores 0.6650. 2,000 is the smallest size where
#: this file tests the thing it is named after, and it ingests in under two seconds.
#:
#: What the floor catches is a rerank that stopped happening or a code path that broke, not a
#: subtle recall regression -- 0.026 of per-build sigma is wider than any regression worth
#: arguing about at this scale. The cross-dataset numbers in `docs/SUPPORT_MATRIX.md` are
#: where recall is actually tracked.
MIN_RECALL = 0.85


def a_vector(i):
    """A point on one of eight clusters, so the corpus has structure for k-means to find.

    Uniform noise would give a codebook with nothing to say and a recall floor that measures
    the corpus rather than the index — three synthetic corpora in this project's history scored
    0.705, 0.535 and 0.505 unfiltered for exactly that reason.
    """
    centre = i % 8
    return [
        round(math.sin(centre * 1.7 + d * 0.9) + 0.08 * math.cos(i * 0.31 + d * 2.1), 4)
        for d in range(DIM)
    ]


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def quantized_table():
    name = f"api_pq_{uuid.uuid4().hex[:8]}"
    post(
        "/tables",
        {
            "name": name,
            "dimension": DIM,
            "distance_metric": "cosine",
            "quantization": {
                "num_subvectors": 4,
                "num_centroids": 16,
                "quantize_at_rows": QUANTIZE_AT,
            },
        },
    )
    records = [
        {
            "id": f"q{i}",
            "vector": a_vector(i),
            "payload": {"cluster": i % 8, "n": i, "tag": "even" if i % 2 == 0 else "odd"},
        }
        for i in range(ROWS)
    ]
    for start in range(0, ROWS, 500):
        post(f"/tables/{name}/records", {"records": records[start : start + 500]})
    yield name, {r["id"]: r for r in records}
    delete(f"/tables/{name}", expect_ok=False)


def exact_top(vectors, query, k):
    scored = sorted(((cosine(query, r["vector"]), rid) for rid, r in vectors.items()), reverse=True)
    return [rid for _, rid in scored[:k]]


def test_the_table_is_actually_quantized(quantized_table):
    """Every other assertion in this file is vacuous if this one fails.

    Quantization is background work: the codebooks train on a sample after the row threshold is
    crossed, so a table well past `quantize_at_rows` can still be an exact graph for a while.
    Reading `quantized` is the only way to know which engine answered -- an A/B of "quantized
    against not" at 300,000 rows in this project was really "not quantized against not
    quantized", and measured a recall difference that was not there.
    """
    name, _ = quantized_table
    stats = get(f"/tables/{name}/stats").json()["data"]
    assert stats["record_count"] == ROWS, f"the table holds {stats['record_count']} of {ROWS}"
    assert stats["quantized"] is True, (
        f"the table did not quantize at {ROWS} rows with quantize_at_rows={QUANTIZE_AT}; "
        f"stats say {stats}. Everything else in this file would pass against the exact graph."
    )


def test_an_exact_search_is_unmoved_by_quantization(quantized_table):
    """`search_mode: "exact"` scores rows, never codes, so it owes the same answer it always did."""
    name, vectors = quantized_table
    query = a_vector(3)
    hits = post(
        f"/tables/{name}/search", {"vector": query, "top_k": TOP_K, "search_mode": "exact"}
    ).json()["data"]["hits"]
    assert [h["id"] for h in hits] == exact_top(vectors, query, TOP_K)
    for h in hits:
        assert h["score"] == pytest.approx(cosine(query, vectors[h["id"]]["vector"]), abs=1e-5), (
            f"{h['id']} scored {h['score']} on the exact path, which does not read codes"
        )


def test_the_graph_on_codes_clears_the_recall_floor(quantized_table):
    name, vectors = quantized_table
    found = 0
    for i in range(QUERIES):
        query = [x + 0.05 for x in a_vector(i * 7)]
        hits = post(f"/tables/{name}/search", {"vector": query, "top_k": TOP_K}).json()["data"]["hits"]
        assert len(hits) == TOP_K, f"query {i} returned {len(hits)} of {TOP_K}"
        found += len(set(h["id"] for h in hits) & set(exact_top(vectors, query, TOP_K)))
    recall = found / (QUERIES * TOP_K)
    assert recall >= MIN_RECALL, f"recall@{TOP_K} on codes is {recall:.4f}, floor {MIN_RECALL}"


def test_every_stored_vector_finds_itself(quantized_table):
    """A record in the store that the graph cannot reach.

    Judged by distance and not by id: this corpus is eight clusters, so two records can hold
    near-identical vectors, and demanding the queried id back would count a correct tie as a
    miss. The measured version of that mistake reported 2.5% of GIST-960 unreachable where the
    truth was 2.0%.
    """
    name, vectors = quantized_table
    stranded = []
    for i in range(0, ROWS, 37):
        rid = f"q{i}"
        hits = post(f"/tables/{name}/search", {"vector": vectors[rid]["vector"], "top_k": 1}).json()["data"]["hits"]
        if not hits or hits[0]["score"] < 1.0 - 1e-3:
            stranded.append((rid, hits[0]["score"] if hits else None))
    assert not stranded, f"searching with a stored vector did not return a record at distance zero: {stranded}"


def test_a_filter_and_the_rerank_agree(quantized_table):
    """The walk reads codes and the rerank reads the row file; a filter crosses both."""
    name, vectors = quantized_table
    f = {"must": [{"key": "cluster", "operation": {"type": "Equal", "value": 3}}]}
    want = sorted(rid for rid, r in vectors.items() if r["payload"]["cluster"] == 3)

    listed = post(f"/tables/{name}/records/list", {"limit": 1000, "filter": f}).json()["data"]
    assert sorted(r["id"] for r in listed["records"]) == want
    assert listed["total"] == len(want)

    hits = post(f"/tables/{name}/search", {"vector": a_vector(3), "top_k": 5, "filter": f}).json()["data"]["hits"]
    assert hits, "a filter matching an eighth of the table returned nothing"
    assert all(h["id"] in want for h in hits), "the graph returned records the filter excludes"
    for h in hits:
        assert h["score"] == pytest.approx(cosine(a_vector(3), vectors[h["id"]]["vector"]), abs=1e-3), (
            f"{h['id']}'s score came from the codes, not from the rerank's row read"
        )


def test_an_update_and_a_delete_reach_the_codes(quantized_table):
    """A quantized graph holds a code, not a vector, so a write has to re-encode.

    The failure this catches is a write that lands in the row file and leaves a stale code in
    the graph: `records/list` would show the new vector and search would keep ranking by the
    old one.
    """
    name, vectors = quantized_table
    rid = "q11"
    moved = [round(-x, 4) for x in vectors[rid]["vector"]]
    put(f"/tables/{name}/records/{rid}", {"vector": moved})

    hits = post(f"/tables/{name}/search", {"vector": moved, "top_k": 1}).json()["data"]["hits"]
    assert hits[0]["score"] > 1.0 - 1e-3, (
        f"after moving {rid}, searching at its new position returned {hits[0]['id']} at "
        f"{hits[0]['score']} -- the graph is still ranking by the code it had before"
    )

    delete(f"/tables/{name}/records", body={"ids": [rid]})
    hits = post(f"/tables/{name}/search", {"vector": moved, "top_k": 5}).json()["data"]["hits"]
    assert rid not in [h["id"] for h in hits], f"{rid} is deleted and the graph still returns it"
    del vectors[rid]  # the module fixture is shared; keep it matching the server
