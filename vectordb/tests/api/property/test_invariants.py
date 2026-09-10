"""Invariants that must hold for any corpus, checked on a generated one.

Borrowed in shape from Chroma's `chromadb/test/property/invariants.py` (Apache 2.0 -- the list
of properties, not the code). Chroma generates data with hypothesis and asserts a fixed set of
invariants after every operation; this is the same idea with a fixed generator, which is enough
to catch the failures worth catching here and reproduces without a shrinker.

What `ann_accuracy` does and does not tell you, since a threshold invites over-reading:

* It compares the index's answer against a brute-force computed in the test, over the same
  corpus. At 2,000 rows the graph is **unquantized** -- `DEFAULT_QUANTIZE_AT_ROWS` is 65,536 --
  so recall sits near 1.0 and the 0.95 threshold is a tripwire for "the index, the rerank, or
  the on-disk vector reader returned the wrong neighbours at all".
* It is **not** evidence about quantization quality. That lever is `make bench-recall` and the
  VectorDBBench cases, which run at a scale where the codes actually exist.
* Raising the threshold to chase more signal would only make it flaky. The way to measure more
  is more rows, which belongs in the benchmark, not here.

Running it in both storage modes is the point rather than a bonus: `rerank_exact` reads vectors
from a mapped row file in disk mode and from the record in memory mode, and this is the only
invariant that would notice a broken row-file reader.
"""

import pytest

from ..helpers import client
from .conftest import DIM, N, vector

pytestmark = pytest.mark.slow

TOP_K = 10
MIN_RECALL = 0.95
QUERIES = 20


def list_page(table, limit, offset):
    data = client.post(
        f"/tables/{table}/records/list", {"limit": limit, "offset": offset}
    ).json()["data"]
    return data.get("records") or []


def test_count_matches_what_was_inserted(big_table):
    data = client.get(f"/tables/{big_table}").json()["data"]
    assert data["record_count"] == N


def test_every_id_comes_back_and_none_twice(big_table, expected_ids):
    """Set equality and duplicate-freedom asserted separately from paging order, so a failure
    says which of the two properties broke."""
    seen = []
    offset = 0
    while True:
        page = list_page(big_table, 1000, offset)
        if not page:
            break
        seen.extend(r["id"] for r in page)
        offset += len(page)
        if offset >= N:
            break

    assert len(seen) == len(set(seen)), "paging returned a record more than once"
    assert set(seen) == expected_ids


def test_paging_is_stable_across_repeated_calls(big_table):
    """The same page twice must be the same page.

    This is not hypothetical: the enumeration sorts by `created_at` descending over the record
    store, and in memory mode that store is a `HashMap`, so any records sharing a timestamp
    would be ordered by hash iteration and could differ between calls -- with the limit applied
    afterwards, that would change *which* records a page contains, not merely their order.
    Measured: at 2,000 records inserted in one request, every `created_at` is distinct, so
    there are no ties to break and the order is stable. This test is what tells us if that
    ceases to be true -- on a faster machine, or with a coarser timestamp.
    """
    first = [r["id"] for r in list_page(big_table, 200, 0)]
    for _ in range(3):
        assert [r["id"] for r in list_page(big_table, 200, 0)] == first


def test_payloads_round_trip(big_table):
    page = list_page(big_table, 50, 0)
    assert page
    for record in page:
        n = int(record["id"][1:])
        payload = record.get("payload")
        if payload is None:
            import json

            payload = json.loads(record["payload_json"])
        assert payload == {"n": n, "bucket": f"b{n % 7}"}


def test_a_record_is_its_own_nearest_neighbour(big_table):
    for i in (0, N // 3, N // 2, N - 1):
        hits = client.post(
            f"/tables/{big_table}/search", {"vector": vector(i), "top_k": 1}
        ).json()["data"]["hits"]
        assert [h["id"] for h in hits] == [f"p{i:06d}"]


def test_ann_accuracy_against_a_brute_force_computed_here(big_table):
    """The index's top-k against the exact top-k, over the same corpus.

    The brute force is computed in the test rather than taken from the server, so a bug that
    affects both would not cancel out.
    """
    corpus = [(f"p{i:06d}", vector(i)) for i in range(N)]

    found = 0
    total = 0
    for q in range(QUERIES):
        # Queries are corpus vectors perturbed in one coordinate, so they are in distribution
        # without being exact members.
        query = list(vector(q * 97 % N))
        query[q % DIM] += 0.05

        norm = sum(x * x for x in query) ** 0.5
        query = [x / norm for x in query]

        exact = sorted(
            ((sum(a * b for a, b in zip(query, v)), rid) for rid, v in corpus),
            reverse=True,
        )[:TOP_K]
        expected = {rid for _, rid in exact}

        hits = client.post(
            f"/tables/{big_table}/search", {"vector": query, "top_k": TOP_K}
        ).json()["data"]["hits"]
        got = {h["id"] for h in hits}

        found += len(expected & got)
        total += len(expected)

    recall = found / total
    assert recall >= MIN_RECALL, (
        f"recall@{TOP_K} was {recall:.4f} over {QUERIES} queries against a brute force "
        f"computed here; below {MIN_RECALL} means the index returned wrong neighbours, not "
        "that quantization lost precision -- at this size nothing is quantized"
    )
