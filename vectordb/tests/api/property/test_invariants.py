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

import json

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


# ---------------------------------------------------------------------------------------------
# A filtered search must agree with the filter.
#
# Three code paths answer one question here, chosen by the estimated selectivity: score the
# filter's matches exactly, walk the graph and keep what passes, or push the filter into the
# traversal. They are supposed to be three ways of computing the same set. Nothing checked that
# they were, and when the payload index disagreed with `Record::matches_filter` about
# `must_not`, `should` and an empty `must`, the same filter returned different answers from
# different endpoints for as long as it took someone to ask.
#
# So the expectation is computed here, from the records, without asking the server anything it
# could be wrong about: take every record, apply the filter in Python, score the survivors by
# hand, sort. Two properties come out of it, and the first is the one that holds for every mode:
#
#   * **Soundness** — nothing comes back that the filter excludes. A half-built index, a
#     traversal that forgot to re-check, an id set from the wrong field: all of them show up as
#     a hit that does not match.
#   * **Completeness** — `search_mode: "exact"` skips the graph, so its answer must *be* the
#     expected one. The approximate modes are allowed to miss, and are held to the same recall
#     threshold as the unfiltered case.
# ---------------------------------------------------------------------------------------------

#: Filters spanning the shapes and the selectivities that pick different paths, including the
#: three that were broken. `expected` is what each should select out of the generated corpus,
#: stated here so a filter that silently stops matching is a failure rather than a vacuous pass.
FILTERS = [
    ("one bucket", {"must": [{"key": "bucket", "operation": {"type": "Equal", "value": "b3"}}]}),
    ("two buckets", {"must": [{"key": "bucket", "operation": {"type": "In", "value": ["b1", "b5"]}}]}),
    ("half by range", {"must": [{"key": "n", "operation": {"type": "GreaterThanOrEqual", "value": N // 2}}]}),
    ("everything but one bucket", {"must_not": [{"key": "bucket", "operation": {"type": "Equal", "value": "b0"}}]}),
    ("a range minus a bucket", {
        "must": [{"key": "n", "operation": {"type": "LessThan", "value": N // 4}}],
        "must_not": [{"key": "bucket", "operation": {"type": "Equal", "value": "b2"}}],
    }),
    ("either of two buckets, as should", {"should": [
        {"key": "bucket", "operation": {"type": "Equal", "value": "b4"}},
        {"key": "bucket", "operation": {"type": "Equal", "value": "b6"}},
    ]}),
    ("a range narrowed by should", {
        "must": [{"key": "n", "operation": {"type": "GreaterThanOrEqual", "value": N // 2}}],
        "should": [{"key": "bucket", "operation": {"type": "Equal", "value": "b3"}}],
    }),
]


def _matches_condition(payload, condition):
    value = payload.get(condition["key"])
    operation = condition["operation"]
    wanted = operation["value"]
    kind = operation["type"]
    if kind == "Equal":
        return value == wanted
    if kind == "NotEqual":
        return value != wanted
    if kind == "In":
        return value in wanted
    if kind == "NotIn":
        return value not in wanted
    if value is None or isinstance(value, str) != isinstance(wanted, str):
        return False
    if kind == "GreaterThan":
        return value > wanted
    if kind == "GreaterThanOrEqual":
        return value >= wanted
    if kind == "LessThan":
        return value < wanted
    if kind == "LessThanOrEqual":
        return value <= wanted
    raise AssertionError(f"the test cannot evaluate {kind}; add it or drop the filter using it")


def matches(payload, filt):
    """`Record::matches_filter`, in Python. Deliberately a second implementation: agreeing with
    the server's own copy would prove only that the copy was consistent with itself."""
    for condition in filt.get("must") or []:
        if not _matches_condition(payload, condition):
            return False
    for condition in filt.get("must_not") or []:
        if _matches_condition(payload, condition):
            return False
    should = filt.get("should") or []
    if should and not any(_matches_condition(payload, c) for c in should):
        return False
    return True


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def corpus(big_table):
    """Every record with its vector and payload, read once."""
    out = {}
    offset = 0
    while True:
        data = client.post(
            f"/tables/{big_table}/records/list",
            {"limit": 1000, "offset": offset, "with_vector": True},
        ).json()["data"]
        records = data.get("records") or []
        if not records:
            break
        for record in records:
            payload = json.loads(record["payload_json"]) if record.get("payload_json") else {}
            out[record["id"]] = (record["vector"], payload)
        offset += len(records)
    assert len(out) == N, f"read {len(out)} records of {N}"
    return out


@pytest.mark.parametrize("name,filt", FILTERS, ids=[f[0] for f in FILTERS])
def test_a_filtered_search_agrees_with_the_filter(big_table, corpus, name, filt):
    query = vector(7)
    eligible = {rid: vec for rid, (vec, payload) in corpus.items() if matches(payload, filt)}
    assert eligible, f"{name!r} selects no records, so it proves nothing"
    assert len(eligible) < N or "must_not" not in filt, f"{name!r} selects everything"

    ranked = sorted(
        ((cosine(query, vec), rid) for rid, vec in eligible.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    expected = [rid for _, rid in ranked[:TOP_K]]
    boundary = ranked[min(TOP_K, len(ranked)) - 1][0]

    for mode in ("exact", "accurate", "normal"):
        body = {"vector": query, "top_k": TOP_K, "limit": TOP_K, "filter": filt,
                "search_mode": mode}
        hits = client.post(f"/tables/{big_table}/search", body).json()["data"]["hits"]
        got = [hit["id"] for hit in hits]

        # Soundness, and it holds for every mode: nothing the filter excludes may appear.
        unmatched = [rid for rid in got if rid not in eligible]
        assert not unmatched, (
            f"{name!r} in {mode} mode returned {unmatched}, which the filter excludes"
        )

        if mode == "exact":
            # No graph, no approximation: the answer is the answer. Ids are compared as a set
            # because two records at the same distance may come back in either order; a
            # difference is only allowed at the boundary score, where the tie actually is.
            missing = [rid for rid in expected if rid not in got]
            allowed = [
                rid for rid in missing
                if abs(cosine(query, corpus[rid][0]) - boundary) < 1e-6
            ]
            assert not set(missing) - set(allowed), (
                f"{name!r} in exact mode missed {sorted(set(missing) - set(allowed))}; "
                f"expected {expected}, got {got}"
            )
        else:
            recall = len(set(got) & set(expected)) / len(expected)
            assert recall >= MIN_RECALL, (
                f"{name!r} in {mode} mode recalled {recall:.2f} of the filtered ground truth"
            )
