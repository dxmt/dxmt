"""Diversity reranking on a search.

A chunked corpus answers a query with several near-copies of one passage. All of them really
are the nearest neighbours, and together they say one thing; the fifth-nearest passage from
somewhere else is usually worth more to a reader than the fourth copy of the first.

The corpus here is built so a plain top-3 is one cluster and a diversified one is not: cluster A
sits 10 degrees from the query and owns the ranking, cluster B sits 40 degrees away in a
different plane -- further from the query than A, and much further from A than A's copies are
from each other. That last part is what makes this a test of MMR rather than of distance: if
every candidate were equally far from every other, there would be nothing to diversify.
"""

import math
import uuid

import pytest

from .helpers.client import delete, post


@pytest.fixture(scope="module")
def clustered():
    name = f"mmr_{uuid.uuid4().hex[:8]}"
    post("/tables", {"name": name, "dimension": 3})
    records = []
    for i in range(4):
        t = math.radians(10) + i * 0.004
        records.append({"id": f"A{i}", "vector": [math.cos(t), math.sin(t), 0.0],
                        "payload": {"c": "A"}})
    for i in range(4):
        t = math.radians(40) + i * 0.004
        records.append({"id": f"B{i}", "vector": [math.cos(t), 0.0, math.sin(t)],
                        "payload": {"c": "B"}})
    post(f"/tables/{name}/records", {"records": records})
    yield name
    delete(f"/tables/{name}", expect_ok=False)


QUERY = [1.0, 0.0, 0.0]


def ids(table, **body):
    body.setdefault("vector", QUERY)
    hits = post(f"/tables/{table}/search", body).json()["data"]["hits"]
    return [h["id"] for h in hits]


def test_a_plain_search_returns_one_cluster_and_a_diversified_one_does_not(clustered):
    plain = ids(clustered, top_k=3)
    assert plain == ["A0", "A1", "A2"], plain
    diverse = ids(clustered, top_k=3, mmr={"lambda_mult": 0.5, "fetch_k": 8})
    assert diverse[0] == "A0", diverse
    assert "B0" in diverse, f"nothing from the other cluster: {diverse}"


def test_lambda_one_is_the_plain_ranking(clustered):
    # The parameter has to reach the arithmetic: at 1.0 the redundancy term is multiplied by
    # zero, so this must be the ranking with no reranking at all.
    assert ids(clustered, top_k=3, mmr={"lambda_mult": 1.0, "fetch_k": 8}) == ids(
        clustered, top_k=3
    )


def test_pages_of_a_diversified_search_tile(clustered):
    whole = ids(clustered, top_k=4, mmr={"lambda_mult": 0.5, "fetch_k": 8})
    first = ids(clustered, top_k=2, mmr={"lambda_mult": 0.5, "fetch_k": 8})
    second = ids(clustered, top_k=2, offset=2, mmr={"lambda_mult": 0.5, "fetch_k": 8})
    # The offset applies to the reranked order, not the ranking underneath it -- otherwise page
    # two of a diversified search would be a page of something else.
    assert first + second == whole, (first, second, whole)


def test_diversity_can_only_come_from_candidates_that_were_fetched(clustered):
    # `fetch_k` is the cost knob and the honest limit: with only three candidates there is
    # nothing from the other cluster to promote, and the answer says so by being the plain one
    # rather than by pretending.
    assert ids(clustered, top_k=3, mmr={"lambda_mult": 0.5, "fetch_k": 3}) == ["A0", "A1", "A2"]


def test_it_composes_with_a_filter(clustered):
    only_b = {"must": [{"key": "c", "operation": {"type": "Equal", "value": "B"}}]}
    got = ids(clustered, top_k=2, filter=only_b, mmr={"lambda_mult": 0.5, "fetch_k": 8})
    assert all(i.startswith("B") for i in got), got
