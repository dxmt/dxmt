"""The fused score, recomputed here from each arm run on its own.

`tests/api/test_full_text.py` checks that hybrid search answers, that `vector_weight` reaches
the right fusion methods and that a misspelled knob is refused. What it does not check is the
arithmetic: whether the number the server calls a fused score is the fusion of the numbers the
two arms produced. Its expectations are hand-written, and a hand-written expectation for a
score is copied from the implementation, so it agrees with the implementation by construction.

LanceDB's reranker tests have the idea this borrows. They run a hybrid query, then run the
vector arm and the text arm separately, align on row id and assert the fused result carries
*exactly* the per-arm scores it was built from. That works because their hybrid response
carries `_distance` and `_score` alongside `_relevance_score`; ours carries one number, so the
identity check is not available. The stronger version is available instead: run each arm
alone, apply the documented fusion here, and compare. That validates the arithmetic and not
only the passing-through of it.

Each formula below is written from its definition, deliberately not transcribed from
`intelligent_indexed_storage.rs`. A test that mirrors the code line by line proves the code
does what it does.

    dbsf   each arm's scores mapped to [0,1] by clipping to mean ± 3σ, then summed
    rrf    weight / (k + rank), rank 1-based per arm, text weight = 1 - vector_weight
    wsum   each arm min-max normalised, then weighted; an arm whose scores span less
           than 0.01 contributes 1.0 for every member

**The arms have to be asked for the rows the fusion saw, not for `limit` of them.** The hybrid
path fetches `limit * 3` per arm without a filter (`limit * 10` with one), fuses, and cuts the
page afterwards -- so the fused top ten legitimately contains rows outside either arm's top
ten. Asking each arm for `limit` made the first run of this file report five "invented" ids
that were nothing of the sort. Every request here is unfiltered and the arms are asked for
`FETCH` = `LIMIT * 3`, which is the number that has to track
`IntelligentIndexedStorage::hybrid_search`.
"""

import math

import pytest

from .helpers.client import post

pytestmark = pytest.mark.skipif(
    not __import__("os").environ.get("DXMT_FEATURES", "").count("text-bm25"),
    reason="hybrid fusion needs the text arm, which is the text-bm25 feature",
)

DIM = 8
LIMIT = 10
#: What each arm is asked for, mirroring `hybrid_search`'s unfiltered `request.limit * 3`.
FETCH = LIMIT * 3
#: Enough rows that each arm returns a spread of scores -- a fusion over three identical
#: scores is satisfied by almost any arithmetic.
ROWS = 40
TOL = 1e-4


def a_vector(i):
    """Distinct directions, so the vector arm produces a real ranking rather than ties."""
    return [round(math.sin(i * 0.7 + d * 1.3), 4) for d in range(DIM)]


WORDS = ["alpha", "beta", "gamma", "delta", "epsilon"]


@pytest.fixture(scope="module")
def fused_table():
    import uuid

    name = f"api_fuse_{uuid.uuid4().hex[:8]}"
    post("/tables", {
        "name": name,
        "dimension": DIM,
        "distance_metric": "cosine",
        "enable_full_text_search": True,
        # The indexed field is named separately from the payload key it is fed from; naming
        # both the same finds nothing, as `test_full_text.py` records.
        "text_fields": [{"name": "content", "source": {"metadata_field": "body"},
                         "analyzer": "standard", "boost": 1.0}],
    })
    records = [
        {
            "id": f"r{i}",
            "vector": a_vector(i),
            # Every row shares one word so the text arm returns a spread rather than a
            # handful, and differs in the rest so the ranking is not flat.
            "payload": {"body": f"alpha {WORDS[i % len(WORDS)]} row {i} {'beta' * (i % 3)}"},
        }
        for i in range(ROWS)
    ]
    post(f"/tables/{name}/records", {"records": records})
    yield name
    post(f"/tables/{name}", None, expect_ok=False)


QUERY_VECTOR = a_vector(3)
QUERY_TEXT = "alpha beta"


def arm_vector(table, limit=FETCH):
    hits = post(f"/tables/{table}/search",
                {"vector": QUERY_VECTOR, "top_k": limit}).json()["data"]["hits"]
    return [(h["id"], h["score"]) for h in hits]


def arm_text(table, limit=FETCH):
    hits = post(f"/tables/{table}/search/text",
                {"query": QUERY_TEXT, "limit": limit}).json()["data"]["hits"]
    return [(h["id"], h["score"]) for h in hits]


def hybrid(table, method, **kw):
    body = {"vector": QUERY_VECTOR, "text": QUERY_TEXT, "limit": LIMIT,
            "fusion_method": method, **kw}
    hits = post(f"/tables/{table}/search/hybrid", body).json()["data"]["hits"]
    return {h["id"]: h["score"] for h in hits}


# ── the three formulas, from their definitions ──────────────────────────────────────────────


def dbsf_norm(arm):
    """Clip to mean ± 3σ and map that window onto [0, 1]."""
    if not arm:
        return {}
    scores = [s for _, s in arm]
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    std = max(math.sqrt(var), 1e-8)
    low, high = mean - 3 * std, mean + 3 * std
    span = max(high - low, 1e-8)
    return {i: (min(max(s, low), high) - low) / span for i, s in arm}


def fuse_dbsf(vector, text):
    out = {}
    for part in (dbsf_norm(vector), dbsf_norm(text)):
        for i, s in part.items():
            out[i] = out.get(i, 0.0) + s
    return out


def fuse_rrf(vector, text, vector_weight, k):
    out = {}
    for arm, weight in ((vector, vector_weight), (text, 1.0 - vector_weight)):
        for rank, (i, _) in enumerate(arm, 1):
            out[i] = out.get(i, 0.0) + weight / (k + rank)
    return out


def minmax(arm, weight):
    if not arm:
        return {}
    scores = [s for _, s in arm]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    # An arm whose scores are all but identical carries no ranking information, and dividing
    # by that span would amplify float noise into a ranking. Everyone gets 1.0.
    return {i: (1.0 if span < 0.01 else (s - lo) / span) * weight for i, s in arm}


def fuse_weighted_sum(vector, text, vector_weight):
    out = dict(minmax(vector, vector_weight))
    for i, s in minmax(text, 1.0 - vector_weight).items():
        out[i] = out.get(i, 0.0) + s
    return out


# ── the tests ───────────────────────────────────────────────────────────────────────────────


def compare(name, got, want):
    """Every id the server returned must carry the score the arms imply.

    Only the ids the server returned are checked, not the whole fused set: the page is the top
    `LIMIT` of a fusion over `FETCH` rows each way, and which rows fall outside the page is a
    tie-break question rather than an arithmetic one.
    """
    assert got, f"{name}: the fusion returned nothing, so this compares two empty sets"
    missing = [i for i in got if i not in want]
    assert not missing, (
        f"{name}: the fusion returned {missing}, which neither arm did -- a fused result can "
        f"only contain ids one of its arms produced"
    )
    wrong = {i: (got[i], want[i]) for i in got if abs(got[i] - want[i]) > TOL}
    assert not wrong, (
        f"{name}: the fused score is not the fusion of the arms' scores.\n"
        + "\n".join(f"  {i}: server {g:.6f}, recomputed {w:.6f}" for i, (g, w) in
                    sorted(wrong.items())[:8])
    )


def test_dbsf_is_the_sum_of_each_arm_clipped_to_three_sigma(fused_table):
    want = fuse_dbsf(arm_vector(fused_table), arm_text(fused_table))
    compare("dbsf", hybrid(fused_table, "dbsf"), want)


@pytest.mark.parametrize("weight", [0.0, 0.3, 0.7, 1.0])
def test_rrf_weights_each_arm_by_rank(fused_table, weight):
    """`vector_weight` is read by RRF, which is easy to miss -- it reads as a `weighted_sum`
    knob and is not. At 0.0 and 1.0 one arm is switched off entirely, which is the case a
    weight that was quietly ignored would still pass."""
    want = fuse_rrf(arm_vector(fused_table), arm_text(fused_table), weight, 60.0)
    compare(f"rrf w={weight}", hybrid(fused_table, "reciprocal_rank_fusion",
                                      vector_weight=weight), want)


@pytest.mark.parametrize("k", [1.0, 60.0, 500.0])
def test_rrf_k_moves_the_scores_the_way_the_formula_says(fused_table, k):
    want = fuse_rrf(arm_vector(fused_table), arm_text(fused_table), 0.7, k)
    compare(f"rrf k={k}", hybrid(fused_table, "reciprocal_rank_fusion", rrf_k=k), want)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_weighted_sum_min_max_normalises_each_arm(fused_table, weight):
    want = fuse_weighted_sum(arm_vector(fused_table), arm_text(fused_table), weight)
    compare(f"weighted_sum w={weight}", hybrid(fused_table, "weighted_sum",
                                               vector_weight=weight), want)


def test_a_fused_id_came_from_an_arm_and_the_order_follows_the_score(fused_table):
    """Two properties no single-method test covers, on the default fusion.

    A fused result may only contain ids an arm produced -- a fusion that invents a row is a
    different bug from one that scores it wrong -- and the page must be sorted by the score it
    reports, which is what a caller reading the first hit depends on.
    """
    body = {"vector": QUERY_VECTOR, "text": QUERY_TEXT, "limit": LIMIT}
    hits = post(f"/tables/{fused_table}/search/hybrid", body).json()["data"]["hits"]
    from_arms = {i for i, _ in arm_vector(fused_table)} | {i for i, _ in arm_text(fused_table)}
    invented = [h["id"] for h in hits if h["id"] not in from_arms]
    assert not invented, f"the fusion returned ids no arm produced: {invented}"

    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), (
        f"the page is not ordered by the score it reports: {scores}"
    )
