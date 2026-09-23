"""`search_mode` over HTTP, and what each value promises.

Four of the five values are the same knob at different settings — they scale HNSW's traversal
budget — and one is not: `exact` skips the index and scores every live row. That difference is
the point. Without it, "what recall is my index giving me on my data" has no answer that does
not involve a benchmark dataset with published ground truth, which is by definition not your
data.

Two things are pinned here that are easy to get wrong and invisible when they break:

* **`exact` is ground truth**, so its answer is compared against the corpus order written down
  in `helpers/corpus.py` — not against another search.
* **The rejection shape for an unknown mode.** axum's `Json` extractor refuses an unknown enum
  variant before any handler runs, so the response is `422` with a **`text/plain`** body, not
  the `{"success": false, "error": {...}}` envelope every other failure uses. Any client that
  probes for a mode's existence reads that response, and one that assumes the envelope breaks
  on it. It is asserted here because nothing else documents it.
"""

import pytest

from .helpers.client import post, succeeded
from .helpers.corpus import BY_ID, CORPUS, ORDER_FROM_R1

MODES = ["fast", "normal", "accurate", "auto", "exact"]


def search(table, body):
    return post(f"/tables/{table}/search", body, expect_ok=False)


@pytest.mark.parametrize("mode", MODES)
def test_every_documented_mode_is_accepted(table, mode):
    response = search(table, {"vector": BY_ID["r1"]["vector"], "top_k": 3, "search_mode": mode})
    assert succeeded(response), f"search_mode {mode!r} was refused: {response.text[:300]}"
    assert len(response.json()["data"]["hits"]) == 3


def test_exact_returns_the_corpus_order_written_down_in_the_fixture(table):
    """The one mode whose answer can be checked against something other than another search."""
    hits = search(
        table, {"vector": BY_ID["r1"]["vector"], "top_k": 5, "search_mode": "exact"}
    ).json()["data"]["hits"]

    assert [h["id"] for h in hits] == ORDER_FROM_R1, (
        "exact search disagrees with the nearest-neighbour order the corpus was built to have"
    )
    assert abs(hits[0]["score"] - 1.0) < 1e-4, "r1 is its own nearest neighbour at similarity 1.0"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), f"hits are not ordered by score: {scores}"


def test_exact_agrees_with_itself_across_every_record(table):
    """Each record is its own nearest neighbour — eight independent checks that the scan is
    reading the row it thinks it is reading. An off-by-one in the row mapping passes a single
    spot check and fails this."""
    for record in CORPUS:
        hits = search(
            table, {"vector": record["vector"], "top_k": 1, "search_mode": "exact"}
        ).json()["data"]["hits"]
        assert hits[0]["id"] == record["id"], (
            f"querying with {record['id']}'s own vector returned {hits[0]['id']}"
        )


def test_exact_applies_a_filter_and_stays_exact(table):
    body = {
        "vector": BY_ID["r1"]["vector"],
        "top_k": 5,
        "search_mode": "exact",
        "filter": {"must": [{"key": "city", "operation": {"type": "Equal", "value": "Berlin"}}]},
    }
    hits = search(table, body).json()["data"]["hits"]
    assert [h["id"] for h in hits] == ["r5", "r6"], (
        "Berlin holds r5 and r6, and r5 is the nearer of the two to r1"
    )


def test_the_approximate_modes_can_be_measured_against_exact(table):
    """The comparison the whole mode exists to make, on a corpus small enough that every mode
    must be perfect. At eight records the index is exact anyway, so this asserts the two answers
    are comparable — same ids, same shape — rather than putting a number on the graph."""
    query = {"vector": BY_ID["r3"]["vector"], "top_k": 4}
    truth = [h["id"] for h in search(table, {**query, "search_mode": "exact"}).json()["data"]["hits"]]

    for mode in ["fast", "normal", "accurate", "auto"]:
        got = [h["id"] for h in search(table, {**query, "search_mode": mode}).json()["data"]["hits"]]
        overlap = len(set(truth) & set(got)) / len(truth)
        assert overlap == 1.0, (
            f"{mode} returned {got} against a ground truth of {truth}; at eight records every "
            f"mode should be exact, so a miss here is a bug rather than a recall trade-off"
        )


def test_an_unknown_mode_is_refused_in_the_error_envelope(table):
    """A capability probe -- "does this build have exact?" -- reads exactly this response.

    It used to be axum's own rejection: a 422 with a `text/plain` body, because the extractor
    refuses an unknown enum variant before the handler runs and there was no `ApiResponse` to
    wrap it. A probe doing `response.json()["error"]["code"]` raised instead of reporting the
    mode as absent. `ApiJson` now answers in the same envelope as everything else, and serde's
    message -- which lists the permitted values -- is the message.
    """
    response = search(table, {"vector": BY_ID["r1"]["vector"], "top_k": 1, "search_mode": "wat"})

    assert response.status_code == 400, (
        f"expected 400 from the body extractor, got {response.status_code}"
    )
    assert response.headers["content-type"].startswith("application/json"), (
        f"expected the JSON error envelope, got {response.headers.get('content-type')!r}"
    )
    error = response.json()["error"]
    assert error["code"] == "INVALID_INPUT", error
    body = error["message"]
    assert "search_mode" in body and "unknown variant" in body, (
        f"the message should name the field and the problem: {body[:300]}"
    )
    for mode in MODES:
        assert mode in body, (
            f"the rejection lists the permitted values, and {mode!r} is missing from it -- "
            f"a mode was added or removed without this test noticing: {body[:300]}"
        )



def test_exact_refuses_a_table_it_would_have_to_read_too_much_of(table):
    """The cap is what makes the mode safe to expose. It cannot be exercised here without a
    huge table, so this only checks that the corpus is comfortably under it and the query runs
    — the refusal itself is covered in `tests/exact_search_test.rs`, where the limit can be
    lowered."""
    response = search(table, {"vector": BY_ID["r1"]["vector"], "top_k": 1, "search_mode": "exact"})
    assert succeeded(response), (
        f"an eight-record table must be under any sane exact-scan cap: {response.text[:300]}"
    )


def test_named_vector_spaces_are_refused_rather_than_silently_approximated(table):
    """Scanning the default space and calling the answer exact would be worse than refusing."""
    response = search(
        table,
        {
            "vector": BY_ID["r1"]["vector"],
            "top_k": 1,
            "search_mode": "exact",
            "vector_name": "title",
        },
    )
    assert not succeeded(response)
    message = str(response.json().get("error", {}).get("message", response.text))
    assert "named vector" in message, f"the refusal should name the reason: {message[:300]}"
