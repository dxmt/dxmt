"""What `PUT /records/{id}` keeps and what it replaces.

"Change what was given, keep what was not" — both ways. A payload-only update keeps the vector,
and a vector-only update keeps the payload. The second did not hold: `{"vector": [...]}` in the
default replace mode replaced the payload with nothing, so re-embedding a table with a new model
erased every record's metadata while answering 200. The HTTP handler and gRPC each had their own
copy of the decision; there is one now, `apply_update`, and this pins it from the outside.
"""

import json

from .helpers.client import get, put


def payload_of(table, rid):
    return json.loads(get(f"/tables/{table}/records/{rid}").json()["data"]["payload_json"])


def vector_of(table, rid):
    return get(f"/tables/{table}/records/{rid}").json()["data"]["vector"]


def test_a_vector_only_update_keeps_the_payload(fresh_table):
    before = payload_of(fresh_table, "r1")
    assert before, "the corpus record should carry a payload for this to mean anything"
    put(f"/tables/{fresh_table}/records/r1", {"vector": [0.0, 0.0, 1.0, 0.0]})
    assert vector_of(fresh_table, "r1") == [0.0, 0.0, 1.0, 0.0]
    assert payload_of(fresh_table, "r1") == before


def test_an_empty_update_changes_nothing(fresh_table):
    before = (payload_of(fresh_table, "r2"), vector_of(fresh_table, "r2"))
    put(f"/tables/{fresh_table}/records/r2", {})
    assert (payload_of(fresh_table, "r2"), vector_of(fresh_table, "r2")) == before


def test_a_payload_only_update_keeps_the_vector_and_replaces_the_payload(fresh_table):
    vector = vector_of(fresh_table, "r3")
    put(f"/tables/{fresh_table}/records/r3", {"payload": {"only": 1}})
    assert payload_of(fresh_table, "r3") == {"only": 1}
    assert vector_of(fresh_table, "r3") == vector


def test_merge_adds_and_overwrites_keys_and_keeps_the_rest(fresh_table):
    before = payload_of(fresh_table, "r4")
    put(f"/tables/{fresh_table}/records/r4", {"payload": {"city": "Tokyo", "new": True}, "merge_mode": "merge"})
    assert payload_of(fresh_table, "r4") == {**before, "city": "Tokyo", "new": True}


def test_an_empty_payload_is_how_to_clear_one(fresh_table):
    put(f"/tables/{fresh_table}/records/r5", {"payload": {}})
    assert payload_of(fresh_table, "r5") == {}
