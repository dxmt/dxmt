"""Table branching, and the health probes.

Branching was probed before this was written (2026-09-11, both storage modes and across a
restart). `POST /tables/{name}/branch` is a full copy -- `create_table`, then `insert_records` of
every row -- so a branch and its source are independent from the moment it returns, and the
branch survives a restart as an ordinary table. The same probe found a branch name checked by
nothing, and a response that said the new branch held no records.

Not asserted here, and where it is covered instead: `/ready` answering 503 needs durability to
degrade, which only an injected WAL sync failure produces
(`wal_sync_failure_enters_read_only_mode` in tests/wal_recovery_test.rs, feature
`test-failpoints`); and surviving a restart needs a server this suite does not own.
"""

import json

import pytest

from .helpers import client
from .helpers.client import delete, get, post, put, succeeded
from .helpers.corpus import BY_ID, CORPUS, DIM, ORDER_FROM_R1, ids_of, query_at

#: The probes live at the root, beside `/api/v1`, where an orchestrator looks for them.
ROOT = client.BASE.removesuffix("/api/v1")


def root_get(path):
    return client._HTTP.get(ROOT + path)


def payload_of(record):
    """`GET /records/{id}` carries the payload as a JSON string; see test_crud.py."""
    raw = record.get("payload_json")
    return json.loads(raw) if isinstance(raw, str) else record.get("payload")


def nearest(table, vector, k=5):
    body = {"vector": list(vector), "top_k": k}
    return ids_of(post(f"/tables/{table}/search", body).json()["data"]["hits"])


def record_count(table):
    return get(f"/tables/{table}/stats").json()["data"]["record_count"]


def table_names():
    return [t["name"] for t in get("/tables").json()["data"]["tables"]]


# ---- Health probes --------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/live", "/health/live"])
def test_liveness_answers_alive(path):
    response = root_get(path)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "alive"
    assert body["service"] == "vectordb"


@pytest.mark.parametrize("path", ["/ready", "/health/ready"])
def test_readiness_is_200_with_no_reason_while_durability_is_intact(path):
    response = root_get(path)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["reason"] is None
    assert body["storage"]["ready"] is True
    assert body["storage"]["durability_degraded"] is False


def test_health_and_status_agree_with_the_readiness_probe():
    ready = root_get("/ready").json()["storage"]["ready"]

    health = root_get("/health").json()["data"]
    assert health["ready"] is ready
    assert health["storage_healthy"] is ready
    assert health["accepting_requests"]["searches"] is True

    status = root_get("/status").json()["data"]
    assert status["status"] == ("ready" if ready else "not_ready")
    assert isinstance(status["version"], str) and status["version"]
    assert status["persistence"]["ready"] is ready


def test_metrics_add_up():
    metrics = root_get("/health/metrics").json()["data"]
    ongoing = metrics["ongoing_operations"]
    assert ongoing["total"] == (
        ongoing["inserts"] + ongoing["searches"] + ongoing["deletes"] + ongoing["table_operations"]
    )
    assert isinstance(metrics["completed_operations"], int)


@pytest.mark.parametrize("path", ["/health", "/health/live", "/health/ready", "/health/metrics"])
def test_the_probes_an_sdk_can_reach_are_mirrored_under_api_v1(path):
    # The Python and JavaScript SDKs put `/api/v1` in their base URL and call `/health` and
    # `/health/metrics` beneath it.
    assert get(path).status_code == 200


# ---- Branching ------------------------------------------------------------------------------


@pytest.fixture
def branch_of(fresh_table):
    """Branch `fresh_table` under a name derived from it; every branch made is dropped after."""
    made = []

    def branch(suffix="br"):
        name = f"{fresh_table}_{suffix}"
        data = post(f"/tables/{fresh_table}/branch", {"branch_name": name}).json()["data"]
        made.append(name)
        return name, data

    yield branch
    for name in made:
        delete(f"/tables/{name}", expect_ok=False)


def test_a_branch_holds_every_record_and_answers_as_its_source_does(fresh_table, branch_of):
    branch, data = branch_of()

    assert data["source"] == fresh_table
    assert data["branch"] == branch
    assert data["table"]["record_count"] == len(CORPUS), (
        "the response described the branch as it was before its records were copied in"
    )
    assert record_count(branch) == len(CORPUS)
    for record in CORPUS:
        stored = get(f"/tables/{branch}/records/{record['id']}").json()["data"]
        assert payload_of(stored) == record["payload"]
    assert nearest(branch, query_at("r1")) == nearest(fresh_table, query_at("r1")) == ORDER_FROM_R1


def test_a_write_to_the_branch_does_not_reach_the_source(fresh_table, branch_of):
    branch, _ = branch_of()
    post(f"/tables/{branch}/records", {"records": [{"id": "only_in_branch", "vector": [0.5] * DIM}]})

    assert succeeded(get(f"/tables/{branch}/records/only_in_branch", expect_ok=False))
    assert get(f"/tables/{fresh_table}/records/only_in_branch", expect_ok=False).status_code == 404
    assert record_count(fresh_table) == len(CORPUS)


def test_a_write_to_the_source_does_not_reach_the_branch(fresh_table, branch_of):
    branch, _ = branch_of()
    put(f"/tables/{fresh_table}/records/r2", {"payload": {"rank": 200}})
    post("/query", {"query": f"DELETE FROM {fresh_table} WHERE id = 'r8'"})

    assert payload_of(get(f"/tables/{branch}/records/r2").json()["data"]) == BY_ID["r2"]["payload"]
    assert succeeded(get(f"/tables/{branch}/records/r8", expect_ok=False))
    assert record_count(branch) == len(CORPUS)


def test_a_branch_is_the_source_as_it_was_when_the_branch_was_taken(fresh_table, branch_of):
    early, _ = branch_of("early")
    put(f"/tables/{fresh_table}/records/r2", {"payload": {"rank": 200}})
    late, _ = branch_of("late")

    assert payload_of(get(f"/tables/{early}/records/r2").json()["data"]) == BY_ID["r2"]["payload"]
    assert payload_of(get(f"/tables/{late}/records/r2").json()["data"]) == {"rank": 200}


def test_a_branch_outlives_its_source(fresh_table, branch_of):
    branch, _ = branch_of()
    delete(f"/tables/{fresh_table}")

    assert record_count(branch) == len(CORPUS)
    assert nearest(branch, query_at("r3"), k=1) == ["r3"]


def test_branching_an_empty_table_gives_an_empty_table(empty_table):
    name = f"{empty_table}_br"
    try:
        data = post(f"/tables/{empty_table}/branch", {"branch_name": name}).json()["data"]
        assert data["branch"] == name
        assert data["table"]["record_count"] == 0
        assert record_count(name) == 0
    finally:
        delete(f"/tables/{name}", expect_ok=False)


def test_branching_a_table_that_does_not_exist_is_a_404_and_creates_nothing():
    response = post(
        "/tables/definitely_not_a_table/branch", {"branch_name": "never_created"}, expect_ok=False
    )
    assert response.status_code == 404, response.text
    assert "never_created" not in table_names()


@pytest.mark.parametrize("target", ["another branch", "the source itself"])
def test_branching_onto_a_name_in_use_is_a_409(fresh_table, branch_of, target):
    name = branch_of()[0] if target == "another branch" else fresh_table
    response = post(f"/tables/{fresh_table}/branch", {"branch_name": name}, expect_ok=False)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "TABLE_ALREADY_EXISTS"


@pytest.mark.parametrize("body", [{}, {"branch_name": 7}, ["a_branch"]])
def test_a_branch_request_without_a_string_name_is_a_400(fresh_table, body):
    response = post(f"/tables/{fresh_table}/branch", body, expect_ok=False)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("name", ["bad name!", "", "9lives", "a" * 64, "a" * 300])
def test_a_branch_name_follows_the_rules_for_a_table_name(fresh_table, name):
    """`POST /tables` has always checked names. The branch route did not: `"bad name!"` and `""`
    became tables that survived a restart, and a 300-character name was a 500 in disk mode
    ("File name too long") and a table in memory mode."""
    response = post(f"/tables/{fresh_table}/branch", {"branch_name": name}, expect_ok=False)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "INVALID_TABLE_NAME"
    assert name not in table_names()
