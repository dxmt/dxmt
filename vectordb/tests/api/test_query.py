"""DXQL over `POST /query`, asserted against the fixed corpus.

Until 2026-09-11 this endpoint answered every statement -- `SELECT * FROM t` included -- with
`400 No valid statement found`. The parser looked for `statement` among the top-level parse
pairs, where only `query` ever is, and everything behind WHERE, INSERT, UPDATE, DELETE, CREATE
and DROP returned placeholders. `api_contract_test.rs` counted the 400 as the route being
served, and the Python SDK's `client.query()` raised on every call.

So every statement kind is exercised here with the rows it must touch written down, and the DXQL
this server does not execute is asserted to say so with a 501, not to answer something
plausible.
"""

import json
import uuid

import pytest

from .helpers.client import delete, get, post, succeeded
from .helpers.corpus import BY_ID, CORPUS

ALL_IDS = sorted(record["id"] for record in CORPUS)


def query(sql, expect_ok=True):
    return post("/query", {"query": sql}, expect_ok=expect_ok)


def data(sql):
    return query(sql).json()["data"]


def ids(sql):
    return [record["id"] for record in data(sql)["records"]]


def payload_of(record):
    """`GET /records/{id}` carries the payload as a JSON string; see test_crud.py."""
    raw = record.get("payload_json")
    return json.loads(raw) if isinstance(raw, str) else record.get("payload")


# ---- SELECT ---------------------------------------------------------------------------------


def test_select_star_returns_every_record_in_id_order_with_its_payload(table):
    answer = data(f"SELECT * FROM {table}")
    assert [r["id"] for r in answer["records"]] == ALL_IDS
    for record in answer["records"]:
        assert record["fields"] == {"id": record["id"], **BY_ID[record["id"]]["payload"]}
        assert record["score"] is None
    assert answer["metadata"]["total_results"] == len(CORPUS)
    assert answer["metadata"]["affected_rows"] == 0
    assert answer["metadata"]["plan_type"] == "SELECT"


def test_projections_and_aliases_name_the_fields_that_come_back(table):
    (record,) = data(f"SELECT id, rank AS r, city FROM {table} WHERE id = 'r4'")["records"]
    assert record["fields"] == {"id": "r4", "r": 4, "city": "Moscow"}


def test_the_vector_can_be_selected(table):
    (record,) = data(f"SELECT vector FROM {table} WHERE id = 'r3'")["records"]
    assert record["fields"]["vector"] == pytest.approx(BY_ID["r3"]["vector"], abs=1e-6)


NOT_R3 = [i for i in ALL_IDS if i != "r3"]
NOT_LONDON = [i for i in ALL_IDS if BY_ID[i]["payload"]["city"] != "London"]


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("rank = 3", ["r3"]),
        ("rank = 3.0", ["r3"]),  # an integer payload equals the same number written as a float
        ("rank != 3", NOT_R3),
        ("rank <> 3", NOT_R3),
        ("rank > 6", ["r7", "r8"]),
        ("rank >= 7", ["r7", "r8"]),
        ("rank < 2", ["r1"]),
        ("rank <= 2", ["r1", "r2"]),
        ("rank > -1.5", ALL_IDS),
        ("city = 'London'", ["r1", "r2"]),
        ('city = "London"', ["r1", "r2"]),
        ("city != 'London'", NOT_LONDON),
        ("city < 'London'", ["r5", "r6"]),  # Berlin sorts before London
        ("id = 'r5'", ["r5"]),
    ],
)
def test_comparisons(table, condition, expected):
    assert ids(f"SELECT id FROM {table} WHERE {condition}") == expected


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("city = 'London' AND tag = 'beta'", ["r2"]),
        ("city = 'London' OR city = 'Paris'", ["r1", "r2", "r7", "r8"]),
        # AND binds tighter than OR, as in SQL: London, or (Paris and beta).
        ("city = 'London' OR city = 'Paris' AND tag = 'beta'", ["r1", "r2", "r8"]),
        ("(city = 'London' OR city = 'Paris') AND tag = 'beta'", ["r2", "r8"]),
        ("city IN ('Berlin', 'Paris')", ["r5", "r6", "r7", "r8"]),
        ("rank IN (1, 8.0)", ["r1", "r8"]),
        ("rank BETWEEN 2 AND 4", ["r2", "r3", "r4"]),  # inclusive at both ends
        ("rank BETWEEN 2 AND 4 AND tag = 'alpha'", ["r3"]),
        ("city LIKE 'Lon%'", ["r1", "r2"]),
        ("city LIKE '%lin'", ["r5", "r6"]),
        ("city LIKE 'Par_s'", ["r7", "r8"]),
        ("city LIKE 'ondon'", []),  # without a wildcard the pattern is the whole string
        ("no_such_field = 1", []),
        ("no_such_field != 1", []),  # a missing field satisfies nothing, != included
    ],
)
def test_logic_sets_and_patterns(table, condition, expected):
    assert ids(f"SELECT id FROM {table} WHERE {condition}") == expected


def test_order_by_a_numeric_field_descending(table):
    assert ids(f"SELECT id FROM {table} ORDER BY rank DESC LIMIT 3") == ["r8", "r7", "r6"]


def test_order_by_several_keys(table):
    got = ids(f"SELECT id FROM {table} ORDER BY city ASC, rank DESC")
    assert got == ["r6", "r5", "r2", "r1", "r4", "r3", "r8", "r7"]


def test_limit_and_offset_page_through_the_table_without_overlap(table):
    pages = [
        ids(f"SELECT id FROM {table} ORDER BY rank LIMIT 3 OFFSET {start}") for start in (0, 3, 6)
    ]
    assert pages == [["r1", "r2", "r3"], ["r4", "r5", "r6"], ["r7", "r8"]]
    # Without ORDER BY the order is by id, so a page is the same page every time.
    assert ids(f"SELECT id FROM {table} LIMIT 2 OFFSET 2") == ALL_IDS[2:4]


def test_keywords_are_case_insensitive_and_a_trailing_semicolon_is_accepted(table):
    sql = f"select id from {table} where rank between 1 and 2 order by rank desc limit 5;"
    assert ids(sql) == ["r2", "r1"]


def test_an_empty_table_selects_nothing(empty_table):
    answer = data(f"SELECT * FROM {empty_table}")
    assert answer["records"] == []
    assert answer["metadata"]["total_results"] == 0


# ---- INSERT / UPDATE / DELETE ---------------------------------------------------------------


def test_insert_writes_a_record_the_rest_api_can_read(fresh_table):
    answer = data(
        f"INSERT INTO {fresh_table} (id, vector, city, rank) "
        "VALUES ('n1', [0.5, 0.5, 0.5, 0.5], 'Rome', 9)"
    )
    assert answer["metadata"]["plan_type"] == "INSERT"
    assert answer["metadata"]["affected_rows"] == 1

    stored = get(f"/tables/{fresh_table}/records/n1").json()["data"]
    assert payload_of(stored) == {"city": "Rome", "rank": 9}
    assert ids(f"SELECT id FROM {fresh_table} WHERE city = 'Rome'") == ["n1"]
    (row,) = data(f"SELECT vector FROM {fresh_table} WHERE id = 'n1'")["records"]
    assert row["fields"]["vector"] == pytest.approx([0.5] * 4)


def test_an_insert_of_the_wrong_dimension_is_refused_and_leaves_nothing(fresh_table):
    response = query(
        f"INSERT INTO {fresh_table} (id, vector) VALUES ('bad', [1.0, 2.0])", expect_ok=False
    )
    assert response.status_code == 422, response.text
    assert not succeeded(get(f"/tables/{fresh_table}/records/bad", expect_ok=False))


def test_an_insert_of_a_non_finite_number_is_refused_and_the_table_still_answers(fresh_table):
    """`1e999` is valid to the grammar and parses to an infinity.

    The record routes reject it (`VectorValidator` runs in the handler), but DXQL reaches
    storage without passing that validator, so this used to be accepted. An infinite coordinate
    makes a cosine distance NaN, which panicked the comparator HNSW sorts with: the next search
    on the table closed the connection with no response at all, and the record was in the WAL,
    so a restart brought it back.
    """
    response = query(
        f"INSERT INTO {fresh_table} (id, vector) VALUES ('inf', [1e999, 0, 0, 0])",
        expect_ok=False,
    )
    assert response.status_code == 422, response.text
    assert not succeeded(get(f"/tables/{fresh_table}/records/inf", expect_ok=False))

    answer = post(f"/tables/{fresh_table}/search", {"vector": [1, 0, 0, 0], "top_k": 3})
    assert answer.status_code == 200, answer.text


def test_update_changes_the_named_fields_and_keeps_the_rest(fresh_table):
    answer = data(f"UPDATE {fresh_table} SET city = 'Oslo', note = 'moved' WHERE id = 'r2'")
    assert answer["metadata"]["affected_rows"] == 1

    stored = get(f"/tables/{fresh_table}/records/r2").json()["data"]
    assert payload_of(stored) == {**BY_ID["r2"]["payload"], "city": "Oslo", "note": "moved"}
    (row,) = data(f"SELECT vector FROM {fresh_table} WHERE id = 'r2'")["records"]
    assert row["fields"]["vector"] == pytest.approx(BY_ID["r2"]["vector"], abs=1e-6)


def test_update_touches_every_matching_record_and_reports_how_many(fresh_table):
    assert data(f"UPDATE {fresh_table} SET rank = 20 WHERE city = 'Paris'")["metadata"][
        "affected_rows"
    ] == 2
    assert ids(f"SELECT id FROM {fresh_table} WHERE rank = 20") == ["r7", "r8"]
    assert data(f"UPDATE {fresh_table} SET rank = 0 WHERE city = 'Rome'")["metadata"][
        "affected_rows"
    ] == 0


def test_delete_removes_exactly_the_matching_records(fresh_table):
    answer = data(f"DELETE FROM {fresh_table} WHERE rank > 6")
    assert answer["metadata"]["affected_rows"] == 2
    assert ids(f"SELECT id FROM {fresh_table}") == ["r1", "r2", "r3", "r4", "r5", "r6"]
    assert get(f"/tables/{fresh_table}/records/r8", expect_ok=False).status_code == 404
    # A DELETE that matches nothing says so, rather than echoing back what it was asked to do.
    assert data(f"DELETE FROM {fresh_table} WHERE rank > 100")["metadata"]["affected_rows"] == 0


def test_delete_without_where_empties_the_table(fresh_table):
    assert data(f"DELETE FROM {fresh_table}")["metadata"]["affected_rows"] == len(CORPUS)
    assert ids(f"SELECT id FROM {fresh_table}") == []


# ---- CREATE / DROP --------------------------------------------------------------------------


def test_create_table_takes_its_dimension_from_the_vector_column_and_drop_removes_it():
    name = f"dxql_{uuid.uuid4().hex[:8]}"
    try:
        answer = data(f"CREATE TABLE {name} (id TEXT PRIMARY KEY, embedding VECTOR(4) NOT NULL)")
        assert answer["metadata"]["plan_type"] == "CREATE"
        # It used to be created with 128 dimensions whatever the statement said.
        assert get(f"/tables/{name}").json()["data"]["table"]["dimension"] == 4
        data(f"INSERT INTO {name} (id, vector) VALUES ('a', [1, 0, 0, 0])")
        assert ids(f"SELECT id FROM {name}") == ["a"]

        assert data(f"DROP TABLE {name}")["metadata"]["plan_type"] == "DROP"
        assert get(f"/tables/{name}", expect_ok=False).status_code == 404
    finally:
        delete(f"/tables/{name}", expect_ok=False)


# ---- What is refused, and with what ---------------------------------------------------------


def test_a_body_without_a_query_is_a_400():
    assert post("/query", {}, expect_ok=False).status_code == 400


@pytest.mark.parametrize(
    "sql",
    [
        "SELEC * FROM t",
        "SELECT * FROM",
        "SELECT * FROM t WHERE rank = 3.",
        "INSERT INTO t (id, vector) VALUES ('a')",
    ],
)
def test_malformed_dxql_is_a_400(sql):
    response = query(sql, expect_ok=False)
    assert response.status_code == 400, response.text
    assert "Parse error" in response.json()["error"]


@pytest.mark.parametrize(
    "template",
    [
        "INSERT INTO {t} (id) VALUES ('no-vector')",
        "UPDATE {t} SET id = 'x' WHERE id = 'r1'",
        "CREATE TABLE {t}_new (id TEXT)",
        # Valid identifiers to the grammar, not valid table names to `POST /tables`.
        "CREATE TABLE `bad name` (embedding VECTOR(4))",
        # `_hidden` used to be here: a leading underscore is a legal file name and is
        # accepted now. A path separator is what the rule is actually for.
        "CREATE TABLE `a/b` (embedding VECTOR(4))",
    ],
)
def test_well_formed_dxql_that_cannot_be_carried_out_as_written_is_a_400(table, template):
    response = query(template.format(t=table), expect_ok=False)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM definitely_not_a_table",
        "DELETE FROM definitely_not_a_table WHERE id = 'x'",
        "UPDATE definitely_not_a_table SET a = 1",
        "INSERT INTO definitely_not_a_table (id, vector) VALUES ('a', [1, 0, 0, 0])",
        "DROP TABLE definitely_not_a_table",
    ],
)
def test_a_table_that_does_not_exist_is_a_404(sql):
    response = query(sql, expect_ok=False)
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "template",
    [
        "SELECT * FROM {t} WHERE vector_similarity(vector, [1, 0, 0, 0]) > 0.9",
        "SELECT * FROM {t} WHERE text_match(city, 'London')",
        "SELECT * FROM {t} ORDER BY hybrid_score(0.5, 0.5, 0)",
        "SELECT * FROM {t} WHERE rank = $1",
        "CREATE HNSW INDEX idx ON {t}(vector)",
        "DROP INDEX idx",
    ],
)
def test_dxql_this_server_does_not_execute_is_a_501_that_says_so(table, template):
    response = query(template.format(t=table), expect_ok=False)
    assert response.status_code == 501, response.text
    assert "Not supported" in response.json()["error"]
