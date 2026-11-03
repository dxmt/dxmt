"""`dxmt.migrate.chroma`: Chroma collections copied into DXMT with their vectors.

The unit half drives `migrate_collection` with a stand-in collection and a mocked client, so
it runs without chromadb or a server. The integration half (`-m integration`) builds a real
Chroma store, migrates it into a running server and searches the copy by vector and by text.
"""

import os
import uuid
from unittest.mock import Mock

import pytest

from dxmt import Client
from dxmt.migrate.chroma import distance_of, migrate_collection, table_name_for


class Collection:
    """The part of a Chroma collection the migration reads."""

    def __init__(self, name, rows, metadata=None, configuration=None):
        self.name = name
        self.metadata = metadata
        self.configuration = configuration
        self.rows = rows  # (id, embedding, document, metadata)

    def count(self):
        return len(self.rows)

    def get(self, limit, offset=0, include=()):
        page = self.rows[offset : offset + limit]
        return {
            "ids": [r[0] for r in page],
            "embeddings": [r[1] for r in page],
            "documents": [r[2] for r in page],
            "metadatas": [r[3] for r in page],
        }


def dxmt(tables=()):
    client = Mock(spec=Client)
    client.list_tables.return_value = list(tables)
    return client


def rows(n):
    return [(f"id{i}", [float(i), 1.0, 0.0], f"doc {i}", {"n": i}) for i in range(n)]


def test_every_record_is_copied_in_pages_with_its_vector_payload_and_document():
    client = dxmt()
    result = migrate_collection(Collection("notes", rows(5)), client, batch_size=2)
    assert (result.table, result.records) == ("notes", 5)
    created = client.create_table.call_args.kwargs
    assert created["dimension"] == 3 and created["distance_metric"] == "euclidean"
    assert created["text_fields"][0]["source"] == {"metadata_field": "text"}
    batches = [c.args[1] for c in client.insert_records.call_args_list]
    assert [len(b) for b in batches] == [2, 2, 1]
    assert batches[0][1] == {"id": "id1", "vector": [1.0, 1.0, 0.0], "payload": {"n": 1, "text": "doc 1"}}


def test_an_existing_table_is_written_into_not_recreated():
    client = dxmt([{"name": "notes", "dimension": 3}])
    migrate_collection(Collection("notes", rows(1)), client)
    client.create_table.assert_not_called()
    client.insert_records.assert_called_once()


def test_an_existing_table_of_another_width_is_refused_before_anything_is_written():
    client = dxmt([{"name": "notes", "dimension": 8}])
    with pytest.raises(ValueError, match="8-dimension"):
        migrate_collection(Collection("notes", rows(1)), client)
    client.insert_records.assert_not_called()


def test_an_empty_collection_is_skipped_and_says_why():
    client = dxmt()
    result = migrate_collection(Collection("empty", []), client)
    assert result.skipped and result.records == 0
    client.create_table.assert_not_called()


def test_a_document_never_overwrites_a_different_metadata_field_of_the_same_name():
    client = dxmt()
    clash = [("a", [1.0, 0.0], "the document", {"text": "something else"})]
    with pytest.raises(ValueError, match="--text-field"):
        migrate_collection(Collection("c", clash), client)
    migrate_collection(Collection("c", clash), dxmt(), text_field="body")


def test_distance_comes_from_configuration_or_from_old_style_metadata():
    assert distance_of(Collection("a", [], configuration={"hnsw": {"space": "ip"}})) == "dot"
    assert distance_of(Collection("a", [], metadata={"hnsw:space": "cosine"})) == "cosine"
    assert distance_of(Collection("a", [])) == "euclidean"


def test_collection_names_become_table_names():
    assert table_name_for("team.notes-2024") == "team_notes-2024"
    with pytest.raises(ValueError, match="--table"):
        table_name_for("x" * 64)


@pytest.mark.integration
def test_a_chroma_store_is_searchable_in_dxmt_after_migrating(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from dxmt.migrate.chroma import main

    store = chromadb.PersistentClient(path=str(tmp_path))
    name = f"mig-{uuid.uuid4().hex[:8]}"
    collection = store.create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    collection.add(
        ids=["tomato", "tax", "plain"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        documents=["grow tomatoes in the garden", "file the taxes in april", None],
        metadatas=[{"kind": "hobby"}, {"kind": "chore"}, None],
    )
    host = os.environ.get("DXMT_TEST_HOST", "localhost")
    port = int(os.environ.get("DXMT_TEST_PORT", "9999"))
    argv = ["--path", str(tmp_path), "--collection", name, "--url", f"http://{host}:{port}"]
    assert main(argv) == 0
    assert main(argv) == 0, "a second run rewrites the same records"

    client = Client(host=host, port=port, api_key=os.environ.get("DXMT_API_KEY"))
    try:
        table = client.get_table(name)
        assert table["record_count"] == 3
        assert client.search(name, vector=[0.0, 1.0, 0.0], limit=1)[0]["id"] == "tax"
        assert client.full_text_search(name, "tomatoes", limit=1)[0]["id"] == "tomato"
    finally:
        client.delete_table(name)
        client.close()
