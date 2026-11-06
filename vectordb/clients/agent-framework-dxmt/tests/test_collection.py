"""The collection and store against a live server (DXMT_TEST_URL)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID, uuid4

import pytest
from agent_framework import Filter, FilterGroup, VectorStoreCollectionDefinition, VectorStoreField, vectorstoremodel
from conftest import SERVER_MARKS

from agent_framework_dxmt import DXMTCollection, DXMTStore

pytestmark = SERVER_MARKS


@pytest.fixture
def collection(definition, table_name, server_url):
    return DXMTCollection(
        dict, definition=definition, collection_name=table_name, url=server_url, api_key=os.getenv("DXMT_TEST_API_KEY")
    )


async def test_lifecycle_crud_and_vectors(collection, record, server_url):
    async with collection, DXMTStore(url=server_url, api_key=os.getenv("DXMT_TEST_API_KEY")) as store:
        assert not await collection.collection_exists()
        assert await collection.get([]) == []
        assert await collection.upsert([], generate_vectors=False) == []
        await collection.delete([])
        await collection.ensure_collection_deleted()
        await collection.ensure_collection_exists()
        await collection.ensure_collection_exists()
        assert await store.collection_exists(collection.collection_name)
        assert collection.collection_name in await store.list_collection_names()
        assert await collection.get() == []
        assert [item async for item in await collection.search(vector=[1, 0, 0])] == []

        values = [record(1), record(2, text="' OR 1=1; [x] \\\\ \"* % _", embedding=[0, 1, 0])]
        assert await collection.upsert(values, generate_vectors=False) == [1, 2]
        loaded = sorted(await collection.get([1, 2, 999], include_vectors=True), key=lambda i: i["id"])
        assert loaded == values
        assert all("embedding" not in item for item in await collection.get([1, 2]))
        await collection.upsert([record(1, text="replacement")], generate_vectors=False)
        assert (await collection.get([1]))[0]["text"] == "replacement"

        page = await collection.get(filter=Filter("number", "gte", 1), top=1)
        assert len(page) == 1
        assert len(await collection.get(top=10)) == 2
        assert len(await collection.get(top=10, skip=1)) == 1
        with pytest.raises(NotImplementedError):
            await collection.get(order_by={"number": True})

        await collection.delete([1])
        assert [i["id"] for i in await collection.get([1, 2])] == [2]
        await collection.ensure_collection_deleted()
        assert not await collection.collection_exists()


async def test_search_scores_filters_threshold_and_paging(collection, record):
    async with collection:
        await collection.ensure_collection_exists()
        await collection.upsert(
            [
                record(1, text="one", embedding=[1, 0, 0], number=1.0, tags=["x"]),
                record(2, text="two", embedding=[0.9, 0.1, 0], number=2.0, tags=["y"]),
                record(3, text="three", embedding=[0, 0, 1], number=3.0, tags=["x", "y"]),
            ],
            generate_vectors=False,
        )
        hits = [h async for h in await collection.search(vector=[1, 0, 0], top=3)]
        assert [h["record"]["id"] for h in hits] == [1, 2, 3]
        assert (
            hits[0]["score"] == pytest.approx(1.0, abs=1e-4) and hits[0]["score"] > hits[1]["score"] > hits[2]["score"]
        )
        assert "embedding" not in hits[0]["record"]
        with_vectors = [h async for h in await collection.search(vector=[1, 0, 0], top=1, include_vectors=True)]
        assert with_vectors[0]["record"]["embedding"] == [1.0, 0.0, 0.0]

        only_x = [
            h async for h in await collection.search(vector=[1, 0, 0], top=3, filter=Filter("tags", "contains", "x"))
        ]
        assert [h["record"]["id"] for h in only_x] == [1, 3]
        ranged = [
            h
            async for h in await collection.search(
                vector=[1, 0, 0],
                top=3,
                filter=FilterGroup("and", [Filter("number", "gt", 1), Filter("tags", "contains_any", ["y"])]),
            )
        ]
        assert [h["record"]["id"] for h in ranged] == [2, 3]
        close = [h async for h in await collection.search(vector=[1, 0, 0], top=3, score_threshold=0.9)]
        assert [h["record"]["id"] for h in close] == [1, 2]
        second = [h async for h in await collection.search(vector=[1, 0, 0], top=1, skip=1)]
        assert [h["record"]["id"] for h in second] == [2]


async def test_keyword_hybrid_search_fuses_text_and_vector(collection, record):
    async with collection:
        await collection.ensure_collection_exists()
        await collection.upsert(
            [
                record(1, text="rust borrows memory instead of collecting it", embedding=[1, 0, 0]),
                record(2, text="go collects garbage instead of borrowing", embedding=[0, 1, 0]),
            ],
            generate_vectors=False,
        )
        hits = [
            h
            async for h in await collection.search(
                "garbage collection",
                search_type="keyword_hybrid",
                vector=[1, 0, 0],
                additional_property_name="text",
                top=2,
            )
        ]
        assert {h["record"]["id"] for h in hits} == {1, 2}
        assert all(h["score"] is not None for h in hits)
        with pytest.raises(NotImplementedError):
            await collection.search(
                "x", search_type="keyword_hybrid", vector=[1, 0, 0], additional_property_name="number"
            )


async def test_uuid_keys_and_a_decorated_model(table_name, server_url):
    @vectorstoremodel(collection_name=table_name)
    @dataclass
    class Doc:
        id: Annotated[UUID, VectorStoreField("key")]
        title: Annotated[str, VectorStoreField("data")]
        embedding: Annotated[list[float] | None, VectorStoreField("vector", dimensions=3)] = None

    key = uuid4()
    async with DXMTStore(url=server_url, api_key=os.getenv("DXMT_TEST_API_KEY")) as store:
        collection = store.get_collection(Doc)
        await collection.ensure_collection_exists()
        assert await collection.upsert([Doc(key, "t", [0, 0, 1])], generate_vectors=False) == [key]
        got = await collection.get([key])
        assert isinstance(got[0], Doc) and got[0].id == key and got[0].title == "t"
        hits = [h async for h in await collection.search(vector=[0, 0, 1], top=1)]
        assert hits[0]["record"].id == key


@pytest.mark.skipif(
    not os.getenv("DXMT_TEST_EMBEDDING"), reason="Set DXMT_TEST_EMBEDDING=1 when the server has a model"
)
async def test_server_side_embedding(table_name, server_url):
    definition = VectorStoreCollectionDefinition(
        [
            VectorStoreField("key", name="id", type_="str"),
            VectorStoreField("data", name="text", type_="str", is_full_text_indexed=True),
            VectorStoreField("vector", name="embedding", dimensions=1024),
        ]
    )
    # "server": the model the server was started with. Records carry text and no vector; a
    # search given a string and no local embedding generator sends the text to the server.
    async with DXMTCollection(
        dict,
        definition=definition,
        collection_name=table_name,
        url=server_url,
        api_key=os.getenv("DXMT_TEST_API_KEY"),
        server_embedding="server",
    ) as collection:
        await collection.ensure_collection_exists()
        await collection.upsert(
            [
                {"id": "rust", "text": "rust borrows memory instead of collecting it", "embedding": None},
                {"id": "go", "text": "go collects garbage instead of borrowing", "embedding": None},
            ],
            generate_vectors=False,
        )
        hits = [h async for h in await collection.search("garbage collection", top=1)]
        assert hits[0]["record"]["id"] == "go"
        hybrid = [h async for h in await collection.search("borrowing", search_type="keyword_hybrid", top=2)]
        assert {h["record"]["id"] for h in hybrid} == {"rust", "go"}


def test_model_rules(definition):
    two_vectors = VectorStoreCollectionDefinition(
        [
            VectorStoreField("key", name="id", type_="str"),
            VectorStoreField("vector", name="a", dimensions=2),
            VectorStoreField("vector", name="b", dimensions=2),
        ]
    )
    with pytest.raises(NotImplementedError, match="exactly one vector"):
        DXMTCollection(dict, definition=two_vectors, collection_name="x", url="http://localhost:9999")
    reserved = VectorStoreCollectionDefinition(
        [
            VectorStoreField("key", name="id", type_="str"),
            VectorStoreField("data", name="content", type_="str"),
            VectorStoreField("vector", name="a", dimensions=2),
        ]
    )
    with pytest.raises(ValueError, match="reserved"):
        DXMTCollection(dict, definition=reserved, collection_name="x", url="http://localhost:9999")
    hamming = VectorStoreCollectionDefinition(
        [
            VectorStoreField("key", name="id", type_="str"),
            VectorStoreField("vector", name="a", dimensions=2, distance_function="hamming"),
        ]
    )
    with pytest.raises(NotImplementedError, match="distance"):
        DXMTCollection(dict, definition=hamming, collection_name="x", url="http://localhost:9999")
