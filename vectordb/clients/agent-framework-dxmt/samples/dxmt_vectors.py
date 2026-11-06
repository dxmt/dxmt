"""DXMT as an Agent Framework vector store.

Run a DXMT server first (``vectordb serve``, or ``docker run -p 127.0.0.1:9999:9999 -v dxmt:/data dxmtai/dxmt``),
then ``python dxmt_vectors.py``. Set ``DXMT_URL`` if it is not on localhost:9999.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated

from agent_framework import Filter, FilterGroup, VectorStoreField, vectorstoremodel

from agent_framework_dxmt import DXMTStore


@vectorstoremodel(collection_name="af_sample_docs")
@dataclass
class Document:
    id: Annotated[str, VectorStoreField("key")]
    title: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    year: Annotated[int, VectorStoreField("data", is_indexed=True)]
    tags: Annotated[list[str], VectorStoreField("data")]
    embedding: Annotated[list[float] | None, VectorStoreField("vector", dimensions=3)] = None


async def main() -> None:
    async with DXMTStore() as store:
        collection = store.get_collection(Document)
        await collection.ensure_collection_deleted()
        await collection.ensure_collection_exists()

        await collection.upsert(
            [
                Document("rust", "rust borrows memory instead of collecting it", 2015, ["systems"], [1.0, 0.0, 0.0]),
                Document("go", "go collects garbage instead of borrowing", 2012, ["systems", "gc"], [0.0, 1.0, 0.0]),
                Document("py", "python counts references and collects cycles", 1991, ["gc"], [0.0, 0.0, 1.0]),
            ],
            generate_vectors=False,
        )

        print("nearest to [1,0,0]:")
        async for item in await collection.search(vector=[1.0, 0.0, 0.0], top=3):
            print(f"  {item['record'].id:5} score={item['score']:.3f}")

        print("after 2000, tagged gc:")
        async for item in await collection.search(
            vector=[1.0, 0.0, 0.0],
            top=3,
            filter=FilterGroup("and", [Filter("year", "gte", 2000), Filter("tags", "contains", "gc")]),
        ):
            print(f"  {item['record'].id:5} {item['record'].title}")

        print("keyword-hybrid for 'garbage collection' (vector + BM25, fused on the server):")
        async for item in await collection.search(
            "garbage collection",
            search_type="keyword_hybrid",
            vector=[0.0, 0.0, 1.0],
            additional_property_name="title",
            top=2,
        ):
            print(f"  {item['record'].id:5} score={item['score']:.3f}")

        await collection.ensure_collection_deleted()


if __name__ == "__main__":
    asyncio.run(main())
