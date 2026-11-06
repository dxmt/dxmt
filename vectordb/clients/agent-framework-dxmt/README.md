# agent-framework-dxmt

[DXMT](https://github.com/dxmt/dxmt), the open-source memory database for personal agents,
for [Microsoft Agent Framework](https://github.com/microsoft/agent-framework): a vector store
with keyword-hybrid search, and memory as a context provider.

```bash
pip install agent-framework-dxmt
```

A DXMT server has to be running (`vectordb serve`, or
`docker run -p 127.0.0.1:9999:9999 -v dxmt:/data dxmtai/dxmt`). Connection settings are
`DXMT_URL` (default `http://localhost:9999`) and `DXMT_API_KEY`, or the `url` / `api_key`
arguments, resolved through Agent Framework's `load_settings` like every other connector.

## Vector store

`DXMTCollection` and `DXMTStore` implement `BaseVectorCollection` / `BaseVectorSearch` and
`BaseVectorStore`. A collection is one DXMT table; the record's single vector field is the
row's vector and its data fields are the payload.

```python
from dataclasses import dataclass
from typing import Annotated

from agent_framework import Filter, VectorStoreField, vectorstoremodel
from agent_framework_dxmt import DXMTStore


@vectorstoremodel
@dataclass
class Document:
    id: Annotated[str, VectorStoreField("key")]
    title: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    year: Annotated[int, VectorStoreField("data", is_indexed=True)]
    embedding: Annotated[list[float] | None, VectorStoreField("vector", dimensions=3)] = None


async with DXMTStore() as store:
    collection = store.get_collection(Document, collection_name="documents")
    await collection.ensure_collection_exists()
    await collection.upsert([Document("1", "Hello DXMT", 2026, [1.0, 0.0, 0.0])], generate_vectors=False)
    results = await collection.search(vector=[1.0, 0.0, 0.0], filter=Filter("year", "gte", 2020), top=3)
    async for item in results:
        print(item["record"].title, item["score"])
```

What is specific to DXMT:

- **Keyword-hybrid search.** A data field marked `is_full_text_indexed` gets the table's BM25
  index, and `collection.search("hello", search_type="keyword_hybrid", additional_property_name="title")`
  is the server's hybrid search (vector and keywords fused on the server). Most in-tree
  connectors declare `{"vector"}` only.
- **Server-side embedding.** `DXMTCollection(..., server_embedding={"model": "text-embedding-3-small"},
  server_embedding_field="title")` has the server embed records whose vector is `None` and
  queries given as text, so no embedding client runs in the agent's process.
- **Filters.** Every portable operator (`eq`, `ne`, ranges, `between`, `in`, `not_in`,
  `is_null`, `is_not_null`, `exists`, `contains`, `contains_any`, `contains_all`,
  `starts_with`, `ends_with`, `contains_text`) and `and` / `or` / `not` groups, translated
  exactly to the server's flat filter; a shape the server has no flat form for (`(a and b) or c`)
  raises `NotImplementedError` rather than being approximated.
- Scores are similarities (1.0 is identical under cosine); `score_threshold` is in those units.
  One vector field per collection; `cosine_similarity` (default), `dot_prod`, `euclidean_distance`.

## Memory as a context provider

`DXMTContextProvider` recalls memories before a run (by meaning *and* keyword) and remembers
the exchange after it, using DXMT's own memory API. Scope is tags: memories are written under
`user_id` / `agent_id` / `application_id` and recalled by the separate `search_*` ids, the
split the mem0 provider uses so a memory written under a shared agent is not read back for an
unrelated user by accident.

```python
from agent_framework import Agent
from agent_framework_dxmt import DXMTContextProvider

memory = DXMTContextProvider(user_id="alice", search_user_id="alice")
agent = Agent(chat_client, instructions="...", context_providers=[memory])
await agent.run("I prefer answers in Chinese.")
await agent.run("How should you answer me?")   # the preference is recalled into the context
```

`memory.remember(text, kind=..., supersedes=[...])`, `memory.recall(query)` and
`memory.forget(ids)` expose the three operations directly. A memory said twice is kept once;
a new fact can supersede an old one, which is kept as history and left out of recall. The
same store is what every other agent on the machine sees over MCP and REST.

## Tests

```bash
pytest                                  # unit tests, client mocked
DXMT_TEST_URL=http://127.0.0.1:9999 pytest -m integration   # against a server
```
