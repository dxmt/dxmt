# crewai-dxmt

[DXMT](https://github.com/dxmt/dxmt), the open-source memory database for personal agents, as
[CrewAI](https://github.com/crewAIInc/crewAI)'s memory storage backend, knowledge storage and
search tool.

```bash
pip install crewai-dxmt
```

A DXMT server has to be running: `vectordb serve`, or
`docker run -p 127.0.0.1:9999:9999 -v dxmt:/data dxmtai/dxmt`.

## Memory

CrewAI's unified `Memory` does the thinking (an LLM infers scope, categories and importance;
recall blends similarity, recency and importance) and hands the storage records with their
embeddings. `DXMTStorageBackend` implements CrewAI's `StorageBackend` protocol over one DXMT
table, so every selection the protocol asks for -- scope prefix, categories, metadata,
`older_than` -- is a server filter.

```python
from crewai import Memory
from crewai_dxmt import DXMTStorageBackend

memory = Memory(storage=DXMTStorageBackend("crewai_memory", host="localhost", port=9999))
memory.remember("We decided to use PostgreSQL for the user database.")
memory.recall("What database did we choose?")
```

`Crew(memory=True)` builds its `Memory` without a `storage` argument, so to route *every*
crew to DXMT register the backend once at startup:

```python
import crewai_dxmt
crewai_dxmt.install(host="localhost", port=9999)
```

Why DXMT rather than the LanceDB default: a server process, so agent memory and crew memory,
or several crews on one machine, share one store with no file lock; a write-ahead log, so
`kill -9` loses nothing acknowledged; and the same store is reachable over MCP, REST and the
Python, JavaScript, Rust and Go clients.

## Knowledge

`DXMTKnowledgeStorage` is a `BaseKnowledgeStorage`. It embeds on the server
(`embedding={"model": ...}`, any OpenAI-compatible endpoint via `base_url`) or with a function
(`embedder=`); with `enable_full_text_search=True` queries are hybrid (vector + BM25).

```python
from crewai import Knowledge
from crewai_dxmt import DXMTKnowledgeStorage

storage = DXMTKnowledgeStorage(table_name="docs", embedding={"model": "text-embedding-3-small"})
knowledge = Knowledge(collection_name="docs", sources=[...], storage=storage)
```

`install()` registers a knowledge factory too, using the embedder CrewAI resolved for the
collection, or `knowledge_embedding={...}` for server-side embedding.

## Tool

```python
from crewai import Agent
from crewai_dxmt import DXMTVectorSearchTool

tool = DXMTVectorSearchTool(table_name="docs", search_mode="hybrid", limit=5)
agent = Agent(role="Researcher", goal="...", backstory="...", tools=[tool])
```

The query is embedded by the table's server-side model, or by `embedder=`; `search_mode` is
`vector`, `hybrid` or `text`.

## Tests

`tests/test_memory_backend.py` drives CrewAI's real `Memory` engine end to end (remember,
recall with composite scoring, scopes, categories, forget, update) with a deterministic
embedder and a mocked LLM, against a live server: `DXMT_HOST=127.0.0.1 DXMT_PORT=9999 pytest`.
