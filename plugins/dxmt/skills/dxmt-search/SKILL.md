---
name: dxmt-search
description: Store documents in DXMT and search them by meaning, by keyword, or both (hybrid), with payload filters -- over MCP, REST or the Python/JavaScript clients. Use when the user wants a local knowledge base or RAG store, asks to index notes or files for search, or needs vector or keyword search with filters from code.
---

# Search with DXMT

A **table** holds records: an id, a vector, a JSON payload, and optionally text. With an
embedding model on the server (see the `dxmt-setup` skill) you send text and the server embeds
it; otherwise you bring your own vectors. Every route is under `http://127.0.0.1:9999/api/v1`.

## Quickest path: the `kb` table

With a model configured, the `kb` table is made by its first write, takes text, and has both
halves of hybrid search:

```bash
curl -s -X POST localhost:9999/api/v1/tables/kb/records -H 'content-type: application/json' \
  -d '{"records":[{"id":"n1","text":"HNSW is a graph index for nearest-neighbour search","payload":{"lang":"en"}}]}'
curl -s -X POST localhost:9999/api/v1/tables/kb/search/hybrid -H 'content-type: application/json' \
  -d '{"text":"how do graph indexes work","limit":5}'
```

Over MCP: `insert_vector` and `vector_search` accept `text` instead of `vector`;
`hybrid_search` takes `text`; `add_document` stores a long text as chunks (and
`delete_document` removes them all).

## Which search

| route | finds | use for |
| --- | --- | --- |
| `POST /tables/{t}/search` with `text` or `vector` | nearest by meaning | paraphrases, other languages |
| `POST /tables/{t}/search/text` with `query` | BM25 keyword match | exact names, codes, error strings |
| `POST /tables/{t}/search/hybrid` with `text` | both, fused | the default when unsure |

`search` takes `top_k`; the other two take `limit`. All three take `filter`.

## Filters

```json
{"must": [
   {"key": "lang", "operation": {"type": "Equal", "value": "en"}},
   {"key": "year", "operation": {"type": "GreaterThanOrEqual", "value": 2024}},
   {"key": "tags", "operation": {"type": "In", "value": ["rust", "db"]}}
 ],
 "must_not": [{"key": "draft", "operation": {"type": "Equal", "value": true}}]}
```

Keys may be paths (`meta.author`). `should` is "any of". A filter on a field the table declined
to index is refused with a 400, never answered with an empty page.

## Your own table

```json
POST /tables
{"name": "docs", "embedding": {"model": "bge-m3", "base_url": "http://127.0.0.1:11434/v1"},
 "enable_full_text_search": true,
 "text_fields": [{"name": "content", "source": {"metadata_field": "text"}}]}
```

- With `embedding`, `dimension` is inferred from the model. Without it, give `dimension` and
  send vectors.
- **The full-text field's `name` must be `content`** (the index's own field names are fixed);
  `source` is the payload key that holds the text. Named anything else, nothing is indexed and
  keyword search answers empty.

## From code

```python
# pip install dxmt-python-sdk
from dxmt import Client, Field
db = Client(host="localhost", port=9999)
db.insert_records("kb", [{"id": "1", "text": "...", "payload": {"lang": "en"}}])
db.hybrid_search("kb", text="...", limit=5)
db.search("kb", text="...", limit=5, filter=Field("lang") == "en")
```

JavaScript: `npm install dxmt-js-sdk`, `new DXMTClient({ host, port })`, the same methods.
LangChain, LlamaIndex, Haystack, CrewAI and Microsoft Agent Framework packages wrap the same
tables; see the README's "Clients and integrations".
