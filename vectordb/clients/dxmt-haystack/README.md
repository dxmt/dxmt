# dxmt-haystack

[DXMT](https://github.com/dxmt/dxmt) as a [Haystack](https://haystack.deepset.ai) document
store, with an embedding retriever, a BM25 retriever and a hybrid retriever.

```bash
pip install dxmt-haystack
```

A DXMT server has to be running (`vectordb serve`, or `docker run -p 9999:9999 dxmtai/dxmt`).

```python
from haystack import Document, Pipeline
from haystack_integrations.document_stores.dxmt import DXMTDocumentStore
from haystack_integrations.components.retrievers.dxmt import (
    DXMTBM25Retriever, DXMTEmbeddingRetriever, DXMTHybridRetriever,
)

store = DXMTDocumentStore(table_name="docs", embedding_dim=768, enable_full_text_search=True)
store.write_documents([
    Document(content="rust borrows memory instead of collecting it", meta={"lang": "rust"}, embedding=[...]),
    Document(content="go collects garbage instead of borrowing",    meta={"lang": "go"},   embedding=[...]),
])

store.filter_documents({"field": "meta.lang", "operator": "==", "value": "rust"})
DXMTEmbeddingRetriever(store, top_k=5).run(query_embedding=[...])
DXMTBM25Retriever(store, top_k=5).run(query="garbage")
DXMTHybridRetriever(store, top_k=5).run(query="garbage", query_embedding=[...])
```

## What the store does

- **One document, one record.** `id` is the record id, `embedding` the vector, `sparse_embedding`
  the sparse vector, and the payload holds `content` and `meta` as they are, `meta` nested, so a
  filter on `meta.page` is the server path `meta.page`.
- **Documents without an embedding are kept**, as in every Haystack store. The server wants a
  vector per record, so such a document is written with a fixed stand-in vector and
  `__has_embedding: false`; it comes back with `embedding=None` and the embedding retriever never
  ranks it. With `server_embedding={"model": ...}` the server embeds `content` instead and the
  embedding retriever takes `query_text`.
- **Filters.** The full Haystack filter language is supported by `filter_documents`: a filter
  the server has a flat form for (`==`, `!=`, numeric ranges, `in`, `not in`, `AND`, `OR`, `NOT`
  of one condition) runs on the server; anything else (dates, strings under `>`, `None`, nested
  `AND` inside `OR`, `NOT` over several conditions) is evaluated with Haystack's own
  `document_matches_filter` over a scroll of the table, which is correct and slower. The
  retrievers only take filters with a server form and refuse the rest by name, because a
  ranked page cannot be filtered afterwards without changing what it means.
- **Duplicate policies** `NONE` and `OVERWRITE` upsert, `SKIP` writes only the missing ids, `FAIL`
  raises `DuplicateDocumentError`.
- `blob` is not stored (DXMT payloads are JSON); a warning says so.

## Tests

`tests/test_document_store.py` runs Haystack's own `DocumentStoreBaseTests` (count, write, delete
and 39 filter cases, with the in-memory evaluator as the oracle) against a live server:

```bash
DXMT_HOST=127.0.0.1 DXMT_PORT=9999 pytest
```

`tests/test_filters.py` needs no server and pins which filters get a server form.
