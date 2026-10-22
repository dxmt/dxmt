# Embeddings: sending text instead of vectors

A table can name an embedding model. Records and searches then carry `text`, and the server
turns it into vectors with that table's model. This is `server-embedding`, a default build
feature. You can still compute vectors yourself and send them; both work on the same table.

Any provider that speaks the OpenAI `/embeddings` protocol works: OpenAI, a local ollama
(`/v1`), text-embeddings-inference, Infinity, vLLM, LM Studio, llama.cpp, and the hosted
providers listed under "Provider keys" below.

## 1. Create a table with a model

```bash
export OPENAI_API_KEY=...     # in the server's environment; never sent in a request body
./target/release/vectordb serve
```

```bash
# No dimension: the server embeds one probe string and takes the width from the answer.
curl -X POST localhost:9999/api/v1/tables \
  -H 'Content-Type: application/json' \
  -d '{"name": "docs", "embedding": {"model": "text-embedding-3-small"}}'
```

`embedding` takes:

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | required | Passed to the provider verbatim |
| `base_url` | `https://api.openai.com/v1` | An OpenAI-compatible API, without `/embeddings` |
| `api_key_env` | `OPENAI_API_KEY` | The **name** of a server environment variable holding the key |
| `field` | `text` | The payload field the text is stored in |
| `batch_size` | 64 | Texts per provider request |

A local model needs no key: name a variable that is unset.

```bash
curl -X POST localhost:9999/api/v1/tables \
  -H 'Content-Type: application/json' \
  -d '{"name": "notes", "embedding": {"model": "bge-m3",
       "base_url": "http://localhost:11434/v1", "api_key_env": "OLLAMA_API_KEY"}}'
```

## 2. Insert text, search with text

```bash
curl -X POST localhost:9999/api/v1/tables/docs/records \
  -H 'Content-Type: application/json' \
  -d '{"records": [{"id": "1", "text": "rust borrows memory instead of collecting it"},
                   {"id": "2", "text": "go collects garbage instead of borrowing"}]}'

curl -X POST localhost:9999/api/v1/tables/docs/search \
  -H 'Content-Type: application/json' \
  -d '{"text": "memory management", "top_k": 5}'
```

The text is kept in the payload (field `text`), so each hit carries the passage it came from.
A query is embedded with the table's own model, never another one.

If the provider fails, the request fails (503) and nothing is written: there is no fallback
vector. A model that answers at a width other than the table's is refused before anything is
stored. `PATCH /api/v1/tables/{t}` can change `base_url`, `api_key_env`, `field` and
`batch_size`, but not `model` on a table that already holds rows.

## 3. Documents: long texts in, chunks out

A document is split into overlapping chunks (500 characters with 50 shared, by default) on
paragraph and sentence boundaries; each chunk is a record `{doc_id}#{n}` and is searchable on
its own. Writing the same `doc_id` again replaces the document.

```bash
curl -X PUT localhost:9999/api/v1/tables/docs/documents/guide%2Fintro.md \
  -H 'Content-Type: application/json' \
  -d '{"text": "…the whole file…", "metadata": {"path": "guide/intro.md"}}'

curl localhost:9999/api/v1/tables/docs/documents/guide%2Fintro.md        # chunks, in order
curl -X DELETE localhost:9999/api/v1/tables/docs/documents/guide%2Fintro.md
```

`metadata` is copied onto every chunk, so a filter on it selects the document's chunks.
`chunk_size` and `chunk_overlap` override the defaults.

## 4. The `kb` table

When the server has a default model — `DXMT_EMBEDDING_MODEL`, or a local ollama embedding
model it finds on its own — a table named `kb` is created the first time something writes to
or searches it. It embeds `text` with that model and also indexes it for full-text search with
the Chinese-aware analyzer, so `/search/hybrid` works on it with text alone:

```bash
curl -X POST localhost:9999/api/v1/tables/kb/records -H 'Content-Type: application/json' \
  -d '{"records": [{"id": "n1", "text": "HNSW is a graph index for vector search"}]}'
curl -X POST localhost:9999/api/v1/tables/kb/search/hybrid -H 'Content-Type: application/json' \
  -d '{"text": "graph index", "limit": 3}'
```

`DXMT_KB_TABLE` renames it; `DXMT_KB_TABLE=off` disables it. A table created with neither a
`dimension` nor an `embedding` also takes the server's default model.

## 5. From Python

```bash
pip install ./clients/dxmt-python-sdk     # not on PyPI yet
```

```python
from dxmt import Client

client = Client(host="localhost", port=9999)

client.create_table(name="docs", embedding={"model": "text-embedding-3-small"})
client.insert_records("docs", [
    {"id": "1", "text": "rust borrows memory instead of collecting it"},
    {"id": "2", "text": "go collects garbage instead of borrowing"},
])
for hit in client.search("docs", text="memory management", k=5):
    print(hit["id"], hit["score"])

client.put_document("docs", "guide/intro.md", open("intro.md").read(),
                    metadata={"path": "guide/intro.md"})
print(client.get_document("docs", "guide/intro.md")["chunks"][0]["text"])
client.delete_document("docs", "guide/intro.md")

hits = client.hybrid_search("kb", text="graph index", k=3)
```

The MCP server (`vectordb mcp`) takes text in the same way: `create_table` accepts
`embedding`, `insert_vector` and `vector_search` accept `text`, and `add_document` /
`delete_document` manage documents.

## 6. Provider keys

A table's `base_url` and `api_key_env` come from whoever creates the table, so the server
limits where a key may go. A key is sent only if:

- `api_key_env` is a known provider variable (`OPENAI_API_KEY`, `COHERE_API_KEY`,
  `JINA_API_KEY`, `VOYAGE_API_KEY`, `MISTRAL_API_KEY`, `DEEPINFRA_API_KEY`,
  `TOGETHER_API_KEY`, `SILICONFLOW_API_KEY`, `DASHSCOPE_API_KEY`, ...), the one named in
  `DXMT_EMBEDDING_API_KEY_ENV`, or one listed in `DXMT_ALLOWED_KEY_ENVS`; never `DXMT_API_KEY`;
- and the host of `base_url` is that provider's API host, a loopback address, the host of
  `DXMT_EMBEDDING_BASE_URL`, or one listed in `DXMT_ALLOWED_PROVIDER_HOSTS`.

Link-local and cloud metadata addresses are refused whether or not a key is involved. A
variable that is unset sends no key, so a keyless local provider needs no configuration. For a
self-hosted provider that does need a key:

```bash
DXMT_ALLOWED_KEY_ENVS=TEI_KEY DXMT_ALLOWED_PROVIDER_HOSTS=10.0.0.7 \
TEI_KEY=... ./target/release/vectordb serve
```

## Computing vectors yourself

Nothing above is required. A table created with a `dimension` and no `embedding` takes the
vectors you send, and the LangChain and LlamaIndex packages under `clients/` embed on the
client side by default.
