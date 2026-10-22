# LlamaIndex DXMT Vector Store Integration

DXMT VectorDB integration for LlamaIndex, providing a powerful vector store backend for LLM applications.

## Features

- **Vector similarity search** — cosine, euclidean, manhattan and dot.
- **Metadata filtering** — `MetadataFilters` translated to DXMT's filter language, including
  `and`/`or` nesting.
- **MMR** (`VectorStoreQueryMode.MMR`) — reranked **on the server**, not by a client-side
  postprocessor, so the diversification draws on candidates that were never sent over the
  wire. `mmr_threshold` is the usual lambda; `fetch_k=` sets how many candidates it reranks.
- **Sparse search** (`VectorStoreQueryMode.SPARSE`) — pass
  `DXMTVectorStore(..., sparse_encoder=fn)`, where `fn` maps a string to
  `{"indices": [...], "values": [...], "dimension": n}`. The same encoder runs on insert.
- **Hybrid and full-text search** — need a server built with `--features text-bm25`, which is
  **not** the default. Without it they raise `FEATURE_DISABLED` rather than quietly returning
  the vector half.
- **Batch inserts**, sized by `batch_size`.

A query mode this store does not serve — `SEMANTIC_HYBRID`, `SVM`, the regression modes —
raises `NotImplementedError` naming the mode. It used to answer them with a plain dense
search, which a caller cannot tell from a result that honoured the mode.

### What this integration does not do

Listed because the list above used to include them.

- **Async is not really async.** `aquery` and `async_add` call the synchronous methods, so
  every `await` blocks the event loop: twenty concurrent `aquery` calls take 14× the time of
  one, and a coroutine scheduled every millisecond gets zero turns while a query is in
  flight. Use the sync methods and your own thread pool until this is wired to the SDK's
  async client, which does exist. (The LangChain integration in this repository is wired and
  measures 1.0×.)
- **No choice of index type.** `index_type=` is accepted and does not decide anything: the
  server picks the index from the table's size and settings and overwrites the request. There
  is no LSH implementation at all, and `ivf_pq` is refused by the server by name.
- **Object storage is not a property of this package.** It is an HTTP client; where the
  server keeps its data is the server's business, and its S3 backend is a non-default build
  feature.

## Installation

```bash
pip install llama-index-vector-stores-dxmt
```

Or with poetry:

```bash
poetry add llama-index-vector-stores-dxmt
```

Neither this package nor `dxmt-python-sdk` is on PyPI yet. Until the packages are published, install from source: the SDK first,
because this package's dependency on `dxmt-python-sdk` would otherwise be looked up on PyPI and
fail.

```bash
# from a checkout of the repository
pip install ./vectordb/clients/dxmt-python-sdk
pip install ./vectordb/clients/llama-index-vector-stores-dxmt
```

## Quick Start

### Basic Usage

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Initialize vector store
vector_store = DXMTVectorStore(
    table_name="my_documents",
    dimension=1536,  # OpenAI embedding dimension
    host="localhost",
    port=9999,
)

# Create index with vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=OpenAIEmbedding(),
)

# Add documents
documents = [
    Document(text="LlamaIndex is a data framework for LLM applications"),
    Document(text="Vector databases enable semantic search"),
]

for doc in documents:
    index.insert(doc)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is LlamaIndex?")
print(response)
```

### Using Existing Table

```python
from llama_index.vector_stores.dxmt import DXMTVectorStore

# Connect to existing table (auto-detects configuration)
vector_store = DXMTVectorStore.from_existing_table(
    table_name="existing_docs",
    host="localhost",
    port=9999,
)

# Use with LlamaIndex
index = VectorStoreIndex.from_vector_store(vector_store)
```

### Authentication

A server started with `DXMT_API_KEY` answers 401 to everything but its health probes until the key is sent. It goes out as the `api-key` header; send it over TLS, or it is readable at every hop.

```python
store = DXMTVectorStore(table_name="docs", dimension=768,
                        host="db.example.com", ssl=True, api_key="your-key")
```

The key is handed to the client and is not a field of the store, so it does not come back out of
`to_dict()` or a persisted index. `from_existing_table(..., api_key=...)` takes it too.

### Hybrid Search

Combine vector similarity with full-text search:

```python
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)

# Create store with full-text search enabled
vector_store = DXMTVectorStore(
    table_name="hybrid_docs",
    dimension=1536,
    enable_full_text_search=True,  # Enable BM25 search
)

# Perform hybrid search
query = VectorStoreQuery(
    query_embedding=embedding,
    query_str="machine learning",
    mode=VectorStoreQueryMode.HYBRID,
    similarity_top_k=5,
    alpha=0.7,  # 70% vector, 30% text
)

results = vector_store.query(query)
```

### Metadata Filtering

Filter search results using metadata:

```python
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)

# Create filters
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value=2022, operator=FilterOperator.GTE),
        MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ),
    ],
    condition=FilterCondition.AND,
)

# Search with filters
query = VectorStoreQuery(
    query_embedding=embedding,
    similarity_top_k=5,
    filters=filters,
)

results = vector_store.query(query)
```

### Advanced Configuration

```python
from llama_index.vector_stores.dxmt import DXMTVectorStore

vector_store = DXMTVectorStore(
    table_name="advanced_docs",
    dimension=1536,

    # Index configuration
    index_type="hnsw",  # Options: hnsw, ivf, ivf_pq, lsh, brute_force
    distance_metric="cosine",  # Options: cosine, euclidean, manhattan, dot

    # Performance tuning
    batch_size=500,  # Batch size for bulk operations

    # Full-text search
    enable_full_text_search=True,
    text_fields=[
        {
            "name": "content",
            # IMPORTANT: LlamaIndex stores text in "_node_content" field
            "source": {"metadata_field": "_node_content"},
            "analyzer": "standard",
            "boost": 1.0,
        }
    ],

    # Connection
    host="localhost",
    port=9999,
    ssl=False,
)
```

## Configuration Options

### Vector Store Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table_name` | str | `"llamaindex"` | Name of the table in DXMT |
| `dimension` | int | None | Vector dimension (required for new tables) |
| `host` | str | `"localhost"` | DXMT server host |
| `port` | int | `9999` | DXMT server port |
| `ssl` | bool | `False` | Use SSL for connection |
| `client` | DXMTClient | None | Optional existing DXMT client |

### Index Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_type` | str | `"hnsw"` | Index algorithm (hnsw, ivf, ivf_pq, lsh, brute_force) |
| `distance_metric` | str | `"cosine"` | Distance metric (cosine, euclidean, manhattan, dot) |

### Performance Tuning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `100` | Batch size for bulk operations |

**Insert Modes:**
- `standard` - Standard insert with immediate indexing
- `lock_free` - Lock-free insertion for high concurrency
- `streaming` - Streaming insert for large datasets
- `deferred_index` - Defer indexing until later
- `ultimate` - Maximum performance mode
- `batch` - Optimized batch insert
- `concurrent_batch` - Concurrent batch insert
- `adaptive` - Adaptive mode based on workload
- `tiered_storage` - Optimize for tiered storage
- `memory_constrained` - Optimize for limited memory

### Full-Text Search

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_full_text_search` | bool | `False` | Enable BM25 full-text search |
| `text_fields` | List[Dict] | None | Text field configuration |

**Important:** LlamaIndex stores document text in the `_node_content` metadata field. When configuring full-text search, use:

```python
text_fields=[
    {
        "name": "content",
        "source": {"metadata_field": "_node_content"},  # LlamaIndex's text field
        "analyzer": "standard",
        "boost": 1.0,
    }
]
```

## Search Modes

DXMT supports multiple search modes through LlamaIndex's `VectorStoreQueryMode`:

- **DEFAULT** - Standard vector similarity search
- **HYBRID** - Combined vector + full-text search
- **TEXT_SEARCH** - Full-text search only (BM25)

```python
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# Vector search (default)
query = VectorStoreQuery(
    query_embedding=embedding,
    mode=VectorStoreQueryMode.DEFAULT,
)

# Hybrid search
query = VectorStoreQuery(
    query_embedding=embedding,
    query_str="search text",
    mode=VectorStoreQueryMode.HYBRID,
    alpha=0.7,  # Weight: 0=text only, 1=vector only
)

# Text search
query = VectorStoreQuery(
    query_str="search text",
    mode=VectorStoreQueryMode.TEXT_SEARCH,
)
```

## Distance Metrics

Choose the appropriate distance metric for your use case:

- **cosine** - Cosine similarity (best for normalized embeddings, most common)
- **euclidean** - L2 distance (good for dense vectors)
- **manhattan** - L1 distance (good for sparse vectors)
- **dot** - Dot product (fast, requires normalized vectors)

## Index Types

DXMT automatically selects the best index type based on dataset size, but you can override:

- **hnsw** - Hierarchical Navigable Small World (best for <100K vectors, high recall)
- **ivf** - Inverted File Index (good for 100K-1M vectors)
- **ivf_pq** - IVF with Product Quantization (memory efficient for >1M vectors)
- **lsh** - Locality Sensitive Hashing (fast approximate search)
- **brute_force** - Exact search (best for <5K vectors, 100% recall)

## Examples

See the [examples](./examples/) directory for more usage patterns:

- [basic_usage.py](./examples/basic_usage.py) - Basic CRUD operations
- [hybrid_search.py](./examples/hybrid_search.py) - Hybrid search example
- [rag_example.py](./examples/rag_example.py) - Complete RAG pipeline
- [advanced_filters.py](./examples/advanced_filters.py) - Complex filtering

## Development

### Running Tests

```bash
# Install dev dependencies
poetry install --with dev

# Run all tests
pytest

# Run with coverage
pytest --cov=llama_index --cov-report=html

# Run specific test file
pytest tests/test_dxmt_vector_store.py

# Run integration tests
pytest -m integration
```

### Code Quality

```bash
# Format code
black llama_index tests examples

# Lint
ruff check llama_index tests examples

# Type checking
mypy llama_index
```

## Requirements

- Python 3.9+
- llama-index-core >= 0.11.0
- dxmt >= 1.0.0
- DXMT server running (for integration tests)

## Architecture

DXMT is a single-node server. Records and full vectors live in row files beside a
write-ahead log; only the HNSW-PQ graph, the PQ codes and the id map stay resident (about
0.5 KB per 768-dimension row at 10M rows). A table is an exact graph below 65,536 rows and a
quantized one above, with an exact rerank of the candidates from the row file. Durability is
the WAL plus periodic snapshots; `search_mode: "exact"` scores every row and is the ground
truth the approximate modes are measured against. `vectordb/docs/SUPPORT_MATRIX.md` in the
repository says which parts are covered by the 1.0 compatibility promise.

## Contributing

Contributions are welcome! Please see the [main repository](https://github.com/dxmt/dxmt) for contribution guidelines.

## License

Apache License 2.0 - see [LICENSE](./LICENSE) file for details.

## Links

- [DXMT VectorDB](https://github.com/dxmt/dxmt)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [Documentation](https://github.com/dxmt/dxmt/tree/main/vectordb/docs)

## Support

- GitHub Issues: https://github.com/dxmt/dxmt/issues
- Email: support@dxmt.online
