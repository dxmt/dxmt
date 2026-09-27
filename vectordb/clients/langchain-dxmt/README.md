# LangChain DXMT Integration

[![PyPI version](https://badge.fury.io/py/langchain-dxmt.svg)](https://badge.fury.io/py/langchain-dxmt)
[![Python Version](https://img.shields.io/pypi/pyversions/langchain-dxmt)](https://pypi.org/project/langchain-dxmt/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/dxmt/dxmt/actions/workflows/ci.yml/badge.svg)](https://github.com/dxmt/dxmt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dxmt/vectordb/branch/main/graph/badge.svg)](https://codecov.io/gh/dxmt/vectordb)

Official LangChain integration for DXMT VectorDB - a lightweight, horizontally scalable vector database with storage native architecture written in Rust.

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Advanced Usage](#advanced-usage)
- [Configuration Options](#configuration-options)
- [Performance](#performance)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Architecture

`langchain-dxmt` is a **thin adapter layer** that provides LangChain VectorStore integration for DXMT VectorDB. It depends on [`dxmt-python-sdk`](https://github.com/dxmt/dxmt/tree/main/vectordb/clients/dxmt-python-sdk) for all communication with the DXMT server.

```
┌─────────────────────────────────────────────────────────┐
│            User Application Layer                        │
│  (LangChain RAG, Q&A, ChatBots, etc.)                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         langchain-dxmt (This Package)                    │
│  • Implements LangChain VectorStore interface           │
│  • Document ↔ Record conversion                         │
│  • Embeddings integration                               │
└────────────────┬────────────────────────────────────────┘
                 │ depends on
                 ▼
┌─────────────────────────────────────────────────────────┐
│         dxmt-python-sdk (Dependency)                     │
│  • HTTP client (sync & async)                           │
│  • All DXMT API endpoints                               │
│  • Error handling & type definitions                    │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/REST API
                 ▼
┌─────────────────────────────────────────────────────────┐
│            DXMT Server (Rust)                            │
│  • Vector storage & indexing                            │
│  • Search operations & full-text search                 │
└─────────────────────────────────────────────────────────┘
```

**Benefits of this architecture:**
- ✅ **Choice**: Use SDK directly or via LangChain
- ✅ **Consistency**: Same underlying client everywhere
- ✅ **Single Source**: Bug fixes in SDK benefit all integrations
- ✅ **Reduced Duplication**: ~700 lines of code eliminated

For more details, see [ARCHITECTURE.md](./docs/ARCHITECTURE.md).

## Installation

```bash
pip install langchain-dxmt
```

This will automatically install [`dxmt-python-sdk`](https://github.com/dxmt/dxmt/tree/main/vectordb/clients/dxmt-python-sdk) as a dependency.

### Requirements

- Python 3.9+
- DXMT VectorDB server (running locally or remotely)
- OpenAI API key (or other embedding provider)

### Using dxmt-python-sdk Directly

If you don't need LangChain integration, you can use the SDK directly:

```bash
pip install dxmt-python-sdk
```

```python
from dxmt import Client

# Direct API access without LangChain
client = Client(host="localhost", port=9999)
client.create_table(name="docs", dimension=1536)
client.insert_records(table_name="docs", records=[...])
results = client.search(table_name="docs", vector=[...], k=10)
```

## Quick Start

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

# Initialize
embeddings = OpenAIEmbeddings()
vector_store = DXMTVectorDB(
    table_name="my_documents",
    embedding=embeddings,
    dimension=1536,
    host="localhost",
    port=9999,
)

# Add documents
texts = ["Document 1 content", "Document 2 content"]
metadatas = [{"source": "doc1"}, {"source": "doc2"}]
vector_store.add_texts(texts, metadatas)

# Search
results = vector_store.similarity_search("query text", k=3)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### Authentication

A server started with `DXMT_API_KEY` answers 401 to everything but its health probes until the key is sent. It goes out as the `api-key` header; send it over TLS, or it is readable at every hop.

```python
store = DXMTVectorDB(table_name="docs", embedding=emb, dimension=768,
                     host="db.example.com", ssl=True, api_key="your-key")
```

Ignored when you pass `client=` — configure that client instead.

## Features

- ✅ **Vector Similarity Search**: Multiple distance metrics (Cosine, Euclidean, Manhattan, Dot Product)
- ✅ **Multiple Index Types**: HNSW, IVF, IVF-PQ, LSH, Brute Force
- ✅ **Full-Text Search**: BM25 with Tantivy integration
- ✅ **Hybrid Search**: Combine vector similarity + full-text search
- ✅ **Chinese Text Support**: Jieba tokenizer for Chinese word segmentation
- ✅ **Metadata Filtering**: Filter results by metadata
- ✅ **Async Support**: Async operations for better performance
- ✅ **Batch Operations**: Multiple insert modes for optimization
- ✅ **Quantization**: Scalar, Product, Binary quantization support

## Advanced Usage

### Hybrid Search (Vector + Text)

```python
# Enable full-text search during initialization
vector_store = DXMTVectorDB(
    table_name="docs",
    embedding=embeddings,
    dimension=1536,
    enable_full_text_search=True,
)

# Perform hybrid search
results = vector_store.hybrid_search(
    query="machine learning",
    k=5,
    vector_weight=0.7,  # 70% vector, 30% text
)
```

### Chinese Text Search

```python
from langchain_dxmt import DXMTVectorDB

vector_store = DXMTVectorDB(
    table_name="chinese_docs",
    embedding=embeddings,
    dimension=1536,
    enable_full_text_search=True,
    text_fields=[
        {
            "name": "content",
            "source": {"metadata_field": "text"},
            "analyzer": "chinese",
            "boost": 1.0,
        }
    ],
)
```

### Metadata Filtering

```python
results = vector_store.similarity_search(
    query="AI research",
    k=10,
    filter={"category": "machine_learning", "year": 2024}
)
```

### Custom Index Type

```python
vector_store = DXMTVectorDB(
    table_name="large_dataset",
    embedding=embeddings,
    dimension=768,
    index_type="ivf_pq",  # Product Quantization for memory efficiency
    distance_metric="cosine",
)
```

### Async Operations

```python
import asyncio

async def main():
    # Add documents asynchronously
    await vector_store.aadd_texts(texts, metadatas)

    # Search asynchronously
    results = await vector_store.asimilarity_search("query", k=5)
    for doc in results:
        print(doc.page_content)

asyncio.run(main())
```

### As LangChain Retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "filter": {"source": "documentation"}}
)

# Use in chains
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever,
)

answer = qa_chain.invoke("What is DXMT VectorDB?")
```

## Configuration Options

### Connection Parameters

- `host`: DXMT server hostname (default: "localhost")
- `port`: DXMT server port (default: 9999)
- `ssl`: Use HTTPS (default: False)
- `headers`: Additional HTTP headers (optional)

### Index Parameters

- `index_type`: Index algorithm
  - `"hnsw"` (default): Hierarchical Navigable Small World
  - `"ivf"`: Inverted File Index
  - `"ivf_pq"`: IVF with Product Quantization
  - `"lsh"`: Locality-Sensitive Hashing
  - `"brute_force"`: Exact search

- `distance_metric`: Distance calculation method
  - `"cosine"` (default): Cosine similarity
  - `"euclidean"`: Euclidean distance
  - `"manhattan"`: Manhattan distance
  - `"dot"`: Dot product

### Full-Text Search Parameters

- `enable_full_text_search`: Enable BM25 full-text search (default: False)
- `text_fields`: Configuration for text fields (list of dicts)

## Performance

DXMT VectorDB offers excellent performance characteristics:

| Metric | Value | Configuration |
|--------|-------|---------------|
| **Query Latency (p50)** | ~15ms | 1M vectors, 1536-dim, k=10 |
| **Query Latency (p95)** | ~38ms | 1M vectors, 1536-dim, k=10 |
| **Insert Throughput** | ~1500 docs/sec | Single server |
| **Memory Usage** | ~5.5 GB | 1M vectors, HNSW index |
| **Memory Usage (IVF-PQ)** | ~2 GB | 1M vectors, compressed |

*Benchmarks run on standard server hardware. Performance may vary based on configuration and dataset.*

### Performance Tuning

```python
# High-performance configuration
vector_store = DXMTVectorDB(
    table_name="high_perf",
    embedding=embeddings,
    dimension=1536,
    index_type="hnsw",           # Fast approximate search
    enable_full_text_search=True,
)

# Memory-efficient configuration
vector_store = DXMTVectorDB(
    table_name="memory_eff",
    embedding=embeddings,
    dimension=1536,
    index_type="ivf_pq",         # Product quantization
    enable_full_text_search=False,
)
```

## Examples

Complete working examples are available in the [`examples/`](./examples/) directory:

- **[basic_usage.py](./examples/basic_usage.py)** - Getting started with basic operations
- **[hybrid_search.py](./examples/hybrid_search.py)** - Hybrid vector + text search
- **[async_usage.py](./examples/async_usage.py)** - Async operations and concurrent searches
- **[chinese_text_search.py](./examples/chinese_text_search.py)** - Chinese language support
- **[advanced_configuration.py](./examples/advanced_configuration.py)** - Advanced configuration options

## Documentation

- **[Architecture Overview](./docs/ARCHITECTURE.md)** - Package architecture and design decisions
- **[API Reference](./docs/API_REFERENCE.md)** - Complete API documentation
- **[Testing Guide](./tests/README.md)** - Running and writing tests
- **[Contributing Guide](./docs/CONTRIBUTING.md)** - How to contribute
- **[Changelog](./docs/CHANGELOG.md)** - Version history
- **[dxmt-python-sdk Documentation](https://github.com/dxmt/dxmt/tree/main/vectordb/clients/dxmt-python-sdk)** - Core SDK documentation

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy langchain_dxmt
```

## License

Apache License 2.0 - see LICENSE file for details

## Support

- [GitHub Issues](https://github.com/dxmt/dxmt/issues)
- [Documentation](https://github.com/dxmt/dxmt)
- [Community Discord](https://discord.gg/dxmt)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.
