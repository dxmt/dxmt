# DXMT Python SDK

[![PyPI version](https://badge.fury.io/py/dxmt-python-sdk.svg)](https://badge.fury.io/py/dxmt-python-sdk)
[![Python Version](https://img.shields.io/pypi/pyversions/dxmt-python-sdk)](https://pypi.org/project/dxmt-python-sdk/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official Python SDK for DXMT VectorDB - a lightweight, horizontally scalable vector database with storage native architecture written in Rust.

## Features

- 🚀 **Simple API** - Clean, intuitive Python interface
- ⚡ **Async Support** - Full async/await support for high performance
- 🔍 **Vector Search** - Multiple distance metrics (Cosine, Euclidean, Manhattan, Dot Product)
- 📊 **Multiple Index Types** - HNSW, IVF, IVF-PQ, LSH, Brute Force
- 📝 **Full-Text Search** - BM25 with Tantivy integration
- 🔀 **Hybrid Search** - Combine vector similarity + full-text search
- 🇨🇳 **Chinese Support** - Jieba tokenizer for Chinese text
- 🎯 **Type Safe** - Full type hints with mypy support
- 📦 **Zero Dependencies** - Only requires httpx

> **⚠️ Important Note (Updated 2026-01-22):**
> When using full-text search or hybrid search, **you must use the `content` field** in your payload for main text content.
> Using other field names like `"text"` will cause full-text search to return 0 results.
> See [Field Naming Conventions](#field-naming-conventions) for details.

## Installation

```bash
pip install dxmt-python-sdk
```

## Quick Start

### Basic Usage

```python
from dxmt import Client

# Connect to DXMT server
client = Client(host="localhost", port=9999)

# Create a table
client.create_table(
    name="documents",
    dimension=1536,
    distance_metric="cosine",
    index_type="hnsw"
)

# Insert vectors
client.insert_records(
    table_name="documents",
    records=[
        {
            "id": "doc1",
            "vector": [0.1, 0.2, 0.3, ...],  # 1536-dim vector
            "payload": {
                "title": "Document 1",
                "content": "This is document 1",
                "source": "file.txt"
            }
        },
        {
            "id": "doc2",
            "vector": [0.4, 0.5, 0.6, ...],
            "payload": {
                "title": "Document 2",
                "content": "This is document 2",
                "source": "file.txt"
            }
        }
    ]
)

# Search for similar vectors
results = client.search(
    table_name="documents",
    vector=[0.1, 0.2, 0.3, ...],  # query vector
    limit=10
)

for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}")
```

### Async Usage

```python
import asyncio
from dxmt import Client

async def main():
    client = Client(host="localhost", port=9999)

    # Create table
    await client.acreate_table(
        name="documents",
        dimension=1536,
        distance_metric="cosine"
    )

    # Insert records
    await client.ainsert_records(
        table_name="documents",
        records=[...]
    )

    # Search
    results = await client.asearch(
        table_name="documents",
        vector=[0.1, 0.2, ...],
        limit=10
    )

    await client.aclose()

asyncio.run(main())
```

### Context Manager

```python
from dxmt import Client

# Automatic cleanup
with Client(host="localhost", port=9999) as client:
    client.create_table(name="test", dimension=128)
    # ... do work ...

# Async context manager
async with Client(host="localhost", port=9999) as client:
    await client.acreate_table(name="test", dimension=128)
    # ... do work ...
```

### Hybrid Search (Vector + Text)

```python
from dxmt import Client

client = Client(host="localhost", port=9999)

# Create table with full-text search enabled
client.create_table(
    name="documents",
    dimension=1536,
    distance_metric="cosine",
    enable_full_text_search=True  # Enable text search
)

# Insert documents with "content" field for text search
client.insert_records(
    table_name="documents",
    records=[
        {
            "id": "doc1",
            "vector": embedding1,
            "payload": {
                "title": "Machine Learning Guide",
                "content": "Machine learning is...",  # Required for text search
                "category": "AI"
            }
        }
    ]
)

# Hybrid search: combine vector similarity + text matching
results = client.hybrid_search(
    table_name="documents",
    vector=[0.1, 0.2, ...],          # Vector query
    text_query="machine learning",   # Text query
    alpha=0.5,                        # Balance (0=text only, 1=vector only)
    limit=10
)
```

### Table Management

```python
from dxmt import Client

client = Client()

# List all tables
tables = client.list_tables()
print(f"Tables: {tables}")

# Get table info
info = client.get_table("documents")
print(f"Dimension: {info['dimension']}")
print(f"Distance: {info['distance_metric']}")

# Delete table
client.delete_table("documents")
```

### Record Operations

```python
# Get records by IDs
records = client.get_records(
    table_name="documents",
    ids=["doc1", "doc2"]
)

# Delete records
client.delete_records(
    table_name="documents",
    ids=["doc1", "doc2"]
)
```

## Field Naming Conventions

### Standard Payload Fields

DXMT uses specific field names for optimal functionality, especially when using **full-text search** and **hybrid search**:

| Field Name | Purpose | Required | Example |
|-----------|---------|----------|---------|
| `content` | Main searchable text content | **Yes** for text search | Article body, document content |
| `title` | Document title (higher search boost) | No | "Introduction to AI" |
| `description` | Brief description/summary | No | "An overview of..." |
| `tags` | Categorization tags | No | ["AI", "ML"] |

### ⚠️ Important: Use `content` for Text Search

When using full-text search or hybrid search, **you must use the `content` field** for your main text:

```python
# ✅ CORRECT - Use "content"
records = [{
    "id": "doc1",
    "vector": [0.1, 0.2, ...],
    "payload": {
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of AI...",  # ✅
        "category": "AI"
    }
}]

# ❌ WRONG - Don't use "text" or other field names
records = [{
    "id": "doc1",
    "vector": [0.1, 0.2, ...],
    "payload": {
        "title": "Introduction to Machine Learning",
        "text": "Machine learning is...",  # ❌ Won't work!
        "category": "AI"
    }
}]
```

### Why Field Names Matter

The full-text search engine (Tantivy) maps payload fields to its internal schema:

- `content` → Main searchable field (BM25 ranking)
- `title` → Title field (2x boost in search)
- `description` → Description field (0.8x boost)
- `tags` → Tags field (1.5x boost, supports filtering)

**If you use incorrect field names:**
- Full-text search will return 0 results
- Hybrid search text component will not work
- No error is thrown (silent failure)

### Full-Text Search Example

```python
# Create table with full-text search
client.create_table(
    name="articles",
    dimension=128,
    distance_metric="cosine",
    enable_full_text_search=True  # Enable text search
)

# Insert with correct field names
client.insert_records(
    table_name="articles",
    records=[
        {
            "id": "art1",
            "vector": embedding1,
            "payload": {
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks...",  # ✅ Required
                "category": "AI",
                "tags": ["deep-learning", "neural-networks"]
            }
        }
    ]
)

# Full-text search
results = client.full_text_search(
    table_name="articles",
    text_query="neural networks",
    limit=10
)

# Hybrid search (vector + text)
results = client.hybrid_search(
    table_name="articles",
    vector=query_embedding,
    text_query="deep learning",
    alpha=0.5,  # 0.5 = balanced, 0.0 = text only, 1.0 = vector only
    limit=10
)
```

### Custom Metadata Fields

You can add any custom fields alongside the standard fields:

```python
"payload": {
    "content": "Main text here...",      # ✅ Required for text search
    "title": "Document Title",           # Standard field
    "author": "John Doe",                # ✅ Custom field
    "publish_date": "2024-01-15",       # ✅ Custom field
    "rating": 4.5,                       # ✅ Custom field
    "custom_metadata": {...}             # ✅ Any structure
}
```

### Migration from Old Code

If your code uses `"text"` field, update it to `"content"`:

```python
# Old code (before 2026-01-22)
"payload": {"text": "..."}

# New code (after 2026-01-22)
"payload": {"content": "..."}  # ✅ Required change
```

After updating field names, you need to:
1. Re-create tables with full-text search
2. Re-insert all records with corrected field names

## Advanced Features

### Distance Metrics

```python
# Choose your distance metric
client.create_table(
    name="table",
    dimension=128,
    distance_metric="cosine"      # or "euclidean", "manhattan", "dot"
)
```

### Index Types

```python
# Choose optimal index for your use case
client.create_table(
    name="table",
    dimension=128,
    index_type="hnsw"  # Fast, high recall
    # or: "ivf", "ivf_pq" (memory efficient), "lsh", "brute_force"
)
```

### Insert Modes

```python
# Optimize insertion for your workload
client.insert_records(
    table_name="documents",
    records=[...],
    mode="batch"  # Batch insertion
    # or: "streaming", "lock_free", "parallel", etc.
)
```

### Filtering

```python
# Filter search results by metadata
results = client.search(
    table_name="documents",
    vector=[0.1, 0.2, ...],
    limit=10,
    filter={
        "source": {"$eq": "file.txt"},
        "date": {"$gte": "2024-01-01"}
    }
)
```

## Integrations

### mem0 Integration

DXMT can be used as a vector storage backend for [mem0](https://mem0.ai), an intelligent memory layer for AI agents.

**Installation:**
```bash
pip install mem0ai dxmt-python-sdk
```

**Usage:**
```python
from mem0 import Memory
from dxmt import Client
from dxmt.mem0 import DXMTMemoryStore

# Initialize DXMT client
client = Client(host="localhost", port=9999)

# Create DXMT memory store
vector_store = DXMTMemoryStore(
    collection_name="user_memories",
    client=client,
    embedding_model_dims=1536  # OpenAI text-embedding-ada-002
)

# Configure mem0
config = {
    "vector_store": {
        "provider": "custom",
        "config": {"custom_vector_store": vector_store}
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-ada-002"}
    }
}

m = Memory.from_config(config)

# Store memories
m.add("I love Python programming", user_id="user_123")

# Search memories
results = m.search("programming preferences", user_id="user_123")
for mem in results:
    print(f"{mem['memory']} (score: {mem['score']:.3f})")
```

**Features:**
- Multi-scope memory (user_id, agent_id, run_id)
- Automatic deduplication via hash
- Efficient filtering and semantic search
- Timestamp tracking (created_at, updated_at)
- Cost-efficient tiered storage

**Examples:** See [../examples/mem0_integration.py](../examples/mem0_integration.py) for comprehensive usage examples

## API Reference

### Client

#### `__init__(host="localhost", port=9999, ssl=False, timeout=30.0)`

Create a new DXMT client.

**Parameters:**
- `host` (str): Server hostname
- `port` (int): Server port
- `ssl` (bool): Use HTTPS
- `timeout` (float): Request timeout in seconds

#### `create_table(name, dimension, distance_metric="cosine", index_type="hnsw", **kwargs)`

Create a new table.

**Parameters:**
- `name` (str): Table name
- `dimension` (int): Vector dimension
- `distance_metric` (str): Distance metric ("cosine", "euclidean", "manhattan", "dot")
- `index_type` (str): Index type ("hnsw", "ivf", "ivf_pq", "lsh", "brute_force")

#### `insert_records(table_name, records, mode="standard")`

Insert records into a table.

**Parameters:**
- `table_name` (str): Table name
- `records` (list[dict]): Records to insert (each record must have `id`, `vector`, and `payload`)
- `mode` (str): Insert mode ("standard", "batch", "streaming", etc.)

**Record Structure:**
```python
{
    "id": "unique_id",           # Required: Record ID
    "vector": [0.1, 0.2, ...],   # Required: Embedding vector
    "payload": {                 # Required: Metadata
        "content": "...",        # For full-text search
        "title": "...",          # Optional
        # ... custom fields
    }
}
```

#### `search(table_name, vector, limit=10, filter=None, **kwargs)`

Search for similar vectors.

**Parameters:**
- `table_name` (str): Table name
- `vector` (list[float]): Query vector
- `limit` (int): Number of results
- `filter` (dict): Metadata filter

#### `hybrid_search(table_name, vector, text_query, alpha=0.5, limit=10)`

Hybrid search combining vector + text.

**Parameters:**
- `table_name` (str): Table name
- `vector` (list[float]): Query vector
- `text_query` (str): Text query
- `alpha` (float): Weight (0=text only, 1=vector only)
- `limit` (int): Number of results

### Async Methods

All methods have async equivalents with `a` prefix:
- `acreate_table()`, `ainsert_records()`, `asearch()`, etc.

### Metadata Filters

DXMT supports flexible metadata filtering with MongoDB-style operators. See **[FILTERS_GUIDE.md](FILTERS_GUIDE.md)** for comprehensive documentation.

Quick examples:

```python
# Simple equality
filter = {"category": "AI", "year": 2023}

# Comparison operators
filter = {"rating": {"$gte": 8.0}, "year": {"$lt": 2024}}

# Logical operators
filter = {
    "$and": [
        {"year": {"$gte": 2022}},
        {"rating": {"$gte": 8.0}},
        {"category": {"$in": ["AI", "ML", "DL"]}}
    ]
}

# Use in search
results = client.search(
    table_name="documents",
    vector=query_vector,
    limit=10,
    filter=filter
)
```

**Supported operators:**
- Comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- Array: `$in`, `$nin`
- String: `$contains`, `$startswith`, `$endswith`
- Logical: `$and`, `$or`, `$not`

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/dxmt/dxmt.git
cd dxmt/vectordb/dxmt-python-sdk

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=dxmt --cov-report=term-missing

# Run specific test file
pytest tests/test_client.py -v
```

### Code Quality

```bash
# Lint
ruff check dxmt

# Format
ruff format dxmt

# Type check
mypy dxmt
```

## Requirements

- Python 3.9+
- DXMT VectorDB server (running locally or remotely)

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Links

- **Homepage**: https://github.com/dxmt/dxmt
- **Documentation**: https://docs.dxmt.io
- **Issues**: https://github.com/dxmt/dxmt/issues
- **PyPI**: https://pypi.org/project/dxmt-python-sdk/

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
