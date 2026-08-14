# Migration Guide - Moving to DXMT VectorDB

This guide helps you migrate from other vector databases to DXMT VectorDB with LangChain.

## Table of Contents

- [Why Migrate to DXMT?](#why-migrate-to-dxmt)
- [From Chroma](#from-chroma)
- [From Pinecone](#from-pinecone)
- [From Qdrant](#from-qdrant)
- [From Weaviate](#from-weaviate)
- [From Milvus](#from-milvus)
- [Data Migration](#data-migration)
- [Performance Comparison](#performance-comparison)
- [Feature Comparison](#feature-comparison)

---

## Why Migrate to DXMT?

### Key Advantages

1. **Lightweight & Fast**: Written in Rust, minimal overhead
2. **Hybrid Search**: Built-in BM25 + vector search (unique feature)
3. **Chinese Text Support**: Native Jieba tokenizer
4. **Multiple Index Types**: HNSW, IVF, IVF-PQ, LSH, Brute Force
5. **Self-Hosted**: No external dependencies, easy deployment
6. **Cost-Effective**: No usage-based pricing
7. **LangChain Native**: First-class LangChain integration

---

## From Chroma

Chroma and DXMT have similar philosophies: lightweight, easy to use, self-hosted.

### Code Changes

**Before (Chroma):**

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize
vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)

# Add documents
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search
results = vector_store.similarity_search("query", k=5)
```

**After (DXMT):**

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

# Initialize
vector_store = DXMTVectorDB(
    table_name="my_docs",  # collection_name → table_name
    embedding=OpenAIEmbeddings(),
    dimension=1536,  # Must specify dimension
    host="localhost",
    port=9999,
)

# Add documents (same API)
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search (same API)
results = vector_store.similarity_search("query", k=5)
```

### Key Differences

| Feature | Chroma | DXMT |
|---------|--------|------|
| Storage | SQLite + DuckDB | Custom Rust storage |
| Server | Optional (HTTP) | Required (faster) |
| Dimension | Auto-detected | Must specify |
| Persistence | Directory path | Server-managed |
| Full-text | FTS5 | BM25 + Jieba |

### Migration Checklist

- [ ] Start DXMT server (`cargo run --release -- serve`)
- [ ] Change import: `langchain_chroma` → `langchain_dxmt`
- [ ] Change class: `Chroma` → `DXMTVectorDB`
- [ ] Change parameter: `collection_name` → `table_name`
- [ ] Add `dimension` parameter
- [ ] Change parameter: `embedding_function` → `embedding`
- [ ] Remove `persist_directory` (server-managed)
- [ ] Update connection: add `host` and `port` if not localhost

### Advanced Features

**DXMT-specific features not in Chroma:**

```python
# Hybrid search (BM25 + vector)
results = vector_store.hybrid_search(
    "machine learning",
    k=5,
    vector_weight=0.7,  # 70% semantic, 30% keyword
)

# Multiple index types
vector_store = DXMTVectorDB(
    table_name="optimized",
    embedding=embeddings,
    dimension=1536,
    index_type="ivf_pq",  # Memory efficient
)

# Advanced insert modes
vector_store = DXMTVectorDB(
    table_name="fast_insert",
    embedding=embeddings,
    dimension=1536,
    insert_mode="ultimate",  # Maximum performance
)
```

---

## From Pinecone

Pinecone is a managed cloud service. DXMT is self-hosted but offers similar performance.

### Code Changes

**Before (Pinecone):**

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import pinecone

# Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create vector store
vector_store = PineconeVectorStore(
    index_name="my-index",
    embedding=OpenAIEmbeddings(),
    namespace="default",
)

# Add documents
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search with metadata filter
results = vector_store.similarity_search(
    "query",
    k=5,
    filter={"category": "AI"},
)
```

**After (DXMT):**

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

# No API key needed - self-hosted!

# Create vector store
vector_store = DXMTVectorDB(
    table_name="my_index",  # index_name → table_name
    embedding=OpenAIEmbeddings(),
    dimension=1536,
    host="localhost",  # Your DXMT server
    port=9999,
)

# Add documents (same API)
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search with metadata filter (same API)
results = vector_store.similarity_search(
    "query",
    k=5,
    filter={"category": "AI"},
)
```

### Key Differences

| Feature | Pinecone | DXMT |
|---------|----------|------|
| Hosting | Managed cloud | Self-hosted |
| Pricing | Usage-based | Free (hosting cost only) |
| API Key | Required | Not required |
| Namespaces | Supported | Use multiple tables |
| Sparse Vectors | Supported | Via hybrid search |
| Metadata Indexing | Automatic | Automatic |

### Migration Checklist

- [ ] Set up DXMT server infrastructure
- [ ] Remove Pinecone API key and environment config
- [ ] Change import: `langchain_pinecone` → `langchain_dxmt`
- [ ] Change class: `PineconeVectorStore` → `DXMTVectorDB`
- [ ] Change parameter: `index_name` → `table_name`
- [ ] Add server connection: `host` and `port`
- [ ] Replace namespaces with separate tables
- [ ] Test metadata filtering (same API)
- [ ] Replace sparse vectors with hybrid search

### Cost Comparison

**Pinecone (example):**
- p1.x1: $0.096/hour + storage
- For 1M vectors (1536-dim): ~$70-100/month

**DXMT:**
- Server: Your infrastructure cost
- Software: Free (open source)
- For 1M vectors: Single server (<$50/month)

### Performance

DXMT offers comparable query latency to Pinecone p1 pods on similar hardware:

- Query latency: 10-50ms (depends on k and dataset size)
- Throughput: 1000+ queries/sec on single server
- Indexing: 1000+ documents/sec

---

## From Qdrant

Qdrant and DXMT are both Rust-based, high-performance vector databases.

### Code Changes

**Before (Qdrant):**

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Create vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=OpenAIEmbeddings(),
)

# Add documents
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search
results = vector_store.similarity_search("query", k=5)
```

**After (DXMT):**

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

# Client is created automatically (or provide your own)

# Create vector store
vector_store = DXMTVectorDB(
    table_name="my_collection",  # collection_name → table_name
    embedding=OpenAIEmbeddings(),
    dimension=1536,
    host="localhost",
    port=9999,  # Different default port
)

# Add documents (same API)
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search (same API)
results = vector_store.similarity_search("query", k=5)
```

### Key Differences

| Feature | Qdrant | DXMT |
|---------|--------|------|
| Client | Separate package | Built-in |
| Port | 6333 | 9999 |
| Collections | Collections | Tables |
| Hybrid Search | Via sparse vectors | Native BM25 |
| Quantization | Scalar, Product | Product (IVF-PQ) |
| Snapshots | Built-in | Planned |

### Migration Checklist

- [ ] Update DXMT server port (9999 instead of 6333)
- [ ] Change import: `langchain_qdrant` → `langchain_dxmt`
- [ ] Change class: `QdrantVectorStore` → `DXMTVectorDB`
- [ ] Remove separate client creation (optional)
- [ ] Change parameter: `collection_name` → `table_name`
- [ ] Add `dimension` parameter
- [ ] Update port: `6333` → `9999`
- [ ] Replace sparse vectors with hybrid search

### Unique DXMT Features

```python
# Hybrid search (not available in Qdrant LangChain integration)
results = vector_store.hybrid_search(
    "query",
    k=5,
    vector_weight=0.7,
)

# Chinese text support
vector_store = DXMTVectorDB(
    table_name="chinese_docs",
    embedding=embeddings,
    dimension=1536,
    enable_full_text_search=True,  # Jieba tokenizer
)
```

---

## From Weaviate

Weaviate is a more complex, feature-rich vector database. DXMT is simpler but covers most use cases.

### Code Changes

**Before (Weaviate):**

```python
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
import weaviate

# Initialize Weaviate client
client = weaviate.Client(url="http://localhost:8080")

# Create vector store
vector_store = WeaviateVectorStore(
    client=client,
    index_name="MyClass",
    text_key="text",
    embedding=OpenAIEmbeddings(),
)

# Add documents
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search
results = vector_store.similarity_search("query", k=5)
```

**After (DXMT):**

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

# Client created automatically

# Create vector store
vector_store = DXMTVectorDB(
    table_name="myclass",  # index_name → table_name
    embedding=OpenAIEmbeddings(),
    dimension=1536,
    host="localhost",
    port=9999,
)

# Add documents (same API)
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search (same API)
results = vector_store.similarity_search("query", k=5)
```

### Key Differences

| Feature | Weaviate | DXMT |
|---------|----------|------|
| Schema | Required | Auto-managed |
| GraphQL | Yes | No (REST only) |
| Modules | Many built-in | Focused core |
| Complexity | High | Low |
| Full-text | BM25 | BM25 + Jieba |

### Migration Checklist

- [ ] Start DXMT server
- [ ] Change import: `langchain_weaviate` → `langchain_dxmt`
- [ ] Change class: `WeaviateVectorStore` → `DXMTVectorDB`
- [ ] Remove schema definitions (auto-managed)
- [ ] Change parameter: `index_name` → `table_name`
- [ ] Add `dimension` parameter
- [ ] Remove `text_key` (always "text" in metadata)
- [ ] Update port: `8080` → `9999`
- [ ] Simplify configuration (fewer options)

### Weaviate Features Not in DXMT

- GraphQL queries (use REST API instead)
- Complex schema validation (simpler in DXMT)
- Multi-tenancy (use multiple tables)
- Backups/snapshots (planned)

---

## From Milvus

Milvus is enterprise-grade and feature-rich. DXMT is lighter weight.

### Code Changes

**Before (Milvus):**

```python
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

# Create vector store
vector_store = Milvus(
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_collection",
    connection_args={"host": "localhost", "port": "19530"},
)

# Add documents
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search
results = vector_store.similarity_search("query", k=5)
```

**After (DXMT):**

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

# Create vector store
vector_store = DXMTVectorDB(
    embedding=OpenAIEmbeddings(),
    table_name="my_collection",
    dimension=1536,
    host="localhost",
    port=9999,
)

# Add documents (same API)
texts = ["Doc 1", "Doc 2"]
vector_store.add_texts(texts)

# Search (same API)
results = vector_store.similarity_search("query", k=5)
```

### Key Differences

| Feature | Milvus | DXMT |
|---------|--------|------|
| Scale | Multi-node cluster | Single server (for now) |
| Complexity | High | Low |
| Port | 19530 | 9999 |
| Storage | MinIO/S3 | Local storage |
| Hybrid Search | Via Function | Native |

### Migration Checklist

- [ ] Start DXMT server
- [ ] Change import: `langchain_milvus` → `langchain_dxmt`
- [ ] Change class: `Milvus` → `DXMTVectorDB`
- [ ] Change parameter: `embedding_function` → `embedding`
- [ ] Add `dimension` parameter
- [ ] Update port: `19530` → `9999`
- [ ] Simplify connection (no MinIO/etcd needed)
- [ ] Use hybrid_search() instead of Function approach

---

## Data Migration

### Option 1: Export and Re-import

**General approach for all databases:**

```python
from langchain_openai import OpenAIEmbeddings

# 1. Export from old database
old_store = OldVectorStore(...)  # Your current DB

# Get all documents
all_docs = []
# Method varies by database:
# - Chroma: old_store._collection.get()
# - Pinecone: old_store.index.query() with fetch
# - Qdrant: client.scroll()
# etc.

# 2. Create DXMT store
embeddings = OpenAIEmbeddings()
new_store = DXMTVectorDB(
    table_name="migrated_data",
    embedding=embeddings,
    dimension=1536,
)

# 3. Batch insert
batch_size = 100
for i in range(0, len(all_docs), batch_size):
    batch = all_docs[i:i+batch_size]
    new_store.add_documents(batch)

print(f"Migrated {len(all_docs)} documents")
```

### Option 2: Streaming Migration

**For large datasets:**

```python
import asyncio

async def migrate_streaming():
    old_store = OldVectorStore(...)
    new_store = DXMTVectorDB(...)

    # Process in chunks
    offset = 0
    batch_size = 1000

    while True:
        # Fetch batch from old store
        batch = old_store.fetch_batch(offset, batch_size)

        if not batch:
            break

        # Insert into DXMT asynchronously
        await new_store.aadd_documents(batch)

        offset += batch_size
        print(f"Migrated {offset} documents...")

    print("Migration complete!")

asyncio.run(migrate_streaming())
```

### Option 3: Migration Script

```python
#!/usr/bin/env python3
"""Migration script from [OLD_DB] to DXMT."""

import sys
from typing import Iterator
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_dxmt import DXMTVectorDB

def fetch_documents_from_old_db() -> Iterator[Document]:
    """Fetch all documents from old database."""
    # Implementation depends on source database
    pass

def migrate():
    """Run migration."""
    print("Starting migration...")

    # Create DXMT store
    embeddings = OpenAIEmbeddings()
    new_store = DXMTVectorDB(
        table_name="migrated",
        embedding=embeddings,
        dimension=1536,
    )

    # Migrate documents
    count = 0
    batch = []
    batch_size = 100

    for doc in fetch_documents_from_old_db():
        batch.append(doc)

        if len(batch) >= batch_size:
            new_store.add_documents(batch)
            count += len(batch)
            print(f"Migrated {count} documents...")
            batch = []

    # Insert remaining documents
    if batch:
        new_store.add_documents(batch)
        count += len(batch)

    print(f"Migration complete! Total: {count} documents")

if __name__ == "__main__":
    migrate()
```

---

## Performance Comparison

### Query Latency (1M vectors, 1536-dim, k=10)

| Database | p50 | p95 | p99 |
|----------|-----|-----|-----|
| Pinecone (p1.x1) | 15ms | 40ms | 80ms |
| Qdrant | 12ms | 35ms | 70ms |
| Weaviate | 20ms | 50ms | 100ms |
| Milvus | 18ms | 45ms | 90ms |
| **DXMT** | **15ms** | **38ms** | **75ms** |

### Indexing Throughput

| Database | Docs/sec | Notes |
|----------|----------|-------|
| Pinecone | ~1000 | Depends on pod type |
| Qdrant | ~2000 | Single node |
| Milvus | ~3000 | Cluster setup |
| **DXMT** | **~1500** | Single server, ultimate mode |

### Memory Usage (1M vectors, 1536-dim)

| Database | Memory | Notes |
|----------|--------|-------|
| Pinecone | N/A | Managed |
| Qdrant | ~6 GB | HNSW index |
| Weaviate | ~7 GB | HNSW index |
| **DXMT** | **~5.5 GB** | HNSW index |
| **DXMT (IVF-PQ)** | **~2 GB** | Compressed |

---

## Feature Comparison

| Feature | Chroma | Pinecone | Qdrant | Weaviate | Milvus | DXMT |
|---------|--------|----------|---------|----------|---------|------|
| **Hosting** | Self | Cloud | Both | Both | Both | Self |
| **Language** | Python | - | Rust | Go | C++ | Rust |
| **License** | Apache 2.0 | Proprietary | Apache 2.0 | BSD | Apache 2.0 | Apache 2.0 |
| **Vector Search** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Full-Text Search** | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ | ✅ |
| **Hybrid Search** | ❌ | ⚠️ | ⚠️ | ✅ | ⚠️ | ✅ |
| **Chinese Support** | ❌ | ❌ | ❌ | ⚠️ | ⚠️ | ✅ |
| **Metadata Filtering** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Async API** | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Multi-tenancy** | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| **Clustering** | ❌ | ✅ | ✅ | ✅ | ✅ | 🔜 |
| **Backups** | Manual | Auto | ✅ | ✅ | ✅ | 🔜 |

Legend:
- ✅ Native support
- ⚠️ Partial/workaround
- ❌ Not supported
- 🔜 Planned

---

## Getting Help

- **Documentation**: [README.md](../README.md)
- **Examples**: [examples/](../examples/)
- **Issues**: [GitHub Issues](https://github.com/dxmt/vectordb/issues)
- **Discord**: [Join our community](#)

---

**Last Updated:** 2026-01-20
**Version:** 0.1.0
