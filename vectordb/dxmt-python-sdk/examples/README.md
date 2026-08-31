# DXMT Python SDK Examples

This directory contains comprehensive examples demonstrating various use cases and features of the DXMT Python SDK.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Basic Examples](#basic-examples)
- [Advanced Examples](#advanced-examples)
- [Real-World Applications](#real-world-applications)
- [Prerequisites](#prerequisites)
- [Running Examples](#running-examples)

## 🚀 Quick Start

### 1. Start DXMT Server

```bash
# From the repository root
cargo run -- serve --simple --memory-limit 1024
```

### 2. Install Dependencies

```bash
# Basic dependencies
pip install dxmt

# For semantic search examples
pip install sentence-transformers

# For LangChain integration
pip install langchain-dxmt langchain-openai langchain-core
```

### 3. Run Your First Example

```bash
cd examples
python3 basic_usage.py
```

## 🛠️ Common Utilities

### [common_utils.py](common_utils.py) 🔧
**Shared Functions for All Examples**

Reusable utilities to reduce code duplication:
- `create_test_client()` - Initialize DXMT client
- `generate_dummy_data()` - Generate test data
- `print_results()` - Pretty print search results
- `benchmark()` - Performance timing
- `setup_test_table()` - Standard table creation
- `compare_search_methods()` - Compare search strategies

**Usage in your examples:**
```python
from common_utils import create_test_client, print_results

client = create_test_client()
results = client.search(...)
print_results(results)
```

---

## 📚 Basic Examples

### [basic_usage.py](basic_usage.py)
**Fundamentals of DXMT SDK**

Learn the essential operations:
- Creating a client and connecting to server
- Creating tables with different configurations
- Inserting vectors with metadata
- Searching for similar vectors
- Retrieving and deleting records

```bash
python3 basic_usage.py
```

**Use Case:** Getting started with vector database operations

---

### [async_usage.py](async_usage.py)
**Asynchronous Operations for High Performance**

Demonstrates async/await patterns:
- Async table creation and management
- Parallel insertion with `ainsert_records()`
- Concurrent searches with `asyncio.gather()`
- Async record retrieval and deletion

```bash
python3 async_usage.py
```

**Use Case:** High-throughput applications requiring concurrent operations

---

### [hybrid_search.py](hybrid_search.py)
**Combining Vector and Text Search**

Explore hybrid search capabilities:
- Full-text search with BM25
- Vector similarity search
- Hybrid search with adjustable weights (alpha parameter)
- Comparing different search strategies

```bash
python3 hybrid_search.py
```

**Use Case:** Applications requiring both keyword matching and semantic understanding

---

## 🔍 Advanced Examples

### [filtered_search.py](filtered_search.py)
**Metadata Filtering and Complex Queries**

Master filtering techniques:
- Equality filters (`{"field": value}`)
- Comparison operators (`$lt`, `$lte`, `$gt`, `$gte`, `$ne`)
- Logical operators (`$and`, `$or`)
- Combining filters with vector search
- E-commerce product filtering example

```bash
python3 filtered_search.py
```

**Use Case:** E-commerce search, document filtering, constrained recommendations

---

### [batch_operations.py](batch_operations.py)
**High-Performance Batch Processing**

Optimize throughput:
- Batch insert modes (batch, parallel, lockfree)
- Performance benchmarking
- Large-scale data ingestion (10,000+ records)
- Batch retrieval and deletion
- Incremental update patterns

```bash
python3 batch_operations.py
```

**Use Case:** Data migration, ETL pipelines, bulk data processing

---

### [semantic_search.py](semantic_search.py)
**Real Semantic Search with Embeddings**

Build intelligent search:
- Using sentence-transformers for embeddings
- Semantic vs. keyword search comparison
- Technical documentation search
- Filtered semantic search
- Understanding embedding models

```bash
python3 semantic_search.py
```

**Requirements:** `pip install sentence-transformers`

**Use Case:** FAQ systems, documentation search, content discovery

---

### [semantic_search_openai.py](semantic_search_openai.py) ⭐ NEW
**Production-Quality Semantic Search with OpenAI**

Enhanced semantic search with superior understanding:
- Using OpenAI text-embedding-3-small (1536 dimensions)
- **3-5x better semantic understanding** than open-source models
- Higher relevance scores (0.6-0.9 vs 0.4-0.7)
- Better grasp of technical terms and concepts
- Side-by-side quality comparison

```bash
# Set API key
export OPENAI_API_KEY='your-api-key'

# Run example
python3 semantic_search_openai.py
```

**Requirements:**
- `pip install openai`
- OpenAI API key (get from https://platform.openai.com/api-keys)

**Cost:** ~$0.001 per 1000 queries (negligible!)

**Use Case:** Production documentation search, technical support, high-quality recommendations

**Why OpenAI?**
- 🎯 ~85% top-1 accuracy vs ~60% with open-source
- 📈 Higher relevance scores = better user experience
- 💰 Cost: <$1 per 100K queries
- 🚀 No model download needed

**When to use this over semantic_search.py:**
- ✅ Production applications
- ✅ Quality matters more than cost
- ✅ Technical content with domain terminology
- ✅ User-facing search features

---

## 🚀 Advanced Retrieval Techniques

### [mmr_search.py](mmr_search.py) ⭐ NEW
**Maximum Marginal Relevance (MMR) Search**

Balance relevance and diversity in search results:
- MMR algorithm for diverse results
- Lambda parameter tuning (0.0-1.0)
- Avoid redundant/similar results
- Clustered data demonstration
- Use case recommendations

```bash
python3 mmr_search.py
```

**Requirements:** `pip install numpy`

**Use Case:** Content recommendations, news aggregation, product search, research papers

**Why MMR?**
- Avoid showing too many similar results
- Increase result diversity
- Better user experience for exploration
- Perfect for recommendation systems

**Lambda parameter guide:**
- λ = 1.0: Pure relevance (like standard search)
- λ = 0.5: Balanced (recommended)
- λ = 0.0: Maximum diversity

---

### [fusion_search.py](fusion_search.py) ⭐ NEW
**Multi-Query Fusion with RRF**

Combine multiple search strategies using Reciprocal Rank Fusion:
- Multi-query fusion (query variations)
- Filtered fusion (different filters)
- Hybrid fusion (different weights)
- RRF algorithm implementation
- Comparison with single query

```bash
python3 fusion_search.py
```

**Use Case:** Complex queries, ambiguous intent, ensemble search, improving recall

**RRF Benefits:**
- Better coverage (more relevant results)
- Reduced position bias
- Ensemble effect (robust results)
- Combine any search types

**When to use:**
- Query intent is ambiguous
- Need to combine vector + text + filters
- Recall more important than speed

---

### [rag_application.py](rag_application.py) ⭐ NEW
**RAG (Retrieval-Augmented Generation)**

Complete RAG application with LLM integration:
- Document ingestion with embeddings
- Context retrieval from knowledge base
- Response generation with OpenAI
- Source citation
- Interactive Q&A mode

```bash
# Set API key
export OPENAI_API_KEY='your-api-key'

# Run example
python3 rag_application.py
```

**Requirements:**
- `pip install openai sentence-transformers`
- OpenAI API key

**Use Case:** Question answering, chatbots, documentation Q&A, research assistants

**What it demonstrates:**
- Building a complete RAG pipeline
- Vector search for context retrieval
- LLM for answer generation
- Source tracking and citation
- Production-ready patterns

---

### [data_loader.py](data_loader.py) ⭐ NEW
**Batch Data Loading Utility**

Efficient batch data ingestion from various sources:
- Load from text lists
- Load from directory (recursive)
- Load from JSONL files
- Progress tracking
- Error handling
- Automatic embedding generation

```bash
python3 data_loader.py
```

**Requirements:** `pip install sentence-transformers`

**Use Case:** Data migration, ETL pipelines, document ingestion, initial data load

**Supported formats:**
- Text lists with metadata
- Directory of files (.txt, .md, etc.)
- JSONL files (one JSON per line)
- Custom formats (extend DataLoader class)

**Features:**
- Batch processing (configurable size)
- Automatic metadata extraction
- Progress tracking
- Error resilience
- Performance statistics

---

## 🎯 Real-World Applications

### [duplicate_detection.py](duplicate_detection.py)
**Find and Remove Duplicate Content**

Implement deduplication:
- Detecting exact and near-duplicate documents
- Similarity threshold tuning
- Deduplication strategies
- Pre-insert duplicate checking
- Plagiarism detection

```bash
python3 duplicate_detection.py
```

**Requirements:** `pip install sentence-transformers`

**Use Case:** Content moderation, data cleaning, plagiarism detection

---

### [recommendation_system.py](recommendation_system.py)
**Content-Based Recommendations**

Build recommendation engines:
- Content-based filtering
- User profile embeddings
- Filtered recommendations
- Genre-based exploration
- Movie/product recommendations

```bash
python3 recommendation_system.py
```

**Requirements:** `pip install sentence-transformers`

**Use Case:** E-commerce, streaming platforms, content platforms

---

### [chinese_text_search.py](chinese_text_search.py)
**Chinese Text Search with Jieba**

Support Chinese language:
- Full-text search with Jieba tokenization
- Multilingual embeddings
- Hybrid search for Chinese
- Category filtering
- Semantic search in Chinese

```bash
python3 chinese_text_search.py
```

**Requirements:** `pip install sentence-transformers`

**Use Case:** Chinese content platforms, multilingual applications

---

### [chinese_text_search_openai.py](chinese_text_search_openai.py) ⭐ NEW
**Chinese Text Search with OpenAI Embeddings**

Enhanced Chinese search with better abbreviation understanding:
- Using OpenAI's text-embedding-3-small model
- **Better understanding of abbreviations** (NLP ↔ 自然语言处理)
- Improved Chinese-English mixed queries
- Full-text + semantic hybrid search
- Comparison with open-source models

```bash
# Set API key
export OPENAI_API_KEY='your-api-key'

# Run example
python3 chinese_text_search_openai.py
```

**Requirements:**
- `pip install openai`
- OpenAI API key (get from https://platform.openai.com/api-keys)

**Cost:** ~$0.004 per 1000 queries (very affordable!)

**Use Case:** Production Chinese search, technical documentation with abbreviations

**Why OpenAI?** See comparison below showing 3-5x better abbreviation understanding.

---

### [compare_embeddings.py](compare_embeddings.py) 🔬 NEW
**Compare OpenAI vs SentenceTransformer**

Side-by-side comparison of embedding models:
- Test abbreviation understanding (NLP, AI, ML)
- Quantitative similarity scores
- Cost vs quality analysis
- Performance benchmarking
- Recommendations for your use case

```bash
# With OpenAI API key
export OPENAI_API_KEY='your-api-key'
python3 compare_embeddings.py

# Or without (tests only SentenceTransformer)
python3 compare_embeddings.py
```

**Example Output:**
```
NLP vs 自然语言处理:
  OpenAI:           0.8547 ✅
  SentenceT:        0.4521 ⚠️
  Improvement:      +0.4026 (89% better!)
```

**Requirements:** `pip install openai sentence-transformers`

**Use Case:** Choosing the right embedding model for your application

---

### [incremental_updates.py](incremental_updates.py)
**Incremental Data Updates**

Handle data changes:
- Content updates (vector re-embedding)
- Metadata-only updates
- Batch updates from CDC pipelines
- Version tracking
- Update verification

```bash
python3 incremental_updates.py
```

**Requirements:** `pip install sentence-transformers`

**Use Case:** CMS, real-time data pipelines, document versioning

---

## 📦 Prerequisites

### Required

- **DXMT Server:** Running on `localhost:9999` (or configure host/port)
- **Python:** 3.8 or higher
- **dxmt SDK:** `pip install dxmt`

### Optional (for specific examples)

```bash
# For semantic search and embeddings
pip install sentence-transformers

# For OpenAI embeddings (recommended for production)
pip install openai
export OPENAI_API_KEY='sk-...'  # Get from https://platform.openai.com/api-keys

# For LangChain integration (see langchain-dxmt/examples/)
pip install langchain-dxmt langchain-openai langchain-core
```

### 🆕 OpenAI Embeddings vs Open-Source Models

**New in 2026-01:** We now support OpenAI embeddings for superior quality!

| Aspect | SentenceTransformer | OpenAI text-embedding-3-small |
|--------|---------------------|------------------------------|
| **Cost** | Free | $0.02 / 1M tokens (~$0.004/1000 queries) |
| **Abbreviation Understanding** | ⭐⭐ | ⭐⭐⭐⭐⭐ (3-5x better!) |
| **Chinese-English Mixed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup** | Download model (2-5GB) | Just API key |
| **Latency** | 10ms | 50-100ms |
| **Offline Support** | ✅ Yes | ❌ No (needs internet) |

**Example Improvement:**
- Query: "NLP技术应用" (NLP technology applications)
- SentenceTransformer: "HNSW算法" ranks #1 ❌ (wrong!)
- OpenAI: "自然语言处理概述" ranks #1 ✅ (correct!)

**When to use OpenAI:**
- ✅ Production applications requiring high quality
- ✅ Technical documentation with abbreviations (AI, ML, NLP, etc.)
- ✅ Chinese-English mixed queries
- ✅ Budget allows (~$0.004 per 1000 queries)

**When to use SentenceTransformer:**
- ✅ Offline environments
- ✅ Cost is critical concern
- ✅ Latency must be <20ms
- ✅ Privacy-sensitive data

**📚 Learn More:** See [OPENAI_EMBEDDINGS_GUIDE.md](OPENAI_EMBEDDINGS_GUIDE.md)

## 🏃 Running Examples

### Run Individual Examples

```bash
# Basic usage
python3 basic_usage.py

# Async operations
python3 async_usage.py

# Semantic search
python3 semantic_search.py
```

### Server Configuration

If your DXMT server is not on `localhost:9999`, modify the client initialization:

```python
client = Client(host="your-host", port=your-port)
```

### Troubleshooting

**Issue:** `Connection refused`
- **Solution:** Ensure DXMT server is running: `cargo run -- serve`

**Issue:** `ModuleNotFoundError: No module named 'sentence_transformers'`
- **Solution:** Install dependencies: `pip install sentence-transformers`

**Issue:** `DXMTTableError: Table already exists`
- **Solution:** Examples auto-cleanup, but if interrupted, manually delete: `client.delete_table("table_name")`

## 📊 Feature Comparison

| Feature | Basic | Async | Hybrid | Filtered | Batch | Semantic | Duplicate | Recommend | Chinese | Updates |
|---------|-------|-------|--------|----------|-------|----------|-----------|-----------|---------|---------|
| CRUD Operations | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Vector Search | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Text Search | - | - | ✓ | - | - | - | - | - | ✓ | - |
| Metadata Filters | - | - | - | ✓ | - | ✓ | - | ✓ | ✓ | - |
| Embeddings | - | - | - | - | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| Batch Operations | - | - | - | - | ✓ | - | - | - | - | ✓ |
| Async/Await | - | ✓ | - | - | - | - | - | - | - | - |
| Real Embeddings | - | - | - | - | - | ✓ | ✓ | ✓ | ✓ | ✓ |

## 🎓 Learning Path

### Beginner
1. **basic_usage.py** - Learn fundamental operations
2. **async_usage.py** - Understand async patterns
3. **hybrid_search.py** - Explore different search types

### Intermediate
4. **filtered_search.py** - Master metadata filtering
5. **batch_operations.py** - Optimize for performance
6. **semantic_search.py** - Build intelligent search

### Advanced
7. **duplicate_detection.py** - Implement deduplication
8. **recommendation_system.py** - Build recommender systems
9. **chinese_text_search.py** - Support multiple languages
10. **incremental_updates.py** - Handle data changes

## 🔧 Common Patterns

### Creating a Client

```python
from dxmt import Client

client = Client(host="localhost", port=9999)
```

### Creating a Table

```python
client.create_table(
    name="my_table",
    dimension=128,
    distance_metric="cosine",  # or "euclidean", "dot"
    index_type="hnsw",         # or "ivf", "flat"
    enable_full_text_search=False,  # True for text search
)
```

### Inserting Records

```python
records = [
    {
        "id": "doc1",
        "vector": [0.1, 0.2, ...],  # dimension must match table
        "payload": {
            "title": "Document 1",
            "category": "tech",
            # any JSON-serializable metadata
        },
    },
    # more records...
]

client.insert_records(
    table_name="my_table",
    records=records,
    mode="batch"  # or "parallel", "lockfree"
)
```

### Searching

```python
# Vector search
results = client.search(
    table_name="my_table",
    vector=[0.1, 0.2, ...],
    k=10,
    filter={"category": "tech"}  # optional
)

# Hybrid search (if full_text_search enabled)
results = client.hybrid_search(
    table_name="my_table",
    vector=[0.1, 0.2, ...],
    text_query="machine learning",
    alpha=0.5,  # 0.0 = text only, 1.0 = vector only
    limit=10
)
```

### Error Handling

```python
from dxmt import DXMTTableError, DXMTConnectionError

try:
    client.create_table(name="test", dimension=128)
except DXMTTableError as e:
    if "already exists" in str(e):
        print("Table exists, using existing table")
    else:
        raise
except DXMTConnectionError as e:
    print(f"Connection failed: {e}")
```

## 📖 Additional Resources

- **Main Documentation:** [../README.md](../README.md)
- **API Reference:** [DXMT Python SDK Docs](https://docs.dxmt.com/python-sdk)
- **LangChain Integration:** [../langchain-dxmt/examples/](../../langchain-dxmt/examples/)
- **Server Configuration:** [Server Docs](https://docs.dxmt.com/server)

## 🤝 Contributing

Found an issue or want to add an example?

1. Create an issue describing the problem or new example
2. Fork the repository
3. Create your example following the existing patterns
4. Add documentation to this README
5. Submit a pull request

## 📝 License

These examples are part of the DXMT VectorDB project and are released under the same license as the main project.

---

**Need Help?**
- 📧 Email: support@dxmt.com
- 💬 Discord: [DXMT Community](https://discord.gg/dxmt)
- 📚 Docs: [docs.dxmt.com](https://docs.dxmt.com)
- 🐛 Issues: [GitHub Issues](https://github.com/dxmt/vectordb/issues)
