# DXMT Vector Store Examples

This directory contains comprehensive examples demonstrating the DXMT vector store integration with LlamaIndex.

## Prerequisites

### Installation

```bash
# Install the DXMT vector store package
pip install llama-index-vector-stores-dxmt

# Install required dependencies
pip install llama-index-core llama-index-embeddings-openai llama-index-llms-openai

# For RAGAS evaluation example
pip install ragas datasets langchain-openai
```

### Environment Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### DXMT Server

Make sure the DXMT server is running:

```bash
# In the vectordb directory
cargo run -- serve --simple --memory-limit 1024

# Or using make
make run
```

The server should be accessible at `http://localhost:9999`.

## Examples Overview

This directory contains **8 comprehensive examples**, including **3 adapted from LlamaIndex Chroma demos**:

| # | Example | Type | Description |
|---|---------|------|-------------|
| 1 | `basic_usage.py` | Foundation | Basic DXMT operations |
| 2 | `advanced_filters.py` | Foundation | Complex metadata filtering |
| 3 | `rag_example.py` | Foundation | Complete RAG pipeline |
| 4 | `hybrid_search.py` | Advanced | Vector + keyword search |
| 5 | `paul_graham_essay.py` | **Chroma Port** | Classic LlamaIndex demo |
| 6 | `auto_retriever.py` | **Chroma Port** | Auto metadata inference |
| 7 | `crud_operations.py` | **Chroma Port** | Full CRUD lifecycle |
| 8 | `ragas_evaluation.py` | Evaluation | RAG quality metrics |

---

## Examples

### 1. Basic Usage (`basic_usage.py`)

**What it demonstrates:**
- Creating a DXMT vector store
- Adding documents with embeddings
- Basic similarity search
- Metadata handling

**Run:**
```bash
python basic_usage.py
```

**Key concepts:**
- Vector store initialization
- Document insertion
- Query execution
- Result processing

---

### 2. Advanced Filters (`advanced_filters.py`)

**What it demonstrates:**
- Complex metadata filtering
- Combining filters with similarity search
- Multiple filter operators (EQ, NE, GT, LT, IN, etc.)
- Logical conditions (AND, OR)

**Run:**
```bash
python advanced_filters.py
```

**Key concepts:**
- MetadataFilters API
- Filter operators
- Combining semantic search with structured queries
- Using retrievers with filters

**Sample filters:**
```python
# Equality filter
MetadataFilters(filters=[
    MetadataFilter(key="difficulty", value="Beginner", operator=FilterOperator.EQ)
])

# Range filter
MetadataFilters(filters=[
    MetadataFilter(key="year", value=2000, operator=FilterOperator.GTE),
    MetadataFilter(key="popularity", value=80, operator=FilterOperator.GTE)
], condition=FilterCondition.AND)

# IN filter
MetadataFilters(filters=[
    MetadataFilter(key="difficulty", value=["Intermediate", "Advanced"], operator=FilterOperator.IN)
])
```

---

### 3. RAG Example (`rag_example.py`)

**What it demonstrates:**
- Complete RAG (Retrieval Augmented Generation) pipeline
- Document ingestion and chunking
- Semantic retrieval
- Answer generation with LLM
- Source attribution
- Streaming responses

**Run:**
```bash
python rag_example.py
```

**Key concepts:**
- Building a knowledge base
- Query engine creation
- Retrieval + generation workflow
- Source tracking and citation
- Streaming vs. batch responses

**RAG Pipeline:**
```
Documents → Chunking → Embedding → DXMT Storage
                                        ↓
Question → Query Embedding → Similarity Search → Top-K Contexts
                                                      ↓
                                              LLM Generation → Answer
```

---

### 4. Hybrid Search (`hybrid_search.py`)

**What it demonstrates:**
- Combining vector similarity search with keyword search
- BM25 full-text search
- Hybrid ranking strategies
- Text field configuration

**Run:**
```bash
python hybrid_search.py
```

**Key concepts:**
- Full-text search integration
- Vector + keyword fusion
- Ranking and reranking
- Optimizing for different query types

---

### 5. Paul Graham Essay (`paul_graham_essay.py`) ⭐ **CHROMA EQUIVALENT**

**What it demonstrates:**
- Classic LlamaIndex example adapted from Chroma
- Loading and indexing long documents
- DXMT's built-in persistence vs Chroma's modes
- Multiple query patterns
- Side-by-side comparison with Chroma approach

**Run:**
```bash
python paul_graham_essay.py
```

**Key concepts:**
- Document ingestion from files
- DXMT's always-persistent architecture
- Multiple query examples
- Retriever vs query engine

**Chroma equivalent:**
- `reference/llama_index/docs/examples/vector_stores/ChromaIndexDemo.ipynb`

**Highlights:**
```python
# Chroma needs explicit persistence mode
client = chromadb.PersistentClient(path="./chroma_db")

# DXMT is always persistent (server-based)
vector_store = DXMTVectorStore(host="localhost", port=9999)
# Data automatically persists!
```

---

### 6. Auto-Retrieval (`auto_retriever.py`) ⭐ **CHROMA EQUIVALENT**

**What it demonstrates:**
- Automatic metadata filter inference from queries
- Using LLM to understand search intent
- VectorStoreInfo and MetadataInfo configuration
- Comparison with manual filter specification

**Run:**
```bash
python auto_retriever.py
```

**Key concepts:**
- VectorIndexAutoRetriever
- Natural language to structured filters
- Dynamic query optimization
- When to use auto vs manual retrieval

**Chroma equivalent:**
- `reference/llama_index/docs/examples/vector_stores/chroma_auto_retriever.ipynb`

**Example queries:**
```python
# Natural language queries that automatically become filtered searches
"Tell me about two celebrities from United States"
→ Automatically infers: filters={'country': 'United States'}, top_k=2

"Show me Sports celebrities from USA"
→ Automatically infers: filters={'category': 'Sports', 'country': 'United States'}

"Who are some inspiring people?"
→ Pure semantic search (no filters)
```

---

### 7. CRUD Operations (`crud_operations.py`) ⭐ **CHROMA EQUIVALENT**

**What it demonstrates:**
- Complete lifecycle: Create, Read, Update, Delete
- Multiple insertion methods
- Upsert operations
- Soft deletes and versioning patterns

**Run:**
```bash
python crud_operations.py
```

**Key concepts:**
- Document insertion patterns
- Updating with same doc_id (upsert)
- Deletion strategies
- Best practices for production

**Chroma equivalent:**
- `reference/llama_index/docs/examples/vector_stores/ChromaIndexDemo.ipynb` (Update/Delete section)

**CRUD patterns:**
```python
# CREATE
doc = Document(text="...", metadata={...}, doc_id="unique_id")
index.insert(doc)

# READ
retriever = index.as_retriever(similarity_top_k=5, filters=...)
results = retriever.retrieve("query")

# UPDATE (upsert with same ID)
updated_doc = Document(text="...", doc_id="same_id")
index.insert(updated_doc)  # Replaces existing

# DELETE
index.delete_ref_doc("doc_id")
```

---

### 8. RAGAS Evaluation (`ragas_evaluation.py`) ⭐ **NEW**

**What it demonstrates:**
- Systematic RAG evaluation using RAGAS framework
- Multiple evaluation metrics
- Performance benchmarking
- Identifying improvement areas

**Run:**
```bash
python ragas_evaluation.py
```

**Key concepts:**
- Creating evaluation datasets
- Ground truth comparison
- Automated quality assessment
- Data-driven optimization

**RAGAS Metrics:**

| Metric | Description | Measures |
|--------|-------------|----------|
| **Faithfulness** | Answer grounded in context | Hallucination rate |
| **Answer Relevancy** | Answer relevance to question | Query alignment |
| **Context Precision** | Relevant contexts ranked higher | Retrieval quality |
| **Context Recall** | All relevant info retrieved | Coverage |
| **Answer Correctness** | Factual accuracy | Correctness |
| **Answer Similarity** | Similarity to ground truth | Semantic match |

**Example output:**
```
📊 Overall Scores:
   faithfulness             : 0.9234 ✓
   answer_relevancy         : 0.8876 ✓
   context_precision        : 0.8456 ✓
   context_recall           : 0.7654 ✓
   answer_correctness       : 0.8123 ✓
   answer_similarity        : 0.7891 ✓
```

**When to use:**
- Evaluating RAG system before production
- A/B testing different configurations
- Identifying weaknesses in retrieval or generation
- Benchmarking against baseline
- Continuous monitoring in production

---

## Common Patterns

### Vector Store Configuration

```python
from llama_index.vector_stores.dxmt import DXMTVectorStore

vector_store = DXMTVectorStore(
    table_name="my_table",
    dimension=1536,              # Match your embedding model
    host="localhost",
    port=9999,
    index_type="hnsw",           # or "ivf", "brute_force"
    distance_metric="cosine",    # or "euclidean", "dot_product"
)
```

### Embedding Model Setup

```python
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
```

### Query Engine Configuration

```python
query_engine = index.as_query_engine(
    similarity_top_k=3,          # Number of documents to retrieve
    response_mode="compact",     # or "tree_summarize", "simple_summarize"
)
```

## Performance Tips

### 1. Choosing the Right Index Type

- **Brute Force**: Best for <5K vectors, 100% recall
- **HNSW**: Best for 5K-100K vectors, high recall, good performance
- **IVF**: Best for >100K vectors, memory-efficient

### 2. Retrieval Optimization

```python
# Increase top_k for better recall
similarity_top_k=5

# Use reranking for better precision
from llama_index.postprocessor import SentenceTransformerRerank
reranker = SentenceTransformerRerank(top_n=3)

# Combine with filters for precision
filters=MetadataFilters(...)
```

### 3. Embedding Selection

- **text-embedding-ada-002**: General purpose, 1536 dims
- **text-embedding-3-small**: Faster, 512 dims
- **text-embedding-3-large**: Higher quality, 3072 dims

### 4. Chunking Strategy

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,    # Tokens per chunk
    chunk_overlap=50,  # Overlap for context preservation
)
```

## Troubleshooting

### 500 Internal Server Error

```
Server error '500 Internal Server Error' for url '.../search'
```

**Solution:** Table exists with different configuration. Use --clean flag:

```bash
python paul_graham_essay.py --clean
python auto_retriever.py --clean
python crud_operations.py --clean
```

### Connection Error

```
Error: Cannot connect to DXMT server
```

**Solution:** Make sure DXMT server is running on localhost:9999

```bash
# Start the server
cd /path/to/vectordb
cargo run -- serve --simple --memory-limit 1024
```

### OpenAI API Error

```
Error: No API key provided
```

**Solution:** Set your OpenAI API key

```bash
export OPENAI_API_KEY='your-key-here'
```

### RAGAS Import Error

```
ModuleNotFoundError: No module named 'ragas'
```

**Solution:** Install RAGAS

```bash
pip install ragas datasets
```

### Low RAGAS Scores

**Context Recall < 0.7:**
- Increase `similarity_top_k`
- Improve chunking strategy
- Use better embeddings

**Context Precision < 0.7:**
- Add reranking
- Use hybrid search
- Improve metadata filtering

**Faithfulness < 0.7:**
- Lower LLM temperature
- Add stricter prompts
- Use citation mechanisms

## Next Steps

1. **Start with `basic_usage.py`** to understand core concepts
2. **Explore `advanced_filters.py`** for complex queries
3. **Build a RAG system** with `rag_example.py`
4. **Evaluate performance** using `ragas_evaluation.py`
5. **Optimize based on metrics** and deploy to production

## Resources

### Documentation
- [DXMT Documentation](https://github.com/dxmt/vectordb)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

### Troubleshooting Guides
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Complete troubleshooting guide for all examples
- **[RAGAS_TROUBLESHOOTING.md](RAGAS_TROUBLESHOOTING.md)** - RAGAS-specific issues
- **[RAGAS_QUICKFIX.md](RAGAS_QUICKFIX.md)** - Quick fixes for common RAGAS problems
- **[CHROMA_MIGRATION.md](CHROMA_MIGRATION.md)** - Migration guide from Chroma

## Migrating from Chroma

If you're familiar with LlamaIndex + Chroma and want to switch to DXMT, here's a migration guide:

### Basic Migration

**Chroma Code:**
```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# Create Chroma client and collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("my_collection")

# Create vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

**DXMT Code:**
```python
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core import VectorStoreIndex

# Create DXMT vector store (no collection setup needed)
vector_store = DXMTVectorStore(
    table_name="my_collection",
    dimension=1536,
    host="localhost",
    port=9999,
)

# Build index (same as Chroma!)
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)
```

### Key Differences

| Aspect | Chroma | DXMT |
|--------|--------|------|
| **Client Setup** | `chromadb.EphemeralClient()` or `chromadb.PersistentClient()` | `DXMTVectorStore()` (always server-based) |
| **Persistence** | Must choose mode | Always persistent |
| **Collection** | Explicit creation | Auto-created on first use |
| **Querying** | Same LlamaIndex API | Same LlamaIndex API ✓ |
| **Filters** | Same LlamaIndex API | Same LlamaIndex API ✓ |
| **Server** | Optional (Docker) | Required (binary or Docker) |

### Migration Checklist

- [ ] Start DXMT server: `cargo run -- serve --simple --memory-limit 1024`
- [ ] Replace `chromadb` imports with `DXMTVectorStore`
- [ ] Remove client/collection creation code
- [ ] Add `dimension` parameter (required in DXMT)
- [ ] Update connection details (`host`, `port`)
- [ ] Test queries (API is the same!)
- [ ] Verify metadata filters work
- [ ] Check performance (DXMT may be faster for large datasets)

### Side-by-Side Examples

See these examples for direct comparisons:
- `paul_graham_essay.py` - Includes Chroma vs DXMT comparison
- `auto_retriever.py` - Shows identical auto-retrieval functionality
- `crud_operations.py` - CRUD operations work the same way

### Why Migrate?

**Chroma Strengths:**
- ✓ Simple embedded mode
- ✓ Quick prototyping
- ✓ Wide adoption

**DXMT Advantages:**
- ✓ Production-ready architecture (server-based)
- ✓ Automatic tiered storage (Memory → SSD → S3)
- ✓ Cost optimization (70-90% savings vs all-memory)
- ✓ Runtime adaptive indexes
- ✓ Sub-5min recovery from object storage
- ✓ Native hybrid search (vector + text)
- ✓ Built for cloud-native deployments

### Common Migration Issues

**Issue 1: Missing dimension parameter**
```python
# Chroma (infers from embeddings)
vector_store = ChromaVectorStore(chroma_collection=collection)

# DXMT (explicit dimension required)
vector_store = DXMTVectorStore(
    table_name="collection",
    dimension=1536,  # Must specify!
)
```

**Issue 2: Server not running**
```bash
# Error: Connection refused
# Solution: Start DXMT server
cargo run -- serve --simple --memory-limit 1024
```

**Issue 3: Port conflicts**
```python
# If port 9999 is busy, use a different port
vector_store = DXMTVectorStore(
    table_name="collection",
    dimension=1536,
    port=9998,  # Custom port
)

# Start server on same port:
cargo run -- serve --port 9998
```

## Contributing

Found an issue or have a suggestion? Please open an issue or pull request on the [DXMT GitHub repository](https://github.com/dxmt/vectordb).
