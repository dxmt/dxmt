# API Reference - LangChain DXMT

Complete API reference for the `langchain-dxmt` package.

## Table of Contents

- [DXMTVectorDB](#dxmtvectordb)
  - [Initialization](#initialization)
  - [Adding Documents](#adding-documents)
  - [Searching](#searching)
  - [Hybrid Search](#hybrid-search)
  - [Retrieving Documents](#retrieving-documents)
  - [Deleting Documents](#deleting-documents)
  - [Factory Methods](#factory-methods)
  - [Async Methods](#async-methods)
- [DXMTClient](#dxmtclient)
  - [Table Management](#table-management)
  - [Record Operations](#record-operations)
  - [Search Operations](#search-operations)
- [Exceptions](#exceptions)
- [Type Definitions](#type-definitions)

---

## DXMTVectorDB

Main vector store class implementing the LangChain `VectorStore` interface.

### Initialization

```python
DXMTVectorDB(
    table_name: str = "langchain_dxmt",
    embedding: Optional[Embeddings] = None,
    dimension: Optional[int] = None,
    *,
    client: Optional[DXMTClient] = None,
    host: str = "localhost",
    port: int = 9999,
    ssl: bool = False,
    headers: Optional[dict[str, str]] = None,
    index_type: str = "hnsw",
    distance_metric: str = "cosine",
    enable_full_text_search: bool = False,
    text_fields: Optional[list[dict[str, Any]]] = None,
    insert_mode: str = "standard",
    relevance_score_fn: Optional[Callable[[float], float]] = None,
    create_table_if_not_exists: bool = True,
) -> None
```

**Parameters:**

- **table_name** (`str`, default: `"langchain_dxmt"`): Name of the DXMT table.
- **embedding** (`Embeddings`, optional): LangChain embeddings instance for vectorizing text.
- **dimension** (`int`, optional): Vector dimension. Required if `create_table_if_not_exists=True`.
- **client** (`DXMTClient`, optional): Existing DXMT client. If not provided, a new client is created.
- **host** (`str`, default: `"localhost"`): DXMT server hostname.
- **port** (`int`, default: `9999`): DXMT server port.
- **ssl** (`bool`, default: `False`): Whether to use HTTPS.
- **headers** (`dict[str, str]`, optional): Custom HTTP headers (e.g., for authentication).
- **index_type** (`str`, default: `"hnsw"`): Index type. Options:
  - `"hnsw"` - Hierarchical Navigable Small World (recommended)
  - `"ivf"` - Inverted File Index
  - `"ivf_pq"` - IVF with Product Quantization (memory efficient)
  - `"lsh"` - Locality Sensitive Hashing
  - `"brute_force"` - Exact search
- **distance_metric** (`str`, default: `"cosine"`): Distance metric. Options:
  - `"cosine"` - Cosine similarity (recommended for text)
  - `"euclidean"` - Euclidean distance (L2)
  - `"manhattan"` - Manhattan distance (L1)
  - `"dot_product"` - Dot product
- **enable_full_text_search** (`bool`, default: `False`): Enable BM25 full-text search.
- **text_fields** (`list[dict]`, optional): Configuration for text fields (advanced).
- **insert_mode** (`str`, default: `"standard"`): Insert optimization mode. Options:
  - `"standard"` - Default mode
  - `"lockfree"` - Lock-free insertions
  - `"streaming"` - Streaming mode
  - `"memorypool"` - Pre-allocated memory pool
  - `"ultimate"` - All optimizations enabled
- **relevance_score_fn** (`Callable`, optional): Custom score normalization function.
- **create_table_if_not_exists** (`bool`, default: `True`): Auto-create table if it doesn't exist.

**Raises:**

- `ValueError`: If required parameters are missing or invalid.
- `TypeError`: If parameters have incorrect types.
- `DXMTTableError`: If table creation/connection fails.

**Example:**

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

vector_store = DXMTVectorDB(
    table_name="my_documents",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
    index_type="hnsw",
    distance_metric="cosine",
)
```

---

### Adding Documents

#### `add_texts()`

Add texts with optional metadata and IDs.

```python
def add_texts(
    self,
    texts: Iterable[str],
    metadatas: Optional[list[dict]] = None,
    ids: Optional[list[str]] = None,
    **kwargs: Any,
) -> list[str]
```

**Parameters:**

- **texts** (`Iterable[str]`): Texts to add.
- **metadatas** (`list[dict]`, optional): Metadata for each text.
- **ids** (`list[str]`, optional): Custom IDs. Auto-generated if not provided.
- **kwargs**: Additional arguments (reserved for future use).

**Returns:**

- `list[str]`: List of document IDs.

**Raises:**

- `ValueError`: If embedding function not provided or dimension mismatch.
- `DXMTError`: If insertion fails.

**Example:**

```python
texts = ["Document 1", "Document 2"]
metadatas = [{"source": "web"}, {"source": "api"}]

ids = vector_store.add_texts(
    texts=texts,
    metadatas=metadatas,
    ids=["id1", "id2"],
)
```

#### `add_documents()`

Add Document objects.

```python
def add_documents(
    self,
    documents: list[Document],
    ids: Optional[list[str]] = None,
    **kwargs: Any,
) -> list[str]
```

**Parameters:**

- **documents** (`list[Document]`): Documents to add.
- **ids** (`list[str]`, optional): Custom IDs.
- **kwargs**: Additional arguments.

**Returns:**

- `list[str]`: List of document IDs.

**Example:**

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Machine learning basics",
        metadata={"topic": "ml", "level": "beginner"},
    ),
]

ids = vector_store.add_documents(documents)
```

---

### Searching

#### `similarity_search()`

Find similar documents by text query.

```python
def similarity_search(
    self,
    query: str,
    k: int = 4,
    filter: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> list[Document]
```

**Parameters:**

- **query** (`str`): Query text.
- **k** (`int`, default: `4`): Number of results to return.
- **filter** (`dict`, optional): Metadata filter (e.g., `{"category": "AI"}`).
- **kwargs**: Additional search parameters.

**Returns:**

- `list[Document]`: Most similar documents.

**Example:**

```python
results = vector_store.similarity_search(
    query="What is machine learning?",
    k=5,
    filter={"topic": "ml"},
)

for doc in results:
    print(doc.page_content)
```

#### `similarity_search_with_score()`

Search with relevance scores.

```python
def similarity_search_with_score(
    self,
    query: str,
    k: int = 4,
    filter: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> list[tuple[Document, float]]
```

**Parameters:**

- **query** (`str`): Query text.
- **k** (`int`, default: `4`): Number of results.
- **filter** (`dict`, optional): Metadata filter.
- **kwargs**: Additional parameters.

**Returns:**

- `list[tuple[Document, float]]`: Documents with scores (higher is more similar).

**Example:**

```python
results = vector_store.similarity_search_with_score(
    query="deep learning",
    k=3,
)

for doc, score in results:
    print(f"Score: {score:.4f} - {doc.page_content}")
```

#### `similarity_search_by_vector()`

Search using embedding vector directly.

```python
def similarity_search_by_vector(
    self,
    embedding: list[float],
    k: int = 4,
    filter: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> list[Document]
```

**Parameters:**

- **embedding** (`list[float]`): Query embedding vector.
- **k** (`int`, default: `4`): Number of results.
- **filter** (`dict`, optional): Metadata filter.
- **kwargs**: Additional parameters.

**Returns:**

- `list[Document]`: Most similar documents.

**Example:**

```python
# Get embedding from external source
query_vector = [0.1, 0.2, ...]  # 1536-dim vector

results = vector_store.similarity_search_by_vector(
    embedding=query_vector,
    k=5,
)
```

#### `similarity_search_with_score_by_vector()`

Search by vector with scores.

```python
def similarity_search_with_score_by_vector(
    self,
    embedding: list[float],
    k: int = 4,
    filter: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> list[tuple[Document, float]]
```

**Returns:**

- `list[tuple[Document, float]]`: Documents with scores.

---

### Hybrid Search

#### `hybrid_search()`

**DXMT-specific feature**: Combines vector similarity with BM25 full-text search.

```python
def hybrid_search(
    self,
    query: str,
    k: int = 4,
    vector_weight: float = 0.7,
    filter: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> list[Document]
```

**Parameters:**

- **query** (`str`): Query text.
- **k** (`int`, default: `4`): Number of results.
- **vector_weight** (`float`, default: `0.7`): Weight for vector similarity (0.0-1.0).
  - `1.0` = pure vector search
  - `0.0` = pure text search
  - `0.7` = 70% vector, 30% text (recommended)
- **filter** (`dict`, optional): Metadata filter.
- **kwargs**: Additional parameters.

**Returns:**

- `list[Document]`: Most relevant documents.

**Requires:**

- `enable_full_text_search=True` during initialization.

**Raises:**

- `ValueError`: If full-text search not enabled.

**Example:**

```python
# Create store with full-text search
vector_store = DXMTVectorDB(
    table_name="hybrid_docs",
    embedding=embeddings,
    dimension=1536,
    enable_full_text_search=True,
)

# Hybrid search (70% semantic, 30% keyword)
results = vector_store.hybrid_search(
    query="machine learning algorithms",
    k=5,
    vector_weight=0.7,
)
```

---

### Retrieving Documents

#### `get_by_ids()`

Retrieve documents by their IDs.

```python
def get_by_ids(
    self,
    ids: list[str],
) -> list[Document]
```

**Parameters:**

- **ids** (`list[str]`): Document IDs to retrieve.

**Returns:**

- `list[Document]`: Retrieved documents (in same order as IDs).

**Example:**

```python
docs = vector_store.get_by_ids(["id1", "id2", "id3"])

for doc in docs:
    print(f"ID: {doc.id}")
    print(f"Content: {doc.page_content}")
```

---

### Deleting Documents

#### `delete()`

Delete documents by IDs.

```python
def delete(
    self,
    ids: Optional[list[str]] = None,
    **kwargs: Any,
) -> bool
```

**Parameters:**

- **ids** (`list[str]`, optional): IDs to delete. If not provided, uses `kwargs`.
- **kwargs**: Alternative way to specify IDs.

**Returns:**

- `bool`: `True` if deletion succeeded.

**Example:**

```python
# Delete specific documents
success = vector_store.delete(ids=["id1", "id2"])

if success:
    print("Documents deleted successfully")
```

---

### Factory Methods

#### `from_texts()`

Create vector store from list of texts.

```python
@classmethod
def from_texts(
    cls,
    texts: list[str],
    embedding: Embeddings,
    metadatas: Optional[list[dict]] = None,
    ids: Optional[list[str]] = None,
    table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
    dimension: Optional[int] = None,
    **kwargs: Any,
) -> "DXMTVectorDB"
```

**Parameters:**

- **texts** (`list[str]`): Texts to add.
- **embedding** (`Embeddings`): Embeddings instance.
- **metadatas** (`list[dict]`, optional): Metadata for each text.
- **ids** (`list[str]`, optional): Custom IDs.
- **table_name** (`str`): Table name.
- **dimension** (`int`, optional): Vector dimension (inferred from first embedding if not provided).
- **kwargs**: Additional `DXMTVectorDB` initialization parameters.

**Returns:**

- `DXMTVectorDB`: New vector store instance with texts already added.

**Example:**

```python
vector_store = DXMTVectorDB.from_texts(
    texts=["Text 1", "Text 2"],
    embedding=OpenAIEmbeddings(),
    table_name="my_docs",
)
```

#### `from_documents()`

Create vector store from Document objects.

```python
@classmethod
def from_documents(
    cls,
    documents: list[Document],
    embedding: Embeddings,
    ids: Optional[list[str]] = None,
    table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
    dimension: Optional[int] = None,
    **kwargs: Any,
) -> "DXMTVectorDB"
```

**Returns:**

- `DXMTVectorDB`: New vector store with documents added.

**Example:**

```python
documents = [
    Document(page_content="Doc 1", metadata={"source": "web"}),
    Document(page_content="Doc 2", metadata={"source": "api"}),
]

vector_store = DXMTVectorDB.from_documents(
    documents=documents,
    embedding=embeddings,
    table_name="docs",
    dimension=1536,
)
```

#### `from_existing_table()`

Connect to an existing DXMT table.

```python
@classmethod
def from_existing_table(
    cls,
    table_name: str,
    embedding: Embeddings,
    **kwargs: Any,
) -> "DXMTVectorDB"
```

**Parameters:**

- **table_name** (`str`): Name of existing table.
- **embedding** (`Embeddings`): Embeddings instance.
- **kwargs**: Additional parameters (host, port, etc.).

**Returns:**

- `DXMTVectorDB`: Vector store connected to existing table.

**Example:**

```python
# Connect to table created earlier
vector_store = DXMTVectorDB.from_existing_table(
    table_name="my_existing_table",
    embedding=embeddings,
)
```

---

### Async Methods

All major methods have async equivalents with `a` prefix:

- `aadd_texts()` - Async add texts
- `aadd_documents()` - Async add documents
- `asimilarity_search()` - Async similarity search
- `asimilarity_search_with_score()` - Async search with scores
- `asimilarity_search_by_vector()` - Async vector search
- `asimilarity_search_with_score_by_vector()` - Async vector search with scores
- `aget_by_ids()` - Async retrieve by IDs
- `adelete()` - Async delete
- `ahybrid_search()` - Async hybrid search

**Example:**

```python
import asyncio

async def main():
    vector_store = DXMTVectorDB(...)

    # Async operations
    ids = await vector_store.aadd_texts(["Text 1", "Text 2"])
    results = await vector_store.asimilarity_search("query", k=5)
    await vector_store.adelete(ids=ids)

asyncio.run(main())
```

---

## DXMTClient

Low-level HTTP client for DXMT API.

### Initialization

```python
DXMTClient(
    host: str = "localhost",
    port: int = 9999,
    ssl: bool = False,
    headers: Optional[dict[str, str]] = None,
)
```

**Example:**

```python
from langchain_dxmt.client import DXMTClient

client = DXMTClient(host="localhost", port=9999)

# Use as context manager
with DXMTClient() as client:
    tables = client.list_tables()
```

### Table Management

#### `create_table()`

```python
def create_table(
    self,
    name: str,
    dimension: int,
    index_type: str = "hnsw",
    distance_metric: str = "cosine",
    **kwargs: Any,
) -> dict[str, Any]
```

#### `get_table()`

```python
def get_table(self, name: str) -> dict[str, Any]
```

#### `delete_table()`

```python
def delete_table(self, name: str) -> dict[str, Any]
```

#### `list_tables()`

```python
def list_tables(self) -> list[dict[str, Any]]
```

### Record Operations

#### `insert_records()`

```python
def insert_records(
    self,
    table_name: str,
    records: list[dict[str, Any]],
    mode: str = "standard",
) -> dict[str, Any]
```

#### `get_records()`

```python
def get_records(
    self,
    table_name: str,
    ids: list[str],
) -> list[dict[str, Any]]
```

#### `delete_records()`

```python
def delete_records(
    self,
    table_name: str,
    ids: list[str],
) -> dict[str, Any]
```

### Search Operations

#### `search()`

```python
def search(
    self,
    table_name: str,
    vector: list[float],
    k: int = 4,
    filter: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]
```

#### `hybrid_search()`

```python
def hybrid_search(
    self,
    table_name: str,
    vector: list[float],
    text: str,
    vector_weight: float = 0.7,
    k: int = 4,
    filter: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]
```

#### `full_text_search()`

```python
def full_text_search(
    self,
    table_name: str,
    text: str,
    k: int = 4,
    filter: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]
```

---

## Exceptions

### `DXMTError`

Base exception for all DXMT errors.

```python
class DXMTError(Exception):
    """Base exception for DXMT errors."""
    pass
```

### `DXMTConnectionError`

Connection to DXMT server failed.

```python
class DXMTConnectionError(DXMTError):
    """Connection to DXMT server failed."""
    pass
```

### `DXMTTableError`

Table operation error (creation, deletion, etc.).

```python
class DXMTTableError(DXMTError):
    """Table operation error."""
    pass
```

### `DXMTSearchError`

Search operation error.

```python
class DXMTSearchError(DXMTError):
    """Search operation error."""
    pass
```

**Example:**

```python
from langchain_dxmt.client import DXMTConnectionError

try:
    vector_store = DXMTVectorDB(
        table_name="test",
        embedding=embeddings,
        dimension=1536,
        host="invalid-host",
    )
except DXMTConnectionError as e:
    print(f"Connection failed: {e}")
```

---

## Type Definitions

### Common Types

```python
# Distance metrics
DistanceMetric = Literal["cosine", "euclidean", "manhattan", "dot_product"]

# Index types
IndexType = Literal["hnsw", "ivf", "ivf_pq", "lsh", "brute_force"]

# Insert modes
InsertMode = Literal["standard", "lockfree", "streaming", "memorypool", "ultimate"]

# Filter (metadata)
Filter = dict[str, Any]
```

### Document Structure

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The document text",
    metadata={
        "source": "web",
        "author": "John Doe",
        "date": "2024-01-01",
    },
    id="optional-custom-id",
)
```

---

## Configuration Examples

### Minimal Configuration

```python
vector_store = DXMTVectorDB(
    table_name="docs",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
)
```

### Production Configuration

```python
vector_store = DXMTVectorDB(
    table_name="production_docs",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
    host="dxmt.example.com",
    port=443,
    ssl=True,
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    index_type="hnsw",
    distance_metric="cosine",
    insert_mode="ultimate",
    enable_full_text_search=True,
)
```

### Memory-Efficient Configuration

```python
vector_store = DXMTVectorDB(
    table_name="large_docs",
    embedding=embeddings,
    dimension=768,
    index_type="ivf_pq",  # Product quantization
    distance_metric="cosine",
    insert_mode="streaming",
    enable_full_text_search=False,
)
```

---

## See Also

- [README.md](../README.md) - Quick start guide
- [Examples](../examples/) - Code examples
- [Integration Tests](../tests/integration_tests/) - Integration test examples
- [DXMT Documentation](https://github.com/dxmt/dxmt) - DXMT VectorDB docs
- [LangChain Documentation](https://python.langchain.com/) - LangChain docs

---

**Last Updated:** 2026-01-20
**Version:** 0.1.0
