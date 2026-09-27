# DXMT VectorDB AutoGen Integration Guide

**Version:** 1.0.0
**Last Updated:** 2026-01-28
**Status:** Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Usage Patterns](#usage-patterns)
8. [Advanced Features](#advanced-features)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
11. [Migration Guide](#migration-guide)
12. [Best Practices](#best-practices)
13. [FAQ](#faq)

---

## Introduction

### What is DXMT VectorDB?

DXMT is a lightweight, horizontally scalable vector database with object storage native architecture. Unlike traditional vector databases that optimize for in-memory search, DXMT optimizes for cloud-native cost efficiency through tiered storage.

**Key Features:**
- **Tiered Storage:** Memory → SSD → S3 for 98% cost savings
- **Full-Text Search:** Tantivy + BM25 ranking engine
- **Hybrid Search:** Combine vector similarity and text relevance
- **Horizontal Scalability:** Linear performance scaling
- **Fast Recovery:** Sub-5min recovery from object storage

### Why Use DXMT with AutoGen?

1. **Cost Efficiency:** 98% cost reduction vs in-memory-only solutions
2. **Scalability:** Handle millions of memories across thousands of agents
3. **Persistence:** Durable storage with S3 backend
4. **Advanced Search:** Full-text and hybrid search capabilities
5. **Multi-Tenancy:** Built-in support for multi-user scenarios

### What is AutoGen?

AutoGen is Microsoft's multi-agent conversation framework that enables building complex AI agent systems. DXMT provides a Memory backend for AutoGen agents to store and retrieve conversation history, context, and knowledge.

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoGen Application                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Agent 1  │  │ Agent 2  │  │ Agent N  │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │             │             │                          │
│       └─────────────┴─────────────┘                          │
│                     │                                        │
│           ┌─────────▼──────────┐                            │
│           │  Memory Interface  │                            │
│           │  (DXMTVectorMemory)│                            │
│           └─────────┬──────────┘                            │
└─────────────────────┼────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  DXMT Python SDK      │
          │  (HTTP Client)        │
          └───────────┬───────────┘
                      │ HTTP/REST
          ┌───────────▼───────────┐
          │  DXMT Server          │
          │  (Rust Backend)       │
          ├───────────────────────┤
          │ Memory Layer (Hot)    │
          │ SSD Layer (Warm)      │
          │ S3 Layer (Cold)       │
          └───────────────────────┘
```

### Component Layers

#### Layer 1: AutoGen Memory Adapter
- **Location:** `autogen_ext.memory.dxmt`
- **Language:** Pure Python
- **Responsibilities:**
  - Implement AutoGen's Memory interface
  - Convert MemoryContent to DXMT records
  - Score normalization
  - Metadata filtering

#### Layer 2: DXMT Python SDK
- **Location:** `clients/dxmt-python-sdk`
- **Language:** Python
- **Responsibilities:**
  - HTTP client for DXMT server
  - Request/response serialization
  - Connection management

#### Layer 3: DXMT Rust Backend
- **Location:** Core DXMT server
- **Language:** Rust
- **Responsibilities:**
  - Vector indexing and search
  - Tiered storage management
  - Full-text search (Tantivy)
  - Data persistence

### Data Flow

#### Adding Memories
```
1. Application → memory.add(MemoryContent)
2. Adapter → Extract text, metadata
3. Adapter → Generate/use auto-embedding
4. SDK → POST /tables/{name}/records
5. Server → Index vector + metadata
6. Server → Persist to tiered storage
```

#### Querying Memories
```
1. Application → memory.query("query text")
2. Adapter → Choose search method (text/vector/hybrid)
3. SDK → POST /tables/{name}/search/text
4. Server → Execute search (BM25/vector/hybrid)
5. Server → Return top-k results with scores
6. Adapter → Normalize scores to [0,1]
7. Adapter → Convert to MemoryContent objects
8. Application → Process results
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Rust toolchain (for building DXMT server)
- (Optional) Docker for containerized deployment

### Step 1: Install DXMT Server

```bash
# Clone DXMT repository
git clone https://github.com/dxmt/dxmt
cd dxmt/vectordb

# Build release version
cargo build --release

# The binary will be at target/release/vectordb
```

### Step 2: Install Python Dependencies

```bash
# Install AutoGen
pip install autogen-ext autogen-agentchat

# Install DXMT Python SDK
cd clients/dxmt-python-sdk
pip install -e .

# Or install from PyPI (when published)
pip install dxmt-python-sdk
```

### Step 3: Start DXMT Server

```bash
# Simple mode (zero-config)
./target/release/dxmt serve --host 0.0.0.0 --port 9999

# With configuration file
./target/release/dxmt serve --config config.toml
```

### Verification

```bash
# Check server health
curl http://localhost:9999/health

# Expected response:
# {"status":"ok","version":"1.1.0"}
```

---

## Quick Start

### Basic Usage

```python
import asyncio
from autogen_core.memory import MemoryContent
from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig

async def main():
    # Configure DXMT memory
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="my_agent_memory",
        k=5
    )

    memory = DXMTVectorMemory(config=config)

    # Add a memory
    await memory.add(MemoryContent(
        content="User's name is Alice",
        mime_type="text/plain",
        metadata={"user_id": "alice_123"}
    ))

    # Query memory
    results = await memory.query(
        "What is the user's name?",
        user_id="alice_123"
    )

    for result in results.results:
        print(f"{result.content} (score: {result.metadata['score']:.3f})")

    await memory.close()

asyncio.run(main())
```

### With AutoGen Agent

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Configure memory
config = HttpDXMTVectorMemoryConfig(
    host="localhost",
    port=9999,
    table_name="agent_memory"
)

memory = DXMTVectorMemory(config=config)

# Create agent with memory
model_client = OpenAIChatCompletionClient(model="gpt-4")

agent = AssistantAgent(
    name="helpful_assistant",
    model_client=model_client,
    memory=[memory],
    system_message="You are a helpful assistant with persistent memory."
)

# Use the agent
result = await agent.run(task="Remember that I prefer Python over Java")
```

---

## Configuration

### HttpDXMTVectorMemoryConfig

Complete configuration reference for HTTP client mode.

#### Basic Configuration

```python
from autogen_ext.memory.dxmt import HttpDXMTVectorMemoryConfig

config = HttpDXMTVectorMemoryConfig(
    # Connection settings
    host="localhost",           # DXMT server host
    port=9999,                  # DXMT server port
    timeout=30,                 # HTTP request timeout (seconds)

    # Table settings
    table_name="agent_memory",  # Memory table name
    dimension=1536,             # Vector dimension (OpenAI ada-002)
    distance_metric="cosine",   # Distance metric: cosine/euclidean/dot

    # Query settings
    k=5,                        # Number of results to return
    score_threshold=None,       # Minimum similarity score (0-1)

    # Search settings
    enable_full_text_search=True,   # Enable Tantivy full-text search
    enable_hybrid_search=False,     # Enable hybrid search (vector + text)
    hybrid_vector_weight=0.7,       # Vector weight in hybrid (0.0-1.0)
)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client_type` | str | "http" | Client type (always "http" for HTTP mode) |
| `host` | str | "localhost" | DXMT server hostname or IP |
| `port` | int | 9999 | DXMT server port |
| `timeout` | int | 30 | HTTP request timeout in seconds |
| `table_name` | str | "autogen_memory" | Name of the memory table |
| `dimension` | int | 1536 | Vector embedding dimension |
| `distance_metric` | str | "cosine" | Distance metric for similarity |
| `k` | int | 5 | Default number of results to return |
| `score_threshold` | float\|None | None | Minimum similarity score filter |
| `enable_full_text_search` | bool | True | Enable full-text search with Tantivy |
| `enable_hybrid_search` | bool | False | Enable hybrid search mode |
| `hybrid_vector_weight` | float | 0.7 | Weight for vector vs text (0.0-1.0) |

#### Distance Metrics

**Cosine Distance:**
- Range: [0, 2]
- Best for: Normalized vectors, text embeddings
- Normalized to: [1, 0] similarity

**Euclidean Distance:**
- Range: [0, ∞]
- Best for: Geometric distance, image embeddings
- Normalized to: [1, 0] using exp(-distance)

**Dot Product:**
- Range: [-∞, ∞]
- Best for: Dense vectors, when magnitude matters
- Normalized to: [0, 1] assuming unit vectors

### Configuration Examples

#### High-Performance Configuration

```python
config = HttpDXMTVectorMemoryConfig(
    host="localhost",
    port=9999,
    table_name="fast_memory",
    k=3,                        # Fewer results = faster
    timeout=60,                 # Higher timeout for reliability
    score_threshold=0.5,        # Filter low-quality results
)
```

#### Large-Scale Configuration

```python
config = HttpDXMTVectorMemoryConfig(
    host="dxmt-cluster.example.com",
    port=9999,
    table_name="production_memory",
    k=10,
    dimension=1536,
    enable_full_text_search=True,
    enable_hybrid_search=False,  # Requires embedding setup
)
```

#### Multi-User Configuration

```python
config = HttpDXMTVectorMemoryConfig(
    host="localhost",
    port=9999,
    table_name="shared_memory",
    k=5,
    # Users will be filtered via metadata
)
```

### Environment Variables

DXMT server can be configured via environment variables in `.env`:

```bash
# Server settings
DXMT_HOST=0.0.0.0
DXMT_PORT=9999

# Storage settings
DXMT_MEMORY_CACHE_SIZE=512        # Memory cache size in MB

# Object storage (S3)
DXMT_OBJECT_STORAGE_PROVIDER=s3
DXMT_S3_BUCKET=dxmt-vectordb
DXMT_S3_REGION=us-east-1
DXMT_S3_ACCESS_KEY=your_access_key
DXMT_S3_SECRET_KEY=your_secret_key

# Compression
DXMT_COMPRESSION=zstd             # zstd, lz4, none

# Performance
DXMT_WORKER_THREADS=8             # Number of worker threads
```

---

## API Reference

### DXMTVectorMemory

Main class implementing AutoGen's Memory interface.

#### Constructor

```python
def __init__(
    self,
    config: Optional[DXMTVectorMemoryConfigType] = None
) -> None
```

**Parameters:**
- `config` (optional): Configuration object. Defaults to HttpDXMTVectorMemoryConfig()

**Example:**
```python
memory = DXMTVectorMemory(config=HttpDXMTVectorMemoryConfig(
    host="localhost",
    port=9999
))
```

#### Methods

##### add()

Add content to memory.

```python
async def add(
    self,
    content: MemoryContent,
    cancellation_token: Optional[CancellationToken] = None
) -> None
```

**Parameters:**
- `content` (MemoryContent): The memory content to add
- `cancellation_token` (optional): Token to cancel the operation

**Example:**
```python
await memory.add(MemoryContent(
    content="User prefers Python",
    mime_type="text/plain",
    metadata={"user_id": "alice"}
))
```

##### query()

Query memory for relevant content.

```python
async def query(
    self,
    query: str | MemoryContent,
    cancellation_token: Optional[CancellationToken] = None,
    **kwargs: Any
) -> MemoryQueryResult
```

**Parameters:**
- `query` (str | MemoryContent): Query text or content object
- `cancellation_token` (optional): Token to cancel the operation
- `**kwargs`: Additional parameters:
  - `user_id` (str): Filter by user ID
  - `agent_id` (str): Filter by agent ID
  - `session_id` (str): Filter by session ID
  - `limit` (int): Override default k value

**Returns:**
- `MemoryQueryResult`: Object containing list of MemoryContent results

**Example:**
```python
results = await memory.query(
    "What does the user prefer?",
    user_id="alice",
    limit=3
)

for result in results.results:
    print(f"{result.content} (score: {result.metadata['score']:.3f})")
```

##### update_context()

Update model context with relevant memories.

```python
async def update_context(
    self,
    model_context: ChatCompletionContext
) -> UpdateContextResult
```

**Parameters:**
- `model_context` (ChatCompletionContext): The model context to update

**Returns:**
- `UpdateContextResult`: Result containing memories added to context

**Behavior:**
1. Extracts the last message from context
2. Uses it as a query to find relevant memories
3. Formats memories as a numbered list
4. Adds them to context as a SystemMessage

**Example:**
```python
# Called automatically by AutoGen agents
result = await memory.update_context(model_context)
print(f"Added {len(result.memories.results)} memories to context")
```

##### clear()

Clear all content from memory.

```python
async def clear(self) -> None
```

**Warning:** This operation deletes the entire table and recreates it. All data will be lost.

**Example:**
```python
await memory.clear()
```

##### close()

Clean up resources.

```python
async def close(self) -> None
```

**Note:** DXMT HTTP client doesn't require explicit cleanup, so this is a no-op.

**Example:**
```python
await memory.close()
```

### MemoryContent

Data class representing memory content.

```python
class MemoryContent(BaseModel):
    content: str | bytes | Dict[str, Any] | Image
    mime_type: MemoryMimeType | str
    metadata: Optional[Dict[str, Any]] = None
```

**Fields:**
- `content`: The actual content (text, binary, JSON, or image)
- `mime_type`: MIME type of the content
- `metadata`: Optional metadata dictionary

**MIME Types:**
- `"text/plain"`: Plain text
- `"text/markdown"`: Markdown text
- `"application/json"`: JSON data
- `"image/*"`: Image data

### MemoryQueryResult

Result object from query operations.

```python
class MemoryQueryResult(BaseModel):
    results: List[MemoryContent]
```

**Fields:**
- `results`: List of matching MemoryContent objects with scores in metadata

**Example:**
```python
query_result = await memory.query("query text")

for result in query_result.results:
    score = result.metadata.get("score", 0.0)
    created_at = result.metadata.get("created_at")
    print(f"Score: {score:.3f}, Created: {created_at}")
    print(f"Content: {result.content}")
```

---

## Usage Patterns

### Pattern 1: Simple Memory Storage

Store and retrieve simple facts or preferences.

```python
# Add memories
facts = [
    "User's name is Alice",
    "Alice lives in San Francisco",
    "Alice is a software engineer",
]

for fact in facts:
    await memory.add(MemoryContent(
        content=fact,
        mime_type="text/plain"
    ))

# Query
results = await memory.query("Where does the user live?")
```

### Pattern 2: Multi-User Memory

Isolate memories by user ID.

```python
# Add memories for different users
await memory.add(MemoryContent(
    content="Alice prefers Python",
    mime_type="text/plain",
    metadata={"user_id": "alice"}
))

await memory.add(MemoryContent(
    content="Bob prefers Java",
    mime_type="text/plain",
    metadata={"user_id": "bob"}
))

# Query with user filter
results = await memory.query(
    "programming language preference",
    user_id="alice"
)
```

### Pattern 3: Conversation History

Store and retrieve conversation turns.

```python
conversation = [
    ("user", "What is machine learning?"),
    ("assistant", "ML is a subset of AI..."),
    ("user", "Can you give an example?"),
    ("assistant", "Spam detection is a common ML application"),
]

for role, message in conversation:
    await memory.add(MemoryContent(
        content=f"{role}: {message}",
        mime_type="text/plain",
        metadata={"role": role, "type": "conversation"}
    ))

# Retrieve relevant context
results = await memory.query("machine learning examples", limit=2)
```

### Pattern 4: Document Chunks

Store and retrieve document segments.

```python
chunks = [
    "Introduction: Python is a high-level language...",
    "Chapter 1: Variables and data types...",
    "Chapter 2: Functions and scope...",
]

for i, chunk in enumerate(chunks):
    await memory.add(MemoryContent(
        content=chunk,
        mime_type="text/plain",
        metadata={
            "doc_id": "python_guide",
            "chunk_index": i,
            "type": "document"
        }
    ))

# Semantic search across chunks
results = await memory.query("Python functions")
```

### Pattern 5: Multi-Agent Collaboration

Share memory across multiple agents.

```python
# Create shared memory
shared_memory = DXMTVectorMemory(config=config)

# Create agents with shared memory
researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    memory=[shared_memory]
)

writer = AssistantAgent(
    name="writer",
    model_client=model_client,
    memory=[shared_memory]
)

# Agents can query each other's findings
# Researcher adds findings
await shared_memory.add(MemoryContent(
    content="Research shows Python is most popular for AI",
    mime_type="text/plain",
    metadata={"agent_id": "researcher"}
))

# Writer can access researcher's findings
results = await shared_memory.query(
    "What did research find?",
    agent_id="researcher"
)
```

### Pattern 6: Session-Based Memory

Isolate memories by session ID.

```python
# Session 1
await memory.add(MemoryContent(
    content="User asked about Python",
    mime_type="text/plain",
    metadata={"session_id": "session_001"}
))

# Session 2
await memory.add(MemoryContent(
    content="User asked about Java",
    mime_type="text/plain",
    metadata={"session_id": "session_002"}
))

# Query specific session
results = await memory.query(
    "what did user ask",
    session_id="session_001"
)
```

---

## Advanced Features

### Full-Text Search

DXMT supports full-text search using Tantivy + BM25 ranking.

**Enable full-text search:**
```python
config = HttpDXMTVectorMemoryConfig(
    enable_full_text_search=True
)
```

**How it works:**
1. Text content is indexed in Tantivy
2. BM25 algorithm ranks results by relevance
3. Results include highlighting (if available)

**Best for:**
- Exact phrase matching
- Keyword-based search
- Document retrieval

### Hybrid Search

Combine vector similarity and text relevance (requires embedding configuration).

**Enable hybrid search:**
```python
config = HttpDXMTVectorMemoryConfig(
    enable_full_text_search=True,
    enable_hybrid_search=True,
    hybrid_vector_weight=0.7  # 70% vector, 30% text
)
```

**Weight configuration:**
- `0.0`: Pure text search (BM25 only)
- `0.5`: Balanced (equal weights)
- `0.7`: Default (favor vector similarity)
- `1.0`: Pure vector search

**Best for:**
- Semantic + keyword matching
- RAG applications
- Complex queries

### Score Threshold Filtering

Filter out low-relevance results.

```python
config = HttpDXMTVectorMemoryConfig(
    score_threshold=0.5  # Only return scores >= 0.5
)
```

**Benefits:**
- Reduces noise in results
- Improves response quality
- Saves processing time

### Complex Metadata Filtering

Currently supported filters:
- `user_id`: User-specific memories
- `agent_id`: Agent-specific memories
- `session_id`: Session-specific memories

**Example:**
```python
results = await memory.query(
    "query text",
    user_id="alice",
    agent_id="researcher",
    limit=10
)
```

**Future enhancements:**
- Time range filtering
- Custom metadata fields
- Compound conditions

---

## Performance Tuning

### Configuration Optimization

#### Reduce Latency

```python
config = HttpDXMTVectorMemoryConfig(
    k=3,                    # Fewer results
    score_threshold=0.3,    # Filter early
    timeout=60,             # Allow more time
)
```

#### Increase Throughput

```python
# DXMT server .env
DXMT_WORKER_THREADS=16        # More workers
DXMT_MEMORY_CACHE_SIZE=2048   # Larger cache
```

#### Optimize for Cost

```python
# Use tiered storage
DXMT_MEMORY_CACHE_SIZE=512
DXMT_S3_BUCKET=my-bucket
```

### DXMT Server Tuning

#### Memory Cache Size

```bash
# .env
DXMT_MEMORY_CACHE_SIZE=1024  # MB
```

**Guidelines:**
- Small datasets (<10K): 256MB
- Medium datasets (10K-100K): 512MB
- Large datasets (>100K): 1024MB+

#### Storage Type

```bash
```

**Options:**
- `memory`: All in RAM (fastest, expensive)
- `indexed`: SSD-backed (balanced)
- `object`: S3-backed (cheapest)
- `tiered`: Auto-tiering (recommended)

#### Compression

```bash
DXMT_COMPRESSION=zstd
```

**Options:**
- `zstd`: Best compression (recommended)
- `lz4`: Faster, less compression
- `none`: No compression

### Query Optimization

#### Batch Queries

When querying multiple times, batch if possible:

```python
# Bad: Multiple sequential queries
for query in queries:
    await memory.query(query)

# Better: Batch related queries
results = await asyncio.gather(*[
    memory.query(q) for q in queries
])
```

#### Use Metadata Filters

Narrow search scope with filters:

```python
# Slower: Search all memories
results = await memory.query("query")

# Faster: Filter by user
results = await memory.query("query", user_id="alice")
```

#### Adjust k Value

Return only what you need:

```python
# Default
results = await memory.query("query", limit=5)

# For summaries, fewer is better
results = await memory.query("query", limit=2)
```

### Monitoring and Profiling

```python
import time

# Measure query latency
start = time.time()
results = await memory.query("query")
latency = time.time() - start

print(f"Query latency: {latency*1000:.1f}ms")
print(f"Results: {len(results.results)}")
```

---

## Troubleshooting

### Connection Issues

**Problem:** `ConnectionRefusedError: Connection refused`

**Solution:**
1. Check DXMT server is running:
   ```bash
   ps aux | grep dxmt
   ```

2. Verify server is listening:
   ```bash
   curl http://localhost:9999/health
   ```

3. Check firewall rules:
   ```bash
   sudo netstat -tlnp | grep 9999
   ```

### Empty Results

**Problem:** Queries return no results

**Solutions:**

1. **Check if data was added:**
   ```bash
   curl http://localhost:9999/tables/your_table_name/stats
   ```

2. **Lower score threshold:**
   ```python
   config = HttpDXMTVectorMemoryConfig(
       score_threshold=None  # Remove threshold
   )
   ```

3. **Verify metadata filters:**
   ```python
   # Remove filters to see all results
   results = await memory.query("query")  # No user_id filter
   ```

### Slow Queries

**Problem:** Queries take too long

**Solutions:**

1. **Reduce k value:**
   ```python
   results = await memory.query("query", limit=3)
   ```

2. **Add score threshold:**
   ```python
   config = HttpDXMTVectorMemoryConfig(
       score_threshold=0.5
   )
   ```

3. **Enable tiered storage:**
   ```bash
   # .env
   DXMT_MEMORY_CACHE_SIZE=1024
   ```

### Memory Leaks

**Problem:** Memory usage keeps growing

**Solutions:**

1. **Always close memory:**
   ```python
   try:
       await memory.query("query")
   finally:
       await memory.close()
   ```

2. **Clear old memories:**
   ```python
   # Periodically clear old data
   await memory.clear()
   ```

3. **Use connection pooling:**
   ```python
   # Reuse memory instance
   shared_memory = DXMTVectorMemory(config=config)
   ```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'dxmt'`

**Solution:**
```bash
cd clients/dxmt-python-sdk
pip install -e .
```

**Problem:** `ModuleNotFoundError: No module named 'autogen_ext'`

**Solution:**
```bash
pip install autogen-ext autogen-agentchat
```

---

## Migration Guide

### From ChromaDB

**ChromaDB code:**
```python
from autogen_ext.memory.chromadb import ChromaDBVectorMemory

memory = ChromaDBVectorMemory(
    persistence_path="./chroma_db",
    collection_name="my_memory"
)
```

**DXMT equivalent:**
```python
from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig

memory = DXMTVectorMemory(config=HttpDXMTVectorMemoryConfig(
    host="localhost",
    port=9999,
    table_name="my_memory"
))
```

**Key differences:**
- DXMT uses HTTP client (requires running server)
- DXMT supports full-text and hybrid search
- DXMT has tiered storage for cost efficiency

### From mem0

**mem0 code:**
```python
from autogen_ext.memory.mem0 import Mem0Memory

memory = Mem0Memory(
    user_id="alice",
    is_cloud=False,
    config={"path": ":memory:"}
)
```

**DXMT equivalent:**
```python
from autogen_ext.memory.dxmt import DXMTVectorMemory, HttpDXMTVectorMemoryConfig

memory = DXMTVectorMemory(config=HttpDXMTVectorMemoryConfig(
    host="localhost",
    port=9999,
    table_name="alice_memory"
))

# Add user_id to all operations
await memory.add(MemoryContent(
    content="...",
    mime_type="text/plain",
    metadata={"user_id": "alice"}
))

results = await memory.query("...", user_id="alice")
```

**Key differences:**
- DXMT doesn't have built-in user_id, use metadata
- DXMT requires running server
- DXMT has more configuration options

---

## Best Practices

### 1. Configuration Management

**Use environment-specific configs:**
```python
import os

def get_memory_config():
    if os.getenv("ENV") == "production":
        return HttpDXMTVectorMemoryConfig(
            host="dxmt-prod.example.com",
            port=9999,
            table_name="prod_memory",
            timeout=60
        )
    else:
        return HttpDXMTVectorMemoryConfig(
            host="localhost",
            port=9999,
            table_name="dev_memory"
        )
```

### 2. Error Handling

**Always wrap operations in try-except:**
```python
try:
    results = await memory.query("query")
    if not results.results:
        print("No results found")
except Exception as e:
    logger.error(f"Query failed: {e}")
    # Handle error gracefully
```

### 3. Resource Management

**Use context managers (if available) or try-finally:**
```python
memory = DXMTVectorMemory(config=config)

try:
    await memory.add(content)
    results = await memory.query("query")
finally:
    await memory.close()
```

### 4. Metadata Organization

**Use consistent metadata structure:**
```python
# Good
metadata = {
    "user_id": "alice",
    "session_id": "session_123",
    "timestamp": datetime.utcnow().isoformat(),
    "source": "conversation"
}

# Bad (inconsistent naming)
metadata = {
    "userId": "alice",  # Mixed case
    "SessionID": "session_123",  # Inconsistent
}
```

### 5. Query Design

**Be specific in queries:**
```python
# Good: Specific query
results = await memory.query("What is the user's preferred programming language?")

# Bad: Too generic
results = await memory.query("user info")
```

### 6. Memory Lifecycle

**Clean up old memories periodically:**
```python
# Add timestamp to all memories
await memory.add(MemoryContent(
    content="...",
    mime_type="text/plain",
    metadata={
        "created_at": datetime.utcnow().isoformat(),
        "ttl_days": 30
    }
))

# Later: Clean up old memories
# (Requires custom logic to filter and delete)
```

### 7. Testing

**Write tests for memory operations:**
```python
import pytest

@pytest.mark.asyncio
async def test_memory_add_and_query():
    memory = DXMTVectorMemory(config=test_config)

    await memory.add(MemoryContent(
        content="Test content",
        mime_type="text/plain"
    ))

    results = await memory.query("test")

    assert len(results.results) > 0
    assert "test" in results.results[0].content.lower()

    await memory.clear()
    await memory.close()
```

---

## FAQ

### Q: Do I need to run a separate DXMT server?

**A:** Yes, DXMT requires a running server. You can run it locally during development or deploy it to production.

### Q: Can I use DXMT with multiple agents?

**A:** Yes! Multiple agents can share the same DXMTVectorMemory instance or use `agent_id` metadata to separate memories.

### Q: How much does it cost to run DXMT?

**A:** With tiered storage:
- Memory: $10/GB/month
- SSD: $0.50/GB/month
- S3: $0.023/GB/month

A 10GB dataset costs ~$0.23/month with tiered storage vs $100/month all in memory.

### Q: What's the maximum number of memories?

**A:** Single node: ~10M memories. Cluster mode: Unlimited.

### Q: How do I backup my data?

**A:** DXMT automatically persists to S3 when using tiered storage. You can also:
```bash
# Backup S3 bucket
aws s3 sync s3://dxmt-bucket s3://dxmt-backup
```

### Q: Can I use DXMT without internet access?

**A:** Yes, DXMT can run fully offline with local storage (memory or SSD mode).

### Q: Is DXMT production-ready?

**A:** Yes, DXMT v1.1.0 with AutoGen integration is production-ready.

### Q: How do I scale DXMT?

**A:** Use cluster mode with multiple DXMT nodes and a load balancer.

### Q: Does DXMT support embeddings?

**A:** DXMT can auto-embed text if configured with an embedding model. External embedding support is coming in v1.2.0.

### Q: Can I migrate from ChromaDB to DXMT?

**A:** Yes, see the [Migration Guide](#migration-guide) section.

---

## Support and Resources

### Documentation
- **Project README:** `README.md`
- **Quick Start:** `AUTOGEN_QUICKSTART.md`
- **Design Document:** `DXMT_AUTOGEN_INTEGRATION_DESIGN.md`

### Examples
- Basic usage: `examples/autogen_dxmt_basic.py`
- Multi-agent: `examples/autogen_dxmt_multi_agent.py`
- Advanced: `examples/autogen_dxmt_advanced.py`

### Community
- GitHub Issues: https://github.com/dxmt/dxmt/issues
- Discussions: https://github.com/dxmt/dxmt/discussions

### Commercial Support
Contact: support@dxmt.online

---

**Document Version:** 1.0.0
**Last Updated:** 2026-01-28
**License:** Same as DXMT VectorDB project
