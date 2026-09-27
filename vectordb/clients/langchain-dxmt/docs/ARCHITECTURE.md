# langchain-dxmt Architecture

## Overview

`langchain-dxmt` is a **thin adapter layer** that provides LangChain VectorStore integration for DXMT VectorDB. It depends on `dxmt-python-sdk` for all communication with the DXMT server.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│            User Application Layer                        │
│  (LangChain RAG, Q&A, ChatBots, etc.)                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         langchain-dxmt (This Package)                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │  DXMTVectorDB                                     │  │
│  │  - Implements LangChain VectorStore interface    │  │
│  │  - Document ↔ Record conversion                  │  │
│  │  - Embeddings integration                        │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────┘
                 │ depends on
                 ▼
┌─────────────────────────────────────────────────────────┐
│         dxmt-python-sdk (Dependency)                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Client                                           │  │
│  │  - HTTP client (sync & async)                    │  │
│  │  - All DXMT API endpoints                        │  │
│  │  - Error handling                                │  │
│  │  - Type definitions                              │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/REST API
                 ▼
┌─────────────────────────────────────────────────────────┐
│            DXMT Server (Rust)                            │
│  - Vector storage & indexing                            │
│  - Search operations                                     │
│  - Full-text search                                      │
└─────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. **Single Responsibility**

Each package has a clear, focused responsibility:

- **dxmt-python-sdk**: HTTP client and API wrapper (universal Python SDK)
- **langchain-dxmt**: LangChain VectorStore adapter (LangChain-specific)

### 2. **Separation of Concerns**

```python
# dxmt-python-sdk provides low-level API access
from dxmt import Client
client = Client()
client.search(table_name="docs", vector=[...], k=10)

# langchain-dxmt provides high-level LangChain integration
from langchain_dxmt import DXMTVectorDB
vectorstore = DXMTVectorDB(table_name="docs", embedding=embeddings)
vectorstore.similarity_search("query", k=10)
```

### 3. **Code Reuse**

By depending on `dxmt-python-sdk`, we:
- ✅ Eliminate ~700 lines of duplicate code
- ✅ Reduce maintenance burden (fix bugs once)
- ✅ Ensure consistency across integrations
- ✅ Enable standalone SDK usage

### 4. **Backward Compatibility**

We re-export SDK classes for backward compatibility:

```python
# These all work:
from dxmt import Client                    # Direct SDK import
from langchain_dxmt import DXMTClient      # Re-exported from SDK
from langchain_dxmt import DXMTVectorDB    # LangChain integration
```

## Package Contents

### Core Module

```
langchain_dxmt/
├── __init__.py          # Package exports and re-exports
└── vectorstores.py      # DXMTVectorDB implementation (~200 lines)
```

### Key Classes

#### `DXMTVectorDB` (vectorstores.py)

The main VectorStore implementation that:
- Extends `langchain_core.vectorstores.VectorStore`
- Uses `dxmt.Client` internally for all API calls
- Handles document ↔ record conversion
- Integrates with LangChain embeddings
- Supports both sync and async operations

**Key methods:**
- `add_texts()` - Add documents with embeddings
- `similarity_search()` - Vector similarity search
- `hybrid_search()` - Combined vector + full-text search
- `delete()` - Delete documents by ID
- `from_texts()` - Class method to create from texts
- `from_existing_table()` - Connect to existing table

## Dependencies

### Runtime Dependencies

```toml
dependencies = [
    "langchain-core>=0.3.0,<0.4.0",   # LangChain abstractions
    "dxmt-python-sdk>=0.1.0,<1.0.0",  # DXMT HTTP client
    "pydantic>=2.0.0,<3.0.0",         # Data validation
    "numpy>=1.24.0,<2.0.0",           # Vector operations
]
```

### Why This Architecture?

This follows the **industry standard pattern** used by:
- `langchain-pinecone` → depends on `pinecone-client`
- `langchain-chroma` → depends on `chromadb`
- `langchain-milvus` → depends on `pymilvus`

## Usage Examples

### Scenario 1: LangChain Integration (Most Common)

```python
from langchain_dxmt import DXMTVectorDB
from langchain_openai import OpenAIEmbeddings

# LangChain VectorStore
vectorstore = DXMTVectorDB(
    table_name="documents",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
)

# LangChain operations
docs = vectorstore.similarity_search("query", k=5)
vectorstore.add_texts(["text1", "text2"])
```

### Scenario 2: Direct SDK Usage (Without LangChain)

```python
from dxmt import Client

# Direct API access
client = Client(host="localhost", port=9999)
client.create_table(name="docs", dimension=1536)
client.insert_records(table_name="docs", records=[...])
results = client.search(table_name="docs", vector=[...], k=10)
```

### Scenario 3: Mixed Usage

```python
from langchain_dxmt import DXMTVectorDB, DXMTClient
from langchain_openai import OpenAIEmbeddings

# Create client directly
client = DXMTClient(host="localhost", port=9999)

# Use client with VectorStore
vectorstore = DXMTVectorDB(
    table_name="docs",
    embedding=OpenAIEmbeddings(),
    client=client,  # Reuse existing client
    create_table_if_not_exists=False,
)
```

## Testing Strategy

### Unit Tests
- Mock `dxmt.Client` for testing VectorStore logic
- Test document conversion, embedding integration
- Located in `tests/unit_tests/`

### Integration Tests
- Require running DXMT server
- Test end-to-end functionality
- Located in `tests/integration_tests/`

## Migration from Old Architecture

### Before (Duplicated Code)

```
langchain-dxmt/
├── langchain_dxmt/
│   ├── client.py        # 697 lines - DUPLICATE ❌
│   └── vectorstores.py  # 500 lines
```

### After (Clean Architecture)

```
langchain-dxmt/
├── langchain_dxmt/
│   └── vectorstores.py  # 200 lines ✅

dxmt-python-sdk/
└── dxmt/
    └── client.py        # 704 lines ✅
```

**Result**: Eliminated 700 lines of duplicate code!

## Benefits

### For Users

1. **Choice**: Use SDK directly or via LangChain
2. **Consistency**: Same underlying client everywhere
3. **Stability**: Semantic versioning for both packages
4. **Updates**: SDK improvements benefit all integrations

### For Maintainers

1. **Single Source**: Fix bugs in one place
2. **Faster Development**: Changes propagate automatically
3. **Clear Boundaries**: Each package has clear scope
4. **Better Testing**: Test client and adapter separately

## Version Management

### Versioning Strategy

```
dxmt-python-sdk:
  0.1.0 → Initial release
  0.2.0 → New features
  1.0.0 → Production ready

langchain-dxmt:
  0.1.0 → Initial release (requires dxmt-python-sdk>=0.1.0,<1.0.0)
  0.2.0 → New LangChain features
  1.0.0 → Production ready
```

### Compatibility

langchain-dxmt uses **semantic versioning** for dxmt-python-sdk:
```toml
"dxmt-python-sdk>=0.1.0,<1.0.0"  # Compatible with 0.x versions
```

## Future Considerations

### Potential Enhancements

1. **Caching Layer**: Add optional caching between adapter and SDK
2. **Batch Optimizations**: Optimize bulk operations in adapter
3. **Custom Serialization**: Add custom document serializers
4. **Advanced Filtering**: Enhanced metadata filtering in adapter

### Package Evolution

As the ecosystem grows:
- `dxmt-python-sdk` remains the core SDK
- `langchain-dxmt` stays focused on LangChain integration
- Other integrations (e.g., `llamaindex-dxmt`) can reuse SDK

## References

- [LangChain Architecture Blog](https://blog.langchain.com/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/)
- [dxmt-python-sdk Repository](https://github.com/dxmt/dxmt/tree/main/vectordb/clients/dxmt-python-sdk)
- [LangChain VectorStore Interface](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
