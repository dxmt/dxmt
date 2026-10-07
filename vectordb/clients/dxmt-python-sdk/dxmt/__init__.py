"""DXMT Python SDK.

Official Python SDK for DXMT VectorDB - a lightweight single-node vector database
written in Rust.

Example:
    >>> from dxmt import Client
    >>> client = Client(host="localhost", port=9999)
    >>> client.create_table(
    ...     name="my_table",
    ...     dimension=1536,
    ...     distance_metric="cosine"
    ... )
    >>> client.insert_records(
    ...     table_name="my_table",
    ...     records=[{
    ...         "id": "doc1",
    ...         "vector": [0.1, 0.2, ...],
    ...         "metadata": {"text": "content"}
    ...     }]
    ... )
    >>> results = client.search(
    ...     table_name="my_table",
    ...     vector=[0.1, 0.2, ...],
    ...     limit=10
    ... )
"""

from .client import DXMTClient, UploadFailure, UploadResult
from .client import DXMTClient as Client  # Alias for backward compatibility
from .exceptions import (
    DXMTConnectionError,
    DXMTError,
    DXMTRecordError,
    DXMTSearchError,
    DXMTTableError,
    DXMTTimeoutError,
    DXMTValidationError,
)
from .filters import Field
from .types import (
    DenseVector,
    DistanceMetric,
    Filter,
    IndexType,
    InsertMode,
    QuantizationType,
    Record,
    SearchResult,
    SparseVector,
    TableConfig,
)

# Optional LangGraph integration (requires langgraph package)
try:
    from .langgraph_store import DXMTLangGraphStore
    LANGGRAPH_AVAILABLE = True
except (ImportError, RuntimeError, Exception):
    # Catch all exceptions during import (including pydantic validation errors)
    LANGGRAPH_AVAILABLE = False
    DXMTLangGraphStore = None  # type: ignore

# Optional mem0 integration (requires mem0ai package)
try:
    from .mem0 import DXMTMemoryStore
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    DXMTMemoryStore = None  # type: ignore

# Optional Haystack integration (requires haystack-ai package)
try:
    from .haystack import DXMTDocumentStore
    from .haystack_retrievers import DXMTEmbeddingRetriever, DXMTBM25Retriever
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    DXMTDocumentStore = None  # type: ignore
    DXMTEmbeddingRetriever = None  # type: ignore
    DXMTBM25Retriever = None  # type: ignore

# Optional CrewAI integration (requires crewai package)
try:
    from .crewai_tool import DXMTVectorSearchTool, DXMTToolConfig
    from .crewai_storage import DXMTStorage, DXMTStorageConfig
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    DXMTVectorSearchTool = None  # type: ignore
    DXMTToolConfig = None  # type: ignore
    DXMTStorage = None  # type: ignore
    DXMTStorageConfig = None  # type: ignore

# Optional MetaGPT/LlamaIndex integration (requires llama-index package)
try:
    from .metagpt_vector_store import DXMTVectorStore as DXMTLlamaIndexVectorStore
    from .metagpt_retriever import DXMTRetriever, create_dxmt_retriever
    METAGPT_AVAILABLE = True
except ImportError:
    METAGPT_AVAILABLE = False
    DXMTLlamaIndexVectorStore = None  # type: ignore
    DXMTRetriever = None  # type: ignore
    create_dxmt_retriever = None  # type: ignore

# Optional Bisheng compatibility layer (requires langchain). For LangChain itself use the
# `langchain-dxmt` package; see the docstring of `langchain_vectorstore.py`.
try:
    from .langchain_vectorstore import DXMTVectorStore as DXMTLangChainVectorStore
    from .bisheng_integration import initialize_dxmt, initial_dxmt
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    DXMTLangChainVectorStore = None  # type: ignore
    initialize_dxmt = None  # type: ignore
    initial_dxmt = None  # type: ignore

__version__ = "1.3.2"

__all__ = [
    "UploadResult",
    "UploadFailure",
    # Main client
    "Client",
    "DXMTClient",  # Also export under original name
    # Exceptions
    "DXMTError",
    "DXMTConnectionError",
    "DXMTTableError",
    "DXMTRecordError",
    "DXMTSearchError",
    "DXMTValidationError",
    "DXMTTimeoutError",
    # Types
    "DenseVector",
    "SparseVector",
    "DistanceMetric",
    "IndexType",
    "InsertMode",
    "QuantizationType",
    "Record",
    "SearchResult",
    "TableConfig",
    "Filter",
    "Field",
    # LangGraph integration (optional)
    "DXMTLangGraphStore",
    # mem0 integration (optional)
    "DXMTMemoryStore",
    # Haystack integration (optional)
    "DXMTDocumentStore",
    "DXMTEmbeddingRetriever",
    "DXMTBM25Retriever",
    # CrewAI integration (optional)
    "DXMTVectorSearchTool",
    "DXMTToolConfig",
    "DXMTStorage",
    "DXMTStorageConfig",
    # MetaGPT/LlamaIndex integration (optional)
    "DXMTLlamaIndexVectorStore",
    "DXMTRetriever",
    "create_dxmt_retriever",
    # Bisheng/LangChain integration (optional)
    "DXMTLangChainVectorStore",
    "initialize_dxmt",
    "initial_dxmt",
]
