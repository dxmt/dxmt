"""DXMT Python SDK.

Python SDK for DXMT, the open-source memory database for personal agents: agent
memory (remember / recall / forget), vectors and full text, served by one Rust binary.

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

# The Haystack integration is its own package since 1.7: `pip install dxmt-haystack`, then
# `from haystack_integrations.document_stores.dxmt import DXMTDocumentStore`. The copy that
# lived here passed `dataframe=` to a Document that no longer has the field and invented a
# vector for documents without one; it is gone rather than kept beside the real one.

# The CrewAI integration is its own package since 1.7: `pip install crewai-dxmt`. The copy
# that lived here (`dxmt.crewai_storage`) implemented `crewai.memory.storage.interface.Storage`,
# which CrewAI removed with its unified memory system, so it had been a stub for months.

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

__version__ = "1.6.0"

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
    # MetaGPT/LlamaIndex integration (optional)
    "DXMTLlamaIndexVectorStore",
    "DXMTRetriever",
    "create_dxmt_retriever",
    # Bisheng/LangChain integration (optional)
    "DXMTLangChainVectorStore",
    "initialize_dxmt",
    "initial_dxmt",
]
