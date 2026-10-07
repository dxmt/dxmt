"""LangChain integration for DXMT VectorDB.

This package provides a LangChain VectorStore implementation for DXMT VectorDB.
It uses the dxmt-python-sdk for all communication with the DXMT server.

Architecture:
    - dxmt-python-sdk: Core HTTP client and API wrapper
    - langchain-dxmt: LangChain VectorStore adapter (this package)

For direct API access without LangChain, use dxmt-python-sdk directly:
    from dxmt import Client

For LangChain integration, use this package:
    from langchain_dxmt import DXMTVectorDB
"""

# Re-export from dxmt-python-sdk for backward compatibility
from dxmt import (
    Client as DXMTClient,
    DXMTConnectionError,
    DXMTError,
    DXMTRecordError,
    DXMTSearchError,
    DXMTTableError,
    DXMTTimeoutError,
    DXMTValidationError,
)

# Main LangChain integration
from langchain_dxmt.vectorstores import DXMTVectorDB

__version__ = "1.3.0"

__all__ = [
    # LangChain integration
    "DXMTVectorDB",
    # Re-exported from dxmt-python-sdk (for backward compatibility)
    "DXMTClient",
    "DXMTError",
    "DXMTConnectionError",
    "DXMTTableError",
    "DXMTSearchError",
    "DXMTRecordError",
    "DXMTTimeoutError",
    "DXMTValidationError",
    # Metadata
    "__version__",
]
