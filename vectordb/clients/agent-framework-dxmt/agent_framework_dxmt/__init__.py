"""DXMT for Microsoft Agent Framework: a vector store, and memory as a context provider."""

import importlib.metadata

from ._context_provider import DXMTContextProvider
from ._settings import DXMTSettings
from ._vector_store import DXMTCollection, DXMTStore

try:
    __version__ = importlib.metadata.version("agent-framework-dxmt")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["DXMTCollection", "DXMTContextProvider", "DXMTSettings", "DXMTStore", "__version__"]
