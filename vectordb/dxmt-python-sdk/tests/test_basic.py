"""Basic tests for DXMT SDK functionality."""

from dxmt import Client
from dxmt.exceptions import (
    DXMTConnectionError,
    DXMTError,
    DXMTRecordError,
    DXMTSearchError,
    DXMTTableError,
    DXMTTimeoutError,
    DXMTValidationError,
)


def test_imports() -> None:
    """Test that all main classes can be imported."""
    assert Client is not None


def test_client_initialization() -> None:
    """Test client initialization with default settings."""
    client = Client()
    assert client.base_url == "http://localhost:9999/api/v1"
    client.close()


def test_client_initialization_custom() -> None:
    """Test client initialization with custom settings."""
    client = Client(host="example.com", port=8080, ssl=True)
    assert client.base_url == "https://example.com:8080/api/v1"
    client.close()


def test_client_context_manager() -> None:
    """Test client can be used as context manager."""
    with Client() as client:
        assert client is not None
        assert client.base_url == "http://localhost:9999/api/v1"


def test_exceptions_exist() -> None:
    """Test that all exception classes exist."""
    assert DXMTError is not None
    assert DXMTConnectionError is not None
    assert DXMTTableError is not None
    assert DXMTRecordError is not None
    assert DXMTSearchError is not None
    assert DXMTValidationError is not None
    assert DXMTTimeoutError is not None


def test_exception_inheritance() -> None:
    """Test exception inheritance."""
    assert issubclass(DXMTConnectionError, DXMTError)
    assert issubclass(DXMTTableError, DXMTError)
    assert issubclass(DXMTSearchError, DXMTError)


def test_client_has_methods() -> None:
    """Test that client has expected methods."""
    client = Client()

    # Table management
    assert hasattr(client, "create_table")
    assert hasattr(client, "delete_table")
    assert hasattr(client, "list_tables")
    assert hasattr(client, "get_table")

    # Record operations
    assert hasattr(client, "insert_records")
    assert hasattr(client, "delete_records")
    assert hasattr(client, "get_records")

    # Search operations
    assert hasattr(client, "search")
    assert hasattr(client, "hybrid_search")
    assert hasattr(client, "full_text_search")

    # Async equivalents
    assert hasattr(client, "acreate_table")
    assert hasattr(client, "asearch")
    assert hasattr(client, "ainsert_records")

    client.close()


def test_types_module() -> None:
    """Test that types can be imported."""
    from dxmt.types import (
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

    # Check that type aliases exist
    assert DenseVector is not None
    assert SparseVector is not None
    assert DistanceMetric is not None
    assert IndexType is not None
    assert InsertMode is not None
    assert QuantizationType is not None
    assert Record is not None
    assert SearchResult is not None
    assert TableConfig is not None
    assert Filter is not None
