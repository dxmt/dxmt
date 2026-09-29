"""LangChain official VectorStoreIntegrationTests for DXMT.

This module uses LangChain's official VectorStoreIntegrationTests base class
to ensure full compliance with LangChain standards.

Run with:
    pytest tests/langchain_standard/test_vectorstore_integration_tests.py -v

Prerequisites:
- DXMT server running on localhost:9999
- langchain-standard-tests package installed
"""

import os
import time
from typing import Generator

import pytest

try:
    from langchain_core.vectorstores import VectorStore
    from langchain_tests.integration_tests.vectorstores import (
        VectorStoreIntegrationTests,
    )

    LANGCHAIN_TESTS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TESTS_AVAILABLE = False
    # Create dummy base class if tests not available
    class VectorStoreIntegrationTests:
        pass

from langchain_dxmt import DXMTVectorDB


# Configuration
DXMT_HOST = os.getenv("DXMT_HOST", "localhost")
DXMT_PORT = int(os.getenv("DXMT_PORT", "9999"))

# The import above already says `langchain_tests`; these messages named
# `langchain-standard-tests`, which is what the package used to be called and is not on PyPI
# under that name -- so anyone who followed the instruction got "No matching distribution"
# and the official conformance suite stayed skipped.
pytestmark = pytest.mark.skipif(
    not LANGCHAIN_TESTS_AVAILABLE,
    reason="langchain-tests not installed. Install with: pip install langchain-tests",
)


@pytest.mark.skipif(
    not LANGCHAIN_TESTS_AVAILABLE,
    reason="langchain-tests not available",
)
class TestDXMTVectorStoreIntegrationTests(VectorStoreIntegrationTests):
    """Official LangChain VectorStore integration tests for DXMT.

    This class inherits from LangChain's VectorStoreIntegrationTests
    to ensure DXMT implements the standard VectorStore interface correctly.

    The base class provides comprehensive tests for:
    - add_texts() and add_documents()
    - similarity_search() and variants
    - delete() and get_by_ids()
    - Metadata handling and filtering
    - Score calculations
    - Document integrity
    """

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore for unit tests.

        This fixture is required by VectorStoreIntegrationTests.
        It must yield a clean, empty vectorstore instance.
        """
        # Generate unique table name to avoid conflicts
        timestamp = int(time.time() * 1000)
        table_name = f"langchain_test_{timestamp}"

        # Create vectorstore with deterministic embeddings
        store = DXMTVectorDB(
            table_name=table_name,
            embedding=self.get_embeddings(),
            dimension=self.get_embeddings().embed_query("test").__len__(),
            host=DXMT_HOST,
            port=DXMT_PORT,
            create_table_if_not_exists=True,
        )

        try:
            yield store
        finally:
            # Cleanup: delete table after test
            try:
                store._client.delete_table(table_name)
            except Exception:
                pass

    @property
    def has_sync(self) -> bool:
        """Whether this vectorstore supports sync operations."""
        return True

    @property
    def has_async(self) -> bool:
        """Whether this vectorstore supports async operations."""
        return True

    # Override specific tests if needed
    # def test_specific_case(self, vectorstore: VectorStore) -> None:
    #     """Override or skip specific tests if incompatible."""
    #     pytest.skip("Not applicable for DXMT")


# Additional compliance note
def test_compliance_note(capsys) -> None:
    """Print compliance information."""
    if not LANGCHAIN_TESTS_AVAILABLE:
        print("\n⚠️  langchain-standard-tests not installed.")
        print("Install with: pip install langchain-standard-tests")
        print("This package provides official LangChain VectorStore compliance tests.")
    else:
        print("\n✅ Using LangChain official VectorStoreIntegrationTests")
        print("This ensures full compliance with LangChain standards.")
