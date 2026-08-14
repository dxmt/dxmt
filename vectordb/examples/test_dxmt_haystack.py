"""
Basic DXMT Haystack Integration Test

Tests the DXMT Haystack integration without requiring a running DXMT server.
Uses mock client to verify functionality.

Run with: python3 examples/test_dxmt_haystack.py
"""

import sys
from unittest.mock import Mock
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

from dxmt.haystack import DXMTDocumentStore
from dxmt.haystack_retrievers import DXMTEmbeddingRetriever, DXMTBM25Retriever
from dxmt.exceptions import DXMTError

# Import Document and DuplicatePolicy from DXMT's haystack module
# This ensures we use the same classes whether haystack is installed or not
from dxmt.haystack import Document, DuplicatePolicy

# Check if real Haystack is available
try:
    import haystack
    HAYSTACK_AVAILABLE = True
except ImportError:
    print("WARNING: haystack-ai not installed, using stub classes from dxmt.haystack")
    HAYSTACK_AVAILABLE = False


def create_mock_client():
    """Create a mock DXMT client for testing."""
    client = Mock()

    # Mock table operations
    client.get_table.side_effect = DXMTError("Table not found")
    client.create_table.return_value = {"name": "test", "dimension": 384}

    # Mock insert operations
    client.insert_records.return_value = None
    client.update_record.return_value = None

    # Mock search operations
    client.search.return_value = [
        {
            "id": "doc1",
            "score": 0.95,
            "vector": [0.1] * 384,
            "metadata": {"content": "Test document 1", "type": "test"},
        },
        {
            "id": "doc2",
            "score": 0.85,
            "vector": [0.2] * 384,
            "metadata": {"content": "Test document 2", "type": "test"},
        },
    ]

    client.full_text_search.return_value = [
        {
            "id": "doc1",
            "score": 10.5,
            "metadata": {"content": "Test document 1", "type": "test"},
        },
    ]

    # Mock get/delete operations
    client.get_records.return_value = []
    client.delete_records.return_value = None

    return client


def test_document_store_initialization():
    """Test DXMTDocumentStore initialization."""
    print("\n=== Test 1: Document Store Initialization ===")

    client = create_mock_client()
    store = DXMTDocumentStore(
        client=client,
        table_name="test_table",
        embedding_dim=384,
        distance_metric="cosine",
    )

    assert store.table_name == "test_table"
    assert store.embedding_dim == 384
    assert store.distance_metric == "cosine"
    assert client.create_table.called

    print("✓ Document store initialized successfully")


def test_write_documents():
    """Test writing documents to store."""
    print("\n=== Test 2: Write Documents ===")

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)

    documents = [
        Document(
            id="doc1",
            content="Test document 1",
            meta={"type": "test"},
            embedding=[0.1] * 384,
        ),
        Document(
            id="doc2",
            content="Test document 2",
            meta={"type": "test"},
            embedding=[0.2] * 384,
        ),
    ]

    written = store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    assert written == 2
    assert client.insert_records.called

    print(f"✓ Wrote {written} documents")


def test_count_documents():
    """Test counting documents."""
    print("\n=== Test 3: Count Documents ===")

    client = create_mock_client()
    # Override get_table to return count
    client.get_table.side_effect = None
    client.get_table.return_value = {"name": "test", "record_count": 42}

    store = DXMTDocumentStore(client=client, embedding_dim=384)
    count = store.count_documents()

    assert count == 42
    print(f"✓ Document count: {count}")


def test_embedding_retrieval():
    """Test embedding-based retrieval."""
    print("\n=== Test 4: Embedding Retrieval ===")

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384, distance_metric="cosine")

    query_embedding = [0.1] * 384
    results = store.embedding_retrieval(
        query_embedding=query_embedding,
        top_k=5,
        scale_score=True,
    )

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert all(isinstance(doc, Document) for doc in results), "Not all results are Document instances"
    assert all(hasattr(doc, "score") and doc.score is not None for doc in results), "Not all documents have scores"
    assert client.search.called, "search was not called"

    print(f"✓ Retrieved {len(results)} documents")
    if results and results[0].score is not None:
        print(f"  Top score: {results[0].score:.3f}")


def test_bm25_retrieval():
    """Test BM25 full-text retrieval."""
    print("\n=== Test 5: BM25 Retrieval ===")

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)

    results = store.bm25_retrieval(
        query="test query",
        top_k=5,
        scale_score=False,
    )

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert isinstance(results[0], Document), "Result is not a Document instance"
    assert hasattr(results[0], "score") and results[0].score is not None, "Document does not have a score"
    assert client.full_text_search.called, "full_text_search was not called"

    print(f"✓ Retrieved {len(results)} documents")
    if results and results[0].score is not None:
        print(f"  BM25 score: {results[0].score:.2f}")


def test_filter_conversion():
    """Test Haystack filter conversion."""
    print("\n=== Test 6: Filter Conversion ===")

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)

    # Simple filter
    haystack_filter = {"field": "meta.type", "operator": "==", "value": "article"}
    dxmt_filter = store._convert_haystack_filters(haystack_filter)

    # New MongoDB format: {"type": "article"}
    assert dxmt_filter["type"] == "article", f"Expected {{'type': 'article'}}, got {dxmt_filter}"

    # Complex filter
    complex_filter = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.year", "operator": ">=", "value": 2020},
        ],
    }
    dxmt_complex = store._convert_haystack_filters(complex_filter)

    # New MongoDB format: {"$and": [{"type": "article"}, {"year": {"$gte": 2020}}]}
    assert "$and" in dxmt_complex, f"Expected $and operator, got {dxmt_complex}"
    assert len(dxmt_complex["$and"]) == 2, f"Expected 2 conditions, got {len(dxmt_complex['$and'])}"
    assert dxmt_complex["$and"][0]["type"] == "article"
    assert dxmt_complex["$and"][1]["year"]["$gte"] == 2020

    print("✓ Filter conversion working correctly")


def test_embedding_retriever():
    """Test DXMTEmbeddingRetriever component."""
    print("\n=== Test 7: Embedding Retriever Component ===")

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)
    retriever = DXMTEmbeddingRetriever(
        document_store=store,
        top_k=5,
        scale_score=True,
    )

    assert retriever.document_store == store
    assert retriever.top_k == 5
    assert retriever.scale_score == True

    # Test run method
    query_embedding = [0.1] * 384
    result = retriever.run(query_embedding=query_embedding)

    assert "documents" in result
    assert len(result["documents"]) == 2

    print(f"✓ Retriever returned {len(result['documents'])} documents")


def test_bm25_retriever():
    """Test DXMTBM25Retriever component."""
    print("\n=== Test 8: BM25 Retriever Component ===")

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)
    retriever = DXMTBM25Retriever(
        document_store=store,
        top_k=10,
        scale_score=False,
    )

    assert retriever.document_store == store
    assert retriever.top_k == 10
    assert retriever.scale_score == False

    # Test run method
    result = retriever.run(query="test query")

    assert "documents" in result
    assert len(result["documents"]) == 1

    print(f"✓ BM25 retriever returned {len(result['documents'])} documents")


def test_serialization():
    """Test document store serialization."""
    print("\n=== Test 9: Serialization ===")

    client = create_mock_client()
    store = DXMTDocumentStore(
        client=client,
        table_name="test_table",
        embedding_dim=768,
        distance_metric="euclidean",
    )

    # Serialize
    store_dict = store.to_dict()
    assert store_dict["type"] == "dxmt.haystack.DXMTDocumentStore"
    assert store_dict["init_parameters"]["table_name"] == "test_table"
    assert store_dict["init_parameters"]["embedding_dim"] == 768
    assert store_dict["init_parameters"]["distance_metric"] == "euclidean"

    print("✓ Document store serialization working")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running DXMT Haystack Integration Tests")
    print("=" * 60)

    if not HAYSTACK_AVAILABLE:
        print("\nNOTE: haystack-ai not installed, using mock Document class")
        print("Install with: pip install haystack-ai\n")

    tests = [
        test_document_store_initialization,
        test_write_documents,
        test_count_documents,
        test_embedding_retrieval,
        test_bm25_retrieval,
        test_filter_conversion,
        test_embedding_retriever,
        test_bm25_retriever,
        test_serialization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✅ All tests passed!")
        print("\nDXMT Haystack Integration verified:")
        print("  ✓ Document store operations")
        print("  ✓ BM25 full-text retrieval")
        print("  ✓ Embedding vector search")
        print("  ✓ Metadata filtering")
        print("  ✓ Retriever components")
        print("  ✓ Serialization support")
        print("\nNext steps:")
        print("  1. Start DXMT server: cargo run -- serve --port 9999")
        print("  2. Install Haystack: pip install haystack-ai")
        print("  3. Run examples: python examples/dxmt_haystack_rag.py")
    else:
        print(f"\n❌ {failed} tests failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
