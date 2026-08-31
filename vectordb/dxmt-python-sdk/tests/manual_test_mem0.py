"""
Manual test for DXMT mem0 adapter.

Tests basic functionality without requiring pytest.
Run with: python3 tests/manual_test_mem0.py
"""

import sys
from unittest.mock import Mock

# Import the adapter
try:
    from dxmt.mem0 import DXMTMemoryStore
    from dxmt.exceptions import DXMTError
    print("✓ Successfully imported DXMTMemoryStore")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


def create_mock_client():
    """Create a mock DXMT client for testing."""
    client = Mock()

    # Mock table operations
    client.get_table.side_effect = DXMTError("Table not found")
    client.create_table.return_value = {"name": "test", "dimension": 384}
    client.list_tables.return_value = [{"name": "test"}]
    client.delete_table.return_value = None

    # Mock record operations
    client.insert_records.return_value = None
    client.search.return_value = []
    client.get_records.return_value = []
    client.update_record.return_value = None
    client.delete_records.return_value = None

    return client


def test_initialization():
    """Test memory store initialization."""
    print("\n=== Test 1: Initialization ===")

    client = create_mock_client()
    store = DXMTMemoryStore(
        collection_name="test_collection",
        client=client,
        embedding_model_dims=384,
        distance_metric="cosine"
    )

    assert store.collection_name == "test_collection"
    assert store.embedding_model_dims == 384
    assert store.distance_metric == "cosine"

    # Verify collection creation was attempted
    assert client.create_table.called
    print("✓ Initialization test passed")


def test_insert():
    """Test insert operation."""
    print("\n=== Test 2: Insert ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    vectors = [[0.1] * 384, [0.2] * 384]
    payloads = [
        {"data": "memory 1", "user_id": "user1"},
        {"data": "memory 2", "user_id": "user2"}
    ]
    ids = ["mem1", "mem2"]

    store.insert(vectors=vectors, payloads=payloads, ids=ids)

    # Verify insert was called
    assert client.insert_records.called
    call_args = client.insert_records.call_args
    records = call_args[0][1]

    assert len(records) == 2
    assert records[0]["id"] == "mem1"
    assert records[0]["payload"]["data"] == "memory 1"
    assert "created_at" in records[0]["payload"]
    assert "updated_at" in records[0]["payload"]

    print("✓ Insert test passed")


def test_search():
    """Test search operation."""
    print("\n=== Test 3: Search ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Mock search results
    client.search.return_value = [
        {"id": "mem1", "score": 0.95, "metadata": {"data": "memory 1"}},
        {"id": "mem2", "score": 0.85, "metadata": {"data": "memory 2"}},
    ]

    query_vector = [[0.1] * 384]
    results = store.search(query="test", vectors=query_vector, limit=5)

    assert len(results) == 2
    assert results[0]["id"] == "mem1"
    assert results[0]["payload"]["data"] == "memory 1"
    assert 0 <= results[0]["score"] <= 1  # Should be normalized

    # Verify search was called with correct parameters
    assert client.search.called
    call_args = client.search.call_args
    assert call_args[1]["table_name"] == "test"
    assert call_args[1]["k"] == 5

    print("✓ Search test passed")


def test_filter_conversion():
    """Test filter conversion."""
    print("\n=== Test 4: Filter Conversion ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test simple equality filter
    mem0_filters = {"user_id": "user1", "agent_id": "agent1"}
    dxmt_filter = store._convert_filters(mem0_filters)

    assert dxmt_filter is not None
    assert "must" in dxmt_filter
    assert len(dxmt_filter["must"]) == 2

    # Verify filter structure
    user_filter = next(f for f in dxmt_filter["must"] if f["key"] == "user_id")
    assert user_filter["operation"]["type"] == "Equal"
    assert user_filter["operation"]["value"] == "user1"

    print("✓ Filter conversion test passed")


def test_filter_with_operators():
    """Test filter conversion with operators."""
    print("\n=== Test 5: Filter with Operators ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test range filter
    mem0_filters = {
        "created_at": {
            "gte": "2024-01-01",
            "lte": "2024-12-31"
        }
    }
    dxmt_filter = store._convert_filters(mem0_filters)

    assert len(dxmt_filter["must"]) == 2
    operations = [c["operation"]["type"] for c in dxmt_filter["must"]]
    assert "GreaterThanOrEqual" in operations
    assert "LessThanOrEqual" in operations

    print("✓ Filter with operators test passed")


def test_score_normalization():
    """Test score normalization."""
    print("\n=== Test 6: Score Normalization ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test cosine similarity normalization (DXMT returns similarity, not distance)
    store.distance_metric = "cosine"
    assert store._normalize_score(1.0) == 1.0   # Perfect match
    assert store._normalize_score(0.9) == 0.9   # High similarity — must not be compressed
    assert store._normalize_score(0.0) == 0.0   # No similarity

    # Test euclidean distance normalization
    store.distance_metric = "euclidean"
    import math
    assert store._normalize_score(0.0) == 1.0
    assert abs(store._normalize_score(1.0) - math.exp(-1)) < 0.01

    print("✓ Score normalization test passed")


def test_update():
    """Test update operation."""
    print("\n=== Test 7: Update ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test update payload
    new_payload = {"data": "updated memory"}
    store.update("mem1", vector=None, payload=new_payload)

    assert client.update_record.called
    call_args = client.update_record.call_args
    assert call_args[1]["record_id"] == "mem1"
    assert "updated_at" in call_args[1]["payload"]

    print("✓ Update test passed")


def test_delete():
    """Test delete operation."""
    print("\n=== Test 8: Delete ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test delete single record
    store.delete("mem1")
    assert client.delete_records.called
    call_args = client.delete_records.call_args
    assert "mem1" in call_args[0][1]

    print("✓ Delete test passed")


def test_get():
    """Test get operation."""
    print("\n=== Test 9: Get ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test get existing record
    client.get_records.return_value = [{
        "id": "mem1",
        "metadata": {"data": "memory 1", "user_id": "user1"}
    }]

    result = store.get("mem1")
    assert result is not None
    assert result["id"] == "mem1"
    assert result["payload"]["data"] == "memory 1"
    assert result["score"] is None  # No score for direct get

    # Test get non-existent record
    client.get_records.return_value = []
    result = store.get("nonexistent")
    assert result is None

    print("✓ Get test passed")


def test_list():
    """Test list operation."""
    print("\n=== Test 10: List ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Mock list results
    client.search.return_value = [
        {"id": "mem1", "score": 0.9, "metadata": {"data": "memory 1"}},
        {"id": "mem2", "score": 0.8, "metadata": {"data": "memory 2"}},
    ]

    results = store.list(filters=None, limit=100)

    assert len(results) == 2
    assert results[0]["id"] == "mem1"
    assert results[0]["score"] is None  # No scores for listing

    print("✓ List test passed")


def test_collection_operations():
    """Test collection operations."""
    print("\n=== Test 11: Collection Operations ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test list collections
    client.list_tables.return_value = [
        {"name": "col1"},
        {"name": "col2"},
        {"name": "test"}
    ]
    collections = store.list_cols()
    assert len(collections) == 3
    assert "test" in collections

    # Test collection info
    # Clear the side_effect first, then set return value
    client.get_table.side_effect = None
    client.get_table.return_value = {
        "name": "test",
        "dimension": 384,
        "record_count": 100
    }
    info = store.col_info()
    assert info["name"] == "test"
    assert info["dimension"] == 384

    print("✓ Collection operations test passed")


def test_dimension_validation():
    """Test dimension validation."""
    print("\n=== Test 12: Dimension Validation ===")

    client = create_mock_client()
    store = DXMTMemoryStore("test", client, 384)

    # Test insert with wrong dimension
    try:
        vectors = [[0.1, 0.2, 0.3]]  # Only 3 dims instead of 384
        store.insert(vectors=vectors, payloads=[{"data": "test"}], ids=["mem1"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dimension mismatch" in str(e)

    # Test search with wrong dimension
    try:
        query_vector = [[0.1, 0.2, 0.3]]
        store.search(query="", vectors=query_vector, limit=5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dimension mismatch" in str(e)

    print("✓ Dimension validation test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running DXMT mem0 Adapter Tests")
    print("="*60)

    tests = [
        test_initialization,
        test_insert,
        test_search,
        test_filter_conversion,
        test_filter_with_operators,
        test_score_normalization,
        test_update,
        test_delete,
        test_get,
        test_list,
        test_collection_operations,
        test_dimension_validation,
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
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
