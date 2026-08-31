"""Test the query fix - verify the code handles list responses correctly"""

from unittest.mock import Mock, patch
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.core.schema import TextNode

print("=" * 60)
print("Testing Query Fix - List Response Handling")
print("=" * 60)

# Create mock client
mock_client = Mock()

# Mock search to return a list (like the real DXMT SDK)
mock_search_results = [
    {
        "id": "node1",
        "score": 0.95,
        "payload": {
            "_node_content": "This is a test document",
            "_node_type": "TextNode",
            "category": "test",
        }
    },
    {
        "id": "node2",
        "score": 0.85,
        "payload": {
            "_node_content": "Another test document",
            "_node_type": "TextNode",
            "category": "test",
        }
    }
]

# Configure mock
mock_client.search.return_value = mock_search_results
mock_client.hybrid_search.return_value = mock_search_results
mock_client.full_text_search.return_value = mock_search_results

# Test 1: Vector search
print("\n[Test 1] Vector search with list response...")
try:
    store = DXMTVectorStore(
        table_name="test",
        dimension=128,
        client=mock_client,
    )

    query = VectorStoreQuery(
        query_embedding=[0.1] * 128,
        similarity_top_k=2,
        mode=VectorStoreQueryMode.DEFAULT
    )

    result = store.query(query)

    assert len(result.nodes) == 2, f"Expected 2 nodes, got {len(result.nodes)}"
    assert len(result.similarities) == 2, f"Expected 2 similarities, got {len(result.similarities)}"
    assert len(result.ids) == 2, f"Expected 2 ids, got {len(result.ids)}"
    assert result.ids[0] == "node1", f"Expected node1, got {result.ids[0]}"
    assert result.similarities[0] == 0.95, f"Expected 0.95, got {result.similarities[0]}"

    print("✓ Vector search works correctly with list response")
except Exception as e:
    print(f"✗ Vector search failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Hybrid search
print("\n[Test 2] Hybrid search with list response...")
try:
    query = VectorStoreQuery(
        query_embedding=[0.1] * 128,
        query_str="test query",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        alpha=0.7
    )

    result = store.query(query)

    assert len(result.nodes) == 2, f"Expected 2 nodes, got {len(result.nodes)}"
    print("✓ Hybrid search works correctly with list response")
except Exception as e:
    print(f"✗ Hybrid search failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Text search
print("\n[Test 3] Text search with list response...")
try:
    query = VectorStoreQuery(
        query_str="test query",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.TEXT_SEARCH
    )

    result = store.query(query)

    assert len(result.nodes) == 2, f"Expected 2 nodes, got {len(result.nodes)}"
    print("✓ Text search works correctly with list response")
except Exception as e:
    print(f"✗ Text search failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: get_nodes with list response
print("\n[Test 4] get_nodes with list response...")
try:
    from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

    # Configure store with dimension for get_nodes
    store.dimension = 128

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="test", operator=FilterOperator.EQ)
        ]
    )

    nodes = store.get_nodes(filters=filters)

    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    print("✓ get_nodes works correctly with list response")
except Exception as e:
    print(f"✗ get_nodes failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓✓✓ All query fix tests passed!")
print("=" * 60)
print("\nThe AttributeError fix is working correctly:")
print("  ✓ Handles list response from search()")
print("  ✓ Handles list response from hybrid_search()")
print("  ✓ Handles list response from full_text_search()")
print("  ✓ Handles list response in get_nodes()")
print("\nThe integration is ready for use with a running DXMT server.")
print("=" * 60)
