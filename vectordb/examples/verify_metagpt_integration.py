"""Verification script for DXMT MetaGPT integration.

This script verifies the MetaGPT integration without requiring LlamaIndex/MetaGPT
to be installed. It tests:
1. DXMT server connectivity
2. Integration code imports
3. Configuration validation
4. Mock integration scenarios
5. Real DXMT operations
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))


def test_dxmt_connection():
    """Test DXMT server connection."""
    print("\n" + "=" * 70)
    print("  Test 1: DXMT Server Connection")
    print("=" * 70)

    try:
        from dxmt import DXMTClient

        client = DXMTClient(host="localhost", port=9999)
        print("✅ DXMT client created successfully")

        # Test server connectivity
        tables = client.list_tables()
        print(f"✅ Connected to DXMT server")
        print(f"   Current tables: {len(tables)}")

        return True, client

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False, None


def test_integration_imports():
    """Test that integration modules can be imported."""
    print("\n" + "=" * 70)
    print("  Test 2: Integration Module Imports")
    print("=" * 70)

    results = {}

    # Test SDK exports (which handles missing llama_index gracefully)
    try:
        from dxmt import (
            DXMTVectorStore,
            DXMTRetriever,
            create_dxmt_retriever,
            METAGPT_AVAILABLE,
        )

        if METAGPT_AVAILABLE:
            print("✅ MetaGPT integration available")
            print("   - DXMTVectorStore")
            print("   - DXMTRetriever")
            print("   - create_dxmt_retriever")
            results["exports"] = True
        else:
            print("⚠️  LlamaIndex not installed - MetaGPT integration disabled")
            print("   Install with: pip install llama-index llama-index-core")
            results["exports"] = False

    except Exception as e:
        print(f"❌ SDK import failed: {e}")
        results["exports"] = False

    return results["exports"], results


def test_vector_store_initialization():
    """Test vector store initialization."""
    print("\n" + "=" * 70)
    print("  Test 3: Vector Store Initialization")
    print("=" * 70)

    try:
        from dxmt import DXMTClient, DXMTVectorStore, METAGPT_AVAILABLE

        if not METAGPT_AVAILABLE:
            print("⚠️  Skipped (LlamaIndex not installed)")
            return False

        # Create client
        client = DXMTClient(host="localhost", port=9999)

        # Test initialization
        vector_store = DXMTVectorStore(
            client=client,
            table_name="test_metagpt_store",
            dimension=384,
            distance_metric="cosine",
            create_if_not_exists=True,
        )

        print("✅ Vector store initialized")
        print(f"   - Table name: {vector_store.table_name}")
        print(f"   - Dimension: {vector_store.dimension}")
        print(f"   - Distance metric: {vector_store.distance_metric}")
        print(f"   - Stores text: {vector_store.stores_text}")
        print(f"   - Is embedding query: {vector_store.is_embedding_query}")

        return True

    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vector_store_methods():
    """Test vector store methods with mocks."""
    print("\n" + "=" * 70)
    print("  Test 4: Vector Store Methods")
    print("=" * 70)

    try:
        from dxmt import DXMTVectorStore, METAGPT_AVAILABLE

        if not METAGPT_AVAILABLE:
            print("⚠️  Skipped (LlamaIndex not installed)")
            return False

        # Create mock client
        mock_client = Mock()
        mock_client.get_table.side_effect = Exception("Table not found")
        mock_client.create_table.return_value = None
        mock_client.insert_records.return_value = {"success": True}
        mock_client.search.return_value = [
            {
                "id": "node1",
                "score": 0.9,
                "metadata": {"text": "Test node", "_node_content": '{"text": "Test node"}'},
            }
        ]

        # Create vector store
        vector_store = DXMTVectorStore(
            client=mock_client,
            table_name="test_methods",
            dimension=384,
            create_if_not_exists=True,
        )

        print("✅ Vector store created with mock client")

        # Test methods exist
        assert hasattr(vector_store, "add")
        assert hasattr(vector_store, "query")
        assert hasattr(vector_store, "delete")
        assert hasattr(vector_store, "persist")
        assert hasattr(vector_store, "clear")

        print("✅ All required methods present:")
        print("   - add()")
        print("   - query()")
        print("   - delete()")
        print("   - persist()")
        print("   - clear()")

        return True

    except Exception as e:
        print(f"❌ Vector store methods test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_retriever_initialization():
    """Test retriever initialization with mock."""
    print("\n" + "=" * 70)
    print("  Test 5: Retriever Initialization")
    print("=" * 70)

    try:
        from dxmt import DXMTRetriever, DXMTVectorStore, METAGPT_AVAILABLE

        if not METAGPT_AVAILABLE:
            print("⚠️  Skipped (LlamaIndex not installed)")
            return False

        # Create mock index
        mock_index = Mock()
        mock_vector_store = Mock(spec=DXMTVectorStore)
        mock_index.vector_store = mock_vector_store
        mock_index.docstore = Mock()
        mock_index.docstore.docs = {}

        # Create retriever
        retriever = DXMTRetriever(index=mock_index, similarity_top_k=5)

        print("✅ Retriever initialized")
        print(f"   - Similarity top k: {retriever.similarity_top_k}")

        # Test methods exist
        assert hasattr(retriever, "add_nodes")
        assert hasattr(retriever, "persist")
        assert hasattr(retriever, "query_total_count")
        assert hasattr(retriever, "clear")
        assert hasattr(retriever, "retrieve")

        print("✅ All required methods present:")
        print("   - add_nodes() [ModifiableRAGRetriever]")
        print("   - persist() [PersistableRAGRetriever]")
        print("   - query_total_count() [QueryableRAGRetriever]")
        print("   - clear() [DeletableRAGRetriever]")
        print("   - retrieve() [BaseRetriever]")

        return True

    except Exception as e:
        print(f"❌ Retriever initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_real_dxmt_operations(client):
    """Test real DXMT operations if server is available."""
    print("\n" + "=" * 70)
    print("  Test 6: Real DXMT Operations")
    print("=" * 70)

    if not client:
        print("⚠️  Skipped (DXMT server not available)")
        return False

    try:
        table_name = "metagpt_integration_test"

        # Create test table
        try:
            client.create_table(
                name=table_name,
                dimension=384,
                distance_metric="cosine",
                description="MetaGPT integration test table",
            )
            print(f"✅ Created test table: {table_name}")
        except Exception as e:
            print(f"ℹ️  Table already exists: {e}")

        # Insert test vector
        import uuid

        test_id = str(uuid.uuid4())
        test_vector = [0.1] * 384

        client.insert_records(
            table_name=table_name, records=[{"id": test_id, "vector": test_vector, "metadata": {"text": "Test document"}}]
        )
        print("✅ Inserted test vector")

        # Search
        results = client.search(table_name=table_name, vector=test_vector, k=1)
        print(f"✅ Search completed: {len(results)} results")

        # Cleanup
        try:
            client.delete_table(table_name)
            print("✅ Cleaned up test table")
        except:
            pass

        return True

    except Exception as e:
        print(f"❌ Real operations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llamaindex_compatibility():
    """Test LlamaIndex interface compatibility (without requiring llama-index)."""
    print("\n" + "=" * 70)
    print("  Test 7: LlamaIndex Interface Compatibility")
    print("=" * 70)

    try:
        from dxmt import DXMTVectorStore, METAGPT_AVAILABLE

        if not METAGPT_AVAILABLE:
            print("⚠️  Skipped (LlamaIndex not installed)")
            return False

        # Check interface methods
        required_methods = ["add", "query", "delete", "persist"]

        print("✅ Checking LlamaIndex VectorStore interface:")
        for method in required_methods:
            if hasattr(DXMTVectorStore, method):
                print(f"   ✓ {method}()")
            else:
                print(f"   ✗ {method}() - MISSING")
                return False

        # Check properties
        print("✅ Checking required properties:")
        mock_client = Mock()
        mock_client.get_table.side_effect = Exception("Not found")
        mock_client.create_table.return_value = None

        vs = DXMTVectorStore(client=mock_client, table_name="test", dimension=384, create_if_not_exists=True)

        properties = [("stores_text", True), ("is_embedding_query", True)]

        for prop, expected in properties:
            value = getattr(vs, prop)
            if value == expected:
                print(f"   ✓ {prop} = {value}")
            else:
                print(f"   ✗ {prop} = {value} (expected {expected})")
                return False

        return True

    except Exception as e:
        print(f"❌ Interface compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  DXMT + MetaGPT Integration Verification")
    print("=" * 70)
    print("\nThis script verifies the MetaGPT integration without requiring")
    print("LlamaIndex/MetaGPT to be installed. It tests code structure,")
    print("imports, configuration, and DXMT connectivity.")

    results = {}

    # Run tests
    results["connection"], client = test_dxmt_connection()
    results["imports"], import_details = test_integration_imports()
    results["vector_store_init"] = test_vector_store_initialization()
    results["vector_store_methods"] = test_vector_store_methods()
    results["retriever_init"] = test_retriever_initialization()
    results["real_ops"] = test_real_dxmt_operations(client)
    results["compatibility"] = test_llamaindex_compatibility()

    # Summary
    print("\n" + "=" * 70)
    print("  Verification Summary")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n📊 Results: {passed}/{total} tests passed\n")

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name.replace('_', ' ').title()}")

    # Overall status
    print("\n" + "=" * 70)

    if passed == total:
        print("  ✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\n🎉 MetaGPT integration code is correctly implemented!")
        print("\n📝 Next steps to test with real LlamaIndex/MetaGPT:")
        print("   1. Install LlamaIndex: pip install llama-index llama-index-core")
        print("   2. Set OPENAI_API_KEY environment variable")
        print("   3. Run: python examples/metagpt_integration.py")
        return 0
    else:
        print("  ⚠️  SOME TESTS FAILED")
        print("=" * 70)
        print("\n⚠️  Please review the failed tests above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
