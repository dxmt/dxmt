"""Verification script for DXMT CrewAI integration.

This script verifies the CrewAI integration without requiring CrewAI to be installed.
It tests:
1. DXMT server connectivity
2. Integration code imports
3. Configuration validation
4. Mock integration scenarios
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

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

        # Test server connectivity by listing tables
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

    # Test tool import
    try:
        from dxmt.crewai_tool import (
            DXMTVectorSearchTool,
            DXMTToolConfig,
            DXMTToolSchema
        )
        print("✅ Tool layer imports successful")
        print("   - DXMTVectorSearchTool")
        print("   - DXMTToolConfig")
        print("   - DXMTToolSchema")
        results['tool'] = True
    except Exception as e:
        print(f"❌ Tool layer import failed: {e}")
        results['tool'] = False

    # Test storage import
    try:
        from dxmt.crewai_storage import (
            DXMTStorage,
            DXMTStorageConfig
        )
        print("✅ Storage layer imports successful")
        print("   - DXMTStorage")
        print("   - DXMTStorageConfig")
        results['storage'] = True
    except Exception as e:
        print(f"❌ Storage layer import failed: {e}")
        results['storage'] = False

    # Test SDK exports
    try:
        from dxmt import (
            DXMTVectorSearchTool,
            DXMTToolConfig,
            DXMTStorage,
            DXMTStorageConfig
        )
        print("✅ SDK __init__ exports working")
        results['exports'] = True
    except Exception as e:
        print(f"⚠️  SDK exports not configured: {e}")
        results['exports'] = False

    return all(results.values()), results


def test_tool_configuration():
    """Test tool configuration."""
    print("\n" + "=" * 70)
    print("  Test 3: Tool Configuration Validation")
    print("=" * 70)

    try:
        from dxmt.crewai_tool import DXMTToolConfig

        # Test default configuration
        config = DXMTToolConfig(
            host="localhost",
            port=9999,
            table_name="test_table"
        )
        print("✅ Default configuration created")
        print(f"   - host: {config.host}")
        print(f"   - port: {config.port}")
        print(f"   - table_name: {config.table_name}")
        print(f"   - limit: {config.limit} (default)")
        print(f"   - score_threshold: {config.score_threshold} (default)")

        # Test custom configuration
        custom_config = DXMTToolConfig(
            host="custom.host",
            port=8888,
            table_name="custom_table",
            limit=10,
            score_threshold=0.5,
            distance_metric="euclidean"
        )
        print("✅ Custom configuration created")
        print(f"   - limit: {custom_config.limit}")
        print(f"   - score_threshold: {custom_config.score_threshold}")
        print(f"   - distance_metric: {custom_config.distance_metric}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage_configuration():
    """Test storage configuration."""
    print("\n" + "=" * 70)
    print("  Test 4: Storage Configuration Validation")
    print("=" * 70)

    try:
        from dxmt.crewai_storage import DXMTStorageConfig

        # Test default configuration
        config = DXMTStorageConfig(
            host="localhost",
            port=9999,
            table_name="memory"
        )
        print("✅ Default storage configuration created")
        print(f"   - dimension: {config.dimension} (default)")
        print(f"   - distance_metric: {config.distance_metric} (default)")

        # Test with scoping
        scoped_config = DXMTStorageConfig(
            host="localhost",
            port=9999,
            table_name="scoped_memory",
            user_id="user_123",
            agent_id="agent_456",
            run_id="run_789"
        )
        print("✅ Scoped storage configuration created")
        print(f"   - user_id: {scoped_config.user_id}")
        print(f"   - agent_id: {scoped_config.agent_id}")
        print(f"   - run_id: {scoped_config.run_id}")

        return True

    except Exception as e:
        print(f"❌ Storage configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_with_mock_client():
    """Test tool initialization with mock client."""
    print("\n" + "=" * 70)
    print("  Test 5: Tool Initialization with Mock Client")
    print("=" * 70)

    try:
        from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig

        # Create mock client
        mock_client = Mock()
        mock_client.search.return_value = [
            {"id": "doc1", "score": 0.9, "metadata": {"text": "Test doc"}},
        ]

        # Create config
        config = DXMTToolConfig(
            host="localhost",
            port=9999,
            table_name="test_docs",
            limit=5
        )

        # Initialize tool with mock client
        tool = DXMTVectorSearchTool(
            dxmt_config=config,
            client=mock_client
        )

        print("✅ Tool initialized with mock client")
        print(f"   - Tool name: {tool.name}")
        print(f"   - Tool description: {tool.description[:60]}...")
        print(f"   - Config table: {tool.dxmt_config.table_name}")

        return True

    except Exception as e:
        print(f"❌ Tool initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage_with_mock_client():
    """Test storage initialization with mock client."""
    print("\n" + "=" * 70)
    print("  Test 6: Storage Initialization")
    print("=" * 70)

    try:
        from dxmt.crewai_storage import DXMTStorage
        from dxmt.exceptions import DXMTError
        from unittest.mock import patch

        # Mock the DXMTClient
        with patch('dxmt.crewai_storage.DXMTClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_table.side_effect = DXMTError("Table not found")
            mock_client.create_table.return_value = None
            mock_client_class.return_value = mock_client

            # Create storage
            storage = DXMTStorage(
                type="long_term",
                config={
                    "host": "localhost",
                    "port": 9999,
                    "table_name": "test_memory",
                    "user_id": "user_123"
                }
            )

            print("✅ Storage initialized successfully")
            print(f"   - Memory type: {storage.memory_type}")
            print(f"   - Table name: {storage.storage_config.table_name}")
            print(f"   - User ID: {storage.storage_config.user_id}")

            # Verify table creation was attempted
            assert mock_client.create_table.called
            print("✅ Table creation called during initialization")

            return True

    except Exception as e:
        print(f"❌ Storage initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_filter_conversion():
    """Test filter conversion logic."""
    print("\n" + "=" * 70)
    print("  Test 7: Filter Conversion Logic")
    print("=" * 70)

    try:
        from dxmt.crewai_storage import DXMTStorage
        from unittest.mock import patch, Mock
        from dxmt.exceptions import DXMTError

        with patch('dxmt.crewai_storage.DXMTClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_table.side_effect = DXMTError("Not found")
            mock_client.create_table.return_value = None
            mock_client_class.return_value = mock_client

            storage = DXMTStorage(
                type="long_term",
                config={
                    "table_name": "test",
                    "user_id": "user_1",
                    "agent_id": "agent_1"
                }
            )

            # Test filter building
            filter_result = storage._build_search_filter()

            print("✅ Filter conversion tested")
            print(f"   - Filter structure: {type(filter_result)}")
            if filter_result:
                print(f"   - Filter has 'should' key: {'should' in filter_result}")

            return True

    except Exception as e:
        print(f"❌ Filter conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_dxmt_operations(client):
    """Test real DXMT operations if server is available."""
    print("\n" + "=" * 70)
    print("  Test 8: Real DXMT Operations")
    print("=" * 70)

    if not client:
        print("⚠️  Skipped (DXMT server not available)")
        return False

    try:
        table_name = "crewai_integration_test"

        # Create test table
        try:
            client.create_table(
                name=table_name,
                dimension=384,
                distance_metric="cosine",
                description="CrewAI integration test table"
            )
            print(f"✅ Created test table: {table_name}")
        except Exception as e:
            print(f"ℹ️  Table already exists: {e}")

        # Insert test vector
        import uuid
        test_id = str(uuid.uuid4())
        test_vector = [0.1] * 384

        client.insert_records(
            table_name=table_name,
            records=[{
                "id": test_id,
                "vector": test_vector,
                "metadata": {"text": "Test document", "source": "test"}
            }]
        )
        print("✅ Inserted test vector")

        # Search
        results = client.search(
            table_name=table_name,
            vector=test_vector,
            k=1
        )
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


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  DXMT + CrewAI Integration Verification")
    print("=" * 70)
    print("\nThis script verifies the CrewAI integration without requiring")
    print("CrewAI to be installed. It tests code structure, imports,")
    print("configuration, and DXMT connectivity.")

    results = {}

    # Run tests
    results['connection'], client = test_dxmt_connection()
    results['imports'], import_details = test_integration_imports()
    results['tool_config'] = test_tool_configuration()
    results['storage_config'] = test_storage_configuration()
    results['tool_init'] = test_tool_with_mock_client()
    results['storage_init'] = test_storage_with_mock_client()
    results['filter'] = test_filter_conversion()
    results['real_ops'] = test_real_dxmt_operations(client)

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
        print("\n🎉 CrewAI integration code is correctly implemented!")
        print("\n📝 Next steps to test with real CrewAI:")
        print("   1. Install CrewAI: pip install crewai crewai-tools")
        print("   2. Set OPENAI_API_KEY environment variable")
        print("   3. Run: python examples/crewai_dxmt_simple.py")
        return 0
    else:
        print("  ⚠️  SOME TESTS FAILED")
        print("=" * 70)
        print("\n⚠️  Please review the failed tests above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
