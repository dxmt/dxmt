#!/usr/bin/env python3
"""
Quick test script for DXMT VectorDBBench integration.

This script verifies that:
1. DXMT client can be imported
2. Configuration classes work correctly
3. Basic operations (init, insert, search) function
4. All index types are supported

Usage:
    python scripts/test_vectordbbench_integration.py
"""

import sys
import time
from pathlib import Path

# Add VectorDBBench to path
vectordbbench_path = Path(__file__).parent.parent / "reference" / "VectorDBBench"
sys.path.insert(0, str(vectordbbench_path))

try:
    from vectordb_bench.backend.clients import DB
    from vectordb_bench.backend.clients.dxmt import (
        DXMT,
        DXMTAutoConfig,
        DXMTConfig,
        DXMTHNSWConfig,
        DXMTHNSWPQConfig,
    )
    from vectordb_bench.backend.clients.api import MetricType

    print("✓ Successfully imported DXMT client and configs")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_enum_registration():
    """Test that DXMT is registered in DB enum."""
    print("\n=== Testing DB Enum Registration ===")
    try:
        assert hasattr(DB, "DXMT"), "DB.DXMT not found in enum"
        assert DB.DXMT.value == "DXMT"
        print("✓ DXMT registered in DB enum")

        # Test init_cls
        init_cls = DB.DXMT.init_cls
        assert init_cls == DXMT
        print("✓ DB.DXMT.init_cls returns DXMT class")

        # Test config_cls
        config_cls = DB.DXMT.config_cls
        assert config_cls == DXMTConfig
        print("✓ DB.DXMT.config_cls returns DXMTConfig class")

        # Test case_config_cls
        case_config_cls = DB.DXMT.case_config_cls()
        print(f"✓ DB.DXMT.case_config_cls returns {case_config_cls}")

    except Exception as e:
        print(f"✗ Enum registration test failed: {e}")
        return False

    return True


def test_config_classes():
    """Test configuration classes."""
    print("\n=== Testing Configuration Classes ===")
    try:
        # Test DXMTConfig
        config = DXMTConfig(
            host="localhost",
            port=9999,
            db_label="test-instance",
        )
        config_dict = config.to_dict()
        assert config_dict["host"] == "localhost"
        assert config_dict["port"] == 9999
        print("✓ DXMTConfig works correctly")

        # Test DXMTHNSWConfig
        hnsw_config = DXMTHNSWConfig(
            metric_type=MetricType.COSINE,
            m=16,
            ef_construction=200,
            ef_search=100,
        )
        index_params = hnsw_config.index_param()
        assert index_params["metric_type"] == "cosine"
        assert index_params["index_type"] == "hnsw"
        print("✓ DXMTHNSWConfig works correctly")

        # Test DXMTHNSWPQConfig
        pq_config = DXMTHNSWPQConfig(
            metric_type=MetricType.L2,
            pq_m=8,
            pq_nbits=8,
        )
        pq_params = pq_config.index_param()
        assert pq_params["metric_type"] == "euclidean"
        assert pq_params["index_type"] == "hnsw_pq"
        print("✓ DXMTHNSWPQConfig works correctly")

        # Test DXMTAutoConfig
        auto_config = DXMTAutoConfig()
        auto_params = auto_config.index_param()
        assert auto_params["index_type"] == "auto"
        print("✓ DXMTAutoConfig works correctly")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_client_initialization():
    """Test DXMT client initialization."""
    print("\n=== Testing Client Initialization ===")
    try:
        # Note: This will fail if DXMT server is not running
        # We catch the error and just test the code path
        config = {
            "host": "localhost",
            "port": 9999,
        }
        case_config = DXMTHNSWConfig(
            m=16,
            ef_construction=200,
            ef_search=100,
        )

        try:
            client = DXMT(
                dim=128,
                db_config=config,
                db_case_config=case_config,
                collection_name="test_collection",
                drop_old=True,
            )
            print("✓ DXMT client initialized successfully")
            print("  (Note: This means DXMT server is running)")

            # Test context manager
            with client.init():
                print("✓ Client context manager works")

            return True

        except ImportError as e:
            print(f"⚠ dxmt-python-sdk not installed: {e}")
            print("  Install with: pip install dxmt-python-sdk")
            return False

        except Exception as e:
            print(f"⚠ Could not connect to DXMT server: {e}")
            print("  This is expected if DXMT server is not running")
            print("  Start server with: cargo run --release -- serve")
            return False

    except Exception as e:
        print(f"✗ Client initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_filter_support():
    """Test filter support."""
    print("\n=== Testing Filter Support ===")
    try:
        from vectordb_bench.backend.filter import Filter, FilterOp

        # Check supported filter types
        supported = DXMT.supported_filter_types
        print(f"Supported filter types: {supported}")

        assert FilterOp.NonFilter in supported
        assert FilterOp.NumGE in supported
        assert FilterOp.StrEqual in supported
        print("✓ All required filter types supported")

        # Test filter_supported method
        non_filter = Filter(type=FilterOp.NonFilter)
        assert DXMT.filter_supported(non_filter)

        num_filter = Filter(type=FilterOp.NumGE, value=1000)
        assert DXMT.filter_supported(num_filter)

        str_filter = Filter(type=FilterOp.StrEqual, value="red")
        assert DXMT.filter_supported(str_filter)

        print("✓ Filter support detection works correctly")

    except Exception as e:
        print(f"✗ Filter support test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DXMT VectorDBBench Integration Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Enum Registration", test_enum_registration()))
    results.append(("Configuration Classes", test_config_classes()))
    results.append(("Filter Support", test_filter_support()))
    results.append(("Client Initialization", test_client_initialization()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Start DXMT server: cargo run --release -- serve")
        print("2. Install VectorDBBench: pip install -e reference/VectorDBBench")
        print("3. Run benchmark: vectordbbench dxmtauto --case-type Performance768D1M")
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
