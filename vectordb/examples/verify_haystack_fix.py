#!/usr/bin/env python3
"""
Quick verification script to test that the optional fields bug fix works.

This script tests that DXMT Haystack integration handles Haystack Documents
correctly whether or not they have the optional 'dataframe' and 'blob' fields.

Run with: python3 examples/verify_haystack_fix.py
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

print("=" * 70)
print("DXMT Haystack Integration - Optional Fields Bug Fix Verification")
print("=" * 70)

# Check if real Haystack is installed
try:
    from haystack.dataclasses import Document
    REAL_HAYSTACK = True
    print("\n✓ Real haystack-ai package detected")
except ImportError:
    from dxmt.haystack import Document
    REAL_HAYSTACK = False
    print("\n⚠ Using stub Document class (haystack-ai not installed)")

from dxmt.haystack import DXMTDocumentStore
from dxmt.exceptions import DXMTError


def create_mock_client():
    """Create a mock DXMT client for testing."""
    client = Mock()
    client.get_table.side_effect = DXMTError("Table not found")
    client.create_table.return_value = {"name": "test", "dimension": 384}
    client.insert_records.return_value = None
    client.search.return_value = [
        {
            "id": "doc1",
            "score": 0.95,
            "vector": [0.1] * 384,
            "metadata": {"content": "Test document 1", "type": "test"},
        }
    ]
    return client


def test_document_without_optional_fields():
    """Test Document without dataframe or blob fields."""
    print("\n" + "=" * 70)
    print("Test 1: Document without optional fields (dataframe, blob)")
    print("=" * 70)

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)

    # Create document WITHOUT dataframe or blob
    doc = Document(
        id="test1",
        content="This is a test document",
        meta={"category": "test"},
        embedding=[0.1] * 384,
    )

    print(f"\nCreated Document:")
    print(f"  - id: {doc.id}")
    print(f"  - content: {doc.content[:50]}...")
    print(f"  - has dataframe: {hasattr(doc, 'dataframe') and doc.dataframe is not None}")
    print(f"  - has blob: {hasattr(doc, 'blob') and doc.blob is not None}")

    try:
        # Convert to DXMT format - this should NOT raise AttributeError
        record = store._haystack_to_dxmt_record(doc)
        print(f"\n✓ Successfully converted to DXMT record")
        print(f"  - record id: {record['id']}")
        print(f"  - has vector: {'vector' in record}")
        print(f"  - payload keys: {list(record['payload'].keys())}")

        # Verify optional fields are NOT in payload
        assert "dataframe" not in record["payload"], "dataframe should not be in payload"
        assert "blob" not in record["payload"], "blob should not be in payload"
        print(f"\n✓ Optional fields correctly omitted from payload")

        return True
    except AttributeError as e:
        print(f"\n✗ FAILED: AttributeError raised: {e}")
        print("  The bug fix did not work correctly!")
        return False
    except Exception as e:
        print(f"\n✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_with_optional_fields():
    """Test Document with dataframe and blob fields."""
    print("\n" + "=" * 70)
    print("Test 2: Document with optional fields (dataframe, blob)")
    print("=" * 70)

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)

    # Create document WITH dataframe and blob (if supported)
    doc_kwargs = {
        "id": "test2",
        "content": "Document with optional fields",
        "meta": {"category": "test"},
        "embedding": [0.2] * 384,
    }

    # Try adding optional fields if Document class supports them
    if REAL_HAYSTACK:
        try:
            import pandas as pd
            from haystack.dataclasses import ByteStream
            doc_kwargs["dataframe"] = pd.DataFrame({"col": [1, 2, 3]})
            doc_kwargs["blob"] = ByteStream(b"test data")
        except:
            pass  # Skip if not available

    doc = Document(**doc_kwargs)

    print(f"\nCreated Document:")
    print(f"  - id: {doc.id}")
    print(f"  - content: {doc.content[:50]}...")
    print(f"  - has dataframe: {hasattr(doc, 'dataframe') and doc.dataframe is not None}")
    print(f"  - has blob: {hasattr(doc, 'blob') and doc.blob is not None}")

    try:
        # Convert to DXMT format
        record = store._haystack_to_dxmt_record(doc)
        print(f"\n✓ Successfully converted to DXMT record")
        print(f"  - record id: {record['id']}")
        print(f"  - has vector: {'vector' in record}")
        print(f"  - payload keys: {list(record['payload'].keys())}")

        # Check if optional fields were included (if they existed)
        if hasattr(doc, 'dataframe') and doc.dataframe is not None:
            assert "dataframe" in record["payload"], "dataframe should be in payload"
            print(f"  - dataframe included: ✓")
        if hasattr(doc, 'blob') and doc.blob is not None:
            assert "blob" in record["payload"], "blob should be in payload"
            print(f"  - blob included: ✓")

        return True
    except AttributeError as e:
        print(f"\n✗ FAILED: AttributeError raised: {e}")
        return False
    except Exception as e:
        print(f"\n✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reverse_conversion():
    """Test converting DXMT result back to Haystack Document."""
    print("\n" + "=" * 70)
    print("Test 3: DXMT result to Haystack Document conversion")
    print("=" * 70)

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)

    # Simulate DXMT search result
    dxmt_result = {
        "id": "doc1",
        "score": 0.95,
        "vector": [0.1] * 384,
        "metadata": {
            "content": "Test document",
            "category": "test",
        }
    }

    print(f"\nDXMT Result:")
    print(f"  - id: {dxmt_result['id']}")
    print(f"  - score: {dxmt_result['score']}")
    print(f"  - metadata keys: {list(dxmt_result['metadata'].keys())}")

    try:
        # Convert to Haystack Document
        doc = store._dxmt_to_haystack_doc(dxmt_result)
        print(f"\n✓ Successfully converted to Haystack Document")
        print(f"  - id: {doc.id}")
        print(f"  - content: {doc.content}")
        print(f"  - has embedding: {doc.embedding is not None}")
        print(f"  - meta: {doc.meta}")

        # Verify Document was created successfully
        assert doc.id == "doc1"
        assert doc.content == "Test document"
        assert doc.embedding is not None

        return True
    except Exception as e:
        print(f"\n✗ FAILED: Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_write_documents():
    """Test writing documents (the operation that originally failed)."""
    print("\n" + "=" * 70)
    print("Test 4: Write documents operation (original failure scenario)")
    print("=" * 70)

    client = create_mock_client()
    store = DXMTDocumentStore(client=client, embedding_dim=384)

    from dxmt.haystack import DuplicatePolicy

    # Create multiple documents
    documents = [
        Document(
            id=f"doc{i}",
            content=f"Test document {i}",
            meta={"index": i},
            embedding=[0.1 * i] * 384,
        )
        for i in range(1, 4)
    ]

    print(f"\nCreated {len(documents)} documents")
    for doc in documents:
        print(f"  - {doc.id}: {doc.content}")

    try:
        # This is the operation that originally failed with AttributeError
        written = store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
        print(f"\n✓ Successfully wrote {written} documents")
        print(f"  - No AttributeError raised!")

        return True
    except AttributeError as e:
        print(f"\n✗ FAILED: AttributeError raised: {e}")
        print("  The bug is still present!")
        return False
    except Exception as e:
        print(f"\n✗ FAILED: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    tests = [
        test_document_without_optional_fields,
        test_document_with_optional_fields,
        test_reverse_conversion,
        test_write_documents,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("\n" + "🎉" * 20)
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe bug fix is working correctly!")
        print("The AttributeError issue with optional fields is resolved.")
        print("\nNext steps:")
        print("  1. Start DXMT server: cargo run -- serve --port 9999")
        print("  2. Run full examples: python3 examples/dxmt_haystack_rag.py")
        print("\n" + "🎉" * 20)
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("\nThe bug fix needs more work.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
