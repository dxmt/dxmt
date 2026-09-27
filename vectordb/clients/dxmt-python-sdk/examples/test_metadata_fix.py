"""
Test script to verify the metadata/payload field mapping fix.

This script demonstrates that the Python SDK now correctly handles both
'metadata' and 'payload' fields, transparently mapping between them.
"""

import sys
from pathlib import Path

# The package root, relative to this file. It was an absolute path on one developer's machine
# (`/localdisk/code/dxmt/...`), so this example ran nowhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dxmt import Client
import json


def test_metadata_field():
    """Test that 'metadata' field works correctly."""
    print("=" * 80)
    print("Test 1: Using 'metadata' field (user-friendly)")
    print("=" * 80)

    client = Client(host="localhost", port=9999)

    # Create test table
    try:
        client.delete_table("test_metadata_field")
    except:
        pass

    client.create_table(
        name="test_metadata_field",
        dimension=3,
        distance_metric="cosine"
    )

    # Insert with 'metadata' field
    records = [{
        "id": "doc1",
        "vector": [0.1, 0.2, 0.3],
        "metadata": {  # ← Using 'metadata'
            "text": "Test document content",
            "source": "test.txt",
            "author": "Test Author"
        }
    }]

    result = client.insert_records("test_metadata_field", records)
    print(f"✓ Insert succeeded: {result['success']}")

    # Search
    results = client.search("test_metadata_field", [0.1, 0.2, 0.3], k=1)

    if results:
        hit = results[0]
        metadata = hit.get('metadata', {})

        print(f"✓ Search returned {len(results)} result(s)")
        print(f"  - Score: {hit.get('score', 0):.3f}")
        print(f"  - Has 'metadata' field: {'metadata' in hit}")
        print(f"  - Has 'payload' field: {'payload' in hit}")
        print(f"  - Metadata content: {json.dumps(metadata, indent=4)}")

        # Verify all fields are present
        assert metadata.get('text') == "Test document content", "Text field missing!"
        assert metadata.get('source') == "test.txt", "Source field missing!"
        assert metadata.get('author') == "Test Author", "Author field missing!"

        print("✓ All metadata fields correctly retrieved")
    else:
        print("✗ FAILED: No results returned")
        return False

    # Cleanup
    client.delete_table("test_metadata_field")
    print()
    return True


def test_payload_field():
    """Test that 'payload' field still works (backward compatibility)."""
    print("=" * 80)
    print("Test 2: Using 'payload' field (backward compatibility)")
    print("=" * 80)

    client = Client(host="localhost", port=9999)

    # Create test table
    try:
        client.delete_table("test_payload_field")
    except:
        pass

    client.create_table(
        name="test_payload_field",
        dimension=3,
        distance_metric="cosine"
    )

    # Insert with 'payload' field
    records = [{
        "id": "doc1",
        "vector": [0.4, 0.5, 0.6],
        "payload": {  # ← Using 'payload' directly
            "title": "Test Document",
            "category": "testing"
        }
    }]

    result = client.insert_records("test_payload_field", records)
    print(f"✓ Insert succeeded: {result['success']}")

    # Search
    results = client.search("test_payload_field", [0.4, 0.5, 0.6], k=1)

    if results:
        hit = results[0]
        metadata = hit.get('metadata', {})

        print(f"✓ Search returned {len(results)} result(s)")
        print(f"  - Metadata: {json.dumps(metadata, indent=4)}")

        # Both 'payload' and 'metadata' should be available
        assert 'metadata' in hit, "'metadata' field missing!"
        assert 'payload' in hit, "'payload' field missing!"
        assert metadata.get('title') == "Test Document", "Title field missing!"

        print("✓ Payload field still works correctly")
    else:
        print("✗ FAILED: No results returned")
        return False

    # Cleanup
    client.delete_table("test_payload_field")
    print()
    return True


def test_rag_scenario():
    """Test the actual RAG scenario that was failing before."""
    print("=" * 80)
    print("Test 3: RAG Scenario (the original bug case)")
    print("=" * 80)

    from rag_application import RAGApplication, create_sample_knowledge_base

    client = Client(host="localhost", port=9999)
    table_name = "test_rag_scenario"

    try:
        client.delete_table(table_name)
    except:
        pass

    client.create_table(name=table_name, dimension=384, distance_metric="cosine")

    # Initialize RAG
    rag = RAGApplication(client, table_name)
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)

    print(f"✓ Added {len(documents)} documents to knowledge base")

    # Test retrieval
    question = "What is DXMT VectorDB?"
    context_docs = rag.retrieve_context(question, k=2)

    print(f"✓ Retrieved {len(context_docs)} documents for: '{question}'")

    # Verify context has text
    success = True
    for i, doc in enumerate(context_docs, 1):
        metadata = doc.get('metadata', {})
        text = metadata.get('text', '')
        source = metadata.get('source', 'unknown')

        if not text:
            print(f"✗ FAILED: Document {i} has no text!")
            success = False
        else:
            print(f"  [{i}] {source}: {text[:80]}...")

    if success:
        print("✓ All documents have text content (RAG will work)")

    # Cleanup
    client.delete_table(table_name)
    print()
    return success


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "DXMT Python SDK - Metadata Fix Test" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    results = []

    try:
        results.append(("metadata field", test_metadata_field()))
        results.append(("payload field", test_payload_field()))
        results.append(("RAG scenario", test_rag_scenario()))
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The metadata/payload fix is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
