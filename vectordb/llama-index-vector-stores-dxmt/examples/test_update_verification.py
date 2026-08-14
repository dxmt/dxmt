"""
UPDATE Functionality Verification Test

This script thoroughly tests the UPDATE operations to verify:
1. Whether updates actually take effect
2. If there are any caching issues
3. Whether upsert properly replaces vs. appends
4. Metadata-only updates
5. Text-only updates
"""

import os
import time
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def print_subsection(title):
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def verify_document_state(index, query_text, expected_results, description):
    """Helper to verify document state"""
    print(f"\n🔍 Verification: {description}")
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(query_text)

    print(f"   Found {len(nodes)} documents")
    for i, node in enumerate(nodes, 1):
        doc_id = node.node.ref_doc_id if hasattr(node.node, 'ref_doc_id') else "N/A"
        text_preview = node.text[:60] + "..." if len(node.text) > 60 else node.text
        metadata = node.metadata
        print(f"   {i}. [{doc_id}] {text_preview}")
        print(f"      Metadata: {metadata}")
        print(f"      Score: {node.score:.4f}")

    return nodes

def main():
    print_section("UPDATE Functionality Deep Verification Test")

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Configure settings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Clean start
    print("Cleaning up any existing test table...")
    try:
        from dxmt import Client
        client = Client(host="localhost", port=9999)
        try:
            client.delete_table("update_verification_test")
            print("   ✓ Cleaned existing table")
            time.sleep(0.5)  # Give server time to process
        except:
            print("   ℹ No existing table to clean")
    except Exception as e:
        print(f"   ℹ Could not clean table: {e}")

    # Create vector store
    print("\nCreating vector store...")
    vector_store = DXMTVectorStore(
        table_name="update_verification_test",
        dimension=1536,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("   ✓ Vector store created")

    # ========================================================================
    # TEST 1: Basic Text Update
    # ========================================================================
    print_section("TEST 1: Basic Text Update (Same doc_id)")

    print("Step 1: Insert initial document")
    doc1_v1 = Document(
        text="Python is a programming language.",
        metadata={"language": "Python", "version": "v1"},
        doc_id="test_doc_1",
    )
    index = VectorStoreIndex.from_documents([doc1_v1], vector_store=vector_store)
    print("   ✓ Inserted document with doc_id='test_doc_1'")

    # Verify initial state
    nodes = verify_document_state(
        index, "Python programming", 1,
        "Initial state - should have original text"
    )
    initial_text = nodes[0].text if nodes else None
    initial_metadata = nodes[0].metadata if nodes else None
    print(f"\n   📝 Initial text: '{initial_text}'")
    print(f"   📋 Initial metadata: {initial_metadata}")

    print("\n⏱️  Waiting 2 seconds before update...")
    time.sleep(2)

    print("\nStep 2: Update document with SAME doc_id (should replace)")
    doc1_v2 = Document(
        text="Python is a high-level, interpreted programming language known for its simplicity.",
        metadata={"language": "Python", "version": "v2", "updated": True},
        doc_id="test_doc_1",  # SAME ID
    )
    index.insert(doc1_v2)
    print("   ✓ Updated document (same doc_id='test_doc_1')")

    print("\n⏱️  Waiting 2 seconds for index to update...")
    time.sleep(2)

    # Verify updated state
    nodes = verify_document_state(
        index, "Python programming", 1,
        "After update - should have new text"
    )
    updated_text = nodes[0].text if nodes else None
    updated_metadata = nodes[0].metadata if nodes else None
    print(f"\n   📝 Updated text: '{updated_text}'")
    print(f"   📋 Updated metadata: {updated_metadata}")

    # Compare
    print("\n🔬 COMPARISON:")
    print(f"   Text changed: {initial_text != updated_text}")
    print(f"   Metadata changed: {initial_metadata != updated_metadata}")

    if initial_text != updated_text:
        print("   ✅ TEXT UPDATE SUCCESSFUL")
    else:
        print("   ❌ TEXT UPDATE FAILED - Text is still the same!")

    if updated_metadata.get("version") == "v2" and updated_metadata.get("updated") == True:
        print("   ✅ METADATA UPDATE SUCCESSFUL")
    else:
        print("   ❌ METADATA UPDATE FAILED - Metadata not updated!")

    # Check for duplicates
    print("\n🔍 Checking for duplicates...")
    all_nodes = verify_document_state(
        index, "Python", 10,
        "All documents with 'Python'"
    )
    python_docs = [n for n in all_nodes if "Python" in n.text]
    print(f"\n   Total documents with 'Python': {len(python_docs)}")
    if len(python_docs) == 1:
        print("   ✅ NO DUPLICATES - Upsert worked correctly")
    else:
        print(f"   ⚠️  FOUND {len(python_docs)} DOCUMENTS - Upsert may have created duplicates!")
        for i, node in enumerate(python_docs, 1):
            ref_id = node.node.ref_doc_id if hasattr(node.node, 'ref_doc_id') else "N/A"
            print(f"      {i}. ref_doc_id={ref_id}, text='{node.text[:50]}...'")

    # ========================================================================
    # TEST 2: Metadata-Only Update
    # ========================================================================
    print_section("TEST 2: Metadata-Only Update (Preserve text)")

    print("Step 1: Get current state")
    retriever = index.as_retriever(similarity_top_k=1)
    current_nodes = retriever.retrieve("Python")
    if not current_nodes:
        print("   ❌ No documents found!")
        return

    current_node = current_nodes[0]
    current_text = current_node.text
    current_metadata = current_node.metadata.copy()
    ref_doc_id = current_node.node.ref_doc_id if hasattr(current_node.node, 'ref_doc_id') else None

    print(f"   Current text: '{current_text[:50]}...'")
    print(f"   Current metadata: {current_metadata}")
    print(f"   Current ref_doc_id: {ref_doc_id}")

    print("\n⏱️  Waiting 2 seconds...")
    time.sleep(2)

    print("\nStep 2: Update ONLY metadata (keep text)")
    updated_metadata_only = {
        **current_metadata,
        "version": "v3",
        "last_modified": "2026-01-24",
        "author": "Test Script"
    }

    doc1_v3 = Document(
        text=current_text,  # SAME TEXT
        metadata=updated_metadata_only,  # NEW METADATA
        doc_id="test_doc_1",
    )
    index.insert(doc1_v3)
    print("   ✓ Updated metadata only")

    print("\n⏱️  Waiting 2 seconds for index to update...")
    time.sleep(2)

    # Verify
    nodes = verify_document_state(
        index, "Python", 1,
        "After metadata-only update"
    )
    new_text = nodes[0].text if nodes else None
    new_metadata = nodes[0].metadata if nodes else None

    print("\n🔬 COMPARISON:")
    print(f"   Text preserved: {current_text == new_text}")
    print(f"   Metadata changed: {current_metadata != new_metadata}")

    if current_text == new_text:
        print("   ✅ TEXT PRESERVED (as expected)")
    else:
        print("   ⚠️  TEXT CHANGED (unexpected!)")

    if new_metadata.get("version") == "v3" and new_metadata.get("author") == "Test Script":
        print("   ✅ METADATA UPDATE SUCCESSFUL")
    else:
        print("   ❌ METADATA UPDATE FAILED")

    # ========================================================================
    # TEST 3: Multiple Documents - Selective Update
    # ========================================================================
    print_section("TEST 3: Multiple Documents - Selective Update")

    print("Step 1: Insert multiple documents")
    docs = [
        Document(
            text="JavaScript is the language of the web.",
            metadata={"language": "JavaScript", "version": "v1"},
            doc_id="test_doc_js",
        ),
        Document(
            text="Go is a statically typed language.",
            metadata={"language": "Go", "version": "v1"},
            doc_id="test_doc_go",
        ),
        Document(
            text="Rust is a systems programming language.",
            metadata={"language": "Rust", "version": "v1"},
            doc_id="test_doc_rust",
        ),
    ]

    for doc in docs:
        index.insert(doc)
    print(f"   ✓ Inserted {len(docs)} additional documents")

    print("\n⏱️  Waiting 2 seconds...")
    time.sleep(2)

    # Verify all documents
    all_nodes = verify_document_state(
        index, "programming language", 10,
        "All documents after insertion"
    )
    print(f"\n   Total documents: {len(all_nodes)}")

    print("\nStep 2: Update ONLY JavaScript document")
    updated_js = Document(
        text="JavaScript is a versatile, multi-paradigm programming language primarily used for web development.",
        metadata={"language": "JavaScript", "version": "v2", "updated": True},
        doc_id="test_doc_js",  # Update this one
    )
    index.insert(updated_js)
    print("   ✓ Updated JavaScript document")

    print("\n⏱️  Waiting 2 seconds...")
    time.sleep(2)

    # Verify selective update
    all_nodes = verify_document_state(
        index, "programming", 10,
        "After selective update"
    )

    print("\n🔬 VERIFICATION:")
    js_nodes = [n for n in all_nodes if "JavaScript" in n.text]
    go_nodes = [n for n in all_nodes if "Go" in n.text]
    rust_nodes = [n for n in all_nodes if "Rust" in n.text]

    print(f"   JavaScript documents: {len(js_nodes)}")
    if js_nodes:
        print(f"      Version: {js_nodes[0].metadata.get('version')}")
        print(f"      Updated: {js_nodes[0].metadata.get('updated', False)}")
        if len(js_nodes) == 1 and js_nodes[0].metadata.get('version') == 'v2':
            print("      ✅ JavaScript updated correctly")
        else:
            print("      ⚠️  JavaScript update issue")

    print(f"   Go documents: {len(go_nodes)}")
    if go_nodes:
        print(f"      Version: {go_nodes[0].metadata.get('version')}")
        if go_nodes[0].metadata.get('version') == 'v1':
            print("      ✅ Go unchanged (as expected)")
        else:
            print("      ⚠️  Go unexpectedly changed")

    print(f"   Rust documents: {len(rust_nodes)}")
    if rust_nodes:
        print(f"      Version: {rust_nodes[0].metadata.get('version')}")
        if rust_nodes[0].metadata.get('version') == 'v1':
            print("      ✅ Rust unchanged (as expected)")
        else:
            print("      ⚠️  Rust unexpectedly changed")

    # ========================================================================
    # TEST 4: Rapid Updates (Race Condition Test)
    # ========================================================================
    print_section("TEST 4: Rapid Sequential Updates")

    print("Performing 5 rapid updates to same document...")
    for i in range(1, 6):
        rapid_doc = Document(
            text=f"Python version {i}: This is update number {i}.",
            metadata={"language": "Python", "version": f"v{i}", "update_number": i},
            doc_id="test_doc_1",
        )
        index.insert(rapid_doc)
        print(f"   ✓ Update {i} submitted")
        time.sleep(0.5)  # Small delay

    print("\n⏱️  Waiting 3 seconds for all updates to process...")
    time.sleep(3)

    # Verify final state
    nodes = verify_document_state(
        index, "Python", 10,
        "After rapid updates"
    )

    python_nodes = [n for n in nodes if "Python" in n.text]
    print(f"\n🔬 VERIFICATION:")
    print(f"   Total Python documents: {len(python_nodes)}")

    if python_nodes:
        final_version = python_nodes[0].metadata.get("version")
        final_update_num = python_nodes[0].metadata.get("update_number")
        print(f"   Final version: {final_version}")
        print(f"   Final update number: {final_update_num}")

        if len(python_nodes) == 1:
            print("   ✅ Only one document (no duplicates from rapid updates)")
        else:
            print(f"   ⚠️  Found {len(python_nodes)} documents - rapid updates may cause issues")

        if final_update_num == 5:
            print("   ✅ Latest update applied (update 5)")
        else:
            print(f"   ⚠️  Not the latest update (expected 5, got {final_update_num})")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("FINAL SUMMARY")

    # Count total documents
    all_docs = verify_document_state(
        index, "", 20,
        "Final state of all documents"
    )

    print("\n📊 STATISTICS:")
    print(f"   Total documents in index: {len(all_docs)}")

    # Group by ref_doc_id
    ref_doc_ids = {}
    for node in all_docs:
        ref_id = node.node.ref_doc_id if hasattr(node.node, 'ref_doc_id') else "unknown"
        if ref_id not in ref_doc_ids:
            ref_doc_ids[ref_id] = []
        ref_doc_ids[ref_id].append(node)

    print(f"   Unique ref_doc_ids: {len(ref_doc_ids)}")

    print("\n📋 DOCUMENTS BY ID:")
    for ref_id, nodes in ref_doc_ids.items():
        print(f"   ref_doc_id='{ref_id}':")
        print(f"      Count: {len(nodes)}")
        if len(nodes) > 1:
            print(f"      ⚠️  DUPLICATE! Same ref_doc_id appears {len(nodes)} times")
            for i, node in enumerate(nodes, 1):
                print(f"         {i}. {node.text[:40]}...")
        else:
            node = nodes[0]
            print(f"      Text: {node.text[:60]}...")
            print(f"      Metadata: {node.metadata}")

    print("\n" + "=" * 80)
    print("UPDATE VERIFICATION TEST COMPLETED")
    print("=" * 80)

    # Cleanup
    print("\n🧹 Cleanup...")
    try:
        client.delete_table("update_verification_test")
        print("   ✓ Test table deleted")
    except Exception as e:
        print(f"   ℹ Could not delete table: {e}")

if __name__ == "__main__":
    main()
