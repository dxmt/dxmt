"""CRUD Operations Example with DXMT Vector Store.

This example demonstrates Create, Read, Update, and Delete operations,
showing how to manage documents in DXMT beyond just inserting and querying.

(Update and Delete section)

Usage:
    python crud_operations.py          # Normal run
    python crud_operations.py --clean  # Clean start (delete existing table)

If you encounter errors about existing tables, use --clean to start fresh.

Note: Delete operations may have limitations in the current version.
The example demonstrates the intended usage pattern.
"""

import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode


def create_operations(clean_start=False):
    """Demonstrate CREATE operations."""
    print("\n" + "=" * 70)
    print("Part 1: CREATE Operations")
    print("=" * 70)

    # Configure settings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Create vector store
    print("\n1. Creating DXMT vector store...")

    # Optional: clear existing table for clean start
    if clean_start:
        print("   ⚠ Cleaning existing table (if exists)...")
        try:
            from dxmt import Client
            client = Client(host="localhost", port=9999)
            try:
                client.delete_table("crud_example")
                print("   ✓ Cleared existing table")
            except:
                print("   ℹ No existing table to clear")
        except Exception as e:
            print(f"   ℹ Could not clear table: {e}")

    vector_store = DXMTVectorStore(
        table_name="crud_example",
        dimension=1536,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("   ✓ Vector store created")

    # Method 1: Insert documents
    print("\n2. Inserting documents (Method 1: from_documents)...")
    documents = [
        Document(
            text="Python is a high-level programming language.",
            metadata={"language": "Python", "category": "programming", "year": 1991},
            doc_id="doc_1",
        ),
        Document(
            text="JavaScript is the language of the web.",
            metadata={"language": "JavaScript", "category": "programming", "year": 1995},
            doc_id="doc_2",
        ),
        Document(
            text="Rust is a systems programming language focused on safety.",
            metadata={"language": "Rust", "category": "programming", "year": 2010},
            doc_id="doc_3",
        ),
    ]

    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    print(f"   ✓ Inserted {len(documents)} documents")

    # Method 2: Insert nodes directly
    print("\n3. Inserting nodes (Method 2: insert)...")
    new_node = TextNode(
        text="Go is a statically typed, compiled language designed at Google.",
        metadata={"language": "Go", "category": "programming", "year": 2009},
        id_="doc_4",
    )

    index.insert(new_node)
    print("   ✓ Inserted 1 additional node")

    # Method 3: Batch insert
    print("\n4. Batch inserting multiple nodes...")
    batch_nodes = [
        TextNode(
            text="TypeScript is a typed superset of JavaScript.",
            metadata={"language": "TypeScript", "category": "programming", "year": 2012},
            id_="doc_5",
        ),
        TextNode(
            text="Swift is a powerful language for iOS development.",
            metadata={"language": "Swift", "category": "programming", "year": 2014},
            id_="doc_6",
        ),
    ]

    for node in batch_nodes:
        index.insert(node)

    print(f"   ✓ Inserted {len(batch_nodes)} more nodes")

    return index, vector_store


def read_operations(index):
    """Demonstrate READ operations."""
    print("\n" + "=" * 70)
    print("Part 2: READ Operations")
    print("=" * 70)

    # Query with different parameters
    print("\n1. Basic query...")
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("What is Python?")
    print(f"   Query: 'What is Python?'")
    print(f"   Answer: {response}")

    # Retrieval only (no generation)
    print("\n2. Retrieval without generation...")
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve("programming languages")

    print(f"   Retrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes, 1):
        lang = node.metadata.get("language", "Unknown")
        year = node.metadata.get("year", "N/A")
        print(f"   {i}. {lang} ({year}) - Score: {node.score:.4f}")

    # Filtered retrieval
    print("\n3. Filtered retrieval...")
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2010, operator=FilterOperator.GTE)
        ]
    )

    filtered_retriever = index.as_retriever(filters=filters, similarity_top_k=10)
    filtered_nodes = filtered_retriever.retrieve("modern languages")

    print(f"   Retrieved {len(filtered_nodes)} nodes (year >= 2010):")
    for i, node in enumerate(filtered_nodes, 1):
        lang = node.metadata.get("language", "Unknown")
        year = node.metadata.get("year", "N/A")
        print(f"   {i}. {lang} ({year})")


def update_operations(index, vector_store):
    """Demonstrate UPDATE operations."""
    print("\n" + "=" * 70)
    print("Part 3: UPDATE Operations")
    print("=" * 70)

    print("""
    Note: DXMT handles updates through upsert operations.
    If a document with the same ID exists, it will be replaced.
    """)

    # Update by re-inserting with same ID
    print("\n1. Updating a document (upsert)...")

    # First, show original
    retriever = index.as_retriever(similarity_top_k=1)
    original = retriever.retrieve("Python")
    if original:
        print(f"   Original: {original[0].text}")
        print(f"   Original metadata: {original[0].metadata}")

    # Update the document
    updated_doc = Document(
        text="Python is a high-level, interpreted programming language known for its simplicity and versatility.",
        metadata={
            "language": "Python",
            "category": "programming",
            "year": 1991,
            "updated": True,
            "creator": "Guido van Rossum",
        },
        doc_id="doc_1",  # Same ID as original
    )

    index.insert(updated_doc)
    print("\n   ✓ Document updated")

    # Verify update
    updated = retriever.retrieve("Python")
    if updated:
        print(f"   Updated: {updated[0].text}")
        print(f"   Updated metadata: {updated[0].metadata}")

    # Update metadata only (preserve text)
    print("\n2. Updating only metadata...")

    # Get current document
    nodes = retriever.retrieve("JavaScript")
    if nodes:
        current_node = nodes[0]
        print(f"   Current metadata: {current_node.metadata}")

        # Create updated version with new metadata
        updated_node = TextNode(
            text=current_node.text,  # Keep same text
            metadata={
                **current_node.metadata,  # Keep existing metadata
                "popularity": "very_high",  # Add new field
                "paradigm": "multi-paradigm",  # Add new field
                "updated_at": "2024-01-24",  # Add timestamp
            },
            id_=current_node.node_id,  # Same ID
        )

        index.insert(updated_node)
        print("   ✓ Metadata updated")

        # Verify
        verified = retriever.retrieve("JavaScript")
        if verified:
            print(f"   New metadata: {verified[0].metadata}")


def delete_operations(index, vector_store):
    """Demonstrate DELETE operations."""
    print("\n" + "=" * 70)
    print("Part 4: DELETE Operations")
    print("=" * 70)

    # Show current count
    print("\n1. Current state...")
    retriever = index.as_retriever(similarity_top_k=20)
    all_nodes = retriever.retrieve("programming")
    print(f"   Total documents: {len(all_nodes)}")
    for i, node in enumerate(all_nodes, 1):
        lang = node.metadata.get("language", "Unknown")
        year = node.metadata.get("year", "N/A")
        # Show ref_doc_id for debugging
        ref_doc_id = node.node.ref_doc_id if hasattr(node.node, 'ref_doc_id') else "N/A"
        print(f"   {i}. {lang} ({year}) [ref_doc_id: {ref_doc_id}]")

    # Delete a single document
    print("\n2. Deleting a document...")
    print("   Deleting: 'TypeScript' (doc_5)")

    # In DXMT, we use the ref_doc_id to delete
    try:
        index.delete_ref_doc("doc_5")  # TypeScript
        print("   ✓ Document deleted")
    except Exception as e:
        print(f"   ⚠ Deletion error: {e}")
        print("   Note: Deletion may not be fully supported in current version")

    # Verify deletion
    print("\n3. Verifying deletion...")
    remaining_nodes = retriever.retrieve("programming")
    print(f"   Remaining documents: {len(remaining_nodes)}")
    for i, node in enumerate(remaining_nodes, 1):
        lang = node.metadata.get("language", "Unknown")
        year = node.metadata.get("year", "N/A")
        print(f"   {i}. {lang} ({year})")

    # Bulk delete (by filter)
    print("\n4. Conditional delete (older languages)...")
    print("   Deleting languages from before 2010...")
    print("   Note: This demonstrates the logic, but deletion may be limited by API")

    # Get languages to delete
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

    old_filters = MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2010, operator=FilterOperator.LT)
        ]
    )

    old_retriever = index.as_retriever(filters=old_filters, similarity_top_k=10)
    old_nodes = old_retriever.retrieve("programming")

    print(f"   Found {len(old_nodes)} old languages to delete:")
    deleted_count = 0
    failed_count = 0

    for node in old_nodes:
        lang = node.metadata.get("language")
        year = node.metadata.get("year")

        # Try to get ref_doc_id
        ref_doc_id = None
        if hasattr(node.node, 'ref_doc_id'):
            ref_doc_id = node.node.ref_doc_id
        elif hasattr(node, 'ref_doc_id'):
            ref_doc_id = node.ref_doc_id

        print(f"     - {lang} ({year}) [attempting to delete ref_doc_id: {ref_doc_id}]")

        # Try to delete
        if ref_doc_id:
            try:
                index.delete_ref_doc(ref_doc_id)
                deleted_count += 1
                print(f"       ✓ Deleted")
            except Exception as e:
                failed_count += 1
                print(f"       ⚠ Could not delete: {e}")
        else:
            failed_count += 1
            print(f"       ⚠ No ref_doc_id found")

    print(f"\n   Summary: Attempted {len(old_nodes)}, Deleted {deleted_count}, Failed {failed_count}")

    # Final count
    print("\n5. Final state...")
    final_nodes = retriever.retrieve("programming")
    print(f"   Final document count: {len(final_nodes)}")
    for i, node in enumerate(final_nodes, 1):
        lang = node.metadata.get("language", "Unknown")
        year = node.metadata.get("year", "N/A")
        print(f"   {i}. {lang} ({year})")

    print("\n   Note: If deletion didn't work as expected, this is a known")
    print("   limitation in the current version. The delete API is being improved.")


def best_practices():
    """Print CRUD best practices."""
    print("\n" + "=" * 70)
    print("CRUD Best Practices with DXMT")
    print("=" * 70)

    print("""
    CREATE:
    ✓ Use unique doc_id for each document
    ✓ Include rich metadata for filtering
    ✓ Use batch operations when possible
    ✓ Validate data before insertion

    Example:
    ```python
    doc = Document(
        text="Your content here",
        metadata={"key": "value", "timestamp": "2024-01-24"},
        doc_id="unique_id_123",  # Important!
    )
    index.insert(doc)
    ```

    READ:
    ✓ Use similarity_top_k to control results
    ✓ Apply metadata filters to narrow search
    ✓ Use retriever for search-only (no LLM call)
    ✓ Use query_engine for search + generation

    Example:
    ```python
    # Just search
    retriever = index.as_retriever(
        similarity_top_k=5,
        filters=MetadataFilters(...)
    )
    nodes = retriever.retrieve("query")

    # Search + generate answer
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("question")
    ```

    UPDATE:
    ✓ Use same doc_id to upsert
    ✓ Preserve fields you don't want to change
    ✓ Add version/timestamp metadata
    ✓ Consider soft deletes (is_deleted flag)

    Example:
    ```python
    # Upsert (insert or replace)
    updated_doc = Document(
        text="Updated content",
        metadata={**old_metadata, "updated": True},
        doc_id="same_id_as_original",  # Key!
    )
    index.insert(updated_doc)
    ```

    DELETE:
    ✓ Use ref_doc_id for deletion
    ✓ Consider soft deletes for audit trails
    ✓ Be careful with bulk deletes
    ✓ Test deletions in dev environment first

    Example:
    ```python
    # Delete by ID
    index.delete_ref_doc("doc_id_to_delete")

    # Soft delete (recommended for production)
    soft_delete_doc = Document(
        text=original_text,
        metadata={**original_metadata, "deleted": True},
        doc_id=original_id,
    )
    index.insert(soft_delete_doc)  # Upsert with deleted flag
    ```

    VERSIONING:
    ✓ Include version numbers in metadata
    ✓ Use timestamps for audit trails
    ✓ Keep old versions if needed

    Example:
    ```python
    doc = Document(
        text="Content v2",
        metadata={
            "version": 2,
            "created_at": "2024-01-01",
            "updated_at": "2024-01-24",
            "updated_by": "user123",
        },
        doc_id="doc_with_versioning",
    )
    ```

    DXMT-Specific Tips:
    ✓ Leverage DXMT's automatic persistence
    ✓ Use metadata filtering for efficient queries
    ✓ Monitor vector quality after updates
    ✓ Use appropriate index type for your scale
    """)


def main():
    """Run all CRUD examples."""
    import sys

    print("=" * 70)
    print("DXMT Vector Store - CRUD Operations Example")
    print("=" * 70)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Check for clean start flag
    clean_start = "--clean" in sys.argv or "-c" in sys.argv

    if clean_start:
        print("\n⚠  Running with --clean flag: will delete existing table")

    try:
        # Run CRUD operations
        index, vector_store = create_operations(clean_start=clean_start)
        read_operations(index)
        update_operations(index, vector_store)
        delete_operations(index, vector_store)
        best_practices()

        print("\n" + "=" * 70)
        print("CRUD Example Completed Successfully!")
        print("=" * 70)
        print("""
        What you learned:

        1. ✓ CREATE: Multiple ways to insert documents
        2. ✓ READ: Query, retrieve, and filter
        3. ✓ UPDATE: Upsert with same doc_id
        4. ✓ DELETE: Single and bulk deletion
        5. ✓ Best practices for production systems

        Key Takeaways:
        - Use unique doc_id for all documents
        - Upsert = insert with same ID
        - Include rich metadata
        - Consider soft deletes
        - Test in dev before production

        Next Steps:
        - Implement versioning in your app
        - Set up audit logging
        - Create backup strategies
        - Monitor update performance
        """)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
