"""Paul Graham Essay Example — long-document indexing and querying with DXMT.

Loads a long essay, indexes it, and runs several query patterns against it.


Usage:
    python paul_graham_essay.py          # Normal run
    python paul_graham_essay.py --clean  # Clean start (delete existing table)

If you encounter errors about existing tables, use --clean to start fresh.
"""

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.dxmt import DXMTVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display


def download_essay():
    """Download Paul Graham's essay if not already present."""
    import subprocess

    os.makedirs("data/paul_graham", exist_ok=True)

    essay_path = "data/paul_graham/paul_graham_essay.txt"
    if not os.path.exists(essay_path):
        print("Downloading Paul Graham essay...")
        url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt"
        subprocess.run(["wget", url, "-O", essay_path], check=True)
        print(f"✓ Essay downloaded to {essay_path}")
    else:
        print(f"✓ Essay already exists at {essay_path}")


def basic_example(clean_start=False):
    """Basic example: index and query Paul Graham's essay."""
    print("\n" + "=" * 70)
    print("Part 1: Basic Example")
    print("=" * 70)

    # 1. Configure settings
    print("\n1. Configuring LlamaIndex settings...")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    print("   ✓ Settings configured")

    # 2. Download essay
    print("\n2. Preparing data...")
    download_essay()

    # 3. Create DXMT vector store
    print("\n3. Creating DXMT vector store...")

    # Optional: clear existing table for clean start
    if clean_start:
        print("   ⚠ Cleaning existing table (if exists)...")
        try:
            from dxmt import Client  # Correct import name
            client = Client(host="localhost", port=9999)
            try:
                client.delete_table("paul_graham_essay")
                print("   ✓ Cleared existing table")
            except:
                print("   ℹ No existing table to clear")
        except Exception as e:
            print(f"   ℹ Could not clear table: {e}")

    vector_store = DXMTVectorStore(
        table_name="paul_graham_essay",
        dimension=1536,
        host="localhost",
        port=9999,
        index_type="hnsw",
        distance_metric="cosine",
    )
    print("   ✓ Vector store created")

    # 4. Load documents
    print("\n4. Loading documents...")
    documents = SimpleDirectoryReader("data/paul_graham/").load_data()
    print(f"   ✓ Loaded {len(documents)} document(s)")

    # 5. Create index
    print("\n5. Creating index and embedding documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
    )
    print("   ✓ Index created")

    # 6. Query the index
    print("\n6. Querying the index...")
    query_engine = index.as_query_engine()

    question = "What did the author do growing up?"
    print(f"\n   Question: {question}")

    response = query_engine.query(question)
    print(f"\n   Answer:\n   {response}\n")

    return vector_store, index


def persistent_example(existing_index):
    """Example showing how DXMT naturally persists data.

    The server handles persistence automatically once it is connected to a
    database backend; the client does not choose a persistence mode.
    """
    print("\n" + "=" * 70)
    print("Part 2: Persistence (Built-in with DXMT)")
    print("=" * 70)

    print("""
    DXMT Persistence Advantages:

    1. Server-Based Architecture:
       - DXMT runs as a server process
       - All data is automatically persisted
       - No need for special "persistent" vs "ephemeral" clients

    2. Production-Ready:
       - Data survives client disconnections
       - Multiple clients can connect simultaneously
       - No risk of clients "stomping" each other's work

    3. Flexible Storage:
       - Object storage (S3/MinIO) as source of truth
       - SSD and memory as cache layers
       - Automatic tiering based on access patterns

    ```python
    # Just connect to the server; data is persisted automatically.
    vector_store = DXMTVectorStore(host="localhost", port=9999)
    ```
    """)

    # Use the existing index (data is already persisted on server)
    print("\n1. Using existing index...")
    print("   ✓ Index already connected to DXMT server")
    print("   ✓ Data is automatically persisted on the server side")

    # Query to verify data is accessible
    print("\n2. Verifying data is accessible...")
    query_engine = existing_index.as_query_engine()
    response = query_engine.query("What did the author do at Y Combinator?")
    print(f"\n   Answer: {response}\n")
    print("   ✓ Data successfully retrieved from persistent storage")

    print("\n3. Persistence demo:")
    print("   Note: If you restart this script, you can reconnect like this:")
    print("   ```python")
    print("   vector_store = DXMTVectorStore(")
    print("       table_name='paul_graham_essay',")
    print("       dimension=1536,")
    print("       host='localhost',")
    print("       port=9999,")
    print("   )")
    print("   index = VectorStoreIndex.from_vector_store(vector_store)")
    print("   # All your data is still there!")
    print("   ```")


def multi_query_example(index):
    """Demonstrate multiple queries with different parameters."""
    print("\n" + "=" * 70)
    print("Part 3: Multiple Query Examples")
    print("=" * 70)

    # Define queries
    queries = [
        ("What did the author do growing up?", 2),
        ("What did the author do at Y Combinator?", 3),
        ("What were the author's main projects?", 4),
    ]

    for i, (question, top_k) in enumerate(queries, 1):
        print(f"\n{'-' * 70}")
        print(f"Query {i}: {question}")
        print(f"Using top_k={top_k}")
        print('-' * 70)

        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(question)

        print(f"\nAnswer:\n{response}")

        # Show source nodes
        print(f"\nRetrieved {len(response.source_nodes)} source chunks:")
        for j, node in enumerate(response.source_nodes, 1):
            print(f"  {j}. Score: {node.score:.4f} | "
                  f"Text: {node.text[:80]}...")


def retriever_example(index):
    """Demonstrate using retriever directly without generation."""
    print("\n" + "=" * 70)
    print("Part 4: Retriever Example (Retrieval Only)")
    print("=" * 70)

    # Create retriever
    retriever = index.as_retriever(similarity_top_k=3)

    query = "What programming languages did the author use?"
    print(f"\nQuery: {query}")
    print("\nRetrieved chunks (no generation):\n")

    nodes = retriever.retrieve(query)

    for i, node in enumerate(nodes, 1):
        print(f"{i}. Score: {node.score:.4f}")
        print(f"   Text: {node.text[:200]}...")
        print()


def main():
    """Run all examples."""
    import sys

    print("=" * 70)
    print("DXMT Vector Store - Paul Graham Essay Example")
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
        # Run examples
        vector_store, index = basic_example(clean_start=clean_start)
        persistent_example(index)
        multi_query_example(index)
        retriever_example(index)

        print("\n" + "=" * 70)
        print("Example Completed Successfully!")
        print("=" * 70)
        print("""
        What you learned:

        1. ✓ Loading and indexing documents (Paul Graham essay)
        2. ✓ Querying with different parameters
        3. ✓ DXMT's built-in persistence
        4. ✓ Using retrievers for semantic search only

        Next steps:
        - Try modifying the queries
        - Experiment with different similarity_top_k values
        - Add your own documents
        - Explore hybrid search with text filters
        """)

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {e}")

        # Provide helpful hints for common errors
        if "500" in error_msg or "Internal Server Error" in error_msg:
            print("\n💡 Hint: This might be due to an existing table with different configuration.")
            print("   Try running with --clean flag to start fresh:")
            print("   $ python paul_graham_essay.py --clean")
        elif "Connection" in error_msg or "refused" in error_msg:
            print("\n💡 Hint: Make sure DXMT server is running:")
            print("   $ cargo run -- serve")

        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
