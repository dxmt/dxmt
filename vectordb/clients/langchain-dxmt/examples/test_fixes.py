"""Quick test script to verify example fixes."""

import os
import sys

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    try:
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_dxmt import DXMTVectorDB
        print("✅ Basic imports OK")
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False

    # Test MMR method exists
    try:
        from langchain_dxmt import DXMTVectorDB
        assert hasattr(DXMTVectorDB, 'max_marginal_relevance_search')
        assert hasattr(DXMTVectorDB, 'max_marginal_relevance_search_by_vector')
        print("✅ MMR methods exist")
    except (ImportError, AssertionError) as e:
        print(f"❌ MMR methods missing: {e}")
        return False

    return True

def test_mmr_basic():
    """Test basic MMR functionality."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping MMR test (no OPENAI_API_KEY)")
        return True

    print("\nTesting MMR functionality...")

    try:
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings
        from langchain_dxmt import DXMTVectorDB
        from dxmt import Client

        # Clean up
        try:
            client = Client(host="localhost", port=9999)
            client.delete_table("test_mmr")
        except:
            pass

        # Create vectorstore
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = DXMTVectorDB(
            table_name="test_mmr",
            embedding=embeddings,
            dimension=1536,
            host="localhost",
            port=9999,
        )

        # Add test documents
        docs = [
            Document(page_content="Python programming", metadata={"id": "1"}),
            Document(page_content="Python language", metadata={"id": "2"}),
            Document(page_content="Java programming", metadata={"id": "3"}),
        ]
        vectorstore.add_documents(docs)

        # Test MMR search
        results = vectorstore.max_marginal_relevance_search(
            "Python",
            k=2,
            fetch_k=3,
            lambda_mult=0.5
        )

        print(f"✅ MMR search returned {len(results)} results")

        # Cleanup
        client.delete_table("test_mmr")

        return True

    except Exception as e:
        print(f"❌ MMR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing Example Fixes")
    print("=" * 60)

    all_pass = True

    # Test imports
    if not test_imports():
        all_pass = False

    # Test MMR
    if not test_mmr_basic():
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
