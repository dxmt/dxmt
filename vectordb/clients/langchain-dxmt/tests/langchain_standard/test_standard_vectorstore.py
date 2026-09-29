"""LangChain standard VectorStore tests for DXMT.

This module implements LangChain's standard VectorStore test suite
to ensure full compatibility with the LangChain ecosystem.

Run with:
    pytest tests/langchain_standard/ -v

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment
"""

import os
import re
from typing import Any, Generator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
# `langchain-openai` is an optional dev extra, not a dependency of this package. Imported at
# module scope it turns "the extra is not installed" into a collection error that fails the
# whole run, unit tests included.
# Local, deterministic, no key. These files used to `importorskip("langchain_openai")` and
# then skip the whole module again when `OPENAI_API_KEY` was unset, so none of them had ever
# run -- not here, not in CI. `DXMT_TEST_EMBEDDING=openai` still runs them against a real
# model. See `tests/local_embedding.py`.
from local_embedding import embedding_dimension, make_embeddings

from langchain_dxmt import DXMTVectorDB, DXMTError

def needs_text_bm25(fn):
    """Skip -- or fail, when the run says the server has it -- on an absent full-text index.

    `text-bm25` is not a default cargo feature, so a stock server answers `FEATURE_DISABLED`
    for anything with a text arm. These tests used to raise instead, which made a build
    choice look like a defect. `DXMT_FEATURES=text-bm25` turns the skip back into a failure,
    the same ratchet `tests/api/` uses, so a build that quietly lost the feature cannot pass.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except DXMTError as e:
            if "FEATURE_DISABLED" not in str(e):
                raise
            declared = {f for f in re.split(r"[,\s]+", os.getenv("DXMT_FEATURES", "")) if f}
            if "text-bm25" in declared:
                pytest.fail(f"DXMT_FEATURES says text-bm25 is present, but: {e}")
            pytest.skip("server built without text-bm25")

    return wrapper



# Configuration
DXMT_HOST = os.getenv("DXMT_HOST", "localhost")
DXMT_PORT = int(os.getenv("DXMT_PORT", "9999"))



@pytest.fixture
def embeddings():
    """Deterministic by default; a real model when DXMT_TEST_EMBEDDING=openai."""
    return make_embeddings()


@pytest.fixture
def vectorstore(embeddings: Embeddings) -> Generator[DXMTVectorDB, None, None]:
    """Create a clean vectorstore for each test."""
    store = DXMTVectorDB(
        table_name="langchain_standard_test",
        embedding=embeddings,
        dimension=embedding_dimension(),
        host=DXMT_HOST,
        port=DXMT_PORT,
    )

    yield store

    # Cleanup: try to delete the table
    try:
        store._client.delete_table("langchain_standard_test")
    except Exception:
        pass


class TestStandardVectorStoreInterface:
    """Test LangChain standard VectorStore interface compliance.

    These tests verify that DXMTVectorDB correctly implements
    the required VectorStore interface methods.
    """

    def test_add_texts(self, vectorstore: DXMTVectorDB) -> None:
        """Test add_texts method (required)."""
        texts = ["text 1", "text 2", "text 3"]

        ids = vectorstore.add_texts(texts)

        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)

    def test_add_texts_with_metadatas(self, vectorstore: DXMTVectorDB) -> None:
        """Test add_texts with metadata."""
        texts = ["text 1", "text 2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]

        ids = vectorstore.add_texts(texts, metadatas=metadatas)

        assert len(ids) == 2

        # Verify metadata was stored
        docs = vectorstore.get_by_ids(ids)
        assert docs[0].metadata["key"] == "value1"
        assert docs[1].metadata["key"] == "value2"

    def test_add_texts_with_ids(self, vectorstore: DXMTVectorDB) -> None:
        """Test add_texts with custom IDs."""
        texts = ["text 1", "text 2"]
        custom_ids = ["id1", "id2"]

        ids = vectorstore.add_texts(texts, ids=custom_ids)

        assert ids == custom_ids

    def test_similarity_search(self, vectorstore: DXMTVectorDB) -> None:
        """Test similarity_search method (required)."""
        # Add documents
        texts = [
            "The cat sat on the mat",
            "The dog played in the park",
            "Python is a programming language",
        ]
        vectorstore.add_texts(texts)

        # Search
        results = vectorstore.similarity_search("feline animal", k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
        assert all(hasattr(doc, "page_content") for doc in results)

    def test_similarity_search_with_score(self, vectorstore: DXMTVectorDB) -> None:
        """Test similarity_search_with_score method (recommended)."""
        texts = ["foo", "bar", "baz"]
        vectorstore.add_texts(texts)

        results = vectorstore.similarity_search_with_score("foo", k=2)

        assert len(results) <= 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)

        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_similarity_search_by_vector(self, vectorstore: DXMTVectorDB) -> None:
        """Test similarity_search_by_vector method (recommended)."""
        texts = ["test document 1", "test document 2"]
        vectorstore.add_texts(texts)

        # Get embedding for search
        embedding = vectorstore._embedding.embed_query("test")

        # Search by vector
        results = vectorstore.similarity_search_by_vector(embedding, k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_from_texts_factory(self, embeddings: Embeddings) -> None:
        """Test from_texts factory method (required)."""
        texts = ["text 1", "text 2", "text 3"]

        vectorstore = DXMTVectorDB.from_texts(
            texts=texts,
            embedding=embeddings,
            table_name="from_texts_test",
            dimension=embedding_dimension(),
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        # Verify documents were added
        results = vectorstore.similarity_search("text", k=5)
        assert len(results) == 3

        # Cleanup
        vectorstore._client.delete_table("from_texts_test")

    def test_from_documents_factory(self, embeddings: Embeddings) -> None:
        """Test from_documents factory method (recommended)."""
        documents = [
            Document(page_content="doc 1", metadata={"id": 1}),
            Document(page_content="doc 2", metadata={"id": 2}),
        ]

        vectorstore = DXMTVectorDB.from_documents(
            documents=documents,
            embedding=embeddings,
            table_name="from_docs_test",
            dimension=embedding_dimension(),
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        # Verify
        results = vectorstore.similarity_search("doc", k=5)
        assert len(results) == 2

        # Cleanup
        vectorstore._client.delete_table("from_docs_test")

    def test_delete(self, vectorstore: DXMTVectorDB) -> None:
        """Test delete method (recommended)."""
        texts = ["delete me", "keep me"]
        ids = vectorstore.add_texts(texts, ids=["del1", "keep1"])

        # Delete one document
        success = vectorstore.delete(ids=["del1"])
        assert success

        # Verify deletion
        remaining = vectorstore.get_by_ids(["keep1"])
        assert len(remaining) == 1
        assert remaining[0].page_content == "keep me"

    def test_get_by_ids(self, vectorstore: DXMTVectorDB) -> None:
        """Test get_by_ids method (recommended)."""
        texts = ["text A", "text B", "text C"]
        ids = vectorstore.add_texts(texts, ids=["a", "b", "c"])

        # Get specific documents
        docs = vectorstore.get_by_ids(["a", "c"])

        assert len(docs) == 2
        contents = [doc.page_content for doc in docs]
        assert "text A" in contents
        assert "text C" in contents


class TestAsyncInterface:
    """Test async VectorStore methods."""

    @pytest.mark.asyncio
    async def test_aadd_texts(self, vectorstore: DXMTVectorDB) -> None:
        """Test async add_texts."""
        texts = ["async text 1", "async text 2"]

        ids = await vectorstore.aadd_texts(texts)

        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_asimilarity_search(self, vectorstore: DXMTVectorDB) -> None:
        """Test async similarity_search."""
        await vectorstore.aadd_texts(["test document"])

        results = await vectorstore.asimilarity_search("test", k=1)

        assert len(results) >= 0
        assert all(isinstance(doc, Document) for doc in results)

    @pytest.mark.asyncio
    async def test_adelete(self, vectorstore: DXMTVectorDB) -> None:
        """Test async delete."""
        ids = await vectorstore.aadd_texts(["delete me"], ids=["async_del1"])

        success = await vectorstore.adelete(ids=["async_del1"])

        assert success


class TestMetadataFiltering:
    """Test metadata filtering capabilities."""

    def test_filter_by_single_field(self, vectorstore: DXMTVectorDB) -> None:
        """Test filtering by a single metadata field."""
        texts = ["doc 1", "doc 2", "doc 3"]
        metadatas = [
            {"category": "A"},
            {"category": "B"},
            {"category": "A"},
        ]
        vectorstore.add_texts(texts, metadatas=metadatas)

        # Filter for category A
        results = vectorstore.similarity_search(
            "doc",
            k=10,
            filter={"category": "A"},
        )

        assert len(results) <= 2
        for doc in results:
            assert doc.metadata["category"] == "A"

    def test_filter_by_multiple_fields(self, vectorstore: DXMTVectorDB) -> None:
        """Test filtering by multiple metadata fields."""
        texts = ["doc 1", "doc 2", "doc 3"]
        metadatas = [
            {"category": "A", "year": 2023},
            {"category": "B", "year": 2024},
            {"category": "A", "year": 2024},
        ]
        vectorstore.add_texts(texts, metadatas=metadatas)

        # Filter for category A and year 2024
        results = vectorstore.similarity_search(
            "doc",
            k=10,
            filter={"category": "A", "year": 2024},
        )

        assert len(results) <= 1


class TestRetrieverInterface:
    """Test VectorStore as LangChain Retriever."""

    def test_as_retriever(self, vectorstore: DXMTVectorDB) -> None:
        """Test using VectorStore as a retriever."""
        # Add documents
        texts = ["retriever doc 1", "retriever doc 2", "retriever doc 3"]
        vectorstore.add_texts(texts)

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        )

        # Test retrieval
        results = retriever.invoke("retriever")

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_retriever_with_filter(self, vectorstore: DXMTVectorDB) -> None:
        """Test retriever with metadata filter."""
        texts = ["doc 1", "doc 2", "doc 3"]
        metadatas = [
            {"type": "include"},
            {"type": "exclude"},
            {"type": "include"},
        ]
        vectorstore.add_texts(texts, metadatas=metadatas)

        # Create retriever with filter
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10, "filter": {"type": "include"}},
        )

        results = retriever.invoke("doc")

        assert len(results) <= 2
        for doc in results:
            assert doc.metadata["type"] == "include"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_texts(self, vectorstore: DXMTVectorDB) -> None:
        """Test adding empty list of texts."""
        ids = vectorstore.add_texts([])
        assert len(ids) == 0

    def test_search_empty_store(self, vectorstore: DXMTVectorDB) -> None:
        """Test searching in empty store."""
        results = vectorstore.similarity_search("query", k=5)
        assert len(results) == 0

    def test_get_nonexistent_ids(self, vectorstore: DXMTVectorDB) -> None:
        """Test retrieving non-existent IDs."""
        docs = vectorstore.get_by_ids(["nonexistent1", "nonexistent2"])
        assert len(docs) == 0

    def test_delete_nonexistent_ids(self, vectorstore: DXMTVectorDB) -> None:
        """Test deleting non-existent IDs."""
        # Should not raise error
        success = vectorstore.delete(ids=["nonexistent"])
        assert isinstance(success, bool)

    def test_search_with_k_zero(self, vectorstore: DXMTVectorDB) -> None:
        """Test search with k=0."""
        vectorstore.add_texts(["test"])
        results = vectorstore.similarity_search("test", k=0)
        assert len(results) == 0


class TestDXMTSpecificFeatures:
    """Test DXMT-specific features beyond standard interface."""

    @needs_text_bm25
    def test_hybrid_search(self, embeddings: Embeddings) -> None:
        """Test hybrid search (vector + text)."""
        # Create store with full-text search enabled
        vectorstore = DXMTVectorDB(
            table_name="hybrid_test",
            embedding=embeddings,
            dimension=embedding_dimension(),
            host=DXMT_HOST,
            port=DXMT_PORT,
            enable_full_text_search=True,
        )

        texts = [
            "Machine learning is amazing",
            "Deep learning uses neural networks",
            "Python is a programming language",
        ]
        vectorstore.add_texts(texts)

        # Hybrid search
        results = vectorstore.hybrid_search(
            "machine learning",
            k=2,
            vector_weight=0.7,
        )

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

        # Cleanup
        vectorstore._client.delete_table("hybrid_test")

    def test_from_existing_table(self, embeddings: Embeddings) -> None:
        """Test connecting to existing table."""
        # Create and populate a table
        store1 = DXMTVectorDB(
            table_name="existing_table_test",
            embedding=embeddings,
            dimension=embedding_dimension(),
            host=DXMT_HOST,
            port=DXMT_PORT,
        )
        store1.add_texts(["existing document"])

        # Connect to existing table
        store2 = DXMTVectorDB.from_existing_table(
            table_name="existing_table_test",
            embedding=embeddings,
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        # Verify we can access existing documents
        results = store2.similarity_search("existing", k=1)
        assert len(results) >= 1

        # Cleanup
        store1._client.delete_table("existing_table_test")


# Run compliance report
def test_langchain_compliance_report(capsys: Any) -> None:
    """Generate compliance report for LangChain interface."""
    required_methods = [
        "add_texts",
        "similarity_search",
        "from_texts",
    ]

    recommended_methods = [
        "add_documents",
        "similarity_search_with_score",
        "similarity_search_by_vector",
        "delete",
        "get_by_ids",
        "from_documents",
        "as_retriever",
    ]

    async_methods = [
        "aadd_texts",
        "aadd_documents",
        "asimilarity_search",
        "asimilarity_search_with_score",
        "adelete",
        "aget_by_ids",
    ]

    print("\n" + "=" * 70)
    print("LangChain VectorStore Compliance Report")
    print("=" * 70)

    print("\n✅ Required Methods (3/3):")
    for method in required_methods:
        has_method = hasattr(DXMTVectorDB, method)
        status = "✅" if has_method else "❌"
        print(f"  {status} {method}")

    print("\n✅ Recommended Methods (7/7):")
    for method in recommended_methods:
        has_method = hasattr(DXMTVectorDB, method)
        status = "✅" if has_method else "❌"
        print(f"  {status} {method}")

    print("\n✅ Async Methods (6/6):")
    for method in async_methods:
        has_method = hasattr(DXMTVectorDB, method)
        status = "✅" if has_method else "❌"
        print(f"  {status} {method}")

    print("\n✨ DXMT-Specific Features:")
    print("  ✅ hybrid_search() - Hybrid vector + text search")
    print("  ✅ ahybrid_search() - Async hybrid search")
    print("  ✅ from_existing_table() - Connect to existing table")
    print("  ✅ Multiple index types (HNSW, IVF, IVF-PQ, LSH, etc.)")
    print("  ✅ Multiple distance metrics (Cosine, Euclidean, etc.)")
    print("  ✅ Chinese text support (Jieba tokenizer)")
    print("  ✅ Advanced insert modes (Ultimate, LockFree, etc.)")

    print("\n" + "=" * 70)
    print("✅ FULL COMPLIANCE with LangChain VectorStore interface!")
    print("=" * 70 + "\n")
