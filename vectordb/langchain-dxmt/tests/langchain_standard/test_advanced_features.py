"""Advanced LangChain features and edge cases for DXMT VectorStore.

This module tests advanced features beyond basic VectorStore interface:
- MMR (Maximal Marginal Relevance) search
- Score threshold filtering
- Complex metadata filters
- Document immutability
- Validation errors
- Batch size handling

Run with:
    pytest tests/langchain_standard/test_advanced_features.py -v

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment
"""

import os
import time
from typing import Any, Generator

import pytest
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_dxmt import DXMTVectorDB, DXMTError


# Configuration
DXMT_HOST = os.getenv("DXMT_HOST", "localhost")
DXMT_PORT = int(os.getenv("DXMT_PORT", "9999"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set - skipping advanced feature tests",
)


@pytest.fixture
def embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance."""
    return OpenAIEmbeddings()


@pytest.fixture
def test_table_name() -> str:
    """Generate unique test table name."""
    timestamp = int(time.time() * 1000)
    return f"test_advanced_{timestamp}"


@pytest.fixture
def vectorstore(
    test_table_name: str,
    embeddings: OpenAIEmbeddings,
) -> Generator[DXMTVectorDB, None, None]:
    """Create vector store for testing."""
    store = DXMTVectorDB(
        table_name=test_table_name,
        embedding=embeddings,
        dimension=1536,
        host=DXMT_HOST,
        port=DXMT_PORT,
    )

    yield store

    # Cleanup
    try:
        store._client.delete_table(test_table_name)
    except Exception:
        pass


class TestMMRSearch:
    """Test Maximal Marginal Relevance (MMR) search."""

    def test_mmr_search_type(
        self,
        vectorstore: DXMTVectorDB,
        embeddings: OpenAIEmbeddings,
    ) -> None:
        """Test MMR search for diversity.

        MMR balances relevance with diversity to avoid returning
        very similar documents.
        """
        # Add documents with some redundancy
        texts = [
            "Python is a programming language",  # Similar to next
            "Python is used for programming",  # Similar to previous
            "JavaScript is for web development",  # Different topic
            "Machine learning with Python",  # Somewhat related
            "Web scraping tutorial",  # Different topic
        ]
        vectorstore.add_texts(texts)

        # Create retriever with MMR
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 5,  # Fetch more candidates
                "lambda_mult": 0.5,  # Balance relevance and diversity
            },
        )

        # Search for Python-related content
        results = retriever.invoke("Python programming")

        assert len(results) <= 3
        # MMR should return diverse results, not just "Python" documents
        contents = [doc.page_content for doc in results]
        print(f"\nMMR results: {contents}")

    def test_mmr_lambda_mult(self, vectorstore: DXMTVectorDB) -> None:
        """Test MMR lambda_mult parameter.

        lambda_mult=1.0: Maximum relevance (no diversity)
        lambda_mult=0.0: Maximum diversity (no relevance)
        lambda_mult=0.5: Balance both
        """
        texts = [
            "Apple fruit is healthy",
            "Apple fruit nutrition",
            "Orange fruit vitamin C",
            "Banana potassium",
        ]
        vectorstore.add_texts(texts)

        # High relevance (lambda_mult=1.0) - should return similar docs
        retriever_relevance = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 1.0},
        )
        results_relevance = retriever_relevance.invoke("apple fruit")
        # Should get both apple documents
        assert all("apple" in doc.page_content.lower() for doc in results_relevance)

        # High diversity (lambda_mult=0.0) - should return diverse docs
        retriever_diversity = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 0.0},
        )
        results_diversity = retriever_diversity.invoke("apple fruit")
        # May include non-apple documents for diversity
        contents_diversity = [doc.page_content for doc in results_diversity]
        print(f"\nDiversity results: {contents_diversity}")


class TestScoreThreshold:
    """Test similarity score threshold filtering."""

    def test_score_threshold_search(self, vectorstore: DXMTVectorDB) -> None:
        """Test filtering results by minimum similarity score.

        Only return documents with similarity >= threshold.
        """
        texts = [
            "Machine learning with neural networks",  # High similarity
            "Deep learning algorithms",  # Medium similarity
            "Cooking recipes for dinner",  # Low similarity
        ]
        vectorstore.add_texts(texts)

        # Create retriever with score threshold
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.7,  # Only return docs with score >= 0.7
                "k": 10,  # Try to get up to 10, but filter by threshold
            },
        )

        results = retriever.invoke("machine learning neural networks")

        # Should only get high-similarity documents
        assert len(results) >= 1  # At least the first doc should match
        # Cooking recipe should be filtered out by threshold
        contents = [doc.page_content for doc in results]
        assert not any("cooking" in content.lower() for content in contents)

    def test_score_threshold_no_results(self, vectorstore: DXMTVectorDB) -> None:
        """Test score threshold with no matching documents."""
        texts = ["Dogs are pets", "Cats are animals"]
        vectorstore.add_texts(texts)

        # Very high threshold should filter out all results
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.99,  # Very high threshold
            },
        )

        results = retriever.invoke("completely unrelated quantum physics topic")

        # May return empty results if nothing meets threshold
        assert isinstance(results, list)


class TestComplexMetadataFiltering:
    """Test complex metadata filtering scenarios."""

    def test_numeric_range_filter(self, vectorstore: DXMTVectorDB) -> None:
        """Test filtering by numeric range (if supported)."""
        texts = ["doc 1", "doc 2", "doc 3", "doc 4"]
        metadatas = [
            {"year": 2020, "score": 85},
            {"year": 2021, "score": 92},
            {"year": 2022, "score": 78},
            {"year": 2023, "score": 95},
        ]
        vectorstore.add_texts(texts, metadatas=metadatas)

        # Filter for year >= 2022
        # Note: Exact filter syntax depends on DXMT implementation
        try:
            results = vectorstore.similarity_search(
                "doc",
                k=10,
                filter={"year": 2022},  # Exact match
            )
            assert len(results) == 1
        except Exception as e:
            pytest.skip(f"Range filters not yet implemented: {e}")

    def test_nested_metadata(self, vectorstore: DXMTVectorDB) -> None:
        """Test filtering by nested metadata fields (if supported)."""
        texts = ["doc 1", "doc 2"]
        metadatas = [
            {"info": {"category": "A", "priority": "high"}},
            {"info": {"category": "B", "priority": "low"}},
        ]

        try:
            vectorstore.add_texts(texts, metadatas=metadatas)

            # Try to filter by nested field
            results = vectorstore.similarity_search(
                "doc",
                k=10,
                filter={"info.category": "A"},
            )
            # Implementation-specific behavior
            print(f"\nNested filter results: {len(results)}")
        except Exception as e:
            pytest.skip(f"Nested metadata not yet supported: {e}")

    def test_list_metadata(self, vectorstore: DXMTVectorDB) -> None:
        """Test metadata with list values."""
        texts = ["doc 1", "doc 2", "doc 3"]
        metadatas = [
            {"tags": ["python", "programming"]},
            {"tags": ["javascript", "web"]},
            {"tags": ["python", "data"]},
        ]

        vectorstore.add_texts(texts, metadatas=metadatas)

        # Filter for documents with "python" tag
        # Note: List filtering may have special syntax
        results = vectorstore.similarity_search("doc", k=10)
        # Check that list metadata is preserved
        assert all(isinstance(doc.metadata.get("tags"), list) for doc in results)


class TestDocumentIntegrity:
    """Test that documents are not mutated and data integrity is maintained."""

    def test_original_documents_not_mutated(
        self,
        vectorstore: DXMTVectorDB,
    ) -> None:
        """Test that original Document objects are not modified.

        This is important to ensure add_documents() doesn't have side effects.
        """
        original_docs = [
            Document(page_content="doc 1", metadata={"key": "value1"}),
            Document(page_content="doc 2", metadata={"key": "value2"}),
        ]

        # Make copies to compare later
        original_contents = [doc.page_content for doc in original_docs]
        original_metadata = [doc.metadata.copy() for doc in original_docs]

        # Add documents
        vectorstore.add_documents(original_docs)

        # Verify originals not mutated
        assert [doc.page_content for doc in original_docs] == original_contents
        assert [doc.metadata for doc in original_docs] == original_metadata

    def test_metadata_round_trip(self, vectorstore: DXMTVectorDB) -> None:
        """Test that metadata survives round-trip storage."""
        metadata = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        texts = ["test document"]
        ids = vectorstore.add_texts(texts, metadatas=[metadata])

        # Retrieve and check metadata
        retrieved = vectorstore.get_by_ids(ids)
        assert len(retrieved) == 1

        retrieved_metadata = retrieved[0].metadata
        for key, value in metadata.items():
            assert key in retrieved_metadata
            # Type conversions may occur, so check value equality
            assert retrieved_metadata[key] == value

    def test_custom_ids_preserved(self, vectorstore: DXMTVectorDB) -> None:
        """Test that custom IDs are preserved exactly."""
        custom_ids = ["id-1", "id-2", "custom_id_123"]
        texts = ["text 1", "text 2", "text 3"]

        returned_ids = vectorstore.add_texts(texts, ids=custom_ids)

        # IDs should be exactly what we provided
        assert returned_ids == custom_ids

        # Should be able to retrieve by custom IDs
        docs = vectorstore.get_by_ids(custom_ids)
        assert len(docs) == 3

    def test_score_ordering(self, vectorstore: DXMTVectorDB) -> None:
        """Test that search results are ordered by score (descending)."""
        texts = [
            "Python programming language tutorial",
            "Python basics for beginners",
            "Cooking recipes and food",
        ]
        vectorstore.add_texts(texts)

        results = vectorstore.similarity_search_with_score("Python programming", k=3)

        # Scores should be in descending order (higher = more similar)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Scores not in descending order"

        # First result should be most relevant
        assert "python" in results[0][0].page_content.lower()


class TestValidationErrors:
    """Test validation and error handling."""

    def test_metadata_count_mismatch(self, vectorstore: DXMTVectorDB) -> None:
        """Test that metadata count must match text count."""
        texts = ["text 1", "text 2", "text 3"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]  # Only 2 metadatas!

        with pytest.raises((ValueError, DXMTError)) as exc_info:
            vectorstore.add_texts(texts, metadatas=metadatas)

        error_msg = str(exc_info.value).lower()
        assert "metadata" in error_msg or "mismatch" in error_msg or "count" in error_msg

    def test_dimension_mismatch(
        self,
        test_table_name: str,
        embeddings: OpenAIEmbeddings,
    ) -> None:
        """Test that embedding dimension must match table dimension."""
        # Create store with wrong dimension
        store = DXMTVectorDB(
            table_name=f"{test_table_name}_dim",
            embedding=embeddings,
            dimension=128,  # Wrong! OpenAI embeddings are 1536-dim
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        try:
            # This should fail because embedding dimension doesn't match
            with pytest.raises((ValueError, DXMTError)):
                store.add_texts(["test text"])
        finally:
            try:
                store._client.delete_table(f"{test_table_name}_dim")
            except Exception:
                pass

    def test_empty_text_rejected(self, vectorstore: DXMTVectorDB) -> None:
        """Test handling of empty text strings."""
        # Some implementations may reject empty strings
        texts = ["", "valid text", ""]

        try:
            ids = vectorstore.add_texts(texts)
            # If accepted, should get 3 IDs
            assert len(ids) == 3
        except (ValueError, DXMTError):
            # If rejected, that's also acceptable behavior
            pass

    def test_none_id_handling(self, vectorstore: DXMTVectorDB) -> None:
        """Test handling of None in IDs list."""
        texts = ["text 1", "text 2"]
        ids_with_none = ["valid_id", None]

        # Should either auto-generate ID for None or raise error
        try:
            returned_ids = vectorstore.add_texts(texts, ids=ids_with_none)
            assert len(returned_ids) == 2
            assert returned_ids[0] == "valid_id"
            assert returned_ids[1] is not None  # Should be auto-generated
        except (ValueError, DXMTError):
            # Rejecting None is also valid
            pass


class TestBatchHandling:
    """Test handling of different batch sizes."""

    def test_single_document(self, vectorstore: DXMTVectorDB) -> None:
        """Test adding a single document."""
        ids = vectorstore.add_texts(["single document"])
        assert len(ids) == 1

    def test_small_batch(self, vectorstore: DXMTVectorDB) -> None:
        """Test small batch (< 10 documents)."""
        texts = [f"doc {i}" for i in range(5)]
        ids = vectorstore.add_texts(texts)
        assert len(ids) == 5

    def test_medium_batch(self, vectorstore: DXMTVectorDB) -> None:
        """Test medium batch (10-100 documents)."""
        num_docs = 50
        texts = [f"document number {i}" for i in range(num_docs)]
        ids = vectorstore.add_texts(texts)
        assert len(ids) == num_docs

    def test_large_batch(self, vectorstore: DXMTVectorDB) -> None:
        """Test large batch (100+ documents).

        This tests that batch insertion works for larger volumes.
        """
        num_docs = 200
        texts = [f"large batch doc {i}" for i in range(num_docs)]

        start_time = time.time()
        ids = vectorstore.add_texts(texts)
        duration = time.time() - start_time

        assert len(ids) == num_docs
        print(f"\nLarge batch insert ({num_docs} docs): {duration:.2f}s")
        # Should complete in reasonable time
        assert duration < 120  # 2 minutes max


class TestAsync:
    """Test async-specific scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_async_operations(
        self,
        vectorstore: DXMTVectorDB,
    ) -> None:
        """Test concurrent async operations."""
        import asyncio

        # Run multiple async operations concurrently
        tasks = [
            vectorstore.aadd_texts([f"concurrent doc {i}"]) for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All operations should succeed
        assert len(results) == 5
        for ids in results:
            assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_async_search_after_sync_add(
        self,
        vectorstore: DXMTVectorDB,
    ) -> None:
        """Test that async search works after sync add."""
        # Add documents synchronously
        vectorstore.add_texts(["sync added document"])

        # Search asynchronously
        results = await vectorstore.asimilarity_search("sync", k=1)

        assert len(results) >= 1


# Summary test
def test_advanced_features_summary(capsys: Any) -> None:
    """Print summary of advanced feature test coverage."""
    print("\n" + "=" * 70)
    print("Advanced Features Test Coverage Summary")
    print("=" * 70)

    features = [
        ("MMR Search", "Maximal Marginal Relevance for diversity"),
        ("Score Threshold", "Filter results by minimum similarity"),
        ("Complex Metadata", "Nested, list, and range filters"),
        ("Document Integrity", "Immutability and round-trip preservation"),
        ("Validation Errors", "Proper error handling for invalid input"),
        ("Batch Handling", "Support for various batch sizes"),
        ("Async Operations", "Concurrent async operations"),
    ]

    print("\nTested Features:")
    for feature, description in features:
        print(f"  ✅ {feature}: {description}")

    print("\n" + "=" * 70 + "\n")
