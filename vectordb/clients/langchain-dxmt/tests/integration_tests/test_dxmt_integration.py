"""Integration tests for DXMT VectorStore.

These tests require a running DXMT server instance.
Run with: pytest tests/integration_tests/ -v

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment (OPENAI_API_KEY)
"""

import os
import re
import time
from typing import Any

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

from langchain_dxmt import DXMTVectorDB, DXMTClient, DXMTError, DXMTConnectionError

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



# Check if we can run integration tests
DXMT_HOST = os.getenv("DXMT_HOST", "localhost")
DXMT_PORT = int(os.getenv("DXMT_PORT", "9999"))



@pytest.fixture
def embeddings() -> Embeddings:
    """Deterministic by default; a real model when DXMT_TEST_EMBEDDING=openai."""
    return make_embeddings()


@pytest.fixture
def test_table_name() -> str:
    """Generate unique test table name."""
    timestamp = int(time.time() * 1000)
    return f"test_integration_{timestamp}"


@pytest.fixture
def vector_store(
    test_table_name: str,
    embeddings: Embeddings,
) -> DXMTVectorDB:
    """Create vector store for testing."""
    store = DXMTVectorDB(
        table_name=test_table_name,
        embedding=embeddings,
        dimension=embedding_dimension(),
        host=DXMT_HOST,
        port=DXMT_PORT,
        index_type="hnsw",
        distance_metric="cosine",
    )

    yield store

    # Cleanup: delete table after test
    try:
        store._client.delete_table(test_table_name)
    except Exception:
        pass


@pytest.fixture
def vector_store_with_text(
    test_table_name: str,
    embeddings: Embeddings,
) -> DXMTVectorDB:
    """Create vector store with full-text search enabled."""
    table_name = f"{test_table_name}_text"
    store = DXMTVectorDB(
        table_name=table_name,
        embedding=embeddings,
        dimension=embedding_dimension(),
        host=DXMT_HOST,
        port=DXMT_PORT,
        index_type="hnsw",
        distance_metric="cosine",
        enable_full_text_search=True,
    )

    yield store

    # Cleanup
    try:
        store._client.delete_table(table_name)
    except Exception:
        pass


class TestBasicOperations:
    """Test basic vector store operations."""

    def test_add_texts(self, vector_store: DXMTVectorDB) -> None:
        """Test adding texts to vector store."""
        texts = [
            "LangChain is a framework for developing applications with LLMs.",
            "Vector databases store embeddings for similarity search.",
            "DXMT is a lightweight vector database written in Rust.",
        ]

        ids = vector_store.add_texts(texts)

        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)

    def test_add_documents(self, vector_store: DXMTVectorDB) -> None:
        """Test adding documents with metadata."""
        documents = [
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"source": "ml_guide", "page": 1},
            ),
            Document(
                page_content="Deep learning uses neural networks.",
                metadata={"source": "dl_guide", "page": 2},
            ),
        ]

        ids = vector_store.add_documents(documents)

        assert len(ids) == 2

    def test_similarity_search(self, vector_store: DXMTVectorDB) -> None:
        """Test similarity search."""
        # Add documents
        texts = [
            "Python is a programming language.",
            "JavaScript is used for web development.",
            "Rust is a systems programming language.",
            "Vector databases enable semantic search.",
        ]
        vector_store.add_texts(texts)

        # Search
        results = vector_store.similarity_search(
            "What is Python?",
            k=2,
        )

        assert len(results) == 2
        assert isinstance(results[0], Document)
        # Python-related document should be first
        assert "Python" in results[0].page_content

    def test_similarity_search_with_score(
        self,
        vector_store: DXMTVectorDB,
    ) -> None:
        """Test similarity search with scores."""
        texts = [
            "Natural language processing with transformers.",
            "Computer vision with convolutional networks.",
            "Reinforcement learning with policy gradients.",
        ]
        vector_store.add_texts(texts)

        results = vector_store.similarity_search_with_score(
            "NLP and language models",
            k=3,
        )

        assert len(results) == 3
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

        # Scores should be in descending order (higher is more similar)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_search_by_vector(
        self,
        vector_store: DXMTVectorDB,
    ) -> None:
        """Test search by embedding vector."""
        texts = ["Document 1", "Document 2", "Document 3"]
        vector_store.add_texts(texts)

        # Get embedding for query
        query_embedding = vector_store._embedding.embed_query("Document 1")

        # Search by vector
        results = vector_store.similarity_search_by_vector(
            query_embedding,
            k=2,
        )

        assert len(results) == 2

    def test_get_by_ids(self, vector_store: DXMTVectorDB) -> None:
        """Test retrieving documents by IDs."""
        texts = ["Text A", "Text B", "Text C"]
        ids = vector_store.add_texts(texts, ids=["id_a", "id_b", "id_c"])

        # Retrieve subset
        docs = vector_store.get_by_ids(["id_a", "id_c"])

        assert len(docs) == 2
        contents = [doc.page_content for doc in docs]
        assert "Text A" in contents
        assert "Text C" in contents

    def test_delete(self, vector_store: DXMTVectorDB) -> None:
        """Test deleting documents."""
        texts = ["Delete me", "Keep me", "Delete me too"]
        ids = vector_store.add_texts(
            texts,
            ids=["del1", "keep1", "del2"],
        )

        # Delete two documents
        success = vector_store.delete(ids=["del1", "del2"])
        assert success

        # Verify only one document remains
        remaining = vector_store.get_by_ids(["keep1"])
        assert len(remaining) == 1
        assert remaining[0].page_content == "Keep me"


class TestAdvancedFeatures:
    """Test advanced features."""

    def test_metadata_filtering(self, vector_store: DXMTVectorDB) -> None:
        """Test search with metadata filters."""
        documents = [
            Document(
                page_content="Python tutorial",
                metadata={"language": "python", "level": "beginner"},
            ),
            Document(
                page_content="Advanced Python",
                metadata={"language": "python", "level": "advanced"},
            ),
            Document(
                page_content="JavaScript basics",
                metadata={"language": "javascript", "level": "beginner"},
            ),
        ]
        vector_store.add_documents(documents)

        # Filter by language
        results = vector_store.similarity_search(
            "programming tutorial",
            k=5,
            filter={"language": "python"},
        )

        assert len(results) <= 2
        for doc in results:
            assert doc.metadata["language"] == "python"

    @needs_text_bm25
    def test_hybrid_search(
        self,
        vector_store_with_text: DXMTVectorDB,
    ) -> None:
        """Test hybrid search (vector + text)."""
        documents = [
            Document(
                page_content="Machine learning models learn patterns from data.",
                metadata={"topic": "ml"},
            ),
            Document(
                page_content="HNSW algorithm for approximate nearest neighbor search.",
                metadata={"topic": "algorithms"},
            ),
            Document(
                page_content="Vector embeddings represent semantic meaning.",
                metadata={"topic": "embeddings"},
            ),
        ]
        vector_store_with_text.add_documents(documents)

        # Hybrid search
        results = vector_store_with_text.hybrid_search(
            "machine learning",
            k=3,
            vector_weight=0.6,
        )

        assert len(results) <= 3
        assert isinstance(results[0], Document)

    def test_different_index_types(
        self,
        test_table_name: str,
        embeddings: Embeddings,
    ) -> None:
        """Test different index configurations."""
        index_types = ["hnsw", "ivf", "brute_force"]

        for index_type in index_types:
            table_name = f"{test_table_name}_{index_type}"

            store = DXMTVectorDB(
                table_name=table_name,
                embedding=embeddings,
                dimension=embedding_dimension(),
                host=DXMT_HOST,
                port=DXMT_PORT,
                index_type=index_type,
            )

            # Add and search
            store.add_texts(["Test document"])
            results = store.similarity_search("Test query", k=1)

            assert len(results) >= 0

            # Cleanup
            try:
                store._client.delete_table(table_name)
            except Exception:
                pass

    def test_different_distance_metrics(
        self,
        test_table_name: str,
        embeddings: Embeddings,
    ) -> None:
        """Test different distance metrics."""
        metrics = ["cosine", "euclidean", "dot_product"]

        for metric in metrics:
            table_name = f"{test_table_name}_{metric}"

            store = DXMTVectorDB(
                table_name=table_name,
                embedding=embeddings,
                dimension=embedding_dimension(),
                host=DXMT_HOST,
                port=DXMT_PORT,
                distance_metric=metric,
            )

            # Add and search
            store.add_texts(["Test document for distance metrics"])
            results = store.similarity_search("Test query", k=1)

            assert len(results) >= 0

            # Cleanup
            try:
                store._client.delete_table(table_name)
            except Exception:
                pass


class TestAsyncOperations:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_aadd_texts(
        self,
        vector_store: DXMTVectorDB,
    ) -> None:
        """Test async add texts."""
        texts = ["Async text 1", "Async text 2"]

        ids = await vector_store.aadd_texts(texts)

        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_asimilarity_search(
        self,
        vector_store: DXMTVectorDB,
    ) -> None:
        """Test async similarity search."""
        # Add documents
        await vector_store.aadd_texts(["Async document for search"])

        # Search
        results = await vector_store.asimilarity_search("async", k=1)

        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_adelete(
        self,
        vector_store: DXMTVectorDB,
    ) -> None:
        """Test async delete."""
        ids = await vector_store.aadd_texts(
            ["Delete async"],
            ids=["async_del1"],
        )

        success = await vector_store.adelete(ids=["async_del1"])

        assert success


class TestLangChainIntegration:
    """Test LangChain integration features."""

    def test_as_retriever(self, vector_store: DXMTVectorDB) -> None:
        """Test using vector store as retriever."""
        # Add documents
        documents = [
            Document(page_content="LangChain framework"),
            Document(page_content="Vector database"),
            Document(page_content="Similarity search"),
        ]
        vector_store.add_documents(documents)

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        )

        # Retrieve
        results = retriever.invoke("LangChain")

        assert len(results) == 2
        assert isinstance(results[0], Document)

    def test_from_texts_factory(
        self,
        test_table_name: str,
        embeddings: Embeddings,
    ) -> None:
        """Test creating store from texts."""
        texts = ["Factory text 1", "Factory text 2"]
        metadatas = [{"source": "factory"}, {"source": "factory"}]

        table_name = f"{test_table_name}_factory"

        store = DXMTVectorDB.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            table_name=table_name,
            dimension=embedding_dimension(),
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        # Verify documents were added
        results = store.similarity_search("factory", k=2)
        assert len(results) == 2

        # Cleanup
        try:
            store._client.delete_table(table_name)
        except Exception:
            pass

    def test_from_documents_factory(
        self,
        test_table_name: str,
        embeddings: Embeddings,
    ) -> None:
        """Test creating store from documents."""
        documents = [
            Document(
                page_content="Document factory test",
                metadata={"type": "test"},
            ),
        ]

        table_name = f"{test_table_name}_doc_factory"

        store = DXMTVectorDB.from_documents(
            documents=documents,
            embedding=embeddings,
            table_name=table_name,
            dimension=embedding_dimension(),
            host=DXMT_HOST,
            port=DXMT_PORT,
        )

        # Verify
        results = store.similarity_search("factory", k=1)
        assert len(results) >= 0

        # Cleanup
        try:
            store._client.delete_table(table_name)
        except Exception:
            pass


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_table_name(self, embeddings: Embeddings) -> None:
        """Test creating store with invalid table name."""
        # Empty table name should be handled gracefully
        # (either use default or raise clear error)
        try:
            store = DXMTVectorDB(
                table_name="",
                embedding=embeddings,
                dimension=embedding_dimension(),
                host=DXMT_HOST,
                port=DXMT_PORT,
            )
            # If no error, should use default table name
            assert store._table_name != ""
        except ValueError as e:
            # Clear error message
            assert "table_name" in str(e).lower()

    def test_missing_dimension(
        self,
        test_table_name: str,
        embeddings: Embeddings,
    ) -> None:
        """Test creating store without dimension."""
        with pytest.raises(ValueError) as exc_info:
            DXMTVectorDB(
                table_name=test_table_name,
                embedding=embeddings,
                dimension=None,  # Missing dimension
                host=DXMT_HOST,
                port=DXMT_PORT,
                create_table_if_not_exists=True,
            )

        assert "dimension" in str(exc_info.value).lower()

    def test_connection_error(
        self,
        test_table_name: str,
        embeddings: Embeddings,
    ) -> None:
        """An unreachable server says so, and does not say the table is missing.

        This test could not pass as written: with `create_table_if_not_exists=False` the
        constructor verifies the table, so the failure happens there, outside the
        `pytest.raises` block that expected it on `add_texts`. It failed before this change
        too -- with `ValueError: Table ... does not exist: [Errno -3] Temporary failure in
        name resolution`, which was the defect: a blanket `except Exception` turned DNS
        failures, refused connections and timeouts into "your table does not exist", sending
        whoever mistyped a host to look for a table that is sitting there fine.

        Now the constructor fails fast with the error that actually happened.
        """
        with pytest.raises(DXMTConnectionError):
            DXMTVectorDB(
                table_name=test_table_name,
                embedding=embeddings,
                dimension=embedding_dimension(),
                host="invalid-host-12345",
                port=99999,
                create_table_if_not_exists=False,
            )


class TestPerformance:
    """Test performance characteristics."""

    def test_batch_insert_performance(
        self,
        vector_store: DXMTVectorDB,
    ) -> None:
        """Test batch insert performance."""
        # Create large batch
        num_docs = 100
        texts = [f"Document {i} with some content" for i in range(num_docs)]

        start_time = time.time()
        ids = vector_store.add_texts(texts)
        end_time = time.time()

        assert len(ids) == num_docs

        # Should complete in reasonable time
        duration = end_time - start_time
        print(f"\nBatch insert {num_docs} docs: {duration:.2f}s")
        assert duration < 60  # Should be under 1 minute

    def test_search_performance(
        self,
        vector_store: DXMTVectorDB,
    ) -> None:
        """Test search performance."""
        # Add documents
        num_docs = 50
        texts = [f"Performance test document {i}" for i in range(num_docs)]
        vector_store.add_texts(texts)

        # Run multiple searches
        num_searches = 10
        start_time = time.time()

        for i in range(num_searches):
            results = vector_store.similarity_search(
                f"query {i}",
                k=5,
            )
            assert len(results) <= 5

        end_time = time.time()
        duration = end_time - start_time
        avg_latency = duration / num_searches

        print(f"\nAverage search latency: {avg_latency*1000:.1f}ms")
        assert avg_latency < 1.0  # Should be under 1 second per search
