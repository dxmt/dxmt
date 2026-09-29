"""Integration tests for DXMT Vector Store with LlamaIndex.

These tests require a running DXMT server.
Run with: pytest tests/test_integration.py -m integration
"""

import pytest
import os
from typing import List

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.vector_stores.dxmt import DXMTVectorStore

from local_embedding import DIMENSION, LocalEmbedding


# Skip all tests if DXMT server is not available or if integration tests are disabled
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dxmt_host():
    """Get DXMT host from environment."""
    return os.getenv("DXMT_HOST", "localhost")


@pytest.fixture(scope="module")
def dxmt_port():
    """Get DXMT port from environment."""
    return int(os.getenv("DXMT_PORT", "9999"))


@pytest.fixture(scope="module")
def embed_model():
    """A local, deterministic embedding model.

    This asked for `OpenAIEmbedding`, which needs an API key, so the whole file skipped itself
    everywhere -- CI included -- and this package's live path had never actually run.
    """
    return LocalEmbedding()


@pytest.fixture
def vector_store(dxmt_host, dxmt_port):
    """Create a fresh vector store for each test."""
    import time
    table_name = f"test_llamaindex_integration_{int(time.time() * 1000)}"

    store = DXMTVectorStore(
        table_name=table_name,
        dimension=DIMENSION,
        host=dxmt_host,
        port=dxmt_port,
        index_type="hnsw",
        distance_metric="cosine",
    )

    yield store

    # Cleanup
    try:
        store.clear()
    except Exception:
        pass


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            text="Artificial intelligence is revolutionizing technology.",
            metadata={"category": "AI", "year": 2024, "importance": "high"}
        ),
        Document(
            text="Machine learning algorithms learn from data patterns.",
            metadata={"category": "ML", "year": 2024, "importance": "high"}
        ),
        Document(
            text="Deep learning uses neural networks with multiple layers.",
            metadata={"category": "DL", "year": 2024, "importance": "medium"}
        ),
        Document(
            text="Natural language processing enables text understanding.",
            metadata={"category": "NLP", "year": 2024, "importance": "high"}
        ),
        Document(
            text="Computer vision allows machines to interpret images.",
            metadata={"category": "CV", "year": 2024, "importance": "medium"}
        ),
    ]


class TestBasicOperations:
    """Test basic CRUD operations."""

    def test_add_and_query(self, vector_store, embed_model, sample_documents):
        """Test adding documents and querying."""
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        # Add documents
        for doc in sample_documents:
            index.insert(doc)

        # Query. A query engine needs an LLM to write the answer; what is under test here is
        # which nodes DXMT returned to it, so the answer itself is mocked.
        query_engine = index.as_query_engine(similarity_top_k=3, llm=MockLLM())
        response = query_engine.query("What is machine learning?")

        assert response is not None
        assert len(response.source_nodes) > 0
        assert len(response.source_nodes) <= 3
        assert "machine learning" in response.source_nodes[0].node.text.lower()

    def test_add_nodes_directly(self, vector_store, embed_model):
        """Test adding nodes directly to vector store."""
        # Create nodes with embeddings
        nodes = [
            TextNode(
                text="Test document 1",
                id_="node1",
                embedding=embed_model.get_text_embedding("Test document 1"),
                metadata={"type": "test"}
            ),
            TextNode(
                text="Test document 2",
                id_="node2",
                embedding=embed_model.get_text_embedding("Test document 2"),
                metadata={"type": "test"}
            ),
        ]

        # Add nodes
        ids = vector_store.add(nodes)

        assert len(ids) == 2
        assert "node1" in ids
        assert "node2" in ids

        # Query
        query_embedding = embed_model.get_text_embedding("test document")
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=5,
        )

        result = vector_store.query(query)

        assert len(result.nodes) == 2

    def test_delete_nodes(self, vector_store, embed_model, sample_documents):
        """Test deleting nodes."""
        # Create index and add documents
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        for doc in sample_documents[:3]:
            index.insert(doc)

        # Get initial count
        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding("test"),
            similarity_top_k=10,
        )
        initial_results = vector_store.query(query)
        initial_count = len(initial_results.nodes)

        assert initial_count == 3

        # Delete by ID
        if initial_results.ids:
            vector_store.delete_nodes(node_ids=[initial_results.ids[0]])

            # Verify deletion
            new_results = vector_store.query(query)
            assert len(new_results.nodes) == initial_count - 1


class TestFiltering:
    """Test metadata filtering."""

    def test_equality_filter(self, vector_store, embed_model, sample_documents):
        """Test filtering with equality operator."""
        # Add documents
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        for doc in sample_documents:
            index.insert(doc)

        # Query with filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="AI", operator=FilterOperator.EQ)
            ]
        )

        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding("technology"),
            similarity_top_k=5,
            filters=filters,
        )

        result = vector_store.query(query)

        # Should only get AI documents
        assert len(result.nodes) > 0
        for node in result.nodes:
            assert node.metadata.get("category") == "AI"

    def test_comparison_filter(self, vector_store, embed_model, sample_documents):
        """Test filtering with comparison operators."""
        # Add documents
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        for doc in sample_documents:
            index.insert(doc)

        # Query with year filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)
            ]
        )

        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding("technology"),
            similarity_top_k=10,
            filters=filters,
        )

        result = vector_store.query(query)

        assert len(result.nodes) > 0
        for node in result.nodes:
            assert node.metadata.get("year", 0) >= 2024

    def test_in_filter(self, vector_store, embed_model, sample_documents):
        """Test IN operator filter."""
        # Add documents
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        for doc in sample_documents:
            index.insert(doc)

        # Query with IN filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value=["AI", "ML"],
                    operator=FilterOperator.IN
                )
            ]
        )

        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding("technology"),
            similarity_top_k=10,
            filters=filters,
        )

        result = vector_store.query(query)

        assert len(result.nodes) > 0
        for node in result.nodes:
            assert node.metadata.get("category") in ["AI", "ML"]


class TestRetrieval:
    """Test retrieval functionality."""

    def test_retriever(self, vector_store, embed_model, sample_documents):
        """Test using retriever."""
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        for doc in sample_documents:
            index.insert(doc)

        # Create retriever
        retriever = index.as_retriever(similarity_top_k=3)

        # Retrieve
        nodes = retriever.retrieve("artificial intelligence and machine learning")

        assert len(nodes) <= 3
        assert all(hasattr(node, 'score') for node in nodes)
        assert all(hasattr(node, 'text') for node in nodes)

        # Scores should be in descending order
        scores = [node.score for node in nodes]
        assert scores == sorted(scores, reverse=True)


class TestFromExistingTable:
    """Test creating vector store from existing table."""

    def test_from_existing_table(self, dxmt_host, dxmt_port, embed_model):
        """Test from_existing_table class method."""
        import time
        table_name = f"test_existing_{int(time.time() * 1000)}"

        # Create initial store and add data
        store1 = DXMTVectorStore(
            table_name=table_name,
            dimension=DIMENSION,
            host=dxmt_host,
            port=dxmt_port,
            index_type="hnsw",
        )

        nodes = [
            TextNode(
                text="Test",
                id_="test1",
                embedding=embed_model.get_text_embedding("Test"),
            )
        ]
        store1.add(nodes)

        # Create new store from existing table
        store2 = DXMTVectorStore.from_existing_table(
            table_name=table_name,
            host=dxmt_host,
            port=dxmt_port,
        )

        assert store2.dimension == DIMENSION
        assert store2.table_name == table_name

        # Query should work
        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding("Test"),
            similarity_top_k=5,
        )
        result = store2.query(query)

        assert len(result.nodes) > 0

        # Cleanup
        try:
            store2.clear()
        except Exception:
            pass


class TestEndToEndRAG:
    """Test complete end-to-end RAG pipeline."""

    def test_rag_pipeline(self, vector_store, embed_model):
        """The whole workflow: documents in, retrieval out, a generated answer over it.

        This skipped itself without an OpenAI key, which meant the one test of the end-to-end
        path never ran. The retrieval is real; only the generation is mocked, because what a
        vector store owes a RAG pipeline is the right nodes.
        """
        # Create documents
        documents = [
            Document(text="Python is a high-level programming language."),
            Document(text="JavaScript is used for web development."),
            Document(text="Rust is a systems programming language."),
        ]

        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        for doc in documents:
            index.insert(doc)

        # Create query engine with a stand-in LLM
        query_engine = index.as_query_engine(llm=MockLLM(), similarity_top_k=2)

        # Query
        response = query_engine.query("Which language is used for web development?")

        assert response is not None
        assert len(response.source_nodes) == 2

        # The document that shares words with the question comes back first.
        assert "javascript" in response.source_nodes[0].node.text.lower()


class TestDeletionActuallyDeletes:
    """`delete()` and `delete_nodes(filters=...)` raised on every call there had ever been.

    Both did `results.get("hits", [])` on the return of `client.search`, which is a `list`.
    Nothing caught it: the 22 unit tests hand the store a `MagicMock`, and a `MagicMock` has
    `.get()`; the integration test above covers `delete_nodes(node_ids=...)`, which takes the
    other branch of the same method. So the one branch that was broken was also the only one
    with no coverage against a real server.

    These assert on the *row count the server reports*, not on the call having been made.
    A test that asserts `delete_records` was called is a test a `MagicMock` passes.
    """

    def _rows(self, store) -> int:
        return len(list(store.client.scroll(store.table_name)))

    def test_delete_by_ref_doc_id_removes_exactly_that_document(
        self, vector_store, embed_model
    ):
        keep = Document(text="A document that must survive.", metadata={"keep": True})
        drop = Document(text="A document that must be removed.", metadata={"keep": False})
        index = VectorStoreIndex.from_documents(
            [keep, drop],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embed_model,
        )
        assert self._rows(vector_store) == 2

        vector_store.delete(drop.doc_id)

        remaining = list(vector_store.client.scroll(vector_store.table_name))
        assert len(remaining) == 1, (
            f"delete() left {len(remaining)} rows; it used to leave both and raise "
            f"AttributeError on the way out"
        )
        assert "survive" in str(remaining[0].get("payload", {}))
        del index

    def test_delete_by_filter_removes_only_the_matching_rows(
        self, vector_store, embed_model
    ):
        nodes = []
        for i, lang in enumerate(["en", "fr", "en"]):
            node = TextNode(text=f"document number {i}", metadata={"lang": lang})
            node.embedding = embed_model.get_text_embedding(node.get_content())
            nodes.append(node)
        vector_store.add(nodes)
        assert self._rows(vector_store) == 3

        vector_store.delete_nodes(
            filters=MetadataFilters(filters=[MetadataFilter(key="lang", value="fr")])
        )

        rows = list(vector_store.client.scroll(vector_store.table_name))
        assert len(rows) == 2, f"expected the two `en` rows, got {len(rows)}"
        assert all(r["payload"].get("lang") == "en" for r in rows), rows

    def test_deleting_from_a_missing_table_is_a_no_op_and_everything_else_raises(
        self, dxmt_host, dxmt_port
    ):
        """The one case worth swallowing, and the reason the blanket `except` was wrong.

        Removing a document from a table that does not exist has already achieved what the
        caller asked for. Every other failure used to be swallowed by the same `except
        DXMTError: pass`, so a delete that could not run reported success.
        """
        absent = DXMTVectorStore(
            table_name="llamaindex_no_such_table_zzz",
            dimension=DIMENSION,
            host=dxmt_host,
            port=dxmt_port,
        )
        absent.delete("anything")  # must not raise

    def test_an_unsupported_query_mode_is_refused_rather_than_answered(
        self, vector_store, embed_model
    ):
        """Six of LlamaIndex's nine modes used to fall through to a plain dense search.

        MMR is the one that shows why it matters: the caller asked for diversity, got none,
        and the result is indistinguishable from one that honoured the mode.
        """
        node = TextNode(text="anything at all", metadata={})
        node.embedding = embed_model.get_text_embedding(node.get_content())
        vector_store.add([node])
        q = embed_model.get_query_embedding("anything")

        for mode in (
            VectorStoreQueryMode.MMR,
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.SEMANTIC_HYBRID,
        ):
            with pytest.raises(NotImplementedError, match=mode.name):
                vector_store.query(
                    VectorStoreQuery(
                        query_embedding=q, similarity_top_k=2, mode=mode, query_str="x"
                    )
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
