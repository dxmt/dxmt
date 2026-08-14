"""Integration tests for DXMT Vector Store with LlamaIndex.

These tests require a running DXMT server.
Run with: pytest tests/test_integration.py -m integration
"""

import pytest
import os
from typing import List

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.dxmt import DXMTVectorStore


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
    """Create embedding model."""
    return OpenAIEmbedding(model="text-embedding-ada-002")


@pytest.fixture
def vector_store(dxmt_host, dxmt_port):
    """Create a fresh vector store for each test."""
    import time
    table_name = f"test_llamaindex_integration_{int(time.time() * 1000)}"

    store = DXMTVectorStore(
        table_name=table_name,
        dimension=1536,
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

        # Query
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("What is machine learning?")

        assert response is not None
        assert len(response.source_nodes) > 0
        assert len(response.source_nodes) <= 3

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
            dimension=1536,
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

        assert store2.dimension == 1536
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
        """Test complete RAG workflow."""
        # Skip if no OpenAI API key (LLM required for this test)
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key required for RAG test")

        from llama_index.llms.openai import OpenAI

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

        # Create query engine with LLM
        query_engine = index.as_query_engine(
            llm=OpenAI(model="gpt-3.5-turbo"),
            similarity_top_k=2,
        )

        # Query
        response = query_engine.query("What programming languages are mentioned?")

        assert response is not None
        assert len(response.source_nodes) > 0

        # Response should mention at least one language
        response_text = str(response).lower()
        assert any(lang in response_text for lang in ["python", "javascript", "rust"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
