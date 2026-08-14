"""Unit tests for DXMT CrewAI integration.

Tests both the Tool and Storage integrations with CrewAI.
"""

from unittest.mock import Mock, patch, MagicMock
import pytest

# Try to import CrewAI, skip tests if not available
try:
    from crewai.tools import BaseTool
    from crewai.memory.storage.interface import Storage
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="CrewAI not installed")

from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig, DXMTToolSchema
from dxmt.crewai_storage import DXMTStorage, DXMTStorageConfig
from dxmt.exceptions import DXMTError


# ============================================================================
# Tool Integration Tests
# ============================================================================

@pytest.fixture
def mock_dxmt_client_for_tool():
    """Create a mock DXMT client for tool tests."""
    client = Mock()

    # Mock table operations
    client.create_table.return_value = None
    client.get_table.return_value = {"name": "test_table", "dimension": 384}

    # Mock search operation
    client.search.return_value = [
        {
            "id": "doc1",
            "score": 0.95,
            "metadata": {"text": "Test document 1", "source": "test"}
        },
        {
            "id": "doc2",
            "score": 0.85,
            "metadata": {"text": "Test document 2", "source": "test"}
        }
    ]

    # Mock insert operation
    client.insert.return_value = None

    return client


@pytest.fixture
def dxmt_tool_config():
    """Create DXMT tool configuration."""
    return DXMTToolConfig(
        host="localhost",
        port=9999,
        table_name="test_table",
        limit=5,
        score_threshold=0.35
    )


@pytest.fixture
def dxmt_search_tool(mock_dxmt_client_for_tool, dxmt_tool_config):
    """Create DXMT search tool with mocked client."""
    tool = DXMTVectorSearchTool(
        dxmt_config=dxmt_tool_config,
        client=mock_dxmt_client_for_tool
    )
    return tool


class TestDXMTVectorSearchTool:
    """Test DXMT Vector Search Tool."""

    def test_tool_initialization(self, dxmt_tool_config, mock_dxmt_client_for_tool):
        """Test tool initializes correctly."""
        tool = DXMTVectorSearchTool(
            dxmt_config=dxmt_tool_config,
            client=mock_dxmt_client_for_tool
        )

        assert tool.name == "DXMTVectorSearchTool"
        assert tool.dxmt_config.table_name == "test_table"
        assert tool.dxmt_config.limit == 5
        assert tool.client == mock_dxmt_client_for_tool

    def test_tool_schema(self):
        """Test tool input schema."""
        schema = DXMTToolSchema(query="test query")
        assert schema.query == "test query"
        assert schema.filter_by is None
        assert schema.filter_value is None

        schema_with_filter = DXMTToolSchema(
            query="test",
            filter_by="category",
            filter_value="tech"
        )
        assert schema_with_filter.filter_by == "category"
        assert schema_with_filter.filter_value == "tech"

    @patch('dxmt.crewai_tool.DXMTVectorSearchTool._get_openai_embedding')
    def test_run_basic_search(self, mock_embed, dxmt_search_tool):
        """Test basic search execution."""
        # Mock embedding
        mock_embed.return_value = [0.1] * 3072

        # Run search
        result = dxmt_search_tool._run(query="test query")

        # Verify result is JSON string
        assert isinstance(result, str)
        import json
        result_data = json.loads(result)

        # Check result structure
        assert isinstance(result_data, list)
        assert len(result_data) == 2  # Two mock results
        assert "distance" in result_data[0]
        assert "score" in result_data[0]
        assert "metadata" in result_data[0]
        assert "context" in result_data[0]

    @patch('dxmt.crewai_tool.DXMTVectorSearchTool._get_openai_embedding')
    def test_run_with_filter(self, mock_embed, dxmt_search_tool):
        """Test search with metadata filter."""
        mock_embed.return_value = [0.1] * 3072

        # Run search with filter
        result = dxmt_search_tool._run(
            query="test query",
            filter_by="source",
            filter_value="wiki"
        )

        # Verify client.search was called with filter
        call_args = dxmt_search_tool.client.search.call_args
        assert call_args[1]["filter"] is not None
        filter_arg = call_args[1]["filter"]
        assert "must" in filter_arg
        assert filter_arg["must"][0]["key"] == "source"

    @patch('dxmt.crewai_tool.DXMTVectorSearchTool._get_openai_embedding')
    def test_score_threshold_filtering(self, mock_embed, dxmt_search_tool):
        """Test that results are filtered by score threshold."""
        mock_embed.return_value = [0.1] * 3072

        # Set high threshold
        dxmt_search_tool.dxmt_config.score_threshold = 0.9

        # Mock search returns one high score, one low score
        dxmt_search_tool.client.search.return_value = [
            {"id": "doc1", "score": 0.95, "metadata": {"text": "High score"}},
            {"id": "doc2", "score": 0.5, "metadata": {"text": "Low score"}},
        ]

        result = dxmt_search_tool._run(query="test")
        import json
        result_data = json.loads(result)

        # Only high score result should be returned
        assert len(result_data) == 1
        assert result_data[0]["score"] == 0.95

    def test_custom_embedding_function(self, dxmt_tool_config, mock_dxmt_client_for_tool):
        """Test tool with custom embedding function."""
        def custom_embed(text):
            return [0.5] * 3072

        tool = DXMTVectorSearchTool(
            dxmt_config=dxmt_tool_config,
            client=mock_dxmt_client_for_tool,
            custom_embedding_fn=custom_embed
        )

        assert tool.custom_embedding_fn is not None

        # Run search (should use custom embedding)
        result = tool._run(query="test")
        assert isinstance(result, str)

    def test_add_documents(self, dxmt_search_tool):
        """Test adding documents to DXMT."""
        texts = ["Document 1", "Document 2"]
        metadatas = [{"id": 1}, {"id": 2}]

        with patch.object(dxmt_search_tool, '_batch_embed_openai') as mock_batch:
            mock_batch.return_value = [[0.1] * 3072, [0.2] * 3072]

            ids = dxmt_search_tool.add_documents(texts=texts, metadatas=metadatas)

            # Verify insert was called
            assert dxmt_search_tool.client.insert.called
            assert len(ids) == 2


# ============================================================================
# Storage Integration Tests
# ============================================================================

@pytest.fixture
def mock_dxmt_client_for_storage():
    """Create a mock DXMT client for storage tests."""
    client = Mock()

    # Mock table operations
    client.create_table.return_value = None
    client.get_table.side_effect = DXMTError("Table not found")
    client.delete_table.return_value = None

    # Mock insert operation
    client.insert.return_value = None

    # Mock search operation
    client.search.return_value = [
        {
            "id": "mem1",
            "score": 0.9,
            "metadata": {
                "type": "long_term",
                "content": "User prefers Python",
                "user_id": "user_123"
            }
        }
    ]

    return client


@pytest.fixture
def dxmt_storage_config():
    """Create DXMT storage configuration."""
    return {
        "host": "localhost",
        "port": 9999,
        "table_name": "test_memory",
        "user_id": "user_123",
        "agent_id": "test_agent",
        "dimension": 3072
    }


@pytest.fixture
def dxmt_storage(mock_dxmt_client_for_storage, dxmt_storage_config):
    """Create DXMT storage with mocked client."""
    with patch('dxmt.crewai_storage.DXMTClient', return_value=mock_dxmt_client_for_storage):
        storage = DXMTStorage(
            type="long_term",
            config=dxmt_storage_config
        )
        storage.client = mock_dxmt_client_for_storage
        return storage


class TestDXMTStorage:
    """Test DXMT Storage implementation."""

    def test_storage_initialization(self, dxmt_storage_config):
        """Test storage initializes correctly."""
        with patch('dxmt.crewai_storage.DXMTClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_table.side_effect = DXMTError("Not found")
            mock_client_class.return_value = mock_client

            storage = DXMTStorage(
                type="long_term",
                config=dxmt_storage_config
            )

            assert storage.memory_type == "long_term"
            assert storage.storage_config.table_name == "test_memory"
            assert storage.storage_config.user_id == "user_123"

            # Verify table creation was attempted
            assert mock_client.create_table.called

    def test_invalid_storage_type(self, dxmt_storage_config):
        """Test that invalid storage type raises error."""
        with pytest.raises(ValueError, match="Invalid type"):
            DXMTStorage(type="invalid_type", config=dxmt_storage_config)

    @patch('dxmt.crewai_storage.DXMTStorage._get_embedding')
    def test_save_memory(self, mock_embed, dxmt_storage):
        """Test saving memory."""
        mock_embed.return_value = [0.1] * 3072

        # Save memory
        dxmt_storage.save(
            value="User prefers Python",
            metadata={"source": "conversation"}
        )

        # Verify insert was called
        assert dxmt_storage.client.insert.called
        call_args = dxmt_storage.client.insert.call_args
        records = call_args[1]["records"]

        assert len(records) == 1
        assert records[0]["metadata"]["type"] == "long_term"
        assert records[0]["metadata"]["user_id"] == "user_123"

    @patch('dxmt.crewai_storage.DXMTStorage._get_embedding')
    def test_save_with_messages(self, mock_embed, dxmt_storage):
        """Test saving memory with conversation messages."""
        mock_embed.return_value = [0.1] * 3072

        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Final Answer: Python is a programming language"}
        ]

        dxmt_storage.save(
            value="conversation",
            metadata={"messages": messages}
        )

        # Verify insert was called with extracted content
        assert dxmt_storage.client.insert.called

    @patch('dxmt.crewai_storage.DXMTStorage._get_embedding')
    def test_search_memory(self, mock_embed, dxmt_storage):
        """Test searching memory."""
        mock_embed.return_value = [0.1] * 3072

        # Search memory
        results = dxmt_storage.search(
            query="programming preferences",
            limit=5,
            score_threshold=0.6
        )

        # Verify search was called
        assert dxmt_storage.client.search.called

        # Check result format
        assert "results" in results
        assert len(results["results"]) > 0
        assert "content" in results["results"][0]
        assert "memory" in results["results"][0]  # CrewAI compatibility
        assert "score" in results["results"][0]

    @patch('dxmt.crewai_storage.DXMTStorage._get_embedding')
    def test_search_with_filter(self, mock_embed, dxmt_storage):
        """Test search builds correct filter."""
        mock_embed.return_value = [0.1] * 3072

        dxmt_storage.search(query="test", limit=5)

        # Verify filter was built with correct conditions
        call_args = dxmt_storage.client.search.call_args
        filter_arg = call_args[1].get("filter")

        # Should have type and user_id filters
        assert filter_arg is not None

    def test_reset_storage(self, dxmt_storage):
        """Test resetting storage."""
        dxmt_storage.reset()

        # Verify table was deleted and recreated
        assert dxmt_storage.client.delete_table.called
        assert dxmt_storage.client.create_table.call_count >= 1

    def test_extract_user_message(self, dxmt_storage):
        """Test user message extraction."""
        text = "User message: Hello, how are you?"
        result = dxmt_storage._extract_user_message(text)
        assert result == "Hello, how are you?"

        # Test without prefix
        result2 = dxmt_storage._extract_user_message("Just a message")
        assert result2 == "Just a message"

    def test_extract_assistant_message(self, dxmt_storage):
        """Test assistant message extraction."""
        text = "Some thinking... Final Answer: This is the answer"
        result = dxmt_storage._extract_assistant_message(text)
        assert result == "This is the answer"

        # Test without marker
        result2 = dxmt_storage._extract_assistant_message("Just a response")
        assert result2 == "Just a response"

    def test_sanitize_role(self, dxmt_storage):
        """Test role sanitization."""
        role = "Senior Research\nAnalyst / Expert"
        sanitized = dxmt_storage._sanitize_role(role)
        assert "\n" not in sanitized
        assert " " not in sanitized
        assert "/" not in sanitized
        assert sanitized == "SeniorResearchAnalyst_Expert"


# ============================================================================
# Integration Tests
# ============================================================================

class TestCrewAIIntegration:
    """Integration tests for CrewAI components."""

    def test_tool_is_base_tool_subclass(self):
        """Test that DXMTVectorSearchTool is a BaseTool subclass."""
        if CREWAI_AVAILABLE:
            assert issubclass(DXMTVectorSearchTool, BaseTool)

    def test_storage_is_storage_subclass(self):
        """Test that DXMTStorage is a Storage subclass."""
        if CREWAI_AVAILABLE:
            assert issubclass(DXMTStorage, Storage)

    def test_tool_config_validation(self):
        """Test tool configuration validation."""
        config = DXMTToolConfig(
            host="localhost",
            port=9999,
            table_name="test"
        )

        assert config.host == "localhost"
        assert config.port == 9999
        assert config.limit == 5  # default
        assert config.score_threshold == 0.35  # default

    def test_storage_config_validation(self):
        """Test storage configuration validation."""
        config = DXMTStorageConfig(
            host="localhost",
            port=9999,
            table_name="memory"
        )

        assert config.host == "localhost"
        assert config.dimension == 3072  # default
        assert config.distance_metric == "cosine"  # default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
