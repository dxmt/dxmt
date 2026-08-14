"""Tests for DXMT VectorDB AutoGen integration.

These tests verify the DXMT memory implementation works correctly
with AutoGen's Memory interface.

Requirements:
    pip install pytest pytest-asyncio autogen-ext dxmt-python-sdk

Before running:
    1. Start DXMT server: cargo run -- serve --host 0.0.0.0 --port 9999
    2. Run: pytest tests/test_dxmt_autogen.py -v
"""

import sys
from pathlib import Path

import pytest

# Add DXMT SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dxmt-python-sdk"))

try:
    from autogen_core.memory import MemoryContent
    from autogen_ext.memory.dxmt import (
        DXMTVectorMemory,
        HttpDXMTVectorMemoryConfig,
    )
except ImportError:
    pytest.skip("AutoGen not installed", allow_module_level=True)


@pytest.fixture
async def dxmt_memory():
    """Create a DXMT memory instance for testing."""
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="test_autogen_memory",
        k=5,
        enable_full_text_search=True
    )

    memory = DXMTVectorMemory(config=config)

    # Ensure clean state
    try:
        await memory.clear()
    except:
        pass  # Table might not exist yet

    yield memory

    # Cleanup
    try:
        await memory.clear()
        await memory.close()
    except:
        pass


@pytest.mark.asyncio
async def test_add_and_query(dxmt_memory):
    """Test basic add and query operations."""
    # Add a memory
    content = "User prefers Python over Java for web development"
    await dxmt_memory.add(MemoryContent(
        content=content,
        mime_type="text/plain",
        metadata={"user_id": "test_user", "category": "preferences"}
    ))

    # Query the memory
    results = await dxmt_memory.query("What programming language does user prefer?")

    # Assertions
    assert len(results.results) > 0, "Should return at least one result"
    assert "Python" in results.results[0].content, "Should find relevant content"
    assert results.results[0].metadata["score"] > 0.0, "Should have positive score"


@pytest.mark.asyncio
async def test_score_normalization(dxmt_memory):
    """Test that scores are normalized to [0, 1] range."""
    # Add multiple memories
    memories = [
        "Alice likes Python programming",
        "Bob prefers Java development",
        "Charlie enjoys Rust coding",
    ]

    for content in memories:
        await dxmt_memory.add(MemoryContent(
            content=content,
            mime_type="text/plain"
        ))

    # Query
    results = await dxmt_memory.query("programming languages")

    # Check score normalization
    for result in results.results:
        score = result.metadata["score"]
        assert 0.0 <= score <= 1.0, f"Score {score} should be in [0, 1]"


@pytest.mark.asyncio
async def test_metadata_filtering(dxmt_memory):
    """Test filtering by user_id metadata."""
    # Add memories for different users
    await dxmt_memory.add(MemoryContent(
        content="User 1 likes Python",
        mime_type="text/plain",
        metadata={"user_id": "user_1"}
    ))

    await dxmt_memory.add(MemoryContent(
        content="User 2 likes Java",
        mime_type="text/plain",
        metadata={"user_id": "user_2"}
    ))

    # Query with user_id filter
    results = await dxmt_memory.query(
        "What programming language?",
        user_id="user_1"
    )

    # Assertions
    assert len(results.results) > 0, "Should find results for user_1"
    for result in results.results:
        assert "Python" in result.content, "Should only return user_1 memories"
        assert "Java" not in result.content, "Should not return user_2 memories"


@pytest.mark.asyncio
async def test_score_threshold(dxmt_memory):
    """Test score threshold filtering."""
    # Configure memory with score threshold
    config = HttpDXMTVectorMemoryConfig(
        host="localhost",
        port=9999,
        table_name="test_threshold",
        k=10,
        score_threshold=0.5  # Only return results with score >= 0.5
    )

    memory = DXMTVectorMemory(config=config)

    try:
        # Add a memory
        await memory.add(MemoryContent(
            content="Python is a high-level programming language",
            mime_type="text/plain"
        ))

        # Query with somewhat related content (should have lower score)
        results = await memory.query("cooking recipes")

        # All returned results should meet threshold
        for result in results.results:
            assert result.metadata["score"] >= 0.5, "Results should meet threshold"

    finally:
        await memory.clear()
        await memory.close()


@pytest.mark.asyncio
async def test_clear(dxmt_memory):
    """Test clearing all memories."""
    # Add some memories
    for i in range(5):
        await dxmt_memory.add(MemoryContent(
            content=f"Memory number {i}",
            mime_type="text/plain"
        ))

    # Clear all memories
    await dxmt_memory.clear()

    # Query should return no results
    results = await dxmt_memory.query("memory")

    assert len(results.results) == 0, "Should have no results after clear"


@pytest.mark.asyncio
async def test_multiple_memories(dxmt_memory):
    """Test adding and querying multiple memories."""
    # Add multiple memories
    memories = [
        "Alice works as a software engineer at TechCorp",
        "Alice's favorite color is blue",
        "Alice enjoys playing tennis on weekends",
        "Alice is learning machine learning",
        "Alice graduated from Stanford University",
    ]

    for content in memories:
        await dxmt_memory.add(MemoryContent(
            content=content,
            mime_type="text/plain",
            metadata={"user_id": "alice"}
        ))

    # Query with different aspects
    queries = [
        ("What is Alice's job?", "software engineer"),
        ("What does Alice like to do?", "tennis"),
        ("Where did Alice study?", "Stanford"),
    ]

    for query, expected_keyword in queries:
        results = await dxmt_memory.query(query, user_id="alice", limit=3)

        assert len(results.results) > 0, f"Should find results for: {query}"

        # Check if expected keyword appears in top results
        found = any(
            expected_keyword.lower() in result.content.lower()
            for result in results.results
        )
        assert found, f"Should find '{expected_keyword}' for query: {query}"


@pytest.mark.asyncio
async def test_mime_types(dxmt_memory):
    """Test handling different MIME types."""
    # Add memories with different MIME types
    await dxmt_memory.add(MemoryContent(
        content="Plain text content",
        mime_type="text/plain"
    ))

    await dxmt_memory.add(MemoryContent(
        content="# Markdown content\nThis is **bold**",
        mime_type="text/markdown"
    ))

    await dxmt_memory.add(MemoryContent(
        content={"key": "value", "data": "JSON content"},
        mime_type="application/json"
    ))

    # Query should work regardless of MIME type
    results = await dxmt_memory.query("content")

    assert len(results.results) >= 2, "Should find multiple results"

    # Check MIME types are preserved
    mime_types = {result.metadata.get("mime_type") for result in results.results}
    assert "text/plain" in mime_types or "text/markdown" in mime_types


@pytest.mark.asyncio
async def test_empty_query(dxmt_memory):
    """Test behavior with empty query."""
    # Add a memory
    await dxmt_memory.add(MemoryContent(
        content="Test content",
        mime_type="text/plain"
    ))

    # Query with empty string
    results = await dxmt_memory.query("")

    # Should not crash, may return results or empty list
    assert isinstance(results.results, list)


@pytest.mark.asyncio
async def test_limit_parameter(dxmt_memory):
    """Test limit parameter in query."""
    # Add many memories
    for i in range(20):
        await dxmt_memory.add(MemoryContent(
            content=f"Memory about Python programming number {i}",
            mime_type="text/plain"
        ))

    # Query with limit=3
    results = await dxmt_memory.query("Python", limit=3)

    assert len(results.results) <= 3, "Should respect limit parameter"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
