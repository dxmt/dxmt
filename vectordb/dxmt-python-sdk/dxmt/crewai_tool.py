"""DXMT VectorDB Tool for CrewAI.

This module provides a CrewAI BaseTool implementation for DXMT VectorDB,
enabling vector similarity search within CrewAI agents and workflows.

Example:
    ```python
    from crewai import Agent, Task, Crew
    from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig

    # Configure DXMT tool
    config = DXMTToolConfig(
        host="localhost",
        port=9999,
        table_name="documents",
        limit=5,
        score_threshold=0.35
    )

    # Create tool
    search_tool = DXMTVectorSearchTool(dxmt_config=config)

    # Create agent with tool
    agent = Agent(
        role="Research Assistant",
        goal="Find relevant information",
        tools=[search_tool],
        verbose=True
    )
    ```
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from crewai.tools import BaseTool, EnvVar
    CREWAI_AVAILABLE = True
except ImportError:
    # CrewAI not installed, define dummy classes
    CREWAI_AVAILABLE = False

    class BaseTool:  # type: ignore
        """Dummy BaseTool when CrewAI is not installed."""
        def __init__(self, **kwargs: Any) -> None:
            """Initialize dummy BaseTool."""
            for key, value in kwargs.items():
                setattr(self, key, value)

    class EnvVar:  # type: ignore
        """Dummy EnvVar when CrewAI is not installed."""
        pass

from .client import DXMTClient
from .exceptions import DXMTError

logger = logging.getLogger(__name__)


class DXMTToolSchema(BaseModel):
    """Input schema for DXMT Vector Search Tool."""

    query: str = Field(
        ...,
        description="Query to search in DXMT VectorDB - always required."
    )
    filter_by: str | None = Field(
        default=None,
        description="Metadata field to filter by. When filtering, use with filter_value."
    )
    filter_value: Any | None = Field(
        default=None,
        description="Value to filter by. When filtering, use with filter_by."
    )


class DXMTToolConfig(BaseModel):
    """Configuration for DXMT Vector Search Tool."""

    host: str = Field(
        default="localhost",
        description="DXMT server hostname"
    )
    port: int = Field(
        default=9999,
        description="DXMT server port"
    )
    table_name: str = Field(
        ...,
        description="Name of the DXMT table to search"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of results to return"
    )
    score_threshold: float = Field(
        default=0.35,
        description="Minimum similarity score threshold (0-1)"
    )
    filter: dict[str, Any] | None = Field(
        default=None,
        description="DXMT Filter for advanced filtering"
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric (cosine, euclidean, dot)"
    )


class DXMTVectorSearchTool(BaseTool):
    """Vector search tool for DXMT VectorDB.

    This tool enables CrewAI agents to search DXMT VectorDB for relevant
    documents using vector similarity search. It supports filtering,
    custom embedding functions, and configurable search parameters.

    Features:
        - Vector similarity search with cosine/euclidean/dot metrics
        - Metadata filtering (simple and advanced)
        - Custom embedding function support
        - Score threshold filtering
        - Tiered storage benefits (98% cost savings)

    Args:
        dxmt_config: Configuration for DXMT connection and search
        custom_embedding_fn: Optional custom embedding function
        client: Optional pre-initialized DXMT client

    Example:
        ```python
        from dxmt.crewai_tool import DXMTVectorSearchTool, DXMTToolConfig

        config = DXMTToolConfig(
            host="localhost",
            port=9999,
            table_name="documents",
            limit=5
        )

        tool = DXMTVectorSearchTool(dxmt_config=config)

        # Use in agent
        agent = Agent(
            role="Researcher",
            tools=[tool]
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Metadata ---
    name: str = "DXMTVectorSearchTool"
    description: str = (
        "Search DXMT VectorDB for relevant documents using vector similarity search. "
        "Supports filtering by metadata fields and returns top-k most similar results."
    )
    args_schema: type[BaseModel] = DXMTToolSchema
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["dxmt-python-sdk"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for OpenAI embeddings",
                required=True
            )
        ]
    )

    # --- Configuration ---
    dxmt_config: DXMTToolConfig
    custom_embedding_fn: Callable[[str], list[float]] | None = Field(
        default=None,
        description="Optional custom embedding function (query_text -> vector)"
    )
    client: DXMTClient | None = Field(
        default=None,
        description="DXMT client instance (auto-created if None)"
    )

    def __init__(self, **kwargs):
        """Initialize DXMT Vector Search Tool."""
        super().__init__(**kwargs)

        if not CREWAI_AVAILABLE:
            logger.warning(
                "CrewAI is not installed. Install with: pip install crewai"
            )

        # Initialize DXMT client if not provided
        if not self.client:
            self.client = DXMTClient(
                host=self.dxmt_config.host,
                port=self.dxmt_config.port
            )
            logger.info(
                f"Initialized DXMT client: {self.dxmt_config.host}:{self.dxmt_config.port}"
            )

    def _run(
        self,
        query: str,
        filter_by: str | None = None,
        filter_value: Any | None = None,
    ) -> str:
        """Perform vector similarity search.

        Args:
            query: Search query text
            filter_by: Optional metadata field to filter by
            filter_value: Optional value for filter_by field

        Returns:
            JSON string containing search results with score, metadata, and context

        Raises:
            DXMTError: If search fails
        """
        try:
            # Build filter if provided
            search_filter = None
            if filter_by and filter_value is not None:
                # Simple equality filter
                search_filter = {
                    "must": [{
                        "key": filter_by,
                        "operation": {"type": "Equal", "value": filter_value}
                    }]
                }
                logger.debug(f"Applied filter: {filter_by}={filter_value}")
            elif self.dxmt_config.filter:
                # Use pre-configured filter
                search_filter = self.dxmt_config.filter

            # Get query embedding
            if self.custom_embedding_fn:
                query_vector = self.custom_embedding_fn(query)
                logger.debug("Used custom embedding function")
            else:
                # Use OpenAI embeddings
                query_vector = self._get_openai_embedding(query)
                logger.debug("Used OpenAI embedding")

            # Perform search
            results = self.client.search(
                table_name=self.dxmt_config.table_name,
                vector=query_vector,
                k=self.dxmt_config.limit,
                filter=search_filter
            )

            # Filter by score threshold
            filtered_results = [
                r for r in results
                if r.get("score", 0.0) >= self.dxmt_config.score_threshold
            ]

            logger.info(
                f"Search returned {len(filtered_results)}/{len(results)} results "
                f"above threshold {self.dxmt_config.score_threshold}"
            )

            # Format results for CrewAI
            formatted_results = []
            for result in filtered_results:
                formatted_results.append({
                    "distance": 1.0 - result.get("score", 0.0),  # Convert similarity to distance
                    "score": result.get("score", 0.0),
                    "metadata": result.get("metadata", {}),
                    "context": result.get("metadata", {}).get("text", "") or result.get("text", ""),
                    "id": result.get("id", "")
                })

            return json.dumps(formatted_results, indent=2)

        except Exception as e:
            logger.error(f"DXMT search error: {e}")
            raise DXMTError(f"Vector search failed: {e}")

    def _get_openai_embedding(self, text: str) -> list[float]:
        """Get OpenAI embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ImportError: If openai package is not installed
            ValueError: If OPENAI_API_KEY is not set
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package required for default embeddings. "
                "Install with: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when using "
                "default embedding function"
            )

        client = openai.Client(api_key=api_key)
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )

        return response.data[0].embedding

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None
    ) -> list[str]:
        """Add documents to DXMT table.

        This is a utility method for adding documents with automatic embedding.
        Useful for preparing the vector database before using the search tool.

        Args:
            texts: List of document texts to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            List of inserted document IDs

        Example:
            ```python
            tool = DXMTVectorSearchTool(dxmt_config=config)
            tool.add_documents(
                texts=["Document 1 content", "Document 2 content"],
                metadatas=[{"source": "wiki"}, {"source": "docs"}]
            )
            ```
        """
        import uuid

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        if len(texts) != len(metadatas) != len(ids):
            raise ValueError("texts, metadatas, and ids must have the same length")

        # Get embeddings
        if self.custom_embedding_fn:
            vectors = [self.custom_embedding_fn(text) for text in texts]
        else:
            # Batch embed with OpenAI
            vectors = self._batch_embed_openai(texts)

        # Prepare records
        records = []
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
            # Add text to metadata for retrieval
            metadata_with_text = {**metadata, "text": text}

            records.append({
                "id": doc_id,
                "vector": vectors[i],
                "metadata": metadata_with_text
            })

        # Insert into DXMT
        self.client.insert_records(
            table_name=self.dxmt_config.table_name,
            records=records
        )

        logger.info(f"Added {len(records)} documents to {self.dxmt_config.table_name}")
        return ids

    def _batch_embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts using OpenAI.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        client = openai.Client(api_key=api_key)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-large"
        )

        return [data.embedding for data in response.data]


__all__ = [
    "DXMTVectorSearchTool",
    "DXMTToolConfig",
    "DXMTToolSchema",
]
