"""DXMT VectorDB Storage for CrewAI Memory.

This module provides a CrewAI Storage implementation backed by DXMT VectorDB,
enabling persistent memory storage for CrewAI agents with cost-efficient
tiered storage (Memory → SSD → S3).

Example:
    ```python
    from crewai import Agent, Crew
    from dxmt.crewai_storage import DXMTStorage

    # Configure DXMT storage for long-term memory
    storage_config = {
        "host": "localhost",
        "port": 9999,
        "table_name": "crew_memory",
        "user_id": "user_123",
        "agent_id": "research_agent"
    }

    # Create storage
    storage = DXMTStorage(
        type="long_term",
        config=storage_config
    )

    # Use with Crew
    crew = Crew(
        agents=[agent],
        memory=True,
        long_term_memory=storage
    )
    ```
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr

try:
    from crewai.memory.storage.interface import Storage
    CREWAI_STORAGE_AVAILABLE = True
except ImportError:
    # CrewAI not installed, define dummy base class
    CREWAI_STORAGE_AVAILABLE = False

    class Storage:  # type: ignore
        """Dummy Storage when CrewAI is not installed."""
        pass

from .client import DXMTClient
from .exceptions import DXMTError

logger = logging.getLogger(__name__)


class DXMTStorageConfig(BaseModel):
    """Configuration for DXMT Storage."""

    host: str = Field(
        default="localhost",
        description="DXMT server hostname"
    )
    port: int = Field(
        default=9999,
        description="DXMT server port"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="The server's API key, if it was started with DXMT_API_KEY",
        # A credential: hidden in repr (SecretStr) and left out of model_dump(), so it cannot
        # travel wherever this config is logged or serialised.
        exclude=True,
    )
    table_name: str = Field(
        default="crew_memory",
        description="Name of the DXMT table for memory storage"
    )
    user_id: str | None = Field(
        default=None,
        description="User ID for memory scoping"
    )
    agent_id: str | None = Field(
        default=None,
        description="Agent ID for memory scoping"
    )
    run_id: str | None = Field(
        default=None,
        description="Run ID for short-term memory scoping"
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric (cosine, euclidean, dot)"
    )
    dimension: int = Field(
        default=3072,
        description="Vector dimension (3072 for text-embedding-3-large)"
    )


class DXMTStorage(Storage):
    """CrewAI Storage implementation using DXMT VectorDB.

    Implements the CrewAI Storage interface with DXMT backend, providing
    persistent memory storage on a single-node vector database.

    Features:
        - Multi-scope memory (user_id, agent_id, run_id)
        - High-performance vector search (<100ms P99)
        - Support for all memory types (short_term, long_term, entities, external)
        - Automatic embedding generation

    Args:
        type: Memory type - "short_term", "long_term", "entities", or "external"
        crew: Optional Crew instance for agent name extraction
        config: Configuration dict with DXMT connection parameters

    Example:
        ```python
        storage = DXMTStorage(
            type="long_term",
            config={
                "host": "localhost",
                "port": 9999,
                "table_name": "crew_memory",
                "user_id": "user_123"
            }
        )

        # Save memory
        storage.save(
            "User prefers Python over Java",
            {"source": "conversation", "timestamp": "2024-01-28"}
        )

        # Search memory
        results = storage.search("programming preferences", limit=5)
        ```
    """

    def __init__(
        self,
        type: str,
        crew=None,
        config: dict[str, Any] | None = None
    ):
        """Initialize DXMT Storage."""
        super().__init__()

        if not CREWAI_STORAGE_AVAILABLE:
            logger.warning(
                "CrewAI storage interface not installed. "
                "Install with: pip install crewai"
            )

        self._validate_type(type)
        self.memory_type = type
        self.crew = crew
        self.config = config or {}

        # Parse configuration
        self._parse_config()

        # Initialize DXMT client
        self._initialize_client()

        logger.info(
            f"Initialized DXMTStorage: type={type}, "
            f"table={self.storage_config.table_name}"
        )

    def _validate_type(self, type: str):
        """Validate memory type."""
        supported_types = {"short_term", "long_term", "entities", "external"}
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for DXMTStorage. "
                f"Must be one of: {', '.join(supported_types)}"
            )

    def _parse_config(self):
        """Parse configuration from config dict."""
        self.storage_config = DXMTStorageConfig(**self.config)

    def _initialize_client(self):
        """Initialize DXMT client and create table if needed."""
        key = self.storage_config.api_key
        self.client = DXMTClient(
            host=self.storage_config.host,
            port=self.storage_config.port,
            api_key=key.get_secret_value() if key else None,
        )

        # Check if table exists, create if not
        try:
            self.client.get_table(self.storage_config.table_name)
            logger.debug(f"Table {self.storage_config.table_name} already exists")
        except DXMTError:
            # Table doesn't exist, create it
            logger.info(f"Creating table {self.storage_config.table_name}")
            self.client.create_table(
                name=self.storage_config.table_name,
                dimension=self.storage_config.dimension,
                distance_metric=self.storage_config.distance_metric,
                description=f"CrewAI {self.memory_type} memory storage"
            )

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """Save memory to DXMT.

        Args:
            value: Memory content (string or conversation)
            metadata: Additional metadata (messages, timestamp, etc.)

        Example:
            ```python
            storage.save(
                "User prefers dark mode",
                {"source": "conversation", "confidence": 0.95}
            )
            ```
        """
        try:
            # Extract content from value
            if isinstance(value, str):
                content = value
            else:
                content = str(value)

            # Extract messages if present
            messages = metadata.pop("messages", None)
            if messages:
                # Extract last user and assistant messages
                content = self._extract_conversation_content(messages)

            # Generate embedding
            vector = self._get_embedding(content)

            # Prepare metadata
            memory_metadata = {
                "type": self.memory_type,
                "content": content,
                **metadata
            }

            # Add scoping IDs
            if self.storage_config.user_id:
                memory_metadata["user_id"] = self.storage_config.user_id

            if self.storage_config.agent_id:
                memory_metadata["agent_id"] = self.storage_config.agent_id
            elif self.crew:
                memory_metadata["agent_id"] = self._get_agent_name()

            if self.memory_type == "short_term" and self.storage_config.run_id:
                memory_metadata["run_id"] = self.storage_config.run_id

            # Insert into DXMT
            import uuid
            record = {
                "id": str(uuid.uuid4()),
                "vector": vector,
                "metadata": memory_metadata
            }

            self.client.insert_records(
                table_name=self.storage_config.table_name,
                records=[record]
            )

            logger.debug(f"Saved memory: {content[:100]}...")

        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            raise DXMTError(f"Failed to save memory: {e}")

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6
    ) -> list[Any]:
        """Search for relevant memories.

        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of memory results with content and metadata

        Example:
            ```python
            results = storage.search(
                "programming preferences",
                limit=5,
                score_threshold=0.7
            )

            for result in results:
                print(result["content"])
                print(result["score"])
            ```
        """
        try:
            # Generate query embedding
            query_vector = self._get_embedding(query)

            # Build filter
            search_filter = self._build_search_filter()

            # Search DXMT
            results = self.client.search(
                table_name=self.storage_config.table_name,
                vector=query_vector,
                k=limit,
                filter=search_filter
            )

            # Filter by score threshold and format results
            formatted_results = []
            for result in results:
                score = result.get("score", 0.0)
                if score >= score_threshold:
                    metadata = result.get("metadata", {})
                    formatted_results.append({
                        "content": metadata.get("content", ""),
                        "memory": metadata.get("content", ""),  # CrewAI compatibility
                        "score": score,
                        "metadata": metadata,
                        "id": result.get("id", "")
                    })

            logger.debug(
                f"Search returned {len(formatted_results)}/{len(results)} results "
                f"above threshold {score_threshold}"
            )

            # Wrap in results dict for CrewAI compatibility
            return {"results": formatted_results}

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {"results": []}

    def reset(self) -> None:
        """Reset storage by clearing all memories.

        Warning: This will delete all memories in the table!
        """
        try:
            logger.warning(f"Resetting storage: {self.storage_config.table_name}")

            # Delete and recreate table
            self.client.delete_table(self.storage_config.table_name)
            self.client.create_table(
                name=self.storage_config.table_name,
                dimension=self.storage_config.dimension,
                distance_metric=self.storage_config.distance_metric,
                description=f"CrewAI {self.memory_type} memory storage"
            )

            logger.info(f"Storage reset complete: {self.storage_config.table_name}")

        except Exception as e:
            logger.error(f"Error resetting storage: {e}")
            raise DXMTError(f"Failed to reset storage: {e}")

    # Helper methods

    def _build_search_filter(self) -> dict[str, Any] | None:
        """Build DXMT filter for search based on scoping."""
        conditions = []

        # Add memory type filter
        conditions.append({
            "key": "type",
            "operation": {"type": "Equal", "value": self.memory_type}
        })

        # Add scoping filters
        if self.memory_type == "short_term" and self.storage_config.run_id:
            conditions.append({
                "key": "run_id",
                "operation": {"type": "Equal", "value": self.storage_config.run_id}
            })
        else:
            if self.storage_config.user_id and self.storage_config.agent_id:
                # User OR agent filter
                return {
                    "should": [
                        {
                            "must": [
                                {
                                    "key": "user_id",
                                    "operation": {"type": "Equal", "value": self.storage_config.user_id}
                                }
                            ]
                        },
                        {
                            "must": [
                                {
                                    "key": "agent_id",
                                    "operation": {"type": "Equal", "value": self.storage_config.agent_id}
                                }
                            ]
                        }
                    ]
                }
            elif self.storage_config.user_id:
                conditions.append({
                    "key": "user_id",
                    "operation": {"type": "Equal", "value": self.storage_config.user_id}
                })
            elif self.storage_config.agent_id:
                conditions.append({
                    "key": "agent_id",
                    "operation": {"type": "Equal", "value": self.storage_config.agent_id}
                })

        return {"must": conditions} if conditions else None

    def _get_embedding(self, text: str) -> list[float]:
        """Get OpenAI embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for embeddings. "
                "Install with: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        client = openai.Client(api_key=api_key)
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )

        return response.data[0].embedding

    def _extract_conversation_content(self, messages: list[dict]) -> str:
        """Extract content from conversation messages.

        Args:
            messages: List of message dicts with role and content

        Returns:
            Combined content string
        """
        def _last_content(messages, role):
            for msg in reversed(messages):
                if msg.get("role") == role:
                    return msg.get("content", "")
            return ""

        last_user = _last_content(messages, "user")
        last_assistant = _last_content(messages, "assistant")

        # Extract user message
        user_msg = self._extract_user_message(last_user)

        # Extract assistant message
        assistant_msg = self._extract_assistant_message(last_assistant)

        # Combine
        parts = []
        if user_msg:
            parts.append(f"User: {user_msg}")
        if assistant_msg:
            parts.append(f"Assistant: {assistant_msg}")

        return " ".join(parts) if parts else last_user or last_assistant

    def _extract_user_message(self, text: str) -> str:
        """Extract user message from text."""
        pattern = r"User message:\s*(.*)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return text

    def _extract_assistant_message(self, text: str) -> str:
        """Extract assistant message from text."""
        marker = "Final Answer:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text

    def _get_agent_name(self) -> str:
        """Get sanitized agent name from crew."""
        if not self.crew:
            return ""

        agents = self.crew.agents
        agent_roles = [self._sanitize_role(agent.role) for agent in agents]
        return "_".join(agent_roles)

    def _sanitize_role(self, role: str) -> str:
        """Sanitize agent role for use in identifiers."""
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")


__all__ = [
    "DXMTStorage",
    "DXMTStorageConfig",
]
