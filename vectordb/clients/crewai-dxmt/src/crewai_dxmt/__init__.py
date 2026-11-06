"""DXMT for CrewAI: memory storage backend, knowledge storage, and a search tool.

    from crewai_dxmt import install
    install(host="localhost", port=9999)      # every Memory() and Knowledge() now lands in DXMT

or pass instances where CrewAI takes them: ``Memory(storage=DXMTStorageBackend(...))``,
``Knowledge(storage=DXMTKnowledgeStorage(...))``, ``Agent(tools=[DXMTVectorSearchTool(...)])``.
"""

from __future__ import annotations

from typing import Any, Optional

from crewai_dxmt.knowledge import DXMTKnowledgeStorage
from crewai_dxmt.memory import DXMTStorageBackend
from crewai_dxmt.tool import DXMTVectorSearchTool

__all__ = ["DXMTStorageBackend", "DXMTKnowledgeStorage", "DXMTVectorSearchTool", "install", "uninstall"]

__version__ = "1.6.0"


def install(
    *,
    host: str = "localhost",
    port: int = 9999,
    ssl: bool = False,
    api_key: Optional[str] = None,
    memory_table: str = "crewai_memory",
    knowledge_table: str = "crewai_knowledge",
    knowledge_embedding: Optional[dict[str, Any]] = None,
    memory: bool = True,
    knowledge: bool = True,
) -> None:
    """Route CrewAI's default memory and knowledge storage to a DXMT server, process-wide.

    ``Crew(memory=True)`` builds its ``Memory`` without a ``storage`` argument, so passing an
    instance is not enough to reach it; CrewAI's ``set_memory_storage_factory`` is the seam.
    The factory answers every string spec (``"lancedb"``, a path) with a DXMT backend; an
    explicit ``Memory(storage=instance)`` still wins. Knowledge: with ``knowledge_embedding``
    the server embeds; otherwise the embedder CrewAI resolved for the collection is used.
    Call once at startup; ``uninstall()`` restores the built-ins.
    """
    if memory:
        from crewai.memory.storage.factory import set_memory_storage_factory

        def memory_factory(spec: str) -> DXMTStorageBackend:
            # Every spec -- "lancedb", "qdrant-edge", a path -- is answered with the one table.
            return DXMTStorageBackend(memory_table, host=host, port=port, ssl=ssl, api_key=api_key)

        set_memory_storage_factory(memory_factory)
    if knowledge:
        from crewai.knowledge.storage.factory import set_knowledge_storage_factory

        def knowledge_factory(embedder_config: Any, collection_name: Optional[str]) -> Optional[DXMTKnowledgeStorage]:
            table = f"{knowledge_table}_{collection_name}" if collection_name else knowledge_table
            if knowledge_embedding is not None:
                return DXMTKnowledgeStorage(table_name=table, host=host, port=port, ssl=ssl, api_key=api_key,
                                            embedding=knowledge_embedding)
            from crewai.rag.embeddings.factory import build_embedder

            fn = build_embedder(embedder_config) if embedder_config is not None else None
            if fn is None:
                return None  # let CrewAI's built-in storage handle a collection with no embedder
            return DXMTKnowledgeStorage(table_name=table, host=host, port=port, ssl=ssl, api_key=api_key,
                                        embedder=lambda texts: [list(v) for v in fn(texts)])

        set_knowledge_storage_factory(knowledge_factory)


def uninstall() -> None:
    from crewai.knowledge.storage.factory import set_knowledge_storage_factory
    from crewai.memory.storage.factory import set_memory_storage_factory

    set_memory_storage_factory(None)
    set_knowledge_storage_factory(None)
