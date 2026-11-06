"""A CrewAI tool that searches a DXMT table.

The agent writes a question; the tool answers with the nearest records as JSON. The query is
embedded by the server when the table names a model, or by ``embedder`` here; with
``search_mode="hybrid"`` the table's BM25 index is fused in, and ``"text"`` is keyword search
alone. No OpenAI client is built into the tool: the embedding provider is the table's or the
caller's, never a default that needs a key.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from crewai.tools import BaseTool
from dxmt import DXMTClient
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class DXMTSearchSchema(BaseModel):
    query: str = Field(description="What to search for, in plain words")
    filter_by: Optional[str] = Field(default=None, description="Payload field to filter on, e.g. meta.source")
    filter_value: Optional[Any] = Field(default=None, description="The value that field must equal")


class DXMTVectorSearchTool(BaseTool):
    """Search a DXMT table and hand the agent the top matches."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "DXMT search"
    description: str = (
        "Search a DXMT table for records relevant to a question. Returns the top matches as "
        "JSON with id, score and payload. Optionally restrict to records whose payload field "
        "equals a value."
    )
    args_schema: type[BaseModel] = DXMTSearchSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["crewai-dxmt"])

    table_name: str
    host: str = "localhost"
    port: int = 9999
    ssl: bool = False
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    limit: int = 5
    score_threshold: float = 0.0
    search_mode: str = Field(default="vector", description="vector | hybrid | text")
    filter: Optional[dict[str, Any]] = Field(default=None, description="A server filter applied to every search")
    embedder: Optional[Callable[[str], list[float]]] = Field(
        default=None, exclude=True, description="Embeds the query here; omit when the table's server-side model embeds it"
    )
    _client: Optional[DXMTClient] = PrivateAttr(default=None)

    def __init__(self, client: Optional[DXMTClient] = None, **data: Any) -> None:
        super().__init__(**data)
        self._client = client or DXMTClient(host=self.host, port=self.port, ssl=self.ssl, api_key=self.api_key)

    @property
    def client(self) -> DXMTClient:
        assert self._client is not None
        return self._client

    def _run(self, query: str, filter_by: Optional[str] = None, filter_value: Any = None) -> str:
        f = self.filter
        if filter_by and filter_value is not None:
            cond = {"key": filter_by, "operation": {"type": "Equal", "value": filter_value}}
            f = {**(f or {}), "must": list((f or {}).get("must") or []) + [cond]}
        vector = self.embedder(query) if self.embedder is not None else None
        if self.search_mode == "text":
            hits = self.client.full_text_search(self.table_name, query=query, k=self.limit, filter=f)
        elif self.search_mode == "hybrid":
            hits = self.client.hybrid_search(self.table_name, vector=vector, text=query, k=self.limit, filter=f)
        else:
            hits = self.client.search(self.table_name, vector=vector, text=None if vector else query, k=self.limit, filter=f)
        out = [
            {"id": h["id"], "score": float(h.get("score", 0.0)), "payload": h.get("payload") or h.get("metadata") or {}}
            for h in hits
            if float(h.get("score", 0.0)) >= self.score_threshold
        ]
        return json.dumps(out, ensure_ascii=False, indent=2)
