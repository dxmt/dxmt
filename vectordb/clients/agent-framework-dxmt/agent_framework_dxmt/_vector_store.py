"""DXMT vector store and collection for Microsoft Agent Framework.

A collection is one DXMT table. The record's key is the row id (stored as a string; ``int``
and ``UUID`` keys are converted on the way in and back on the way out), its one vector field
is the row's vector, and its data fields are the payload under their storage names. A data
field marked ``is_full_text_indexed`` gets the table's BM25 index, and then
``search(values, search_type="keyword_hybrid", additional_property_name=<that field>)`` is
the server's hybrid search; this is the second in-tree-shaped connector to offer it. With
``server_embedding`` the table names an embedding model and a record whose vector is ``None``
is embedded by the server from ``server_embedding_field``; a search without a local
embedding generator then sends ``values`` as text.

Scores are the server's similarities (1.0 is identical under cosine; larger is better) and
``score_threshold`` is in those units. Supplied clients remain caller-owned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Generic, cast
from uuid import UUID

from agent_framework import (
    BaseVectorCollection,
    BaseVectorSearch,
    BaseVectorStore,
    SearchResults,
    SecretString,
    VectorStoreCollectionDefinition,
)
from agent_framework.exceptions import IntegrationException, IntegrationInvalidResponseException
from dxmt import DXMTClient
from dxmt.exceptions import DXMTError
from typing_extensions import TypeVar

from ._filters import translate
from ._settings import _resolve_client

KeyT = TypeVar("KeyT", default=Any)
ModelT = TypeVar("ModelT", default=Any)

_DISTANCES = {
    "DEFAULT": "cosine",
    "cosine_similarity": "cosine",
    "dot_prod": "dot",
    "euclidean_distance": "euclidean",
}
_BATCH_SIZE = 500


def _not_found(exc: Exception) -> bool:
    return "TABLE_NOT_FOUND" in str(exc) or "NOT_FOUND" in str(exc)


class DXMTCollection(BaseVectorCollection[KeyT, ModelT], BaseVectorSearch[KeyT, ModelT], Generic[KeyT, ModelT]):
    """One DXMT table as an Agent Framework vector collection."""

    supported_key_types: ClassVar[set[str] | None] = {"int", "str", "UUID"}
    supported_vector_types: ClassVar[set[str] | None] = {"float", "float16", "float32", "float64"}
    supported_search_types: ClassVar[set[str]] = {"vector", "keyword_hybrid"}  # type: ignore[assignment]

    def __init__(
        self,
        record_type: type[ModelT],
        *,
        definition: VectorStoreCollectionDefinition | None = None,
        collection_name: str | None = None,
        embedding_generator: Any = None,
        client: DXMTClient | None = None,
        url: str | None = None,
        api_key: str | SecretString | None = None,
        managed_client: bool | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        server_embedding: Mapping[str, Any] | str | None = None,
        server_embedding_field: str | None = None,
        text_analyzer: str = "standard",
    ) -> None:
        """Initialize the collection.

        Args:
            record_type: Registered application model, or dict with an explicit definition.
            definition: Explicit dictionary schema.
            collection_name: Table name, overriding the model's collection name.
            embedding_generator: Default local embedding client.
            client: Existing ``DXMTClient``. When supplied, url, api_key and the .env options are
                ignored and the client remains caller-owned.
            url: Server URL override (``http://host:port``); otherwise ``DXMT_URL``, then localhost:9999.
            api_key: API key override; otherwise ``DXMT_API_KEY``.
            managed_client: Whether to close a connector-created client. Defaults to True.
            env_file_path: Optional .env file to load before process environment variables.
            env_file_encoding: Encoding for the .env file.
            server_embedding: ``{"model": ..., "base_url": ..., "api_key_env": ...}`` to have the
                server embed with that model, or ``"server"`` for the model the server itself was
                started with; the table is created accordingly, and a record whose vector is
                ``None`` is embedded from ``server_embedding_field``.
            server_embedding_field: The data field whose text the server embeds. Defaults to the
                full-text field when there is exactly one.
            text_analyzer: ``standard`` or ``chinese`` for the BM25 index.
        """
        super().__init__(
            record_type,
            definition=definition,
            collection_name=collection_name,
            embedding_generator=embedding_generator,
        )
        resolved, owns = _resolve_client(
            client,
            url=url,
            api_key=api_key,
            managed_client=managed_client,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        self.client = resolved
        self.managed_client = owns
        self.server_embedding: dict[str, Any] | str | None = (
            "server" if server_embedding == "server" else dict(server_embedding) if server_embedding else None
        )
        self.text_analyzer = text_analyzer
        self._closed = False
        self._full_text_field = self._resolve_full_text_field()
        self.server_embedding_field = server_embedding_field or (
            self._full_text_field.name if self._full_text_field is not None else None
        )
        if self.server_embedding is not None and self.server_embedding_field is None:
            raise ValueError(
                "server_embedding needs server_embedding_field: the data field whose text the server embeds."
            )

    # ---- model -----------------------------------------------------------------------

    def _validate_data_model(self) -> None:
        super()._validate_data_model()
        if self.definition.key_field.is_auto_generated:
            raise NotImplementedError("DXMT requires application-provided record ids; it does not generate keys.")
        vectors = self.definition.vector_fields
        if len(vectors) != 1:
            raise NotImplementedError("A DXMT collection holds exactly one vector field.")
        field = vectors[0]
        if not isinstance(field.dimensions, int) or isinstance(field.dimensions, bool):
            raise TypeError("DXMT vector dimensions must be an integer.")
        if field.distance_function not in _DISTANCES:
            raise NotImplementedError(f"DXMT distance function '{field.distance_function}' is not supported.")
        if field.index_kind not in {"default", "hnsw"}:
            raise NotImplementedError(f"DXMT index kind '{field.index_kind}' is not supported.")
        for data in self.definition.data_fields:
            name = data.storage_name or data.name
            if not name or "." in name or name in {"id", "content"}:
                raise ValueError(
                    f"DXMT payload storage name {name!r} is reserved or contains a dot; choose another storage_name."
                )
        if any(f.provider_annotations for f in self.definition.fields):
            raise NotImplementedError("DXMT has no provider annotations.")

    def _resolve_full_text_field(self) -> Any:
        indexed = [f for f in self.definition.data_fields if f.is_full_text_indexed]
        if len(indexed) > 1:
            raise NotImplementedError("DXMT indexes one full-text field per table.")
        return indexed[0] if indexed else None

    @property
    def _vector_field(self) -> Any:
        return self.definition.vector_fields[0]

    def _storage(self, field: Any) -> str:
        return field.storage_name or field.name

    def _key_to_id(self, key: Any) -> str:
        return str(key)

    def _id_to_key(self, value: str) -> KeyT:
        kind = self.definition.key_field.type_
        if kind == "int":
            return cast(KeyT, int(value))
        if kind == "UUID":
            return cast(KeyT, UUID(value))
        return cast(KeyT, value)

    # ---- lifecycle -------------------------------------------------------------------

    async def collection_exists(self, *, operation_options: Mapping[str, Any] | None = None) -> bool:
        try:
            await self.client.aget_table(self.collection_name)
            return True
        except DXMTError as exc:
            if _not_found(exc):
                return False
            raise IntegrationException(f"DXMT table lookup failed: {exc}") from exc

    async def ensure_collection_exists(self, *, operation_options: Mapping[str, Any] | None = None) -> None:
        if await self.collection_exists():
            return
        field = self._vector_field
        text_fields = None
        if self._full_text_field is not None:
            # The index field is called `content` on every table: the server's full-text schema
            # has fixed names and only `content` is the BM25-ranked body. The source is the payload key.
            text_fields = [
                {
                    "name": "content",
                    "source": {"metadata_field": self._storage(self._full_text_field)},
                    "analyzer": self.text_analyzer,
                    "boost": 1.0,
                }
            ]
        try:
            # With a server-side model the server decides the width; "server" means the model it
            # was started with, which a table takes when it names neither a dimension nor a model.
            await self.client.acreate_table(
                name=self.collection_name,
                dimension=None if self.server_embedding is not None else cast(int, field.dimensions),
                distance_metric=_DISTANCES[cast(str, field.distance_function)],
                enable_full_text_search=text_fields is not None,
                text_fields=text_fields,
                embedding=self.server_embedding if isinstance(self.server_embedding, dict) else None,
            )
        except DXMTError as exc:
            raise IntegrationException(f"DXMT could not create table '{self.collection_name}': {exc}") from exc

    async def ensure_collection_deleted(self, *, operation_options: Mapping[str, Any] | None = None) -> None:
        try:
            await self.client.adelete_table(self.collection_name)
        except DXMTError as exc:
            if not _not_found(exc):
                raise IntegrationException(f"DXMT could not delete table '{self.collection_name}': {exc}") from exc

    async def aclose(self) -> None:
        if self.managed_client and not self._closed:
            await self.client.aclose()
            self._closed = True

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.aclose()

    # ---- conversion ------------------------------------------------------------------

    def _serialize_dicts_to_store_models(
        self, records: Sequence[dict[str, Any]], *, context: Mapping[str, Any] | None = None
    ) -> Sequence[dict[str, Any]]:
        key_name = self.definition.key_field_storage_name
        vector_name = self._storage(self._vector_field)
        rows: list[dict[str, Any]] = []
        for record in records:
            payload = {self._storage(f): record.get(self._storage(f)) for f in self.definition.data_fields}
            row: dict[str, Any] = {"id": self._key_to_id(record[key_name]), "payload": payload}
            vector = record.get(vector_name)
            if vector is not None:
                row["vector"] = [float(x) for x in vector]
            elif self.server_embedding is not None and self.server_embedding_field is not None:
                source = self.definition.try_get_field(self.server_embedding_field)
                text = payload.get(self._storage(source)) if source is not None else None
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"Record {row['id']!r} has no vector and no text in '{self.server_embedding_field}' for the server to embed."
                    )
                row["text"] = text
            else:
                raise ValueError(f"Record {row['id']!r} has no vector; set an embedding generator or server_embedding.")
            rows.append(row)
        return rows

    def _deserialize_store_models_to_dicts(
        self, records: Sequence[Any], *, context: Mapping[str, Any] | None = None
    ) -> Sequence[dict[str, Any]]:
        key_name = self.definition.key_field_storage_name
        vector_name = self._storage(self._vector_field)
        out: list[dict[str, Any]] = []
        for row in records:
            if not isinstance(row, Mapping) or "id" not in row:
                raise IntegrationInvalidResponseException("DXMT rows must be mappings with an id.")
            payload = row.get("payload") or row.get("metadata") or {}
            record: dict[str, Any] = {
                self._storage(f): payload.get(self._storage(f)) for f in self.definition.data_fields
            }
            record[key_name] = self._id_to_key(str(row["id"]))
            vector = row.get("vector")
            if vector:
                record[vector_name] = list(vector)
            out.append(record)
        return out

    # ---- CRUD ------------------------------------------------------------------------

    async def _inner_upsert(
        self, records: Sequence[Any], *, operation_options: Mapping[str, Any] | None = None
    ) -> Sequence[KeyT]:
        if not records:
            return []
        try:
            for start in range(0, len(records), _BATCH_SIZE):
                await self.client.ainsert_records(self.collection_name, list(records[start : start + _BATCH_SIZE]))
        except DXMTError as exc:
            raise IntegrationException(f"DXMT upsert failed: {exc}") from exc
        return [self._id_to_key(row["id"]) for row in records]

    async def _inner_get(
        self,
        *,
        keys: Sequence[KeyT] | None = None,
        filter: Any = None,
        top: int = 10,
        skip: int = 0,
        order_by: Mapping[str, bool] | None = None,
        include_vectors: bool = False,
        operation_options: Mapping[str, Any] | None = None,
    ) -> Sequence[Any] | None:
        if order_by:
            raise NotImplementedError("DXMT ordered retrieval is not supported. Omit order_by.")
        try:
            if keys is not None:
                if skip:
                    raise ValueError("DXMT key retrieval cannot be combined with skip.")
                ids = [self._key_to_id(k) for k in keys]
                rows: list[dict[str, Any]] = []
                for start in range(0, len(ids), _BATCH_SIZE):
                    rows.extend(await self.client.aget_records(self.collection_name, ids[start : start + _BATCH_SIZE]))
                if not include_vectors:
                    for row in rows:
                        row.pop("vector", None)
                return rows
            server_filter = translate(filter, self.definition)
            rows = []
            seen = 0
            async for row in self.client.ascroll(
                self.collection_name,
                filter=server_filter,
                batch_size=min(_BATCH_SIZE, skip + top),
                with_vector=include_vectors,
            ):
                if seen < skip:
                    seen += 1
                    continue
                rows.append(row)
                if len(rows) >= top:
                    break
            return rows
        except DXMTError as exc:
            if _not_found(exc):
                return []
            raise IntegrationException(f"DXMT get failed: {exc}") from exc

    async def _inner_delete(self, keys: Sequence[KeyT], *, operation_options: Mapping[str, Any] | None = None) -> None:
        ids = [self._key_to_id(k) for k in keys]
        try:
            for start in range(0, len(ids), _BATCH_SIZE):
                await self.client.adelete_records(self.collection_name, ids[start : start + _BATCH_SIZE])
        except DXMTError as exc:
            if not _not_found(exc):
                raise IntegrationException(f"DXMT delete failed: {exc}") from exc

    # ---- search ----------------------------------------------------------------------

    async def _inner_search(
        self,
        *,
        search_type: str,
        filter: Any = None,
        values: Any | None = None,
        vector: Any | None = None,
        top: int = 3,
        skip: int = 0,
        include_vectors: bool = False,
        vector_property_name: str | None = None,
        additional_property_name: str | None = None,
        score_threshold: float | None = None,
        operation_options: Mapping[str, Any] | None = None,
    ) -> SearchResults[Any]:
        server_filter = translate(filter, self.definition)
        if score_threshold is not None and (
            isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float))
        ):
            raise ValueError("DXMT score_threshold must be a number.")
        text = values if isinstance(values, str) else None
        query_vector = [float(x) for x in vector] if vector is not None else None
        if search_type == "keyword_hybrid":
            if additional_property_name is not None:
                field = self.definition.try_get_field(additional_property_name)
                if field is None or self._full_text_field is None or field.name != self._full_text_field.name:
                    raise NotImplementedError(
                        "DXMT keyword-hybrid search runs over the one full-text indexed field of the collection."
                    )
            if self._full_text_field is None:
                raise NotImplementedError("Keyword-hybrid search needs a data field marked is_full_text_indexed.")
            if text is None:
                raise ValueError("Keyword-hybrid search needs string values.")
            try:
                hits = await self.client.ahybrid_search(
                    self.collection_name, vector=query_vector, text=text, k=top + skip, filter=server_filter
                )
            except DXMTError as exc:
                raise IntegrationException(f"DXMT hybrid search failed: {exc}") from exc
            hits = hits[skip:]
        else:
            if query_vector is None and (text is None or self.server_embedding is None):
                raise ValueError(
                    "DXMT vector search requires a vector, an embedding generator, or server_embedding with string values."
                )
            try:
                hits = await self.client.asearch(
                    self.collection_name,
                    vector=query_vector,
                    text=text if query_vector is None else None,
                    k=top,
                    offset=skip,
                    filter=server_filter,
                    score_threshold=score_threshold,
                )
            except DXMTError as exc:
                raise IntegrationException(f"DXMT search failed: {exc}") from exc
        if score_threshold is not None:
            hits = [h for h in hits if float(h.get("score", 0.0)) >= score_threshold]
        if include_vectors and hits and any(h.get("vector") is None for h in hits):
            # A search hit carries `vector: null`; the rows have it.
            rows = await self.client.aget_records(self.collection_name, [h["id"] for h in hits])
            vectors = {r["id"]: r.get("vector") for r in rows}
            for h in hits:
                h["vector"] = vectors.get(h["id"])
        return SearchResults(hits)

    def _get_record_from_result(self, result: Any) -> Any:
        if not isinstance(result, Mapping):
            raise IntegrationInvalidResponseException("Expected a DXMT hit mapping.")
        return result

    def _get_score_from_result(self, result: Any) -> float | None:
        score = self._get_record_from_result(result).get("score")
        return float(score) if score is not None else None


class DXMTStore(BaseVectorStore):
    """A DXMT server as an Agent Framework vector store; its tables are the collections."""

    def __init__(
        self,
        *,
        client: DXMTClient | None = None,
        url: str | None = None,
        api_key: str | SecretString | None = None,
        embedding_generator: Any = None,
        managed_client: bool | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        resolved, owns = _resolve_client(
            client,
            url=url,
            api_key=api_key,
            managed_client=managed_client,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        super().__init__(embedding_generator=embedding_generator, managed_client=owns)
        self.client = resolved
        self._closed = False

    def get_collection(
        self,
        record_type: type[ModelT],
        *,
        definition: VectorStoreCollectionDefinition | None = None,
        collection_name: str | None = None,
        embedding_generator: Any = None,
        **kwargs: Any,
    ) -> DXMTCollection[Any, ModelT]:
        return DXMTCollection(
            record_type,
            definition=definition,
            collection_name=collection_name,
            embedding_generator=embedding_generator or self.embedding_generator,
            client=self.client,
            managed_client=False,
            **kwargs,
        )

    async def list_collection_names(self, *, operation_options: Mapping[str, Any] | None = None) -> Sequence[str]:
        try:
            tables = await self.client.alist_tables()
        except DXMTError as exc:
            raise IntegrationException(f"DXMT could not list tables: {exc}") from exc
        return [t["name"] for t in tables]

    async def _inner_ensure_collection_deleted(
        self, collection_name: str, *, operation_options: Mapping[str, Any] | None = None
    ) -> None:
        try:
            await self.client.adelete_table(collection_name)
        except DXMTError as exc:
            if not _not_found(exc):
                raise IntegrationException(f"DXMT could not delete table '{collection_name}': {exc}") from exc

    async def aclose(self) -> None:
        if self.managed_client and not self._closed:
            await self.client.aclose()
            self._closed = True

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.aclose()
