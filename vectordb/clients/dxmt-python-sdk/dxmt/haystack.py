"""
DXMT Haystack Integration

This module provides a Haystack-compatible DocumentStore implementation backed by DXMT VectorDB.
Enables using DXMT's tiered storage (Memory → SSD → S3) in Haystack pipelines for RAG and search.

Example usage:
    ```python
    from haystack import Pipeline, Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from dxmt import Client as DXMTClient
    from dxmt.haystack import DXMTDocumentStore

    # Initialize DXMT document store
    client = DXMTClient(host="localhost", port=9999)
    document_store = DXMTDocumentStore(
        client=client,
        table_name="documents",
        embedding_dim=384,
        distance_metric="cosine"
    )

    # Use in Haystack pipelines
    documents = [
        Document(content="DXMT is a vector database"),
        Document(content="Haystack is a framework for LLM applications"),
    ]
    document_store.write_documents(documents)

    # Search
    results = document_store.filter_documents(filters={"field": "meta.type", "operator": "==", "value": "doc"})
    ```
"""

import asyncio
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, List, Dict, Optional, Literal

try:
    from haystack.dataclasses import Document
    from haystack.document_stores.errors import DuplicateDocumentError, DocumentStoreError
    from haystack.document_stores.types import DuplicatePolicy
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    # Define minimal stubs for type hints
    DuplicateDocumentError = Exception
    DocumentStoreError = Exception

    # Stub Document class
    class Document:
        def __init__(self, id=None, content=None, meta=None, embedding=None, dataframe=None, blob=None, **kwargs):
            self.id = id
            self.content = content
            self.meta = meta or {}
            self.embedding = embedding
            self.dataframe = dataframe
            self.blob = blob
            self.score = None

        def to_dict(self, **kwargs):
            return {
                "id": self.id,
                "content": self.content,
                "meta": self.meta,
                "embedding": self.embedding,
                "dataframe": self.dataframe,
                "blob": self.blob,
            }

        @classmethod
        def from_dict(cls, data):
            return cls(**data)

    # Stub DuplicatePolicy enum
    class DuplicatePolicy:
        NONE = "none"
        FAIL = "fail"
        SKIP = "skip"
        OVERWRITE = "overwrite"

from dxmt.client import DXMTClient
from dxmt.exceptions import DXMTError


class DXMTDocumentStore:
    """
    Haystack DocumentStore implementation using DXMT VectorDB as backend.

    This store leverages DXMT's tiered storage architecture (Memory → SSD → S3) for
    cost-efficient document storage and retrieval at scale.

    Features:
        - Full-text search with BM25 ranking
        - Vector similarity search (cosine, euclidean, dot product)
        - Metadata filtering with complex boolean logic
        - High-performance search (3.56M QPS capability)
        - Automatic embedding storage and retrieval
        - Async operation support

    Attributes:
        client: DXMT client instance for backend communication
        table_name: Name of the DXMT table storing documents
        embedding_dim: Dimensionality of document embeddings
        distance_metric: Similarity metric for vector search
        return_embedding: Whether to return embeddings with documents
    """

    def __init__(
        self,
        client: DXMTClient,
        table_name: str = "haystack_documents",
        embedding_dim: int = 768,
        distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine",
        return_embedding: bool = False,
        async_executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initialize the DXMT Document Store.

        Args:
            client: DXMT client instance
            table_name: Name for the document table (default: "haystack_documents")
            embedding_dim: Dimension of document embeddings (default: 768)
            distance_metric: Similarity metric - "cosine", "euclidean", or "dot" (default: "cosine")
            return_embedding: Whether to return embeddings with documents (default: False)
            async_executor: Optional executor for async operations (default: None, creates single-threaded executor)

        Raises:
            ImportError: If haystack-ai is not installed
            DXMTError: If table creation fails

        Note:
            While haystack-ai is recommended, this class can work without it for testing.
            However, real usage requires haystack-ai to be installed.
        """
        # Note: We don't enforce HAYSTACK_AVAILABLE here to allow mock testing
        # Users will get errors when trying to use real Haystack features without it

        self.client = client
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.return_embedding = return_embedding

        # Async executor setup
        self._owns_executor = async_executor is None
        self.executor = (
            ThreadPoolExecutor(
                thread_name_prefix=f"async-dxmt-docstore-executor-{id(self)}",
                max_workers=1
            )
            if async_executor is None
            else async_executor
        )

        # Create table if not exists
        self._ensure_table()

    def __del__(self):
        """Cleanup executor when instance is destroyed."""
        if hasattr(self, "_owns_executor") and self._owns_executor and hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def shutdown(self):
        """Explicitly shutdown the async executor."""
        if self._owns_executor:
            self.executor.shutdown(wait=True)

    def _ensure_table(self) -> None:
        """Create table if it doesn't exist."""
        try:
            self.client.get_table(self.table_name)
        except DXMTError:
            # Table doesn't exist, create it
            try:
                self.client.create_table(
                    name=self.table_name,
                    dimension=self.embedding_dim,
                    distance_metric=self.distance_metric,
                    enable_full_text_search=True,  # Enable BM25 search
                )
            except Exception as e:
                raise DocumentStoreError(f"Failed to create table {self.table_name}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the document store to a dictionary.

        Returns:
            Dictionary with store configuration
        """
        return {
            "type": "dxmt.haystack.DXMTDocumentStore",
            "init_parameters": {
                "table_name": self.table_name,
                "embedding_dim": self.embedding_dim,
                "distance_metric": self.distance_metric,
                "return_embedding": self.return_embedding,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DXMTDocumentStore":
        """
        Deserialize the document store from a dictionary.

        Note: Requires a DXMT client to be provided separately as it cannot be serialized.

        Args:
            data: Serialized store configuration

        Returns:
            DXMTDocumentStore instance

        Raises:
            ValueError: If client is not provided
        """
        init_params = data.get("init_parameters", {})
        if "client" not in init_params:
            raise ValueError(
                "DXMT client must be provided to deserialize DXMTDocumentStore. "
                "Create a client and pass it as client=DXMTClient(...)"
            )
        return cls(**init_params)

    def count_documents(self) -> int:
        """
        Count the total number of documents in the store.

        Returns:
            Number of documents
        """
        try:
            table_info = self.client.get_table(self.table_name)
            return table_info.get("record_count", 0)
        except Exception as e:
            raise DocumentStoreError(f"Failed to count documents: {e}")

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve documents matching the provided filters.

        Haystack filter format is already compatible with DXMT's filter structure:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.rating", "operator": ">=", "value": 3}
            ]
        }
        ```

        Args:
            filters: Filter dict with operator + conditions or field + operator + value

        Returns:
            List of Documents matching the filters
        """
        try:
            # DXMT uses the same filter format as Haystack, can pass through directly
            dxmt_filter = self._convert_haystack_filters(filters) if filters else None

            # Get all matching records
            # Note: DXMT doesn't have a direct filter_all endpoint, so we use search with zero vector
            # This is a workaround - in production you may want to add a scan endpoint to DXMT
            results = self.client.search(
                table_name=self.table_name,
                vector=[0.0] * self.embedding_dim,  # Dummy vector
                k=10000,  # Large number to get all matches
                filter=dxmt_filter,
            )

            # Convert DXMT records to Haystack Documents
            documents = []
            for result in results:
                doc = self._dxmt_to_haystack_doc(result)
                if not self.return_embedding:
                    doc.embedding = None
                documents.append(doc)

            return documents

        except Exception as e:
            raise DocumentStoreError(f"Failed to filter documents: {e}")

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Write documents to the store.

        Args:
            documents: List of Haystack Document objects
            policy: How to handle duplicates (NONE, FAIL, SKIP, OVERWRITE)

        Returns:
            Number of documents written

        Raises:
            DuplicateDocumentError: If policy is FAIL and duplicate exists
            DocumentStoreError: If write operation fails
        """
        if not isinstance(documents, list):
            raise ValueError("documents must be a list of Document objects")

        if not documents:
            return 0

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        written_count = 0

        try:
            for doc in documents:
                # Check for duplicates
                if policy != DuplicatePolicy.OVERWRITE:
                    existing = self._get_document_by_id(doc.id)
                    if existing:
                        if policy == DuplicatePolicy.FAIL:
                            raise DuplicateDocumentError(f"Document with id '{doc.id}' already exists")
                        if policy == DuplicatePolicy.SKIP:
                            continue

                # Convert Haystack Document to DXMT record
                record = self._haystack_to_dxmt_record(doc)

                # Insert or update
                if policy == DuplicatePolicy.OVERWRITE and self._get_document_by_id(doc.id):
                    self.client.update_record(
                        table_name=self.table_name,
                        record_id=doc.id,
                        vector=doc.embedding if doc.embedding else None,
                        payload=record["payload"],
                    )
                else:
                    self.client.insert_records(
                        table_name=self.table_name,
                        records=[record],
                    )

                written_count += 1

        except DuplicateDocumentError:
            raise
        except Exception as e:
            raise DocumentStoreError(f"Failed to write documents: {e}")

        return written_count

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            document_ids: List of document IDs to delete
        """
        if not document_ids:
            return

        try:
            self.client.delete_records(
                table_name=self.table_name,
                ids=document_ids,
            )
        except Exception as e:
            raise DocumentStoreError(f"Failed to delete documents: {e}")

    def bm25_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
    ) -> List[Document]:
        """
        Retrieve documents using BM25 full-text search.

        Args:
            query: Search query string
            filters: Optional metadata filters
            top_k: Number of top documents to return
            scale_score: Whether to scale scores to [0, 1] range

        Returns:
            List of Documents ranked by BM25 relevance
        """
        if not query:
            raise ValueError("Query must be a non-empty string")

        try:
            # Use DXMT's full-text search
            dxmt_filter = self._convert_haystack_filters(filters) if filters else None

            results = self.client.full_text_search(
                table_name=self.table_name,
                query=query,
                k=top_k,
                filter=dxmt_filter,
            )

            # Convert to Haystack Documents with scores
            documents = []
            for result in results:
                doc = self._dxmt_to_haystack_doc(result)
                # BM25 score from DXMT
                score = result.get("score", 0.0)
                if scale_score:
                    # Scale using expit function like Haystack's InMemoryDocumentStore
                    score = self._expit(score / 8.0)  # BM25_SCALING_FACTOR = 8
                doc.score = score

                if not self.return_embedding:
                    doc.embedding = None

                documents.append(doc)

            return documents

        except Exception as e:
            raise DocumentStoreError(f"BM25 retrieval failed: {e}")

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        """
        Retrieve documents by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            filters: Optional metadata filters
            top_k: Number of top documents to return
            scale_score: Whether to scale scores to [0, 1] range
            return_embedding: Override return_embedding setting

        Returns:
            List of Documents ranked by vector similarity
        """
        if not query_embedding or not isinstance(query_embedding[0], float):
            raise ValueError("query_embedding must be a non-empty list of floats")

        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) "
                f"does not match store dimension ({self.embedding_dim})"
            )

        try:
            # Convert Haystack filters to DXMT format
            dxmt_filter = self._convert_haystack_filters(filters) if filters else None

            # Add filter to exclude documents without real embeddings
            # This matches Haystack's behavior of skipping documents without embeddings
            real_embedding_filter = {"_has_real_embedding": True}

            if dxmt_filter:
                # Combine user filter with real embedding filter using AND
                combined_filter = {
                    "$and": [
                        dxmt_filter,
                        real_embedding_filter
                    ]
                }
            else:
                combined_filter = real_embedding_filter

            results = self.client.search(
                table_name=self.table_name,
                vector=query_embedding,
                k=top_k,
                filter=combined_filter,
            )

            # Determine whether to return embeddings
            resolved_return_embedding = (
                self.return_embedding if return_embedding is None else return_embedding
            )

            # Convert to Haystack Documents
            documents = []
            for result in results:
                doc = self._dxmt_to_haystack_doc(result)
                # Convert distance to similarity score
                distance = result.get("score", 0.0)
                score = self._normalize_score(distance, scale_score)
                doc.score = score

                if not resolved_return_embedding:
                    doc.embedding = None

                documents.append(doc)

            return documents

        except Exception as e:
            raise DocumentStoreError(f"Embedding retrieval failed: {e}")

    # Async methods
    async def count_documents_async(self) -> int:
        """Async version of count_documents."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.count_documents
        )

    async def filter_documents_async(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Async version of filter_documents."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.filter_documents(filters=filters)
        )

    async def write_documents_async(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """Async version of write_documents."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.write_documents(documents=documents, policy=policy)
        )

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        """Async version of delete_documents."""
        await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.delete_documents(document_ids=document_ids)
        )

    async def bm25_retrieval_async(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
    ) -> List[Document]:
        """Async version of bm25_retrieval."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.bm25_retrieval(
                query=query, filters=filters, top_k=top_k, scale_score=scale_score
            ),
        )

    async def embedding_retrieval_async(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
    ) -> List[Document]:
        """Async version of embedding_retrieval."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                scale_score=scale_score,
                return_embedding=return_embedding,
            ),
        )

    # Helper methods
    def _convert_haystack_filters(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert Haystack filters to DXMT MongoDB-style format.

        Haystack format:
            {"field": "meta.type", "operator": "==", "value": "programming"}

        DXMT MongoDB format:
            {"type": "programming"}
            or {"type": {"$eq": "programming"}}
        """
        if not filters:
            return None

        # Operator mapping from Haystack to MongoDB
        OPERATOR_MAP = {
            "==": "$eq",
            "!=": "$ne",
            ">": "$gt",
            ">=": "$gte",
            "<": "$lt",
            "<=": "$lte",
            "in": "$in",
            "not in": "$nin",
        }

        def convert_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
            if "field" in condition:
                # Single comparison condition
                field = condition["field"]
                # Strip "meta." prefix if present
                if field.startswith("meta."):
                    field = field[5:]

                operator = condition["operator"]
                value = condition["value"]

                # For equality, use simple form
                if operator == "==":
                    return {field: value}
                else:
                    # For other operators, use MongoDB operator form
                    mongo_op = OPERATOR_MAP.get(operator, operator)
                    return {field: {mongo_op: value}}

            elif "operator" in condition and "conditions" in condition:
                # Logical operator (AND, OR, NOT)
                logical_op = condition["operator"].upper()
                sub_conditions = condition["conditions"]

                if logical_op == "AND":
                    converted = [convert_condition(c) for c in sub_conditions]
                    return {"$and": converted}
                elif logical_op == "OR":
                    converted = [convert_condition(c) for c in sub_conditions]
                    return {"$or": converted}
                elif logical_op == "NOT":
                    # NOT typically has one condition
                    converted = [convert_condition(c) for c in sub_conditions]
                    return {"$not": converted}
                else:
                    # Fallback: treat as AND
                    converted = [convert_condition(c) for c in sub_conditions]
                    return {"$and": converted}
            else:
                # Unknown format, return as-is
                return condition

        return convert_condition(filters)

    def _haystack_to_dxmt_record(self, doc: Document) -> Dict[str, Any]:
        """Convert Haystack Document to DXMT record format."""
        # Generate ID if not present
        doc_id = doc.id if doc.id else str(uuid.uuid4())

        # Prepare payload (metadata + content)
        payload = dict(doc.meta) if doc.meta else {}
        payload["content"] = doc.content

        # Optional fields - only add if present
        if hasattr(doc, 'dataframe') and doc.dataframe is not None:
            payload["dataframe"] = doc.dataframe
        if hasattr(doc, 'blob') and doc.blob is not None:
            payload["blob"] = doc.blob

        # Add content hash for deduplication
        if doc.content:
            content_hash = hashlib.md5(doc.content.encode()).hexdigest()
            payload["content_hash"] = content_hash

        # Prepare record
        record = {
            "id": doc_id,
            "payload": payload,
        }

        # Add embedding if present, otherwise use dummy vector
        if doc.embedding:
            if len(doc.embedding) != self.embedding_dim:
                raise ValueError(
                    f"Document embedding dimension ({len(doc.embedding)}) "
                    f"does not match store dimension ({self.embedding_dim})"
                )
            record["vector"] = doc.embedding
            # Mark that this document has a real embedding (not a dummy)
            payload["_has_real_embedding"] = True
        else:
            # DXMT requires a non-zero vector field
            # Use a small constant vector for BM25-only documents
            # Mark this as a dummy vector so we can filter it out during embedding retrieval
            record["vector"] = [0.001] * self.embedding_dim
            payload["_has_real_embedding"] = False

        return record

    def _dxmt_to_haystack_doc(self, result: Dict[str, Any]) -> Document:
        """Convert DXMT search result to Haystack Document."""
        metadata = result.get("metadata", {})

        # Extract core fields
        content = metadata.pop("content", None)
        dataframe = metadata.pop("dataframe", None)
        blob = metadata.pop("blob", None)
        content_hash = metadata.pop("content_hash", None)
        # Remove internal field (not part of user metadata)
        metadata.pop("_has_real_embedding", None)

        # Get embedding if present
        embedding = result.get("vector", None)

        # Build Document with only supported fields
        doc_kwargs = {
            "id": result.get("id"),
            "content": content,
            "meta": metadata,
            "embedding": embedding,
        }

        # Only add optional fields if they exist and Document supports them
        if dataframe is not None:
            try:
                doc_kwargs["dataframe"] = dataframe
            except:
                pass  # Skip if Document doesn't support dataframe

        if blob is not None:
            try:
                doc_kwargs["blob"] = blob
            except:
                pass  # Skip if Document doesn't support blob

        doc = Document(**doc_kwargs)

        return doc

    def _get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID, return None if not found."""
        try:
            results = self.client.get_records(
                table_name=self.table_name,
                ids=[doc_id],
            )
            return results[0] if results else None
        except:
            return None

    def _normalize_score(self, distance: float, scale: bool) -> float:
        """
        Convert distance to similarity score.

        Args:
            distance: Distance value from DXMT
            scale: Whether to scale to [0, 1] range

        Returns:
            Similarity score
        """
        if not scale:
            return distance

        # Convert distance to similarity based on metric
        if self.distance_metric == "cosine":
            # Cosine distance is in [0, 2], convert to similarity [0, 1]
            # 0 distance = 1 similarity, 2 distance = 0 similarity
            return (distance + 1) / 2
        elif self.distance_metric == "dot":
            # Dot product can be negative, scale using expit
            return self._expit(distance / 100.0)
        else:
            # Euclidean/other: use exponential decay
            import math
            return math.exp(-distance)

    @staticmethod
    def _expit(x: float) -> float:
        """Expit function (inverse of logit), maps R to [0, 1]."""
        import math
        return 1 / (1 + math.exp(-x))
