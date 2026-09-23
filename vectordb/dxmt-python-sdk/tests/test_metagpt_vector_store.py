"""Unit tests for DXMT LlamaIndex / MetaGPT VectorStore integration.

DXMTVectorStore extends BasePydanticVectorStore which declares `client` as a
property, causing `self.client` to return None after super().__init__() when
tested with a Mock.  Rather than fighting Pydantic's metaclass, we test each
method by invoking it on a minimal namespace object (a plain Mock that has
exactly the attributes the method accesses), which tests the exact same code
paths without the Pydantic/property complexity.
"""

import asyncio
from unittest.mock import Mock, patch
import pytest

pytest.importorskip("llama_index", reason="llama_index not installed")

from dxmt.metagpt_vector_store import DXMTVectorStore


def _self(client=None):
    """Build a minimal mock self for DXMTVectorStore method tests."""
    s = Mock(spec=[])  # spec=[] means no auto-attributes; we set them explicitly
    s.table_name = "docs"
    s.dimension = 3
    s.distance_metric = "cosine"
    s.client = client or Mock()
    return s


def _mock_client():
    c = Mock()
    c.get_table.return_value = {"name": "docs", "dimension": 3}
    c.insert_records.return_value = None
    c.search.return_value = []
    c.delete_records.return_value = None
    c.hybrid_search = Mock(return_value=[])
    c.full_text_search = Mock(return_value=[])
    return c


# ---------------------------------------------------------------------------
# P0 fix: delete() must call delete_records(ids=...) not delete_record()
# ---------------------------------------------------------------------------

class TestDeleteUsesDeleteRecords:
    def test_delete_calls_delete_records(self):
        client = _mock_client()
        client.search.return_value = [{"id": "n1"}, {"id": "n2"}]
        self_ = _self(client)

        DXMTVectorStore.delete(self_, "ref_doc_1")

        client.delete_records.assert_called_once_with(table_name="docs", ids=["n1", "n2"])

    def test_delete_no_matching_nodes_skips_delete_records(self):
        client = _mock_client()
        client.search.return_value = []
        self_ = _self(client)

        DXMTVectorStore.delete(self_, "nonexistent_ref")

        client.delete_records.assert_not_called()

    def test_delete_never_calls_delete_record_singular(self):
        """Confirm the nonexistent singular method is never called."""
        client = _mock_client()
        client.search.return_value = [{"id": "x"}]
        self_ = _self(client)

        DXMTVectorStore.delete(self_, "ref")

        # delete_record (singular, without 's') must never be called
        assert not hasattr(client, "delete_record") or not client.delete_record.called


# ---------------------------------------------------------------------------
# Query mode routing
# ---------------------------------------------------------------------------

class TestQueryModeRouting:
    def _make_query(self, mode_name=None, embedding=None, query_str=None, k=5, alpha=None):
        q = Mock(spec=[])
        q.similarity_top_k = k
        q.query_embedding = embedding or [0.1, 0.2, 0.3]
        q.query_str = query_str
        q.filters = None
        q.alpha = alpha
        try:
            from llama_index.core.vector_stores.types import VectorStoreQueryMode
            q.mode = getattr(VectorStoreQueryMode, mode_name) if mode_name else VectorStoreQueryMode.DEFAULT
        except ImportError:
            q.mode = None
        return q

    def test_default_mode_calls_search(self):
        client = _mock_client()
        self_ = _self(client)
        q = self._make_query()
        DXMTVectorStore.query(self_, q)
        client.search.assert_called_once()

    def test_hybrid_mode_calls_hybrid_search(self):
        try:
            from llama_index.core.vector_stores.types import VectorStoreQueryMode
        except ImportError:
            pytest.skip("llama_index not installed")

        client = _mock_client()
        self_ = _self(client)
        q = self._make_query(mode_name="HYBRID", query_str="test query", alpha=0.6)
        DXMTVectorStore.query(self_, q)

        client.hybrid_search.assert_called_once()
        call = client.hybrid_search.call_args
        assert call[1]["alpha"] == 0.6

    def test_semantic_hybrid_passes_no_alpha_when_the_query_has_none(self):
        """It used to substitute 0.7, which is not a neutral default: a weight selects
        weighted fusion, so inventing one changed how every unweighted query was scored --
        and the weight itself was then ignored by the method the SDK actually asked for."""
        try:
            from llama_index.core.vector_stores.types import VectorStoreQueryMode
        except ImportError:
            pytest.skip("llama_index not installed")

        client = _mock_client()
        self_ = _self(client)
        q = self._make_query(mode_name="SEMANTIC_HYBRID", query_str="query", alpha=None)
        DXMTVectorStore.query(self_, q)

        client.hybrid_search.assert_called_once()
        call = client.hybrid_search.call_args
        assert call[1]["alpha"] is None

    def test_text_search_mode_calls_full_text_search(self):
        try:
            from llama_index.core.vector_stores.types import VectorStoreQueryMode
        except ImportError:
            pytest.skip("llama_index not installed")

        client = _mock_client()
        self_ = _self(client)
        q = self._make_query(mode_name="TEXT_SEARCH", query_str="hello world")
        DXMTVectorStore.query(self_, q)

        client.full_text_search.assert_called_once()
        call = client.full_text_search.call_args
        assert call[1]["query"] == "hello world"


# ---------------------------------------------------------------------------
# Filter conversion
# ---------------------------------------------------------------------------

class TestFilterConversion:
    def test_plain_dict_equality(self):
        result = DXMTVectorStore._convert_filters(_self(), {"category": "AI"})
        assert result["must"][0]["operation"]["type"] == "Equal"
        assert result["must"][0]["operation"]["value"] == "AI"

    def test_plain_dict_mongodb_operator(self):
        result = DXMTVectorStore._convert_filters(_self(), {"year": {"$gte": 2022}})
        assert result["must"][0]["operation"]["type"] == "GreaterThanOrEqual"

    def test_plain_dict_ne_operator(self):
        result = DXMTVectorStore._convert_filters(_self(), {"status": {"$ne": "deleted"}})
        assert result["must"][0]["operation"]["type"] == "NotEqual"

    def test_none_returns_none(self):
        assert DXMTVectorStore._convert_filters(_self(), None) is None

    def test_llamaindex_metadata_filters_and(self):
        try:
            from llama_index.core.vector_stores.types import (
                FilterOperator, FilterCondition, MetadataFilters, MetadataFilter,
            )
        except ImportError:
            pytest.skip("llama_index not installed")

        mf = MetadataFilters(
            filters=[
                MetadataFilter(key="year", value=2022, operator=FilterOperator.GTE),
                MetadataFilter(key="cat", value="AI", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        result = DXMTVectorStore._convert_filters(_self(), mf)
        assert "must" in result
        assert len(result["must"]) == 2
        ops = {c["key"]: c["operation"]["type"] for c in result["must"]}
        assert ops["year"] == "GreaterThanOrEqual"
        assert ops["cat"] == "Equal"

    def test_llamaindex_metadata_filters_or(self):
        try:
            from llama_index.core.vector_stores.types import (
                FilterOperator, FilterCondition, MetadataFilters, MetadataFilter,
            )
        except ImportError:
            pytest.skip("llama_index not installed")

        mf = MetadataFilters(
            filters=[
                MetadataFilter(key="cat", value="A", operator=FilterOperator.EQ),
                MetadataFilter(key="cat", value="B", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )
        result = DXMTVectorStore._convert_filters(_self(), mf)
        assert "should" in result

    def test_llamaindex_operator_mapping_in(self):
        try:
            from llama_index.core.vector_stores.types import (
                FilterOperator, FilterCondition, MetadataFilters, MetadataFilter,
            )
        except ImportError:
            pytest.skip("llama_index not installed")

        mf = MetadataFilters(
            filters=[MetadataFilter(key="ids", value=["a", "b"], operator=FilterOperator.IN)],
            condition=FilterCondition.AND,
        )
        result = DXMTVectorStore._convert_filters(_self(), mf)
        assert result["must"][0]["operation"]["type"] == "In"


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------

class TestAsyncVariants:
    def _make_node(self):
        try:
            from llama_index.core.schema import TextNode
            node = TextNode(text="hello", id_="n1")
            node._embedding = [0.1, 0.2, 0.3]
            return node
        except ImportError:
            pytest.skip("llama_index not installed")

    def test_async_add_delegates_to_add(self):
        node = self._make_node()
        # The async wrapper calls self.add(...) in an executor thread.
        # We attach the method directly to the mock self so the lambda can reach it.
        self_ = _self(_mock_client())
        add_mock = Mock(return_value=["n1"])
        self_.add = add_mock

        result = asyncio.run(DXMTVectorStore.async_add(self_, [node]))

        add_mock.assert_called_once()
        assert result == ["n1"]

    def test_adelete_delegates_to_delete(self):
        self_ = _self(_mock_client())
        delete_mock = Mock(return_value=None)
        self_.delete = delete_mock

        asyncio.run(DXMTVectorStore.adelete(self_, "ref_doc"))

        delete_mock.assert_called_once_with("ref_doc")

    def test_aquery_delegates_to_query(self):
        self_ = _self(_mock_client())
        fake_result = Mock()
        q = Mock()
        query_mock = Mock(return_value=fake_result)
        self_.query = query_mock

        result = asyncio.run(DXMTVectorStore.aquery(self_, q))

        query_mock.assert_called_once_with(q)
        assert result is fake_result
