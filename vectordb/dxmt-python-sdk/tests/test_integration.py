"""Integration tests for DXMT SDK.

These tests require a running DXMT server; host/port default to localhost:9999
and can be overridden with DXMT_TEST_HOST / DXMT_TEST_PORT.

Run with:
    pytest tests/test_integration.py -v

Or skip if no server:
    pytest tests/test_integration.py -v -m "not integration"
"""

import asyncio
import os
import random
from typing import Generator

import pytest

from dxmt import Client, Field
from dxmt.exceptions import DXMTSearchError, DXMTTableError

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

TEST_HOST = os.environ.get("DXMT_TEST_HOST", "localhost")
# A server started with DXMT_API_KEY needs it on every request; this suite runs both ways.
TEST_API_KEY = os.environ.get("DXMT_API_KEY")
TEST_PORT = int(os.environ.get("DXMT_TEST_PORT", "9999"))


@pytest.fixture(scope="module")
def test_table_name() -> str:
    """Generate a unique table name for testing."""
    return f"test_table_{random.randint(1000, 9999)}"


@pytest.fixture(scope="module")
def client() -> Generator[Client, None, None]:
    """Create a client for testing."""
    client = Client(host=TEST_HOST, port=TEST_PORT, api_key=TEST_API_KEY)
    yield client
    client.close()


@pytest.fixture(scope="module")
def setup_table(client: Client, test_table_name: str) -> Generator[str, None, None]:
    """Create a test table and clean up after tests."""
    # Delete table if it already exists (from previous test)
    try:
        client.delete_table(test_table_name)
    except Exception:
        pass  # Ignore if table doesn't exist

    # Create table
    try:
        client.create_table(
            name=test_table_name,
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw",
        )
    except Exception as e:
        pytest.skip(f"Cannot create table, DXMT server may not be running: {e}")

    yield test_table_name

    # Cleanup
    try:
        client.delete_table(test_table_name)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture(scope="module")
def setup_hybrid_table(client: Client) -> Generator[str, None, None]:
    """Create table with full-text search enabled for hybrid tests."""
    table_name = f"hybrid_test_{random.randint(1000, 9999)}"

    # Delete table if it already exists
    try:
        client.delete_table(table_name)
    except Exception:
        pass  # Ignore if table doesn't exist

    try:
        client.create_table(
            name=table_name,
            dimension=128,
            distance_metric="cosine",
            enable_full_text_search=True,
        )
    except Exception as e:
        pytest.skip(f"Cannot create hybrid search table: {e}")

    # Insert test data with text
    records = [
        {
            "id": f"hybrid_{i}",
            "vector": [float(i + 1) / 10] * 128,
            "metadata": {
                "text": f"This is document {i} about machine learning",
                "category": "AI",
            },
        }
        for i in range(10)
    ]
    try:
        client.insert_records(table_name=table_name, records=records)
    except Exception as e:
        pytest.skip(f"Cannot insert records into hybrid table: {e}")

    yield table_name

    # Cleanup
    try:
        client.delete_table(table_name)
    except Exception:
        pass


class TestServerConnection:
    """Test server connectivity."""

    def test_server_available(self, client: Client) -> None:
        """Test that server is available."""
        tables = client.list_tables()
        assert isinstance(tables, list)


class TestTableManagement:
    """Test table management operations."""

    def test_create_table(self, client: Client, test_table_name: str) -> None:
        """Test table creation."""
        result = client.create_table(
            name=test_table_name,
            dimension=128,
            distance_metric="cosine",
            index_type="hnsw",
        )
        assert result["success"] is True

    def test_list_tables(self, client: Client, setup_table: str) -> None:
        """Test listing tables."""
        tables = client.list_tables()
        assert isinstance(tables, list)
        table_names = [t["name"] for t in tables]
        assert setup_table in table_names

    def test_get_table(self, client: Client, setup_table: str) -> None:
        """Test getting table info."""
        table_info = client.get_table(setup_table)
        assert table_info["name"] == setup_table
        assert table_info["dimension"] == 128

    def test_delete_table(self, client: Client) -> None:
        """Test table deletion."""
        temp_table = f"temp_table_{random.randint(1000, 9999)}"
        client.create_table(name=temp_table, dimension=64)
        result = client.delete_table(temp_table)
        assert result["success"] is True

    def test_create_duplicate_table_error(
        self, client: Client, setup_table: str
    ) -> None:
        """Test creating duplicate table raises error."""
        with pytest.raises(DXMTTableError):
            client.create_table(name=setup_table, dimension=128)


class TestRecordOperations:
    """Test record insertion and retrieval."""

    def test_insert_single_record(self, client: Client, setup_table: str) -> None:
        """Test inserting a single record."""
        records = [
            {
                "id": "test_1",
                "vector": [0.1] * 128,
                "metadata": {"text": "Test document 1"},
            }
        ]
        result = client.insert_records(table_name=setup_table, records=records)
        assert result["success"] is True

    def test_insert_multiple_records(self, client: Client, setup_table: str) -> None:
        """Test inserting multiple records."""
        records = [
            {
                "id": f"doc_{i}",
                "vector": [float(i + 1) / 10] * 128,
                "metadata": {"index": i, "text": f"Document {i}"},
            }
            for i in range(10)
        ]
        result = client.insert_records(table_name=setup_table, records=records)
        assert result["success"] is True

    def test_get_records_by_ids(self, client: Client, setup_table: str) -> None:
        """Test retrieving records by IDs."""
        # Insert test records
        records = [
            {"id": "get_1", "vector": [0.5] * 128, "metadata": {"type": "test"}},
            {"id": "get_2", "vector": [0.6] * 128, "metadata": {"type": "test"}},
        ]
        client.insert_records(table_name=setup_table, records=records)

        # Retrieve records
        retrieved = client.get_records(table_name=setup_table, ids=["get_1", "get_2"])
        assert len(retrieved) == 2
        ids = [r["id"] for r in retrieved]
        assert "get_1" in ids
        assert "get_2" in ids

    def test_delete_records(self, client: Client, setup_table: str) -> None:
        """Test deleting records."""
        # Insert test record
        records = [
            {"id": "delete_1", "vector": [0.7] * 128, "metadata": {}},
        ]
        client.insert_records(table_name=setup_table, records=records)

        # Delete record
        result = client.delete_records(table_name=setup_table, ids=["delete_1"])
        assert result["success"] is True

    def test_payload_only_update_preserves_vector(self, client: Client, setup_table: str) -> None:
        """A metadata update must not erase the dense vector used for search."""
        client.insert_records(
            table_name=setup_table,
            records=[{"id": "update_1", "vector": [1.0] + [0.0] * 127, "metadata": {"v": 1}}],
        )
        client.update_record(table_name=setup_table, record_id="update_1", payload={"v": 2})
        record = client.get_records(table_name=setup_table, ids=["update_1"])[0]
        assert record["vector"][0] == 1.0
        assert record["metadata"] == {"v": 2}


class TestScrollAndUpload:
    """`upload` in parallel batches, then `scroll` everything back by cursor."""

    def test_upload_then_scroll_returns_every_record_once(self, client: Client) -> None:
        name = f"test_scroll_{random.randint(1000, 9999)}"
        client.create_table(name=name, dimension=8)
        try:
            records = ({"id": f"s{i}", "vector": [random.random() for _ in range(8)],
                        "payload": {"even": i % 2 == 0}} for i in range(250))
            result = client.upload(name, records, batch_size=40, parallel=3)
            assert result.ok, result.failed
            assert (result.uploaded, result.batches) == (250, 7)

            ids = [r["id"] for r in client.scroll(name, batch_size=33)]
            assert len(ids) == 250 and len(set(ids)) == 250

            evens = [r["id"] for r in client.scroll(name, batch_size=20,
                     filter={"must": [{"key": "even", "operation": {"type": "Equal", "value": True}}]})]
            assert sorted(evens) == sorted(f"s{i}" for i in range(0, 250, 2))
        finally:
            client.delete_table(name)


class TestDeleteByFilter:
    """Deleting by filter, which is one request whatever it matches."""

    def test_a_filter_deletes_what_it_matches_and_nothing_else(self, client: Client) -> None:
        name = f"test_delfilter_{random.randint(1000, 9999)}"
        client.create_table(name=name, dimension=2)
        try:
            client.insert_records(name, [
                {"id": f"r{i}", "vector": [1.0, float(i)],
                 "payload": {"tenant": "a" if i % 2 else "b"}}
                for i in range(10)
            ])
            answer = client.delete_records(name, filter=Field("tenant") == "a")
            assert answer["data"]["deleted_count"] == 5

            left = sorted(r["id"] for r in client.scroll(name))
            assert left == ["r0", "r2", "r4", "r6", "r8"]
        finally:
            client.delete_table(name)

    def test_one_selector_or_the_other(self, client: Client) -> None:
        # An empty delete must not read as "delete everything", and two selectors have no
        # obvious meaning -- intersection and union are both defensible, which is the problem.
        with pytest.raises(ValueError):
            client.delete_records("t")
        with pytest.raises(ValueError):
            client.delete_records("t", ["a"], filter=Field("x") == 1)


class TestFieldFiltersAgainstTheServer:
    """What a `Field` expression selects, on arrays, missing fields and mixed number types,
    through both a listing and a search."""

    DOCS = {
        "d1": {"lang": "en", "year": 2021, "tags": ["ai", "ml"]},
        "d2": {"lang": "de", "year": 2019.0, "tags": ["spam"]},
        "d3": {"year": 2023},
        "d4": {"lang": "en", "year": [2019, 2024]},
    }
    CASES = [
        (Field("lang") != "en", {"d2", "d3"}),         # a missing field is "not en"
        (Field("lang").not_in(["en"]), {"d2"}),         # but is not "in something else"
        (Field("tags") == "ai", {"d1"}),                # any element
        (Field("year").between(2020, 2023), {"d1", "d3", "d4"}),  # d4: one element per bound
        (Field("year") == 2019, {"d2", "d4"}),          # 2019.0 is 2019
        ((Field("lang") == "en") & ~Field("tags").is_in(["spam"]), {"d1", "d4"}),
        ((Field("lang") == "de") | (Field("year") >= 2023), {"d2", "d3", "d4"}),
        (Field("id").is_in(["d1", "d3"]), {"d1", "d3"}),
    ]

    def test_listing_and_search_select_the_same_records(self, client: Client) -> None:
        name = f"test_field_{random.randint(1000, 9999)}"
        client.create_table(name=name, dimension=2)
        try:
            client.insert_records(name, [
                {"id": rid, "vector": [1.0, float(i)], "payload": payload}
                for i, (rid, payload) in enumerate(self.DOCS.items())
            ])
            for f, expected in self.CASES:
                listed = {r["id"] for r in client.scroll(name, filter=f)}
                found = {h["id"] for h in client.search(name, [1.0, 0.0], k=10, filter=f)}
                assert (listed, found) == (expected, expected), f
        finally:
            client.delete_table(name)


class TestVectorSearch:
    """Test vector search operations."""

    @pytest.fixture(autouse=True)
    def setup_search_data(self, client: Client, setup_table: str) -> None:
        """Insert test data for search."""
        records = [
            {
                "id": f"search_{i}",
                "vector": [float(i + 1) / 100] * 128,
                "metadata": {"index": i, "category": "test"},
            }
            for i in range(20)
        ]
        client.insert_records(table_name=setup_table, records=records)

    def test_basic_search(self, client: Client, setup_table: str) -> None:
        """Test basic vector search."""
        query_vector = [0.05] * 128
        results = client.search(table_name=setup_table, vector=query_vector, k=5)

        assert len(results) <= 5
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_with_k(self, client: Client, setup_table: str) -> None:
        """Test search with different k values."""
        query_vector = [0.1] * 128

        # Test k=1
        results = client.search(table_name=setup_table, vector=query_vector, k=1)
        assert len(results) <= 1

        # Test k=10
        results = client.search(table_name=setup_table, vector=query_vector, k=10)
        assert len(results) <= 10

    def test_search_results_sorted(self, client: Client, setup_table: str) -> None:
        """Test that search results are sorted by score."""
        query_vector = [0.05] * 128
        results = client.search(table_name=setup_table, vector=query_vector, k=10)

        if len(results) > 1:
            scores = [r["score"] for r in results]
            # Scores should be in descending order (higher is better)
            assert scores == sorted(scores, reverse=True)


class TestAsyncOperations:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_async_table_operations(self, test_table_name: str) -> None:
        """Test async table operations."""
        async with Client(host=TEST_HOST, port=TEST_PORT, api_key=TEST_API_KEY) as client:
            # Create table
            async_table = f"{test_table_name}_async"
            result = await client.acreate_table(
                name=async_table, dimension=64, distance_metric="cosine"
            )
            assert result["success"] is True

            # List tables
            tables = await client.alist_tables()
            assert isinstance(tables, list)

            # Get table
            table_info = await client.aget_table(async_table)
            assert table_info["name"] == async_table

            # Delete table
            await client.adelete_table(async_table)

    @pytest.mark.asyncio
    async def test_async_record_operations(self, setup_table: str) -> None:
        """Test async record operations."""
        async with Client(host=TEST_HOST, port=TEST_PORT, api_key=TEST_API_KEY) as client:
            # Insert records
            records = [
                {"id": "async_1", "vector": [0.8] * 128, "metadata": {}},
                {"id": "async_2", "vector": [0.9] * 128, "metadata": {}},
            ]
            result = await client.ainsert_records(
                table_name=setup_table, records=records
            )
            assert result["success"] is True

            # Get records
            retrieved = await client.aget_records(
                table_name=setup_table, ids=["async_1", "async_2"]
            )
            assert len(retrieved) == 2

            # Delete records
            await client.adelete_records(table_name=setup_table, ids=["async_1"])

    @pytest.mark.asyncio
    async def test_async_search(self, setup_table: str) -> None:
        """Test async search."""
        async with Client(host=TEST_HOST, port=TEST_PORT, api_key=TEST_API_KEY) as client:
            # Insert test data
            records = [
                {"id": f"async_search_{i}", "vector": [float(i + 1) / 10] * 128, "metadata": {}}
                for i in range(5)
            ]
            await client.ainsert_records(table_name=setup_table, records=records)

            # Search
            query_vector = [0.25] * 128
            results = await client.asearch(
                table_name=setup_table, vector=query_vector, k=3
            )
            assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, setup_table: str) -> None:
        """Test concurrent async searches."""
        async with Client(host=TEST_HOST, port=TEST_PORT, api_key=TEST_API_KEY) as client:
            # Perform multiple searches concurrently
            queries = [[float(i + 1) / 100] * 128 for i in range(5)]
            tasks = [
                client.asearch(table_name=setup_table, vector=q, k=3) for q in queries
            ]
            results_list = await asyncio.gather(*tasks)

            assert len(results_list) == 5
            for results in results_list:
                assert isinstance(results, list)
                assert len(results) <= 3


class TestHybridSearch:
    """Test hybrid search (vector + text)."""

    def test_hybrid_search_balanced(
        self, client: Client, setup_hybrid_table: str
    ) -> None:
        """Test hybrid search with balanced weights."""
        query_vector = [0.5] * 128
        results = client.hybrid_search(
            table_name=setup_hybrid_table,
            vector=query_vector,
            text="machine learning",
            vector_weight=0.5,  # Balanced
            k=5,
        )

        assert len(results) <= 5
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)

    def test_the_weight_changes_the_ranking(self, client: Client) -> None:
        """`vector_weight` has to reach a fusion method that reads it.

        The server's default fusion is `dbsf`, which normalises each side by its own
        distribution and ignores the weight -- and this SDK sent the weight without ever
        naming a method, so 0.0 and 1.0 came back with identical rows and identical scores.
        Every test above passes a weight and asserts only the shape of the answer, which is
        how a knob that did nothing stayed in the API and in three integrations' docs.
        """
        name = f"fusion_{random.randint(1000, 9999)}"
        try:
            client.create_table(name=name, dimension=3, enable_full_text_search=True)
        except Exception as e:  # noqa: BLE001 - the feature may not be compiled in
            pytest.skip(f"cannot create a full-text table: {e}")
        try:
            client.insert_records(name, [
                # The vector favours "rust"; the word "python" favours the other document.
                {"id": "vec", "vector": [1.0, 0.0, 0.0],
                 "payload": {"content": "rust systems programming"}},
                {"id": "text", "vector": [0.0, 1.0, 0.0],
                 "payload": {"content": "python data science"}},
            ])

            def top(**kwargs):
                hits = client.hybrid_search(
                    name, vector=[1.0, 0.0, 0.0], text="python", k=2, **kwargs
                )
                if not hits:
                    pytest.skip("hybrid search returned nothing; full-text may be disabled")
                return hits[0]["id"]

            assert top(vector_weight=1.0) == "vec"
            assert top(vector_weight=0.0) == "text"
            # Naming the method explicitly is still possible, and no weight leaves the server
            # on its own default rather than quietly selecting one here.
            assert top(vector_weight=1.0, fusion_method="reciprocal_rank_fusion") in {"vec", "text"}
            assert top() == top(fusion_method="dbsf")
        finally:
            client.delete_table(name)

    def test_hybrid_search_text_only(
        self, client: Client, setup_hybrid_table: str
    ) -> None:
        """Test hybrid search with text-only (vector_weight=0)."""
        query_vector = [0.5] * 128
        results = client.hybrid_search(
            table_name=setup_hybrid_table,
            vector=query_vector,
            text="machine learning",
            vector_weight=0.0,  # Text only
            k=5,
        )

        assert len(results) <= 5

    def test_hybrid_search_vector_only(
        self, client: Client, setup_hybrid_table: str
    ) -> None:
        """Test hybrid search with vector-only (vector_weight=1)."""
        query_vector = [0.5] * 128
        results = client.hybrid_search(
            table_name=setup_hybrid_table,
            vector=query_vector,
            text="machine learning",
            vector_weight=1.0,  # Vector only
            k=5,
        )

        assert len(results) <= 5


class TestFullTextSearch:
    """Test full-text search."""

    def test_full_text_search(self, client: Client, setup_hybrid_table: str) -> None:
        """Test pure full-text search."""
        try:
            results = client.full_text_search(
                table_name=setup_hybrid_table, query="machine learning", k=5
            )
        except DXMTSearchError as exc:
            if "FEATURE_DISABLED" in str(exc):
                pytest.skip("server was built without the text-bm25 feature")
            raise

        assert isinstance(results, list)
        assert len(results) <= 5


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_record_list(self, client: Client, setup_table: str) -> None:
        """Test inserting empty record list."""
        from dxmt.exceptions import DXMTError

        # Server should reject empty record list with 400 Bad Request
        with pytest.raises(DXMTError):
            client.insert_records(table_name=setup_table, records=[])

    def test_search_non_existent_table(self, client: Client) -> None:
        """Test searching in non-existent table."""
        from dxmt.exceptions import DXMTError

        with pytest.raises(DXMTError):
            client.search(
                table_name="non_existent_table_123456", vector=[0.1] * 128, k=5
            )

    def test_large_k_value(self, client: Client, setup_table: str) -> None:
        """Test search with very large k value."""
        query_vector = [0.1] * 128
        results = client.search(table_name=setup_table, vector=query_vector, k=1000)

        # Should return all available records (likely < 1000)
        assert isinstance(results, list)

    def test_health_metrics_expose_storage_state(self, client: Client) -> None:
        body = client.health_metrics()
        assert body["storage"]["ready"] is True
