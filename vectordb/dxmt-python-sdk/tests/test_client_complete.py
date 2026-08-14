"""Complete unit tests for DXMT Client with mocked HTTP."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import Response

from dxmt import Client
from dxmt.exceptions import (
    DXMTConnectionError,
    DXMTSearchError,
    DXMTTableError,
)


@pytest.fixture
def mock_response_success() -> Mock:
    """Create a mock successful response."""
    response = Mock(spec=Response)
    response.status_code = 200
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def mock_response_error() -> Mock:
    """Create a mock error response."""
    from httpx import HTTPStatusError, Request

    response = Mock(spec=Response)
    response.status_code = 400
    response.text = "Error message"
    request = Mock(spec=Request)
    error = HTTPStatusError("Error", request=request, response=response)
    response.raise_for_status = Mock(side_effect=error)
    return response


class TestClientInitialization:
    """Test client initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        client = Client()
        assert client.base_url == "http://localhost:9999/api/v1"
        client.close()

    def test_custom_host_port(self) -> None:
        """Test custom host and port."""
        client = Client(host="example.com", port=8080)
        assert client.base_url == "http://example.com:8080/api/v1"
        client.close()

    def test_ssl_enabled(self) -> None:
        """Test SSL enabled."""
        client = Client(host="example.com", port=443, ssl=True)
        assert client.base_url == "https://example.com:443/api/v1"
        client.close()

    def test_context_manager(self) -> None:
        """Test context manager."""
        with Client() as client:
            assert client is not None
            assert client.base_url == "http://localhost:9999/api/v1"


class TestTableOperations:
    """Test table management operations."""

    @patch("dxmt.client.httpx.Client")
    def test_create_table_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful table creation."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "success": True,
            "data": {"table": {"name": "test", "dimension": 128}},
        }
        mock_instance.post.return_value = mock_response_success

        client = Client()
        result = client.create_table(name="test", dimension=128)

        assert result["success"] is True
        assert result["data"]["table"]["name"] == "test"
        mock_instance.post.assert_called_once()

    @patch("dxmt.client.httpx.Client")
    def test_create_table_error(
        self, mock_client_class: Mock, mock_response_error: Mock
    ) -> None:
        """Test table creation error."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_instance.post.return_value = mock_response_error

        client = Client()
        with pytest.raises(DXMTTableError):
            client.create_table(name="test", dimension=128)

    @patch("dxmt.client.httpx.Client")
    def test_delete_table_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful table deletion."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {"success": True}
        mock_instance.delete.return_value = mock_response_success

        client = Client()
        result = client.delete_table("test")

        assert result["success"] is True
        mock_instance.delete.assert_called_once()

    @patch("dxmt.client.httpx.Client")
    def test_list_tables_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful table listing.

        client.list_tables() returns: data.get("data", {}).get("tables", [])
        """
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "data": {
                "tables": [
                    {"name": "table1", "dimension": 128},
                    {"name": "table2", "dimension": 256},
                ]
            }
        }
        mock_instance.get.return_value = mock_response_success

        client = Client()
        result = client.list_tables()

        assert len(result) == 2
        assert result[0]["name"] == "table1"

    @patch("dxmt.client.httpx.Client")
    def test_get_table_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful table retrieval.

        client.get_table() returns: data.get("data", {}).get("table", {})
        """
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "data": {
                "table": {
                    "name": "test",
                    "dimension": 128,
                    "distance_metric": "cosine",
                }
            }
        }
        mock_instance.get.return_value = mock_response_success

        client = Client()
        result = client.get_table("test")

        assert result["name"] == "test"
        assert result["dimension"] == 128


class TestRecordOperations:
    """Test record operations."""

    @patch("dxmt.client.httpx.Client")
    def test_insert_records_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful record insertion."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "success": True,
            "data": {"inserted": 2},
        }
        mock_instance.post.return_value = mock_response_success

        client = Client()
        records = [
            {"id": "1", "vector": [0.1, 0.2], "metadata": {}},
            {"id": "2", "vector": [0.3, 0.4], "metadata": {}},
        ]
        result = client.insert_records(table_name="test", records=records)

        assert result["success"] is True
        assert result["data"]["inserted"] == 2

    @patch("dxmt.client.httpx.Client")
    def test_delete_records_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful record deletion."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "success": True,
            "deleted": 2,
        }
        # delete_records uses self._client.request("DELETE", ...)
        mock_instance.request.return_value = mock_response_success

        client = Client()
        result = client.delete_records(table_name="test", ids=["1", "2"])

        assert result["success"] is True
        assert result["deleted"] == 2

    @patch("dxmt.client.httpx.Client")
    def test_get_records_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful record retrieval.

        client.get_records() fetches per-ID via GET /tables/{name}/records/{id}.
        Response envelope: data.get("data", {})
        """
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance

        def get_side_effect(url: str) -> Mock:
            record_id = url.split("/")[-1]
            r = Mock(spec=Response)
            r.status_code = 200
            r.raise_for_status = Mock()
            r.json.return_value = {
                "data": {"id": record_id, "vector": [0.1, 0.2], "metadata": {}}
            }
            return r

        mock_instance.get.side_effect = get_side_effect

        client = Client()
        result = client.get_records(table_name="test", ids=["1", "2"])

        assert len(result) == 2
        assert result[0]["id"] == "1"


class TestSearchOperations:
    """Test search operations."""

    @patch("dxmt.client.httpx.Client")
    def test_search_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful vector search.

        client.search() returns: data.get("data", {}).get("hits", [])
        """
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "data": {
                "hits": [
                    {"id": "1", "score": 0.95, "metadata": {"text": "doc1"}},
                    {"id": "2", "score": 0.85, "metadata": {"text": "doc2"}},
                ]
            }
        }
        mock_instance.post.return_value = mock_response_success

        client = Client()
        results = client.search(table_name="test", vector=[0.1, 0.2], k=10)

        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.95

    @patch("dxmt.client.httpx.Client")
    def test_search_error(
        self, mock_client_class: Mock, mock_response_error: Mock
    ) -> None:
        """Test search error."""
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_instance.post.return_value = mock_response_error

        client = Client()
        with pytest.raises(DXMTSearchError):
            client.search(table_name="test", vector=[0.1, 0.2], k=10)

    @patch("dxmt.client.httpx.Client")
    def test_hybrid_search_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful hybrid search.

        client.hybrid_search() returns: data.get("data", {}).get("hits", [])
        """
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "data": {
                "hits": [
                    {"id": "1", "score": 0.92, "metadata": {"text": "doc1"}},
                ]
            }
        }
        mock_instance.post.return_value = mock_response_success

        client = Client()
        results = client.hybrid_search(
            table_name="test",
            vector=[0.1, 0.2],
            text="machine learning",
            vector_weight=0.7,
            k=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "1"

    @patch("dxmt.client.httpx.Client")
    def test_full_text_search_success(
        self, mock_client_class: Mock, mock_response_success: Mock
    ) -> None:
        """Test successful full-text search.

        client.full_text_search() returns: data.get("data", {}).get("hits", [])
        """
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_response_success.json.return_value = {
            "data": {
                "hits": [
                    {"id": "1", "score": 0.88, "metadata": {"text": "doc1"}},
                ]
            }
        }
        mock_instance.post.return_value = mock_response_success

        client = Client()
        results = client.full_text_search(
            table_name="test", query="machine learning", k=10
        )

        assert len(results) == 1
        assert results[0]["id"] == "1"


class TestErrorHandling:
    """Test error handling."""

    @patch("dxmt.client.httpx.Client")
    def test_connection_error(self, mock_client_class: Mock) -> None:
        """Test connection error handling."""
        from httpx import RequestError

        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_instance.post.side_effect = RequestError("Connection refused")

        client = Client()
        with pytest.raises(DXMTConnectionError):
            client.create_table(name="test", dimension=128)

    @patch("dxmt.client.httpx.Client")
    def test_timeout_error(self, mock_client_class: Mock) -> None:
        """Test timeout error handling."""
        from httpx import TimeoutException

        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_instance.post.side_effect = TimeoutException("Timeout")

        client = Client()
        with pytest.raises(DXMTConnectionError):
            client.search(table_name="test", vector=[0.1, 0.2], k=10)


class TestAsyncOperations:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_async_create_table(self) -> None:
        """Test async table creation."""
        with patch("dxmt.client.httpx.AsyncClient") as mock_async_client_class:
            mock_instance = AsyncMock()
            mock_async_client_class.return_value = mock_instance

            mock_response = Mock(spec=Response)
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            # .json() is sync in httpx — use Mock, not AsyncMock
            mock_response.json = Mock(
                return_value={
                    "success": True,
                    "data": {"table": {"name": "test", "dimension": 128}},
                }
            )
            mock_instance.post = AsyncMock(return_value=mock_response)

            client = Client()
            result = await client.acreate_table(name="test", dimension=128)

            assert result["success"] is True
            assert result["data"]["table"]["name"] == "test"

    @pytest.mark.asyncio
    async def test_async_search(self) -> None:
        """Test async search."""
        with patch("dxmt.client.httpx.AsyncClient") as mock_async_client_class:
            mock_instance = AsyncMock()
            mock_async_client_class.return_value = mock_instance

            mock_response = Mock(spec=Response)
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json = Mock(
                return_value={
                    "data": {
                        "hits": [
                            {"id": "1", "score": 0.95, "metadata": {}},
                        ]
                    }
                }
            )
            mock_instance.post = AsyncMock(return_value=mock_response)

            client = Client()
            results = await client.asearch(
                table_name="test", vector=[0.1, 0.2], k=10
            )

            assert len(results) == 1
            assert results[0]["id"] == "1"

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager."""
        async with Client() as client:
            assert client is not None
            assert client.base_url == "http://localhost:9999/api/v1"
