"""Unit tests for DXMT Client."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from httpx import Response

from dxmt import Client
from dxmt.exceptions import DXMTConnectionError, DXMTTableError


@pytest.fixture
def mock_httpx_client() -> Mock:
    """Create a mock httpx client."""
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=None)
    return client


def test_client_initialization() -> None:
    """Test client initialization."""
    client = Client(host="localhost", port=9999)
    assert client.base_url == "http://localhost:9999/api/v1"
    client.close()


def test_client_initialization_with_ssl() -> None:
    """Test client initialization with SSL."""
    client = Client(host="example.com", port=443, ssl=True)
    assert client.base_url == "https://example.com:443/api/v1"
    client.close()


def test_client_context_manager() -> None:
    """Test client as context manager."""
    with Client(host="localhost", port=9999) as client:
        assert client.base_url == "http://localhost:9999/api/v1"


@patch("dxmt.client.httpx.Client")
def test_create_table_success(mock_client_class: Mock) -> None:
    """Test successful table creation."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "success": True,
        "data": {"table": {"name": "test", "dimension": 128}},
    }
    mock_instance.post.return_value = mock_response

    client = Client()
    result = client.create_table(name="test", dimension=128)

    assert result["success"] is True
    assert result["data"]["table"]["name"] == "test"
    mock_instance.post.assert_called_once()


@patch("dxmt.client.httpx.Client")
def test_create_table_error(mock_client_class: Mock) -> None:
    """Test table creation with error."""
    from httpx import HTTPStatusError, Request

    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_response = Mock(spec=Response)
    mock_response.status_code = 400
    mock_response.text = "Table already exists"
    error = HTTPStatusError(
        "Table already exists",
        request=Mock(spec=Request),
        response=mock_response,
    )
    mock_response.raise_for_status = Mock(side_effect=error)
    mock_instance.post.return_value = mock_response

    client = Client()
    with pytest.raises(DXMTTableError, match="Table already exists"):
        client.create_table(name="test", dimension=128)


@patch("dxmt.client.httpx.Client")
def test_insert_records_success(mock_client_class: Mock) -> None:
    """Test successful record insertion."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {"success": True, "inserted": 2}
    mock_instance.post.return_value = mock_response

    client = Client()
    records = [
        {"id": "1", "vector": [0.1, 0.2], "metadata": {}},
        {"id": "2", "vector": [0.3, 0.4], "metadata": {}},
    ]
    result = client.insert_records(table_name="test", records=records)

    assert result["success"] is True
    assert result["inserted"] == 2


@patch("dxmt.client.httpx.Client")
def test_search_success(mock_client_class: Mock) -> None:
    """Test successful search."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    # client.py parses: data.get("data", {}).get("hits", [])
    mock_response.json.return_value = {
        "data": {
            "hits": [
                {"id": "1", "score": 0.95, "metadata": {"text": "doc1"}},
                {"id": "2", "score": 0.85, "metadata": {"text": "doc2"}},
            ]
        }
    }
    mock_instance.post.return_value = mock_response

    client = Client()
    results = client.search(table_name="test", vector=[0.1, 0.2], limit=10)

    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[0]["score"] == 0.95


@patch("dxmt.client.httpx.Client")
def test_delete_table_success(mock_client_class: Mock) -> None:
    """Test successful table deletion."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {"success": True}
    mock_instance.delete.return_value = mock_response

    client = Client()
    result = client.delete_table("test")

    assert result["success"] is True
    mock_instance.delete.assert_called_once()


@patch("dxmt.client.httpx.Client")
def test_list_tables_success(mock_client_class: Mock) -> None:
    """Test successful table listing."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    # client.py parses: data.get("data", {}).get("tables", [])
    mock_response.json.return_value = {
        "data": {
            "tables": [
                {"name": "table1", "dimension": 128},
                {"name": "table2", "dimension": 256},
            ]
        }
    }
    mock_instance.get.return_value = mock_response

    client = Client()
    result = client.list_tables()

    assert len(result) == 2
    assert result[0]["name"] == "table1"


@patch("dxmt.client.httpx.Client")
def test_get_records_success(mock_client_class: Mock) -> None:
    """Test successful record retrieval.

    client.get_records() fetches each ID individually via
    GET /tables/{name}/records/{id}, returning data.get("data", {}) per call.
    """
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    def get_side_effect(url: str) -> Mock:
        record_id = url.split("/")[-1]
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "data": {"id": record_id, "vector": [0.1, 0.2], "metadata": {}}
        }
        return mock_response

    mock_instance.get.side_effect = get_side_effect

    client = Client()
    result = client.get_records(table_name="test", ids=["1", "2"])

    assert len(result) == 2
    assert result[0]["id"] == "1"


@patch("dxmt.client.httpx.Client")
def test_search_sends_top_k_and_normalizes_payload_json(mock_client_class: Mock) -> None:
    """The Rust API uses top_k and serializes GET records as payload_json."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance
    response = Mock(spec=Response)
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "success": True,
        "data": {
            "hits": [{"id": "1", "score": 1.0, "payload": {"kind": "x"}}]
        },
    }
    mock_instance.post.return_value = response

    client = Client()
    assert client.search("test", [1.0, 0.0], k=1)[0]["metadata"] == {"kind": "x"}
    # Search encodes its own body rather than handing httpx a dict, so read what actually
    # goes on the wire. The header has to come with it: `content=` sends bytes and httpx
    # sets no Content-Type of its own, so dropping it would post JSON the server does not
    # parse as JSON.
    kwargs = mock_instance.post.call_args.kwargs
    assert kwargs["headers"]["Content-Type"] == "application/json"
    body = json.loads(kwargs["content"])
    assert body["top_k"] == 1


@patch("dxmt.client.httpx.Client")
def test_get_records_parses_payload_json(mock_client_class: Mock) -> None:
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance
    response = Mock(spec=Response)
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "success": True,
        "data": {"id": "1", "vector": [1.0], "payload_json": '{"x": 2}'},
    }
    mock_instance.get.return_value = response

    record = Client().get_records("test", ["1"])[0]
    assert record["payload"] == {"x": 2}
    assert record["metadata"] == {"x": 2}


@patch("dxmt.client.httpx.Client")
def test_delete_records_success(mock_client_class: Mock) -> None:
    """Test successful record deletion."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {"success": True, "deleted": 2}
    # delete_records uses self._client.request("DELETE", ...)
    mock_instance.request.return_value = mock_response

    client = Client()
    result = client.delete_records(table_name="test", ids=["1", "2"])

    assert result["success"] is True
    assert result["deleted"] == 2


@pytest.mark.asyncio
async def test_async_create_table() -> None:
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
async def test_async_search() -> None:
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
        results = await client.asearch(table_name="test", vector=[0.1, 0.2], limit=10)

        assert len(results) == 1
        assert results[0]["id"] == "1"


@pytest.mark.asyncio
async def test_async_update_preserves_update_options() -> None:
    with patch("dxmt.client.httpx.AsyncClient") as mock_async_client_class:
        mock_instance = AsyncMock()
        mock_async_client_class.return_value = mock_instance
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={"success": True, "data": {}})
        mock_instance.put = AsyncMock(return_value=mock_response)

        client = Client()
        await client.aupdate_record(
            "test", "r1", payload={"x": 1}, merge_mode="merge", upsert=True
        )
        body = mock_instance.put.call_args.kwargs["json"]
        assert body["merge_mode"] == "merge"
        assert body["upsert"] is True


@patch("dxmt.client.httpx.Client")
def test_connection_error(mock_client_class: Mock) -> None:
    """Test connection error handling."""
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance

    mock_instance.post.side_effect = httpx.RequestError("Connection refused")

    client = Client()
    with pytest.raises(DXMTConnectionError, match="Connection refused"):
        client.create_table(name="test", dimension=128)


@patch("dxmt.client.httpx.Client")
def test_every_search_method_converts_the_filter_it_is_given(mock_client_class: Mock) -> None:
    """The MongoDB-style filter is a client-side convenience; the server reads a different
    shape. `full_text_search` passed it through raw while `search` and `hybrid_search`
    converted, so the same filter worked on two methods out of three and silently matched
    nothing on the third.

    Asserted for every method rather than only the one that was broken, so the next method
    added has to answer for itself.
    """
    mock_instance = Mock()
    mock_client_class.return_value = mock_instance
    response = Mock(spec=Response)
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {"data": {"hits": []}}
    mock_instance.post.return_value = response

    client = Client()
    written = {"year": {"$gte": 2022}}
    expected = {
        "must": [
            {"key": "year", "operation": {"type": "GreaterThanOrEqual", "value": 2022}}
        ]
    }

    calls = [
        lambda: client.search(table_name="t", vector=[0.1, 0.2], filter=written),
        lambda: client.hybrid_search(table_name="t", vector=[0.1, 0.2], filter=written),
        lambda: client.full_text_search(table_name="t", query="x", filter=written),
    ]
    for call in calls:
        mock_instance.post.reset_mock()
        call()
        kwargs = mock_instance.post.call_args.kwargs
        # Two send shapes in one client: `search` and `hybrid_search` serialise through
        # `_encode_json` into `content=` (the orjson fast path), `full_text_search` hands the
        # dict to httpx as `json=`. Worth noticing rather than smoothing over -- the fast path
        # exists because insert bodies are mostly floats, and the text path simply never got it.
        sent = kwargs["json"] if "json" in kwargs else json.loads(kwargs["content"])
        assert sent["filter"] == expected, f"sent {sent['filter']!r} instead of the converted form"


def test_no_async_method_has_drifted_from_its_sync_twin() -> None:
    """Every `a`-prefixed method must take the parameters its sync twin takes.

    Not a style rule: `acreate_table` had lost `memory_optimization` and `asearch` had lost
    `search_mode`, so async callers silently could not reach two options the sync API
    documents. A drift like that is invisible until someone passes the argument and gets a
    TypeError, which is the wrong moment to find out.

    Asserted over every pair rather than the two that had drifted, so the next method added in
    one flavour and not the other fails here.
    """
    import inspect

    from dxmt.client import DXMTClient

    pairs = [
        (name, "a" + name)
        for name in dir(DXMTClient)
        if not name.startswith("_") and callable(getattr(DXMTClient, name, None))
        and hasattr(DXMTClient, "a" + name)
    ]
    assert pairs, "no sync/async pairs found; this test would pass vacuously"

    drifted = {}
    for sync, asyn in pairs:
        s = set(inspect.signature(getattr(DXMTClient, sync)).parameters)
        a = set(inspect.signature(getattr(DXMTClient, asyn)).parameters)
        if s != a:
            drifted[f"{sync}/{asyn}"] = {"sync only": sorted(s - a), "async only": sorted(a - s)}

    assert not drifted, f"async methods have drifted from their sync twins: {drifted}"
