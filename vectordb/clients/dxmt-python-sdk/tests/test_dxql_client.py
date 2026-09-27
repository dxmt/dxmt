"""Unit tests for DXMT client DXQL query method."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import Response, HTTPStatusError, Request

from dxmt import Client
from dxmt.exceptions import DXMTError


@pytest.fixture
def mock_httpx_client():
    instance = Mock()
    instance.post.return_value = Mock(
        spec=Response,
        status_code=200,
        raise_for_status=Mock(),
        json=Mock(return_value={
            "success": True,
            "data": {"records": [{"id": "1", "category": "AI"}], "metadata": {}}
        }),
    )
    return instance


class TestClientQueryMethod:
    """client.query() sends POST /query and returns data dict."""

    @patch("dxmt.client.httpx.Client")
    def test_query_posts_to_query_endpoint(self, mock_client_class):
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_instance.post.return_value = Mock(
            spec=Response,
            status_code=200,
            raise_for_status=Mock(),
            json=Mock(return_value={
                "success": True,
                "data": {"records": [], "metadata": {}}
            }),
        )

        client = Client()
        client.query("SELECT * FROM my_table LIMIT 10")

        mock_instance.post.assert_called_once()
        call = mock_instance.post.call_args
        assert "/query" in call[0][0] or call[0][0].endswith("query")
        assert call[1]["json"] == {"query": "SELECT * FROM my_table LIMIT 10"}

    @patch("dxmt.client.httpx.Client")
    def test_query_returns_data_field(self, mock_client_class):
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        records = [{"id": "r1", "category": "AI"}]
        mock_instance.post.return_value = Mock(
            spec=Response,
            status_code=200,
            raise_for_status=Mock(),
            json=Mock(return_value={
                "success": True,
                "data": {"records": records, "metadata": {"count": 1}}
            }),
        )

        client = Client()
        result = client.query("SELECT * FROM docs")

        assert "records" in result
        assert result["records"] == records

    @patch("dxmt.client.httpx.Client")
    def test_query_raises_dxmt_error_on_400(self, mock_client_class):
        from httpx import HTTPStatusError, Request as HttpxRequest

        mock_instance = Mock()
        mock_client_class.return_value = mock_instance

        mock_response = Mock(spec=Response)
        mock_response.status_code = 400
        mock_response.text = "Parse error"
        mock_instance.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "Parse error",
            request=Mock(spec=HttpxRequest),
            response=mock_response,
        )

        client = Client()
        with pytest.raises(DXMTError):
            client.query("INVALID SYNTAX !!!!")

    @patch("dxmt.client.httpx.Client")
    def test_query_select_with_filter_in_sql(self, mock_client_class):
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_instance.post.return_value = Mock(
            spec=Response,
            status_code=200,
            raise_for_status=Mock(),
            json=Mock(return_value={
                "success": True,
                "data": {"records": [{"id": "1"}], "metadata": {}}
            }),
        )

        client = Client()
        sql = "SELECT id, category FROM docs WHERE category = 'AI' LIMIT 5"
        result = client.query(sql)

        call = mock_instance.post.call_args
        assert call[1]["json"]["query"] == sql


class TestClientQuerySQLVariants:
    """Test various DXQL statements are passed through unmodified."""

    @patch("dxmt.client.httpx.Client")
    def _run_query(self, sql, mock_client_class):
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_instance.post.return_value = Mock(
            spec=Response,
            status_code=200,
            raise_for_status=Mock(),
            json=Mock(return_value={"success": True, "data": {}}),
        )
        client = Client()
        client.query(sql)
        return mock_instance.post.call_args[1]["json"]["query"]

    def test_insert_sql_passthrough(self):
        sql = "INSERT INTO docs (id, vector) VALUES ('id1', [0.1, 0.2])"
        assert self._run_query(sql) == sql

    def test_delete_sql_passthrough(self):
        sql = "DELETE FROM docs WHERE id = 'id1'"
        assert self._run_query(sql) == sql

    def test_create_table_sql_passthrough(self):
        sql = "CREATE TABLE new_table (dimension 128, metric cosine)"
        assert self._run_query(sql) == sql
