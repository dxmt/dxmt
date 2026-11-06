from __future__ import annotations

from functools import partial
from unittest.mock import MagicMock, patch

import pytest
from agent_framework import SecretString, load_settings
from dxmt import DXMTClient

from agent_framework_dxmt import DXMTCollection, DXMTContextProvider, DXMTSettings, DXMTStore


@pytest.fixture(params=["collection", "store", "provider"])
def connector_factory(request, definition):
    if request.param == "collection":
        return partial(DXMTCollection, dict, definition=definition, collection_name="test")
    if request.param == "store":
        return DXMTStore
    return partial(DXMTContextProvider, user_id="u")


def test_settings_wrap_environment_secrets(monkeypatch):
    monkeypatch.setenv("DXMT_URL", "https://environment.example:8443")
    monkeypatch.setenv("DXMT_API_KEY", "environment-key")
    settings = load_settings(DXMTSettings, env_prefix="DXMT_")
    assert settings["url"] == "https://environment.example:8443"
    secret = settings["api_key"]
    assert isinstance(secret, SecretString)
    assert secret.get_secret_value() == "environment-key"
    assert "environment-key" not in repr(settings)


async def test_constructor_uses_environment_settings(connector_factory, monkeypatch):
    monkeypatch.setenv("DXMT_URL", "https://environment.example:8443")
    monkeypatch.setenv("DXMT_API_KEY", "environment-key")
    client = MagicMock(spec=DXMTClient)
    with patch("agent_framework_dxmt._settings.DXMTClient", return_value=client) as factory:
        async with connector_factory():
            pass
    factory.assert_called_once_with(host="environment.example", port=8443, ssl=True, api_key="environment-key")


async def test_explicit_arguments_win_and_default_is_localhost(connector_factory, monkeypatch):
    monkeypatch.setenv("DXMT_URL", "http://environment.example:1")
    client = MagicMock(spec=DXMTClient)
    with patch("agent_framework_dxmt._settings.DXMTClient", return_value=client) as factory:
        async with connector_factory(url="http://explicit.example:9000", api_key=SecretString("k")):
            pass
    factory.assert_called_once_with(host="explicit.example", port=9000, ssl=False, api_key="k")
    monkeypatch.delenv("DXMT_URL")
    with patch("agent_framework_dxmt._settings.DXMTClient", return_value=client) as factory:
        async with connector_factory():
            pass
    factory.assert_called_once_with(host="localhost", port=9999, ssl=False, api_key=None)


async def test_an_injected_client_is_never_closed(connector_factory):
    client = MagicMock(spec=DXMTClient)
    client.aclose = MagicMock()
    async with connector_factory(client=client) as connector:
        assert connector.managed_client is False
    client.aclose.assert_not_called()


def test_a_bad_url_is_refused(definition):
    with pytest.raises(ValueError, match="http://host:port"):
        DXMTStore(url="not a url")
