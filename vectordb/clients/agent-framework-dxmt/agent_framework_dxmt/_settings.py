"""Connection settings and client ownership, in the shape the in-tree connectors use."""

from __future__ import annotations

from typing import TypedDict
from urllib.parse import urlparse

from agent_framework import SecretString, load_settings
from dxmt import DXMTClient


class DXMTSettings(TypedDict, total=False):
    """Connection settings loaded through Agent Framework's ``load_settings``.

    Explicit constructor arguments take precedence over an explicitly selected ``.env``
    file, then environment variables prefixed with ``DXMT_``.

    Keys:
        url: Server URL from ``DXMT_URL`` (``http://host:port``). Defaults to ``http://localhost:9999``.
        api_key: Optional API key from ``DXMT_API_KEY``, held as an AF ``SecretString``.
    """

    url: str | None
    api_key: SecretString | None


DEFAULT_URL = "http://localhost:9999"


def _client_kwargs(url: str | None, api_key: SecretString | str | None) -> dict[str, object]:
    parsed = urlparse(url or DEFAULT_URL)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"DXMT url must look like http://host:port, got {url!r}")
    key = api_key.get_secret_value() if isinstance(api_key, SecretString) else api_key
    return {
        "host": parsed.hostname,
        "port": parsed.port or (443 if parsed.scheme == "https" else 9999),
        "ssl": parsed.scheme == "https",
        "api_key": key,
    }


def _create_client(
    *,
    url: str | None,
    api_key: str | SecretString | None,
    env_file_path: str | None,
    env_file_encoding: str | None,
) -> DXMTClient:
    settings = load_settings(
        DXMTSettings,
        env_prefix="DXMT_",
        url=url,
        api_key=api_key,
        env_file_path=env_file_path,
        env_file_encoding=env_file_encoding,
    )
    return DXMTClient(**_client_kwargs(settings.get("url"), settings.get("api_key")))  # type: ignore[arg-type]


def _resolve_client(
    client: DXMTClient | None,
    *,
    url: str | None,
    api_key: str | SecretString | None,
    managed_client: bool | None,
    env_file_path: str | None,
    env_file_encoding: str | None,
) -> tuple[DXMTClient, bool]:
    """The client and whether this connector closes it.

    An injected client is returned unchanged and stays caller-owned; url, api_key and the
    ``.env`` options are not read. A connector-created client is owned unless
    ``managed_client`` is false.
    """
    if client is not None:
        return client, False
    return (
        _create_client(url=url, api_key=api_key, env_file_path=env_file_path, env_file_encoding=env_file_encoding),
        True if managed_client is None else managed_client,
    )
