"""Plugin configuration.

Defaults live in ``default_config.yaml`` (the single source). User overrides come from
``$HERMES_HOME/config.yaml`` under ``plugins.dxmt``; ``DXMT_URL`` in the environment wins
over the configured server URL.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def _load_defaults() -> dict[str, Any]:
    raw = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return (raw.get("plugins", {}) or {}).get("dxmt", {}) or {}


DEFAULTS: dict[str, Any] = _load_defaults()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
    """Defaults merged with ``plugins.dxmt`` from the profile's config.yaml."""
    config_path = _config_path(hermes_home)
    if config_path is None or not config_path.exists():
        return copy.deepcopy(DEFAULTS)
    try:
        with open(config_path, encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        user_cfg = ((raw.get("plugins") or {}).get("dxmt") or {}) if isinstance(raw, dict) else {}
        return _deep_merge(DEFAULTS, user_cfg)
    except Exception as exc:
        logger.warning("Failed to load dxmt plugin config (%s); using defaults", exc)
        return copy.deepcopy(DEFAULTS)


def _config_path(hermes_home: str | Path | None) -> Path | None:
    if hermes_home:
        return Path(hermes_home) / "config.yaml"
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        return None
    return Path(get_hermes_home()) / "config.yaml"


def server_url(config: dict[str, Any]) -> str:
    """The server URL: ``DXMT_URL`` if set, else ``server.url`` from the config."""
    return (os.environ.get("DXMT_URL") or (config.get("server") or {}).get("url") or DEFAULTS["server"]["url"]).strip()


def api_key(config: dict[str, Any]) -> str | None:
    """The server's API key, read through Hermes's profile-aware secret scope (a bare
    ``os.environ`` read could pick up another profile's key under multiplexing); the
    environment directly only outside Hermes."""
    name = str((config.get("server") or {}).get("api_key_env") or "DXMT_API_KEY")
    try:
        from agent.secret_scope import get_secret
    except ImportError:
        return os.environ.get(name) or None
    return get_secret(name) or None


def save_plugin_config(values: dict[str, Any], hermes_home: str) -> None:
    """Write the plugin config under ``plugins.dxmt`` in the profile's config.yaml."""
    config_path = Path(hermes_home) / "config.yaml"
    existing: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8-sig") as f:
            existing = yaml.safe_load(f) or {}
    if not isinstance(existing, dict):
        existing = {}
    existing.setdefault("plugins", {})
    existing["plugins"]["dxmt"] = _deep_merge(DEFAULTS, values or {})
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing, f, default_flow_style=False, sort_keys=False)
