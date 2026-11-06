from __future__ import annotations

import yaml
from conftest import ROOT, load_plugin


def test_defaults_come_from_default_config_yaml(plugin):
    cfg = __import__(f"{plugin.__name__}.src.config", fromlist=["DEFAULTS"])
    raw = yaml.safe_load((ROOT / "src" / "default_config.yaml").read_text(encoding="utf-8"))
    assert cfg.DEFAULTS == raw["plugins"]["dxmt"]
    assert raw["memory"]["provider"] == "dxmt"


def test_shape_and_environment_override(plugin, monkeypatch, tmp_path):
    cfg = __import__(f"{plugin.__name__}.src.config", fromlist=["load_config"])
    d = cfg.DEFAULTS
    assert d["server"]["url"] == "http://localhost:9999"
    assert d["retrieval"]["isolate_workspace"] is True
    assert d["extraction"]["min_turns"] == 3

    (tmp_path / "config.yaml").write_text(
        "plugins:\n  dxmt:\n    server:\n      url: http://db.example:7000\n    retrieval:\n      top_k: 3\n"
    )
    loaded = cfg.load_config(tmp_path)
    assert loaded["server"]["url"] == "http://db.example:7000"
    assert loaded["retrieval"]["top_k"] == 3
    assert loaded["retrieval"]["prefetch_k"] == 5  # untouched default survives the merge
    monkeypatch.delenv("DXMT_URL", raising=False)
    assert cfg.server_url(loaded) == "http://db.example:7000"
    monkeypatch.setenv("DXMT_URL", "http://env.example:1")
    assert cfg.server_url(loaded) == "http://env.example:1"


def test_api_key_is_read_through_the_secret_scope(plugin, monkeypatch):
    cfg = __import__(f"{plugin.__name__}.src.config", fromlist=["api_key"])
    monkeypatch.delenv("DXMT_API_KEY", raising=False)
    assert cfg.api_key(cfg.DEFAULTS) is None
    monkeypatch.setenv("DXMT_API_KEY", "k1")
    assert cfg.api_key(cfg.DEFAULTS) == "k1"  # single-profile: the scope falls through to the environment
    monkeypatch.setenv("OTHER_KEY", "k2")
    assert cfg.api_key({"server": {"api_key_env": "OTHER_KEY"}}) == "k2"
    import agent.secret_scope as scope

    monkeypatch.setattr(scope, "get_secret", lambda name, default=None: f"scoped:{name}")
    assert cfg.api_key(cfg.DEFAULTS) == "scoped:DXMT_API_KEY"


def test_save_plugin_config_keeps_other_keys(plugin, tmp_path):
    cfg = __import__(f"{plugin.__name__}.src.config", fromlist=["save_plugin_config"])
    (tmp_path / "config.yaml").write_text("memory:\n  provider: dxmt\nplugins:\n  other: {a: 1}\n")
    cfg.save_plugin_config({"server": {"url": "http://x:1"}}, str(tmp_path))
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert raw["memory"]["provider"] == "dxmt"
    assert raw["plugins"]["other"] == {"a": 1}
    assert raw["plugins"]["dxmt"]["server"]["url"] == "http://x:1"
    assert raw["plugins"]["dxmt"]["retrieval"]["top_k"] == 10


def test_manifest_matches_provider(plugin):
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["name"] == "dxmt"
    provider_mod = __import__(f"{plugin.__name__}.src.provider", fromlist=["PLUGIN_VERSION"])
    assert manifest["version"] == provider_mod.PLUGIN_VERSION
    for hook in manifest["hooks"]:
        assert callable(getattr(plugin.DXMTMemoryProvider, hook))
    assert load_plugin() is plugin
