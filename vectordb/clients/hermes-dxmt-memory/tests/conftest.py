"""Test fixtures.

The plugin imports ``agent.memory_provider`` from Hermes, so the Hermes checkout has to be
importable: ``HERMES_AGENT_DIR`` names it (default: a sibling ``hermes-agent`` of this
repository's parent, where the development clone lives). Live tests need ``DXMT_TEST_URL``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _hermes_dir() -> Path | None:
    candidates = [os.environ.get("HERMES_AGENT_DIR")]
    candidates += [ROOT.parents[3] / "reference" / "framework" / "hermes-agent", ROOT.parent / "hermes-agent"]
    for c in candidates:
        if c and Path(c).joinpath("agent", "memory_provider.py").exists():
            return Path(c)
    return None


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs a DXMT server at DXMT_TEST_URL")
    hermes = _hermes_dir()
    if hermes and str(hermes) not in sys.path:
        sys.path.insert(0, str(hermes))


def load_plugin():
    """Import the plugin the way Hermes does: the directory root as a package."""
    name = "dxmt_memory_plugin_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def plugin():
    pytest.importorskip("agent.memory_provider", reason="set HERMES_AGENT_DIR to a hermes-agent checkout")
    return load_plugin()


@pytest.fixture
def live_url() -> str:
    url = os.environ.get("DXMT_TEST_URL")
    if not url:
        pytest.skip("set DXMT_TEST_URL to a disposable DXMT server with an embedding model")
    return url
