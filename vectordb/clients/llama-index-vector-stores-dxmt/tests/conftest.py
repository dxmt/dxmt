"""Test-only: talk to a server started with DXMT_API_KEY.

The SDK reads no environment variable itself, and the stores and clients in these tests are
built without a key. When `DXMT_API_KEY` is set, every `DXMTClient` built while the tests run
gets it as its default key, so the same suite runs against an open server and a keyed one.

Patched at configure time rather than in a function-scoped fixture: a module-scoped fixture
builds its client before any function-scoped one runs, and that client went out keyless.
`test_api_key.py` checks what the argument itself does, and opts out with `own_api_key`.
"""

import functools
import os

import pytest
from dxmt import DXMTClient

KEY = os.environ.get("DXMT_API_KEY")
_inject = True


def pytest_configure(config):
    config.addinivalue_line("markers", "own_api_key: the test sets its clients\x27 keys itself")
    if not KEY:
        return
    original = DXMTClient.__init__

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        if _inject and kwargs.get("api_key") is None:
            kwargs["api_key"] = KEY
        original(self, *args, **kwargs)

    DXMTClient.__init__ = init


@pytest.fixture(autouse=True)
def _own_api_key(request):
    global _inject
    _inject = not request.node.get_closest_marker("own_api_key")
    yield
    _inject = True
