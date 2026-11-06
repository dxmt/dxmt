import hashlib
import math
import os
import re
import uuid

import pytest
from dxmt import DXMTClient

HOST = os.environ.get("DXMT_HOST", "localhost")
PORT = int(os.environ.get("DXMT_PORT", "9999"))
API_KEY = os.environ.get("DXMT_API_KEY")
DIM = 64


def embed(texts: list[str]) -> list[list[float]]:
    """A deterministic bag-of-words embedding: texts sharing words point the same way.

    Enough for "the record about databases comes back for a question about databases", and
    stable across runs, which a real model is not.
    """
    out = []
    for t in texts:
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 8) % 2 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


@pytest.fixture
def client():
    return DXMTClient(host=HOST, port=PORT, api_key=API_KEY)


@pytest.fixture
def table_name(client):
    name = f"crew_{uuid.uuid4().hex[:10]}"
    yield name
    try:
        client.delete_table(name)
    except Exception:
        pass
