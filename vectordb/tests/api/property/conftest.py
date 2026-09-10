"""A generated corpus, big enough that the answers cannot be written down by hand.

Everything here is marked `slow` and skipped by `-m "not slow"`. Sizes are deliberately modest:
CI builds the server with `cargo build` at opt-level 0, where HNSW insertion and the distance
kernels are roughly an order of magnitude slower than release, so this is measured in seconds
rather than kept small out of caution.
"""

import math
import os
import uuid

import pytest

from ..helpers import client
from ..helpers.corpus import create_table

N = int(os.environ.get("DXMT_PROPERTY_N", "2000"))
DIM = int(os.environ.get("DXMT_PROPERTY_DIM", "64"))
CHUNK = 500


def vector(i, dim=DIM):
    """A unit vector that depends on `i` only, so the corpus is the same on every run and a
    failure can be reproduced from the seed alone."""
    out = []
    h = (i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    for d in range(dim):
        h ^= h >> 29
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= (h >> 32) ^ d
        out.append(((h & 0xFFFF) / 65535.0) - 0.5)
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


def record(i):
    return {
        "id": f"p{i:06d}",
        "vector": vector(i),
        "payload": {"n": i, "bucket": f"b{i % 7}"},
    }


@pytest.fixture(scope="module")
def big_table():
    name = f"prop_{os.environ.get('DXMT_STORAGE_MODE', 'memory')}_{uuid.uuid4().hex[:8]}"
    create_table(name, DIM)
    for start in range(0, N, CHUNK):
        batch = [record(i) for i in range(start, min(start + CHUNK, N))]
        client.post(f"/tables/{name}/records", {"records": batch})
    yield name
    client.delete(f"/tables/{name}", expect_ok=False)


@pytest.fixture(scope="module")
def expected_ids():
    return {f"p{i:06d}" for i in range(N)}
