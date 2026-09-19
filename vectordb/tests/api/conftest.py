"""Fixtures: one known table, built the same way for every test, in each storage mode.

Two things are borrowed from Qdrant's `tests/openapi` (Apache 2.0 — the shape, not the code):

* A small fixed corpus, so assertions can name the ids they expect. A test that checks only
  `len(hits) == 3` passes when the wrong three come back, and that is the failure worth
  catching.
* The suite runs once per storage mode. In this database "records and vectors in memory" and
  "records and vectors in row files beside the WAL" are different code paths -- different
  readers, different rerank, `vector_for` against a mapped file rather than the record -- and
  nothing was running the same functional assertions across both.

The storage mode is a property of the server process, not of a request, so it cannot be a
parameter here the way `on_disk_vectors` is for Qdrant. `DXMT_STORAGE_MODE` only says which mode
the server under test was started in, and this file does nothing to bring that about:
`scripts/api_tests.sh` starts the server once per mode, with its own WAL directory each time,
and runs the suite against each. Running `pytest tests/api` directly attaches to whatever
server is already listening, which is what you want while writing a test.

Verified rather than assumed, because two of these are silent failure modes:

* Unset means **disk**, not memory -- `store_on_disk_wanted` returns `true` for `None`. The mode
  that had never been exercised over HTTP is memory.
* Without a WAL directory *both* stores fall back to memory whatever the variables say, so a
  "disk" run without one goes green having tested memory.
* A table keeps the store it already has: an existing row file is used even under
  `DXMT_RECORD_STORE=memory`. Hence a fresh WAL directory per mode.

The evidence that the two modes really are two paths: after one insert, a memory-mode WAL
directory holds only `wal.log`, and a disk-mode one also holds `vectors/<t>.f32` and
`records/<t>.rec`.
"""

import os
import uuid

import pytest

from .helpers import client
from .helpers.corpus import CORPUS, DIM, create_table, insert_corpus

STORAGE_MODE = os.environ.get("DXMT_STORAGE_MODE", "memory")


def pytest_report_header(config):
    return f"dxmt: {client.BASE}  storage_mode={STORAGE_MODE}"


@pytest.fixture(scope="session", autouse=True)
def server():
    client.wait_for_server()


@pytest.fixture(scope="module")
def table():
    """A table with the fixed corpus in it, dropped when the module finishes.

    Module scope, not function scope: building it is a create plus one insert, and a test that
    mutates it says so by asking for `fresh_table` instead.
    """
    name = f"api_{STORAGE_MODE}_{uuid.uuid4().hex[:8]}"
    create_table(name, DIM)
    insert_corpus(name)
    yield name
    client.delete(f"/tables/{name}", expect_ok=False)


@pytest.fixture
def fresh_table():
    """A table of its own, for a test that inserts, updates or deletes."""
    name = f"api_mut_{uuid.uuid4().hex[:8]}"
    create_table(name, DIM)
    insert_corpus(name)
    yield name
    client.delete(f"/tables/{name}", expect_ok=False)


@pytest.fixture
def empty_table():
    """A table with no records, for the answers a caller gets before there is data."""
    name = f"api_empty_{uuid.uuid4().hex[:8]}"
    create_table(name, DIM)
    yield name
    client.delete(f"/tables/{name}", expect_ok=False)


@pytest.fixture
def corpus():
    return CORPUS
