"""Data written by an earlier build, opened by this one.

Each directory under `fixtures/` was written once, by the build named in its `expected.json`, with
`scripts/make_compat_fixture.py`, and is never regenerated. This test opens every one of them with
the binary under test and checks that nothing a user stored was lost or changed: tables and their
configuration, every record's payload, version, created_at, vector, sparse and named vectors, and
that search and filters still answer over it. Then it keeps writing — an upgraded database has to
stay usable, not just readable — crashes, and checks again.

Written before the first release, so the first fixture comes from the same code that reads it and
cannot fail yet. It exists for the release after: a change to the WAL entries, the snapshot, the
row files or an index format that 0.2.0's data does not survive fails here instead of in someone's
upgrade. Qdrant keeps nine releases' data for this (`tests/e2e_tests/test_data_compatibility.py`).

    DXMT_BIN=target/debug/vectordb pytest tests/compat
"""

import glob
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantics import instant, matches, score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("DXMT_BIN") or ("target/debug/vectordb" if os.path.exists("target/debug/vectordb") else None)
FIXTURES = sorted(glob.glob(os.path.join(HERE, "fixtures", "*", "expected.json")))
TOL = 1e-4


class Server:
    def __init__(self, wal):
        self.env = dict(os.environ, DXMT_WAL_DIR=wal, DXMT_SNAPSHOT_INTERVAL_SECS="0",
                        DXMT_EVICTION_INTERVAL_SECS="0", RUST_LOG="warn")
        # Opened with the defaults, so each table keeps the store it was written with.
        for name in ("DXMT_API_KEY", "DXMT_RECORD_STORE", "DXMT_VECTOR_STORE"):
            self.env.pop(name, None)
        self.log = os.path.join(os.path.dirname(wal), "server.log")

    def start(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.proc = subprocess.Popen([BIN, "serve", "--host", "127.0.0.1", "--port", str(self.port)],
                                     env=self.env, stdout=open(self.log, "a"), stderr=subprocess.STDOUT)
        for _ in range(600):
            if self.proc.poll() is not None:
                raise AssertionError(f"the server exited opening this data:\n{self.tail()}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/live", timeout=1)
                return
            except Exception:
                time.sleep(0.05)
        raise AssertionError(f"the server did not come up:\n{self.tail()}")

    def stop(self, sig=signal.SIGKILL):
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
            self.proc.wait(timeout=120)

    def tail(self):
        return "".join(open(self.log).readlines()[-15:]) if os.path.exists(self.log) else ""

    def call(self, method, path, body=None):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1{path}",
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as e:
            raw = e.read()
            return e.code, json.loads(raw) if raw else {}

    def ok(self, method, path, body=None):
        status, response = self.call(method, path, body)
        assert status == 200, f"{method} {path}: {status} {response}"
        return response["data"]


def close(a, b):
    return len(a) == len(b) and all(abs(x - y) < 1e-5 for x, y in zip(a, b))


def check(server, expected, where):
    tables = expected["tables"]
    listed = server.ok("GET", "/tables")
    names = sorted(t["name"] for t in (listed["tables"] if isinstance(listed, dict) else listed))
    assert names == sorted(tables), f"{where}: tables {names}, expected {sorted(tables)}"

    rng = random.Random(7)
    for name, config in tables.items():
        table = server.ok("GET", f"/tables/{name}")["table"]
        assert table["dimension"] == config["dimension"], f"{where}: {name} dimension"
        assert table["distance_metric"] == config["distance_metric"], f"{where}: {name} metric"
        assert sorted(s["name"] for s in table.get("named_vectors") or []) == sorted(config["named_vectors"]), f"{where}: {name} named vector spaces"
        if "description" in config:
            assert table.get("description") == config["description"], f"{where}: {name} description"
        # The 1.x table settings, in fixtures that carry them (written by 1.3.2 and later).
        if "payload_index_except" in config:
            assert (table.get("payload_index") or {}).get("except") == config["payload_index_except"], f"{where}: {name} payload_index {table.get('payload_index')}"
            # The policy is in force, not just reported: a filter on an excluded field is refused.
            field = config["payload_index_except"][0]
            status, _ = server.call("POST", f"/tables/{name}/records/count",
                                    {"filter": {"must": [{"key": field, "operation": {"type": "IsEmpty"}}]}})
            assert status == 400, f"{where}: {name} answered a filter on '{field}', which it does not index"
        if "fast_scan" in config:
            assert (table.get("quantization") or {}).get("fast_scan") == config["fast_scan"], f"{where}: {name} quantization {table.get('quantization')}"
        if "recall_target" in config:
            assert table.get("recall_target") == config["recall_target"], f"{where}: {name} recall_target {table.get('recall_target')}"
        for key, value in (config.get("rerank") or {}).items():
            assert (table.get("rerank") or {}).get(key) == value, f"{where}: {name} rerank {table.get('rerank')}"

        want = expected["records"][name]
        page = server.ok("POST", f"/tables/{name}/records/list", {"limit": 10000, "with_vector": True})
        got = {r["id"]: r for r in page["records"]}
        assert set(got) == set(want), (f"{where}: {name} lost {sorted(set(want) - set(got))}, "
                                       f"gained {sorted(set(got) - set(want))}")
        for rid, w in want.items():
            r = got[rid]
            assert json.loads(r["payload_json"]) == w["payload"], f"{where}: {name}/{rid} payload"
            assert r["version"] == w["version"], f"{where}: {name}/{rid} version {r['version']} != {w['version']}"
            assert r["created_at"] == w["created_at"], f"{where}: {name}/{rid} created_at moved"
            full = server.ok("GET", f"/tables/{name}/records/{rid}")
            assert close(full["vector"], w["vector"]), f"{where}: {name}/{rid} vector {full['vector']} != {w['vector']}"
            if "sparse" in w:
                sv = full.get("sparse_vector") or {}
                assert (sv.get("indices"), sv.get("values")) == (w["sparse"]["indices"], w["sparse"]["values"]), f"{where}: {name}/{rid} sparse"
            if "named" in w:
                for space, v in w["named"].items():
                    assert close((full.get("named_vectors") or {}).get(space, []), v), f"{where}: {name}/{rid} named '{space}'"

        # Newest first, ties by id — by the timestamps the data was written with.
        order = [r["id"] for r in page["records"]]
        assert order == sorted(sorted(want), key=lambda i: instant(want[i]["created_at"]), reverse=True), f"{where}: {name} list order"

        # Exact search: a valid top-k by the metric.
        q = [round(rng.uniform(-1, 1), 3) for _ in range(config["dimension"])]
        k = min(5, len(want))
        hits = server.ok("POST", f"/tables/{name}/search", {"vector": q, "top_k": k, "search_mode": "exact"})["hits"]
        truth = {rid: score(config["distance_metric"], q, w["vector"]) for rid, w in want.items()}
        assert len(hits) == k, f"{where}: {name} exact search returned {len(hits)}"
        for h in hits:
            assert abs(h["score"] - truth[h["id"]]) < TOL, f"{where}: {name}/{h['id']} score {h['score']} != {truth[h['id']]}"
        floor = min(h["score"] for h in hits)
        returned = {h["id"] for h in hits}
        assert not [i for i, t in truth.items() if i not in returned and t > floor + TOL], f"{where}: {name} exact search missed one"

        # Filters, by the documented rules, through the list and exact search.
        C = lambda k, op, v: {"key": k, "operation": {"type": op, "value": v}}  # noqa: E731
        for f in [{"must": [C("tags", "Equal", "rag")]}, {"must": [C("tags", "NotEqual", "llm")]},
                  {"must": [C("n", "GreaterThan", 10)]}, {"must": [C("n", "Equal", 4)]},
                  {"must": [C("scores", "GreaterThanOrEqual", 1000)]}, {"must_not": [C("title", "StartsWith", "doc 1")]}]:
            want_ids = [i for i in order if matches(i, want[i]["payload"], f)]
            listed = server.ok("POST", f"/tables/{name}/records/list", {"limit": 10000, "filter": f})
            assert [r["id"] for r in listed["records"]] == want_ids, f"{where}: {name} list {f}"
            found = server.ok("POST", f"/tables/{name}/search", {"vector": q, "top_k": max(1, len(want)), "filter": f, "search_mode": "exact"})
            assert sorted(h["id"] for h in found["hits"]) == sorted(want_ids), f"{where}: {name} exact search {f}"

        # A named vector space answers by its own vectors.
        for space in config["named_vectors"]:
            rid = sorted(r for r in want if space in want[r].get("named", {}))[0]
            hits = server.ok("POST", f"/tables/{name}/search", {"vector": want[rid]["named"][space], "top_k": 1, "vector_name": space})["hits"]
            assert hits and hits[0]["id"] == rid, f"{where}: {name} '{space}' search for {rid} found {hits}"


@pytest.mark.skipif(BIN is None, reason="no server binary; set DXMT_BIN")
@pytest.mark.parametrize("fixture", FIXTURES, ids=[os.path.basename(os.path.dirname(f)) for f in FIXTURES])
def test_this_build_opens_data_written_by_an_earlier_one(fixture):
    expected = json.load(open(fixture))
    work = tempfile.mkdtemp(prefix="dxmt_compat_")
    with tarfile.open(os.path.join(os.path.dirname(fixture), "data.tar.gz")) as tar:
        tar.extractall(work)
    server = Server(os.path.join(work, "wal"))
    label = f"{expected['label']} (written by {expected['commit']})"
    try:
        server.start()
        check(server, expected, f"{label}, opened")

        # Keep using it: an upgraded database has to take writes, and survive a crash after them.
        docs = expected["records"]["docs"]
        new = {"vector": [0.1] * 8, "payload": {"tags": ["rag"], "n": 99, "after_upgrade": True}}
        server.ok("POST", "/tables/docs/records/bulk", {"records": [{"id": "d_new", **new}]})
        docs["d_new"] = {**new, "version": 1, "created_at": None}
        old = sorted(docs)[0]
        server.ok("PUT", f"/tables/docs/records/{old}", {"vector": [0.2] * 8})
        docs[old] = {**docs[old], "vector": [0.2] * 8, "version": docs[old]["version"] + 1}
        gone = sorted(docs)[1]
        server.ok("DELETE", "/tables/docs/records", {"ids": [gone]})
        del docs[gone]
        listed = server.ok("POST", "/tables/docs/records/list", {"limit": 10000})
        docs["d_new"]["created_at"] = next(r["created_at"] for r in listed["records"] if r["id"] == "d_new")

        server.stop(signal.SIGKILL)
        server.start()
        check(server, expected, f"{label}, written to and crashed")
    finally:
        server.stop()
        log = open(server.log).read() if os.path.exists(server.log) else ""
        shutil.rmtree(work, ignore_errors=True)
    assert "panicked" not in log, f"the server panicked:\n{log[-2000:]}"


def test_there_is_a_fixture():
    # An empty glob parametrises to nothing, and nothing passes. That is how a test disappears.
    assert FIXTURES, "no fixtures under tests/compat/fixtures"
