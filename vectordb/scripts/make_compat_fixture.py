#!/usr/bin/env python3
"""Write a data directory with this build, for `tests/compat` to open with every later one.

    python3 scripts/make_compat_fixture.py <label> [--bin target/debug/vectordb]

Run once per release, before the release, and commit what it writes to
`tests/compat/fixtures/<label>/`. Every fixture stays: the test opens all of them, so a change to
the WAL, the snapshot, the row files or an index format that an earlier release's data cannot
survive fails the build instead of a user's upgrade. Qdrant keeps nine releases' data for the
same reason (`tests/e2e_tests/test_data_compatibility.py`).

What the directory holds, and why each part is there:

  * a snapshot *and* a WAL tail after it — the two recovery paths. The second phase ends with
    SIGKILL, so the tail has to be replayed on top of the snapshot.
  * disk-backed tables (records and vectors in row files) and one in-memory table.
  * cosine, euclidean and dot product; a named vector space; sparse vectors.
  * payloads in every shape the filter rules care about — arrays, numbers as int and float, big
    numbers, nested objects, missing fields, an empty payload.
  * replace, merge and vector-only updates, deletes, a dropped table, an altered table.
  * a table carrying the 1.x table settings -- a payload-index policy, fast-scan off, a rerank
    provider and a recall target -- altered again after the snapshot, so the WAL replays
    `AlterTable` entries with those fields. The log writes these structs as positional
    arrays, which is the format a new field most easily breaks (1.3.0 nearly shipped that).

`expected.json` is the model of what was written, not an answer read back from the server —
otherwise whatever the server got wrong at generation time would be written down as correct. The
one exception is `created_at`, which the server assigns; it is read from a *copy* of the
directory, because starting a server can rewrite the files it opens.
"""

import argparse
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Server:
    def __init__(self, binary, wal, extra_env=None):
        self.binary, self.wal = binary, wal
        self.env = dict(os.environ, DXMT_WAL_DIR=wal, DXMT_SNAPSHOT_INTERVAL_SECS="0",
                        DXMT_EVICTION_INTERVAL_SECS="0", RUST_LOG="warn", **(extra_env or {}))
        self.env.pop("DXMT_API_KEY", None)

    def __enter__(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.proc = subprocess.Popen([self.binary, "serve", "--host", "127.0.0.1", "--port", str(self.port)],
                                     env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/live", timeout=1)
                return self
            except Exception:
                time.sleep(0.05)
        raise SystemExit("server did not start")

    def stop(self, sig):
        self.proc.send_signal(sig)
        self.proc.wait(timeout=120)

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.stop(signal.SIGKILL)

    def call(self, method, path, body=None, ok=True):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1{path}",
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{method} {path} -> {e.code} {e.read()[:300]}")


def _anonymous(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="e.g. 0.2.0 — the directory name under tests/compat/fixtures")
    parser.add_argument("--bin", default=os.path.join(ROOT, "target/debug/vectordb"))
    args = parser.parse_args()

    rng = random.Random(20260922)
    out = os.path.join(ROOT, "tests/compat/fixtures", args.label)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; a fixture is written once and never regenerated")
    work = tempfile.mkdtemp(prefix="dxmt_compat_")
    wal = os.path.join(work, "wal")

    tables = {}    # name -> config the test checks
    records = {}   # table -> id -> {vector, payload, version, sparse?, named?}

    def vec(dim):
        return [round(rng.uniform(-1, 1), 4) for _ in range(dim)]

    def payload(i):
        shapes = [
            {"tags": ["rag", "llm"], "n": i, "title": f"doc {i}"},
            {"tags": "rag", "n": float(i), "scores": [i * 10, i * 100]},
            {"tags": [], "n": 1_700_000_000_000_000 + i, "nested": {"k": i}},
            {"title": f"doc {i}", "flag": i % 2 == 0},
            {},
        ]
        return shapes[i % len(shapes)]

    def insert(server, table, dim, ids, extra=None):
        body = []
        for rid in ids:
            v, p = vec(dim), payload(int(rid.lstrip("dgpmc")))
            item = {"id": rid, "vector": v, "payload": p}
            record = {"vector": v, "payload": p}
            if extra:
                extra(rid, item, record)
            old = records[table].get(rid)
            record["version"] = old["version"] + 1 if old else 1
            records[table][rid] = record
            body.append(item)
        server.call("POST", f"/tables/{table}/records/bulk", {"records": body})

    def update(server, table, rid, body):
        server.call("PUT", f"/tables/{table}/records/{rid}", body)
        r = records[table][rid]
        if "vector" in body:
            r["vector"] = body["vector"]
        if "payload" in body:
            r["payload"] = {**r["payload"], **body["payload"]} if body.get("merge_mode") == "merge" else body["payload"]
        r["version"] += 1

    def delete(server, table, ids):
        server.call("DELETE", f"/tables/{table}/records", {"ids": ids})
        for rid in ids:
            records[table].pop(rid, None)

    def docs_extra(rid, item, record):
        i = int(rid[1:])
        if i % 3 == 0:
            item["sparse_vector"] = record["sparse"] = {"indices": [i % 16, 20 + i % 5], "values": [1.0, 0.5], "dimension": 32}
        item["named_vectors"] = record["named"] = {"title": vec(4)}

    # ── phase 1: disk-backed tables, ending with a clean shutdown and so a snapshot ──────
    with Server(args.bin, wal) as s:
        for name, dim, metric, spaces in [("docs", 8, "cosine", [{"name": "title", "dimension": 4, "distance_metric": "cosine"}]),
                                          ("geo", 3, "euclidean", None), ("dotp", 4, "dot", None), ("dropped", 4, "cosine", None)]:
            body = {"name": name, "dimension": dim, "distance_metric": metric, **({"named_vectors": spaces} if spaces else {})}
            s.call("POST", "/tables", body)
            tables[name] = {"dimension": dim, "distance_metric": metric, "named_vectors": [x["name"] for x in spaces or []]}
            records[name] = {}
        insert(s, "docs", 8, [f"d{i}" for i in range(40)], docs_extra)
        insert(s, "geo", 3, [f"g{i}" for i in range(20)])
        insert(s, "dotp", 4, [f"p{i}" for i in range(20)])
        insert(s, "dropped", 4, [f"m{i}" for i in range(5)])
        insert(s, "docs", 8, [f"d{i}" for i in range(0, 10, 3)], docs_extra)          # upserts
        update(s, "docs", "d1", {"payload": {"replaced": True}})
        update(s, "docs", "d2", {"payload": {"merged": 1}, "merge_mode": "merge"})
        update(s, "docs", "d4", {"vector": vec(8)})                                   # payload kept
        update(s, "geo", "g3", {"vector": vec(3), "payload": {"moved": True}})
        delete(s, "docs", ["d5", "d6", "absent"])
        delete(s, "geo", ["g0"])
        s.call("DELETE", "/tables/dropped")
        del tables["dropped"], records["dropped"]
        s.call("PATCH", "/tables/docs", {"description": "altered before the snapshot"})
        tables["docs"]["description"] = "altered before the snapshot"
        # The table settings 1.x added. The provider is never called: nothing here searches
        # with `rerank`, and the address is a closed port.
        rerank = {"model": "compat-reranker", "base_url": "http://127.0.0.1:9", "api_key_env": "DXMT_COMPAT_UNSET"}
        s.call("POST", "/tables", {"name": "configured", "dimension": 4, "distance_metric": "cosine",
                                   "payload_index": {"except": ["nested"]}, "quantization": {"fast_scan": False},
                                   "rerank": rerank})
        tables["configured"] = {"dimension": 4, "distance_metric": "cosine", "named_vectors": [],
                                "payload_index_except": ["nested"], "fast_scan": False}
        records["configured"] = {}
        insert(s, "configured", 4, [f"c{i}" for i in range(12)])
        s.call("PATCH", "/tables/configured", {"recall_target": 0.9})
        s.stop(signal.SIGTERM)

    # ── phase 2: an in-memory table, and writes after the snapshot, ending in a crash ────
    with Server(args.bin, wal, {"DXMT_RECORD_STORE": "memory", "DXMT_VECTOR_STORE": "memory"}) as s:
        s.call("POST", "/tables", {"name": "mem", "dimension": 4, "distance_metric": "cosine"})
        tables["mem"] = {"dimension": 4, "distance_metric": "cosine", "named_vectors": []}
        records["mem"] = {}
        insert(s, "mem", 4, [f"m{i}" for i in range(15)])
        insert(s, "docs", 8, [f"d{i}" for i in range(40, 50)], docs_extra)
        update(s, "docs", "d7", {"payload": {"after_snapshot": True}})
        update(s, "docs", "d8", {"vector": vec(8)})
        delete(s, "docs", ["d9"])
        delete(s, "mem", ["m3"])
        # Altered after the snapshot: these come back only by replaying the log.
        s.call("PATCH", "/tables/configured", {"recall_target": 0.8,
                                               "rerank": {**rerank, "model": "compat-reranker-2", "top_n": 7}})
        tables["configured"].update(recall_target=0.8, rerank={"model": "compat-reranker-2", "top_n": 7})
        insert(s, "configured", 4, [f"c{i}" for i in range(12, 16)])
        s.stop(signal.SIGKILL)

    os.makedirs(out)
    with tarfile.open(os.path.join(out, "data.tar.gz"), "w:gz") as tar:
        # No local user or group in the archive: it is published.
        tar.add(wal, arcname="wal", filter=_anonymous)

    # created_at is the server's; read it from a copy, since opening the data can rewrite it.
    copy = tempfile.mkdtemp(prefix="dxmt_compat_read_")
    shutil.copytree(wal, os.path.join(copy, "wal"))
    with Server(args.bin, os.path.join(copy, "wal")) as s:
        for table in records:
            listed = s.call("POST", f"/tables/{table}/records/list", {"limit": 10000})["data"]["records"]
            got = {r["id"]: r["created_at"] for r in listed}
            if set(got) != set(records[table]):
                raise SystemExit(f"{table}: the server holds {sorted(got)} and the model {sorted(records[table])}")
            for rid, stamp in got.items():
                records[table][rid]["created_at"] = stamp

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    json.dump({"label": args.label, "commit": commit, "tables": tables, "records": records},
              open(os.path.join(out, "expected.json"), "w"), indent=1, sort_keys=True)
    shutil.rmtree(work)
    shutil.rmtree(copy)
    size = os.path.getsize(os.path.join(out, "data.tar.gz"))
    print(f"wrote {out}: data.tar.gz {size // 1024} KB, {sum(len(r) for r in records.values())} records in {len(tables)} tables, from {commit}")


if __name__ == "__main__":
    main()
