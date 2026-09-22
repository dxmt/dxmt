"""A random walk through writes, restarts and crashes, checked against an independent model.

Every serious bug found in the week this was written passed the whole gate first: a contradictory
filter that ended the process, an update followed by a crash that left a server which could not
start, a vector-only update that erased the payload, five ways the payload index disagreed with
the records. Each was found by driving a real server through a sequence nobody had written down
and comparing with an answer computed some other way. This makes that the gate.

Chroma's `RuleBasedStateMachine` tests are the model (`chromadb/test/property/`): random
operations, invariants after each step, and restart as one of the operations. This is a seeded
walk rather than Hypothesis, so a CI failure prints a seed and a step that reproduce it exactly,
and costs no extra dependency.

The model — `Model` below — shares no code with the server. The filter semantics it implements are
the documented ones: a condition on an array field matches when any element does, `NotEqual` is
true for a missing field and `NotIn` is not, numbers compare as numbers, `id` means the record's
own id.

Run through `scripts/api_tests.sh`, which sets `DXMT_BIN`. Directly:

    DXMT_BIN=target/debug/vectordb pytest tests/stateful
    DXMT_STATEFUL_SEEDS=1-40 DXMT_STATEFUL_STEPS=400 DXMT_BIN=... pytest tests/stateful   # soak
    DXMT_STATEFUL_ALLOW_STALE=1 DXMT_BIN=old/vectordb pytest tests/stateful   # does it catch X?
"""

import json
import math
import os
import random
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import pytest

BIN = os.environ.get("DXMT_BIN") or ("target/debug/vectordb" if os.path.exists("target/debug/vectordb") else None)
DIM = 4
IDS = [f"r{i}" for i in range(40)]  # a small id space, so upserts, deletes and re-inserts collide
TOL = 1e-4


def seeds():
    spec = os.environ.get("DXMT_STATEFUL_SEEDS", "1-3")
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


STEPS = int(os.environ.get("DXMT_STATEFUL_STEPS", "150"))
# Every write and restart, one JSON object a line, for replaying a failure outside the walk.
TRACE = os.environ.get("DXMT_STATEFUL_TRACE")


def stale_binary():
    """The newest source file if it is newer than the binary, else None.

    A walk against a stale binary tests the binary, not the source. Written after exactly that:
    `cargo test` run on a stashed tree rebuilt `target/debug/vectordb` from the stash, the stash
    was popped, and the walk that followed reported a bug the source had already fixed.
    """
    built = os.path.getmtime(BIN)
    newest = max((os.path.join(d, f) for d, _, fs in os.walk("src") for f in fs), key=os.path.getmtime, default=None)
    return newest if newest and os.path.getmtime(newest) > built else None


# ── the server ──────────────────────────────────────────────────────────────────────────────


class Server:
    def __init__(self, mode):
        self.dir = tempfile.mkdtemp(prefix=f"dxmt_walk_{mode}_")
        self.env = dict(os.environ, DXMT_WAL_DIR=f"{self.dir}/wal", DXMT_SNAPSHOT_INTERVAL_SECS="0",
                        DXMT_EVICTION_INTERVAL_SECS="0", DXMT_RECORD_STORE=mode, DXMT_VECTOR_STORE=mode,
                        RUST_LOG="warn")
        self.env.pop("DXMT_API_KEY", None)
        self.proc = None
        self.log = f"{self.dir}/server.log"

    def start(self):
        s = __import__("socket").socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.proc = subprocess.Popen([BIN, "serve", "--host", "127.0.0.1", "--port", str(self.port)],
                                     env=self.env, stdout=open(self.log, "a"), stderr=subprocess.STDOUT)
        for _ in range(600):
            if self.proc.poll() is not None:
                raise AssertionError(f"server exited on start:\n{self.tail()}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/live", timeout=1)
                return
            except Exception:
                time.sleep(0.05)
        raise AssertionError(f"server did not come up:\n{self.tail()}")

    def stop(self, sig):
        if TRACE:
            with open(TRACE, "a") as t:
                t.write(json.dumps({"restart": signal.Signals(sig).name}) + "\n")
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(sig)
            self.proc.wait(timeout=120)

    def tail(self):
        try:
            return "".join(open(self.log).readlines()[-15:])
        except OSError:
            return ""

    def call(self, method, path, body=None):
        if TRACE and method != "GET" and not path.endswith(("/list", "/search")):
            with open(TRACE, "a") as t:
                t.write(json.dumps({"method": method, "path": path, "body": body}) + "\n")
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1{path}",
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as e:
            raw = e.read()
            return e.code, json.loads(raw) if raw else {}


# ── the model ───────────────────────────────────────────────────────────────────────────────


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def same(a, b):
    if is_num(a) and is_num(b):
        return float(a) == float(b)
    return type(a) is type(b) and a == b


def elements(v):
    return v if isinstance(v, list) else [v]


def scalar_ok(x, op, t):
    if op == "Equal":
        return same(x, t)
    if op == "In":
        return any(same(x, y) for y in t)
    if op in ("GreaterThan", "GreaterThanOrEqual", "LessThan", "LessThanOrEqual"):
        if not (is_num(x) and is_num(t)):
            return False
        return {"GreaterThan": x > t, "GreaterThanOrEqual": x >= t, "LessThan": x < t, "LessThanOrEqual": x <= t}[op]
    if not isinstance(x, str):
        return False
    return {"Contains": t in x, "StartsWith": x.startswith(t), "EndsWith": x.endswith(t)}[op]


def cond_ok(rid, payload, c):
    op, t = c["operation"]["type"], c["operation"]["value"]
    if c["key"] == "id":
        present, v = True, rid
    else:
        present, v = c["key"] in payload, payload.get(c["key"])
    if op == "NotEqual":
        return not (present and any(same(x, t) for x in elements(v)))
    if op == "NotIn":
        return present and not any(same(x, y) for x in elements(v) for y in t)
    return present and any(scalar_ok(x, op, t) for x in elements(v))


def matches(rid, payload, f):
    if f.get("must") and not all(cond_ok(rid, payload, c) for c in f["must"]):
        return False
    if f.get("should") and not any(cond_ok(rid, payload, c) for c in f["should"]):
        return False
    if f.get("must_not") and any(cond_ok(rid, payload, c) for c in f["must_not"]):
        return False
    return True


def instant(stamp):
    """An RFC 3339 timestamp as something that orders correctly: the fraction padded to nine
    digits, since a serialiser may drop trailing zeros and "…58.2Z" must not sort after "…58.19Z"."""
    head, _, rest = stamp.rstrip("Z").partition(".")
    return head, (rest + "000000000")[:9]


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class Model:
    """What the server should hold: id -> vector, payload, version, created_at once observed."""

    def __init__(self):
        self.records = {}

    def upsert(self, rid, vector, payload):
        old = self.records.get(rid)
        self.records[rid] = {"vector": vector, "payload": payload,
                             "version": old["version"] + 1 if old else 1,
                             "created_at": old["created_at"] if old else None}

    def update(self, rid, vector, payload, merge):
        r = self.records[rid]
        if vector is not None:
            r["vector"] = vector
        if payload is not None:
            r["payload"] = {**r["payload"], **payload} if merge else payload
        r["version"] += 1


# ── generators ──────────────────────────────────────────────────────────────────────────────

TAGS = ["rag", "llm", "db", "x"]


def gen_vector(rng):
    while True:
        v = [round(rng.uniform(-1, 1), 3) for _ in range(DIM)]
        if any(abs(x) > 0.05 for x in v):
            return v


def gen_payload(rng):
    p = {}
    if rng.random() < 0.8:
        p["tags"] = rng.choice([[], ["rag"], ["rag", "llm"], ["db", "x"], ["rag", "rag"], [["rag"], "x"], ["llm", 1], "rag", "db"])
    if rng.random() < 0.7:
        p["n"] = rng.choice([1, 2, 2.0, 3.5, 7, 1_700_000_000_000_001, -4])
    if rng.random() < 0.5:
        p["scores"] = rng.choice([[], [100, 700], [500], [7, 7.0]])
    if rng.random() < 0.5:
        p["title"] = f"title {rng.randint(0, 30)}"
    if rng.random() < 0.2:
        p["obj"] = {"nested": rng.randint(0, 3)}
    return p


def C(key, op, value):
    return {"key": key, "operation": {"type": op, "value": value}}


def gen_filter(rng):
    choices = [
        lambda: C("tags", "Equal", rng.choice(TAGS)),
        lambda: C("tags", "NotEqual", rng.choice(TAGS)),
        lambda: C("tags", "In", rng.sample(TAGS, 2)),
        lambda: C("tags", "NotIn", [rng.choice(TAGS)]),
        lambda: C("tags", "StartsWith", rng.choice(["r", "l", "d"])),
        lambda: C("n", "Equal", rng.choice([2, 2.0, 7, 1_700_000_000_000_001])),
        lambda: C("n", "NotEqual", rng.choice([2, 7])),
        lambda: C("n", "GreaterThan", rng.choice([0, 2, 3.5, 1_700_000_000_000_000])),
        lambda: C("n", "LessThanOrEqual", rng.choice([-4, 2, 7])),
        lambda: C("scores", "GreaterThanOrEqual", rng.choice([100, 400, 700])),
        lambda: C("scores", "LessThanOrEqual", rng.choice([100, 600])),
        lambda: C("title", "Contains", str(rng.randint(0, 9))),
        lambda: C("id", "Equal", rng.choice(IDS)),
        lambda: C("id", "In", rng.sample(IDS, 3)),
        lambda: C("obj", "NotEqual", "x"),
    ]
    f = {"must": [rng.choice(choices)() for _ in range(rng.randint(1, 2))]}
    if rng.random() < 0.3:
        f["must_not"] = [rng.choice(choices)()]
    if rng.random() < 0.2:
        f["should"] = [rng.choice(choices)() for _ in range(2)]
    return f


# ── checks ──────────────────────────────────────────────────────────────────────────────────


def check(server, model, rng, where):
    """Everything the server says, against the model. `where` names the step for the report."""
    status, body = server.call("POST", "/tables/t/records/list", {"limit": 10000, "with_vector": True})
    assert status == 200, f"{where}: list answered {status} {body}"
    listed = body["data"]["records"]
    got = {r["id"]: r for r in listed}
    assert set(got) == set(model.records), (
        f"{where}: ids differ — missing {sorted(set(model.records) - set(got))}, extra {sorted(set(got) - set(model.records))}")
    assert body["data"]["total"] == len(model.records), f"{where}: total {body['data']['total']}"

    for rid, want in model.records.items():
        r = got[rid]
        payload = json.loads(r["payload_json"])
        assert payload == want["payload"], f"{where}: {rid} payload {payload} != {want['payload']}"
        assert r["version"] == want["version"], f"{where}: {rid} version {r['version']} != {want['version']}"
        if want["created_at"] is None:
            want["created_at"] = r["created_at"]
        assert r["created_at"] == want["created_at"], f"{where}: {rid} created_at moved {want['created_at']} -> {r['created_at']}"
        vector = r["vector"] or server.call("GET", f"/tables/t/records/{rid}")[1]["data"]["vector"]
        assert all(abs(a - b) < 1e-5 for a, b in zip(vector, want["vector"])) and len(vector) == DIM, (
            f"{where}: {rid} vector {vector} != {want['vector']}")

    # Newest first, ties by id: sort by id, then stably by time descending.
    order = [r["id"] for r in listed]
    expected_order = sorted(sorted(order), key=lambda i: instant(got[i]["created_at"]), reverse=True)
    assert order == expected_order, f"{where}: list is not newest-first with ties by id"

    # Exact search: a valid top-k by cosine, whatever the ties.
    if model.records:
        q = gen_vector(rng)
        k = min(8, len(model.records))
        status, body = server.call("POST", "/tables/t/search", {"vector": q, "top_k": k, "search_mode": "exact"})
        assert status == 200, f"{where}: exact search {status} {body}"
        hits = body["data"]["hits"]
        truth = {rid: cosine(q, r["vector"]) for rid, r in model.records.items()}
        assert len(hits) == k, f"{where}: exact search returned {len(hits)} of {k}"
        for h in hits:
            assert abs(h["score"] - truth[h["id"]]) < TOL, f"{where}: {h['id']} score {h['score']} != {truth[h['id']]}"
        floor = min(h["score"] for h in hits)
        missed = [rid for rid in truth if rid not in {h["id"] for h in hits} and truth[rid] > floor + TOL]
        assert not missed, f"{where}: exact search left out {missed}, which score above {floor}"

    # Filters, through every route that answers one.
    for _ in range(4):
        f = gen_filter(rng)
        want = [i for i in order if matches(i, model.records[i]["payload"], f)]
        paged, offset = [], 0
        while True:
            status, body = server.call("POST", "/tables/t/records/list", {"limit": 7, "offset": offset, "filter": f})
            assert status == 200, f"{where}: filtered list {status} {body} for {f}"
            assert body["data"]["total"] == len(want), f"{where}: total {body['data']['total']} != {len(want)} for {f}"
            page = [r["id"] for r in body["data"]["records"]]
            if not page:
                break
            paged += page
            offset += len(page)
        assert paged == want, (f"{where}: records/list for {f}\n  got  {paged}\n  want {want}\n  created_at "
                               + ", ".join(f"{i}={got[i]['created_at'][11:]}" for i in want))

        status, body = server.call("POST", "/tables/t/search", {"vector": gen_vector(rng), "top_k": max(1, len(model.records)),
                                                                "filter": f, "search_mode": "exact"})
        assert status == 200, f"{where}: filtered exact search {status} {body}"
        assert sorted(h["id"] for h in body["data"]["hits"]) == sorted(want), f"{where}: exact search for {f}"

        status, body = server.call("POST", "/tables/t/search", {"vector": gen_vector(rng), "top_k": 10, "filter": f})
        assert status == 200, f"{where}: filtered search {status} {body}"
        wrong = [h["id"] for h in body["data"]["hits"] if h["id"] not in want]
        assert not wrong, f"{where}: search returned records the filter excludes: {wrong} for {f}"


# ── the walk ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(BIN is None, reason="no server binary; set DXMT_BIN")
@pytest.mark.parametrize("mode", ["disk", "memory"])
@pytest.mark.parametrize("seed", seeds())
def test_walk(mode, seed):
    newer = None if os.environ.get("DXMT_STATEFUL_ALLOW_STALE") else stale_binary()
    assert newer is None, f"{BIN} is older than {newer}; run `cargo build --bin vectordb` first"
    rng = random.Random(seed * 7919 + (mode == "disk"))
    server, model, trail = Server(mode), Model(), []
    server.start()
    try:
        status, body = server.call("POST", "/tables", {"name": "t", "dimension": DIM, "distance_metric": "cosine"})
        assert status == 200, body

        def step_insert():
            batch = []
            for rid in rng.sample(IDS, rng.randint(1, 8)):
                vector, payload = gen_vector(rng), gen_payload(rng) if rng.random() < 0.9 else None
                batch.append({"id": rid, "vector": vector, **({"payload": payload} if payload is not None else {})})
                model.upsert(rid, vector, payload if payload is not None else {})
            status, body = server.call("POST", "/tables/t/records/bulk", {"records": batch})
            assert status == 200, body
            return f"insert {[r['id'] for r in batch]}"

        def step_update():
            existing = sorted(model.records)
            if not existing or rng.random() < 0.1:
                rid = rng.choice([i for i in IDS if i not in model.records] or IDS)
                status, _ = server.call("PUT", f"/tables/t/records/{rid}", {"payload": {"ghost": 1}})
                if rid in model.records:
                    model.update(rid, None, {"ghost": 1}, False)
                    assert status == 200
                else:
                    assert status == 404, f"update of missing {rid} answered {status}"
                return f"update missing {rid}"
            rid = rng.choice(existing)
            kind = rng.choice(["replace", "merge", "vector", "both", "empty"])
            body, vector, payload, merge = {}, None, None, False
            if kind in ("vector", "both"):
                vector = gen_vector(rng)
                body["vector"] = vector
            if kind in ("replace", "merge", "both"):
                payload = gen_payload(rng)
                body["payload"] = payload
            if kind == "merge":
                body["merge_mode"] = "merge"
                merge = True
            status, response = server.call("PUT", f"/tables/t/records/{rid}", body)
            assert status == 200, f"update {rid} {body}: {status} {response}"
            model.update(rid, vector, payload, merge)
            return f"update {rid} {kind}"

        def step_delete():
            ids = rng.sample(IDS, rng.randint(1, 5))
            status, body = server.call("DELETE", "/tables/t/records", {"ids": ids})
            assert status == 200, body
            present = [i for i in ids if i in model.records]
            assert body["data"]["deleted_count"] == len(present), f"deleted_count {body['data']} for {ids}, present {present}"
            for i in present:
                del model.records[i]
            return f"delete {ids}"

        def step_snapshot():
            status, body = server.call("POST", "/snapshot", {})
            assert status == 200, body
            return "snapshot"

        def step_restart():
            sig = rng.choice([signal.SIGTERM, signal.SIGKILL])
            server.stop(sig)
            server.start()
            return f"restart {sig.name}"

        steps = [(step_insert, 30), (step_update, 30), (step_delete, 12), (step_snapshot, 4), (step_restart, 6)]
        population, weights = zip(*steps)
        for n in range(STEPS):
            op = rng.choices(population, weights)[0]
            trail.append(f"{n}: {op()}")
            if op is step_restart or n % 15 == 14:
                check(server, model, rng, f"seed {seed} {mode} step {n}")
        check(server, model, rng, f"seed {seed} {mode} end")
    except AssertionError as e:
        raise AssertionError(f"{e}\n\nlast steps:\n  " + "\n  ".join(trail[-20:]) +
                             f"\n\nserver log tail:\n{server.tail()}\nreproduce: DXMT_STATEFUL_SEEDS={seed} pytest tests/stateful -k {mode}") from None
    finally:
        server.stop(signal.SIGKILL)
        assert "panicked" not in open(server.log).read(), f"the server panicked:\n{server.tail()}"
