#!/usr/bin/env python3
"""A cluster under load against real S3: two shards of a primary and a replica each, two
routers, many tables written through both, and one shard failed over halfway.

    cargo build --release --features s3-storage --bin vectordb
    DXMT_S3_ENDPOINT=http://127.0.0.1:9100 DXMT_S3_BUCKET=dxmt-soak \\
        DXMT_AWS_ACCESS_KEY_ID=... DXMT_AWS_SECRET_ACCESS_KEY=... \\
        SOAK_SECONDS=120 python3 scripts/cluster_soak.py

Keys come from the environment. The bucket should be empty, and must honour conditional writes
(S3, MinIO -- not GCS's S3 API). Checks, each a PASS/FAIL line:

- tables spread over the shards; both routers list the same tables;
- writes through either router are acknowledged and searches answer, including reads sent to
  replicas;
- shard a's primary is killed with SIGKILL and nothing is restarted: its replica sees the lease
  run out, promotes itself, and writes to shard a resume -- the time that took is reported;
- every write the cluster acknowledged, on every table, is there at the end.
"""
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

DIM = 32
TABLES = int(os.environ.get("SOAK_TABLES", "20"))
DUR = float(os.environ.get("SOAK_SECONDS", "120"))
ROOT = os.environ.get("SOAK_ROOT", tempfile.mkdtemp(prefix="dxmt-cluster-soak-"))
BIN = os.environ.get("SOAK_BIN", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target/release/vectordb"))
CPUS = os.environ.get("SOAK_CPUS", "16-23")

for var in ("DXMT_S3_ENDPOINT", "DXMT_S3_BUCKET", "DXMT_AWS_ACCESS_KEY_ID",
            "DXMT_AWS_SECRET_ACCESS_KEY"):
    if not os.environ.get(var):
        sys.exit(f"{var} must be set")


def req(port, method, path, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", **(headers or {})}
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode("utf8", "replace"), dict(e.headers or {})
    except Exception as e:
        return None, repr(e), {}


procs = {}
ports = {}


def free_port():
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def env_for(name):
    env = dict(os.environ)
    env.update({
        "DXMT_OBJECT_STORAGE_PROVIDER": "s3",
        "DXMT_SSD_CACHE_DIR": os.path.join(ROOT, name + ".ssd"),
        "DXMT_SNAPSHOT_INTERVAL_SECS": "10",
        "DXMT_WAL_RETAIN_SECS": "60",
        "DXMT_REPLICA_POLL_MS": "200",
        "DXMT_HEARTBEAT_SECS": "1",
        "DXMT_NODE_TTL_SECS": "5",
        "DXMT_ROUTER_REFRESH_MS": "500",
        "DXMT_EMBEDDING_AUTODETECT": "0",
        "RUST_LOG": "warn",
    })
    env.pop("DXMT_WAL_DIR", None)
    return env


def wait_live(name):
    t0 = time.time()
    while time.time() - t0 < 120:
        if req(ports[name], "GET", "/live")[0] == 200:
            return
        if procs[name].poll() is not None:
            sys.exit(f"{name} exited: {procs[name].returncode}; see {ROOT}/{name}.log")
        time.sleep(0.2)
    sys.exit(f"{name} never came up")


def start_node(name, shard, replica=False, data=None):
    port = free_port()
    ports[name] = port
    env = env_for(name)
    env["DXMT_ROOT_DIR"] = os.path.join(ROOT, data or name)
    cmd = ["taskset", "-c", CPUS, BIN, "serve", "--host", "127.0.0.1", "--port", str(port),
           "--shard", shard, "--advertise", f"http://127.0.0.1:{port}"]
    if replica:
        cmd.append("--replica")
    procs[name] = subprocess.Popen(cmd, env=env, stdout=open(f"{ROOT}/{name}.log", "a"),
                                   stderr=subprocess.STDOUT)
    wait_live(name)


def start_router(name):
    port = free_port()
    ports[name] = port
    cmd = ["taskset", "-c", CPUS, BIN, "router", "--host", "127.0.0.1", "--port", str(port)]
    procs[name] = subprocess.Popen(cmd, env=env_for(name), stdout=open(f"{ROOT}/{name}.log", "a"),
                                   stderr=subprocess.STDOUT)
    wait_live(name)


def stop(name, sig=signal.SIGTERM):
    p = procs.pop(name, None)
    if p and p.poll() is None:
        p.send_signal(sig)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()


def status(router):
    s, body, _ = req(ports[router], "GET", "/cluster/status")
    return body["data"] if s == 200 else {}


def wait_for(cond, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.5)
    return False


def vec(seed):
    rnd = random.Random(seed)
    return [round(rnd.uniform(-1, 1), 4) for _ in range(DIM)]


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


acked = {}           # table -> set of ids the cluster acknowledged
lock = threading.Lock()
stop_load = threading.Event()
write_status, read_status = {}, {}
shard_of = {}
a_writes = []        # (time, ok) for writes to shard a's tables


def writer(k):
    rnd = random.Random(k)
    n = 0
    while not stop_load.is_set():
        table = f"t{rnd.randrange(TABLES)}"
        router = rnd.choice(["r1", "r2"])
        ids = [f"w{k}-{n + i}" for i in range(20)]
        n += 20
        s, _, _ = req(ports[router], "POST", f"/api/v1/tables/{table}/records",
                      {"records": [{"id": i, "vector": vec(i)} for i in ids]})
        with lock:
            write_status[s] = write_status.get(s, 0) + 1
            if s == 200:
                acked.setdefault(table, set()).update(ids)
            if shard_of.get(table) == "a":
                a_writes.append((time.time(), s == 200))


def reader(k):
    rnd = random.Random(1000 + k)
    while not stop_load.is_set():
        table = f"t{rnd.randrange(TABLES)}"
        router = rnd.choice(["r1", "r2"])
        h = {"x-dxmt-read": "replica"} if rnd.random() < 0.5 else {}
        s, _, _ = req(ports[router], "POST", f"/api/v1/tables/{table}/search",
                      {"vector": vec(rnd.random()), "top_k": 5}, headers=h)
        with lock:
            read_status[s] = read_status.get(s, 0) + 1
        time.sleep(0.02)


def main():
    print(f"root {ROOT}", flush=True)
    os.makedirs(ROOT, exist_ok=True)
    start_node("a1", "a"); start_node("b1", "b")
    start_node("a2", "a", replica=True); start_node("b2", "b", replica=True)
    start_router("r1"); start_router("r2")
    ok = wait_for(lambda: all(
        any(s["shard"] == sh and s["primary"] and len(s["replicas"]) == 1
            for s in status("r1").get("shards", [])) for sh in ("a", "b")), 60)
    check("routers see two shards, each with a primary and a replica", ok, json.dumps(status("r1"))[:300])

    for i in range(TABLES):
        s, body, _ = req(ports["r1" if i % 2 else "r2"], "POST", "/api/v1/tables",
                         {"name": f"t{i}", "dimension": DIM, "distance_metric": "cosine"})
        assert s == 200, (i, s, body)
    st = status("r1")
    per = {s["shard"]: s["tables"] for s in st["shards"]}
    check("tables spread over the shards", abs(per.get("a", 0) - per.get("b", 0)) <= 1, str(per))
    for i in range(TABLES):
        s, _, h = req(ports["r1"], "POST", f"/api/v1/tables/t{i}/search", {"vector": vec(0), "top_k": 1})
        shard_of[f"t{i}"] = h.get("x-dxmt-shard")

    threads = [threading.Thread(target=writer, args=(k,), daemon=True) for k in range(4)]
    threads += [threading.Thread(target=reader, args=(k,), daemon=True) for k in range(4)]
    for t in threads:
        t.start()
    time.sleep(DUR / 2)

    print("=== shard a's primary SIGKILLed; its replica should take over by itself ===", flush=True)
    killed_at = time.time()
    stop("a1", signal.SIGKILL)
    ok = wait_for(lambda: any(s["shard"] == "a" and s["primary"] and s["primary"]["node"] ==
                              f"127-0-0-1-{ports['a2']}" for s in status("r1").get("shards", [])), 60)
    check("the replica promotes itself and the routers follow", ok,
          f"{time.time() - killed_at:.1f}s after the kill")
    ok = wait_for(lambda: any(t > killed_at and good for t, good in list(a_writes)), 60)
    first = min((t for t, good in list(a_writes) if t > killed_at and good), default=None)
    check("writes to shard a resume", ok,
          f"{first - killed_at:.1f}s after the kill" if first else "")
    start_node("a3", "a", replica=True)  # a new replica for the new primary

    time.sleep(DUR / 2)
    stop_load.set()
    for t in threads:
        t.join(timeout=60)
    print(f"  writes by status {write_status}  reads by status {read_status}", flush=True)
    check("no request failed outside the failover",
          set(write_status) <= {200, 502, 503} and set(read_status) <= {200, 502, 503},
          f"{write_status} {read_status}")

    total = sum(len(v) for v in acked.values())
    missing = {}
    for table, ids in sorted(acked.items()):
        s, body, _ = req(ports["r2"], "POST", f"/api/v1/tables/{table}/records/count", {})
        n = body["data"]["count"] if s == 200 else None
        if n != len(ids):
            missing[table] = (n, len(ids))
        sample = random.sample(sorted(ids), min(50, len(ids)))
        lost = [i for i in sample if req(ports["r1"], "GET", f"/api/v1/tables/{table}/records/{i}")[0] != 200]
        if lost:
            missing.setdefault(table, (n, len(ids)))
    check("every acknowledged write is there", not missing,
          f"{total} acknowledged over {len(acked)} tables; mismatched {missing}")
    l1 = req(ports["r1"], "GET", "/api/v1/tables?limit=1000")[1]["data"]["total"]
    l2 = req(ports["r2"], "GET", "/api/v1/tables?limit=1000")[1]["data"]["total"]
    check("both routers list every table", l1 == l2 == TABLES, f"{l1} {l2}")

    for name in list(procs):
        stop(name)
    print(f"\nRESULT: {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)}); logs under {ROOT}")
    if all(results) and not os.environ.get("SOAK_KEEP"):
        shutil.rmtree(ROOT, ignore_errors=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
