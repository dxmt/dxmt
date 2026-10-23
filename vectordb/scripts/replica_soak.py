#!/usr/bin/env python3
"""Read replicas against real S3: a primary under write load, a replica following it, then a
failover and a returning old primary.

The in-process tests in `tests/durability/replica_test.rs` share one `FilesystemStorage` and
one clock. This runs three `vectordb serve` processes against an S3-compatible bucket, with
the primary snapshotting (and so pruning WAL chunks) every few seconds while it is written to.

    cargo build --release --features s3-storage --bin vectordb
    docker run -d --name dxmt-minio -p 9100:9000 \\
        -e MINIO_ROOT_USER=$DXMT_AWS_ACCESS_KEY_ID -e MINIO_ROOT_PASSWORD=$DXMT_AWS_SECRET_ACCESS_KEY \\
        minio/minio server /data
    docker exec dxmt-minio sh -c 'mc alias set l http://127.0.0.1:9000 "$MINIO_ROOT_USER" \\
        "$MINIO_ROOT_PASSWORD" && mc mb l/dxmt-soak'
    DXMT_S3_ENDPOINT=http://127.0.0.1:9100 DXMT_S3_BUCKET=dxmt-soak SOAK_SECONDS=120 \\
        python3 scripts/replica_soak.py

The keys are read from the environment (`DXMT_AWS_ACCESS_KEY_ID`,
`DXMT_AWS_SECRET_ACCESS_KEY`), never written here. The bucket should be empty.

Phases, each one a pass/fail line:
1. load: writers on the primary, readers on the replica, a probe measuring how long a write
   takes to become readable on the replica. A replica that exits (stale) is restarted, as a
   supervisor would, and counted.
2. converge: writers stop; the replica's count must reach the primary's.
3. failover: the primary is killed with SIGKILL, the replica is stopped and started again as a
   primary on its own directory. Every acknowledged record must be there; new writes must
   reach a fresh replica.
4. return of the old primary: the killed primary starts again on its old directory, as a
   supervisor would. The bucket has moved past it, so it must refuse to start, and the
   promoted primary must be undisturbed.
5. fail back: the promoted primary is stopped, the old machine starts on an empty directory
   and recovers everything from the bucket; the replica follows it.
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

DIM = 64
DUR = float(os.environ.get("SOAK_SECONDS", "120"))
ROOT = os.environ.get("SOAK_ROOT", tempfile.mkdtemp(prefix="dxmt-replica-soak-"))
BIN = os.environ.get("SOAK_BIN", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target/release/vectordb"))
CPUS = os.environ.get("SOAK_CPUS", "16-23")
PORTS = {"a": 9971, "b": 9972, "c": 9973}

for var in ("DXMT_S3_ENDPOINT", "DXMT_S3_BUCKET", "DXMT_AWS_ACCESS_KEY_ID",
            "DXMT_AWS_SECRET_ACCESS_KEY"):
    if not os.environ.get(var):
        sys.exit(f"{var} must be set")


def req(port, method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1{path}", data=data,
                               method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400].decode("utf8", "replace")
    except Exception as e:
        return None, repr(e)


procs = {}


def start(name, replica=False, tag=""):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "DXMT_OBJECT_STORAGE_PROVIDER": "s3",
        "DXMT_WAL_DIR": d,
        "DXMT_SSD_CACHE_DIR": d + ".ssd",
        "DXMT_SNAPSHOT_INTERVAL_SECS": "5",
        "DXMT_REPLICA_POLL_MS": "200",
        # Short enough that chunks are pruned during the run; long enough to cover a
        # replica's restart. The default is 600.
        "DXMT_WAL_RETAIN_SECS": os.environ.get("DXMT_WAL_RETAIN_SECS", "30"),
        "RUST_LOG": "info",
    })
    cmd = ["taskset", "-c", CPUS, BIN, "serve", "--host", "127.0.0.1",
           "--port", str(PORTS[name])]
    if replica:
        cmd.append("--replica")
    log = open(f"{d}{tag}.log", "a")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    procs[name] = p
    t0 = time.time()
    while time.time() - t0 < 300:
        if req(PORTS[name], "GET", "/tables")[0] == 200:
            print(f"  [{name}{' replica' if replica else ''}] serving after "
                  f"{time.time() - t0:.1f}s")
            return p
        if p.poll() is not None:
            print(f"  [{name}] exited during startup: {p.returncode}")
            print(open(f"{d}{tag}.log", errors="replace").read()[-2000:])
            return p
        time.sleep(0.2)
    sys.exit(f"{name} never came up")


def stop(name, sig=signal.SIGTERM):
    p = procs.pop(name, None)
    if p and p.poll() is None:
        p.send_signal(sig)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def vec(seed):
    rnd = random.Random(seed)
    return [round(rnd.uniform(-1, 1), 4) for _ in range(DIM)]


def count(port):
    code, body = req(port, "POST", "/tables/t/records/count", {})
    return body["data"]["count"] if code == 200 else None


def missing_on(port, ids):
    return [i for i in ids if req(port, "GET", f"/tables/t/records/{i}")[0] != 200]


acked, lock, stop_load = set(), threading.Lock(), threading.Event()
stop_supervising = threading.Event()
write_fail, read_fail, lags, replica_restarts = [], [], [], [0]


def writer(prefix, port):
    n = 0
    while not stop_load.is_set():
        ids = [f"{prefix}{n + i}" for i in range(100)]
        code, body = req(port, "POST", "/tables/t/records",
                         {"records": [{"id": i, "vector": vec(i)} for i in ids]})
        if code == 200:
            with lock:
                acked.update(ids)
        else:
            write_fail.append((code, str(body)[:200]))
        n += 100


def reader(port):
    while not stop_load.is_set():
        code, body = req(port, "POST", "/tables/t/search",
                         {"vector": vec(random.randint(0, 10_000)), "top_k": 10})
        if code != 200 and code is not None:
            read_fail.append((code, str(body)[:200]))
        time.sleep(0.05)


def lag_probe():
    n = 0
    while not stop_load.is_set():
        rid = f"probe-{n}"
        n += 1
        if req(PORTS["a"], "POST", "/tables/t/records",
               {"records": [{"id": rid, "vector": vec(rid)}]})[0] != 200:
            continue
        with lock:
            acked.add(rid)
        t0 = time.time()
        while time.time() - t0 < 30 and not stop_load.is_set():
            if req(PORTS["b"], "GET", f"/tables/t/records/{rid}")[0] == 200:
                lags.append(time.time() - t0)
                break
            time.sleep(0.02)
        time.sleep(1)


def supervise_replica():
    """Restart the replica when it exits, as systemd or Kubernetes would."""
    while not stop_supervising.is_set():
        p = procs.get("b")
        if p is not None and p.poll() is not None:
            print(f"  replica exited ({p.returncode}); restarting it")
            replica_restarts[0] += 1
            start("b", replica=True)
        time.sleep(0.5)


def wait_for(cond, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.5)
    return False


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")


def main():
    print(f"root {ROOT}")
    print("=== 1. load: primary a, replica b ===")
    start("a")
    code, _ = req(PORTS["a"], "POST", "/tables",
                  {"name": "t", "dimension": DIM, "distance_metric": "cosine"})
    assert code == 200, code
    start("b", replica=True)
    threads = [threading.Thread(target=writer, args=(f"w{k}-", PORTS["a"]), daemon=True)
               for k in range(2)]
    threads += [threading.Thread(target=reader, args=(PORTS["b"],), daemon=True)
                for _ in range(2)]
    threads += [threading.Thread(target=lag_probe, daemon=True)]
    supervisor = threading.Thread(target=supervise_replica, daemon=True)
    supervisor.start()
    for t in threads:
        t.start()
    t0 = time.time()
    while time.time() - t0 < DUR:
        time.sleep(10)
        with lock:
            n = len(acked)
        print(f"  t={int(time.time() - t0)}s acked={n} primary={count(PORTS['a'])} "
              f"replica={count(PORTS['b'])} write_fail={len(write_fail)} "
              f"read_fail={len(read_fail)} restarts={replica_restarts[0]}")
    stop_load.set()
    for t in threads:
        t.join(timeout=60)
    s = sorted(lags)
    lag = (f"p50={s[len(s) // 2] * 1000:.0f}ms p99={s[int(len(s) * 0.99)] * 1000:.0f}ms "
           f"max={s[-1] * 1000:.0f}ms n={len(s)}") if s else "no samples"
    print(f"  replica lag: {lag}")
    check("writes on the primary", not write_fail, str(write_fail[:3]))
    check("reads on the replica", not read_fail, str(read_fail[:3]))

    print("=== 2. converge ===")
    with lock:
        expect = sorted(acked)
    ok = wait_for(lambda: count(PORTS["b"]) == count(PORTS["a"]) == len(expect), 60)
    check("replica count reaches the primary's", ok,
          f"acked={len(expect)} primary={count(PORTS['a'])} replica={count(PORTS['b'])}")
    sample = random.sample(expect, min(1000, len(expect)))
    miss = missing_on(PORTS["b"], sample)
    check("sampled records readable on the replica", not miss, str(miss[:5]))

    check("the replica never had to restart", replica_restarts[0] == 0,
          f"restarts={replica_restarts[0]}")

    print("=== 3. failover: kill -9 a, promote b ===")
    stop_supervising.set()
    supervisor.join(timeout=60)
    stop("a", signal.SIGKILL)
    stop("b")
    start("b")  # same directory, no --replica: promotion
    check("promoted b holds every acknowledged record", count(PORTS["b"]) == len(expect),
          f"{count(PORTS['b'])} of {len(expect)}")
    miss = missing_on(PORTS["b"], sample)
    check("sampled records on promoted b", not miss, str(miss[:5]))
    new_ids = [f"after-{i}" for i in range(2000)]
    code, body = req(PORTS["b"], "POST", "/tables/t/records",
                     {"records": [{"id": i, "vector": vec(i)} for i in new_ids]}, timeout=120)
    check("promoted b accepts writes", code == 200, str(body)[:200])
    start("c", replica=True)
    total = len(expect) + len(new_ids)
    ok = wait_for(lambda: count(PORTS["c"]) == total, 60)
    check("a new replica c follows promoted b", ok, f"c={count(PORTS['c'])} want {total}")

    print("=== 4. the old primary a comes back on its directory ===")
    p = start("a", tag=".return")
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    log = open(os.path.join(ROOT, "a.return.log"), errors="replace").read()
    check("a refuses to start on a directory the bucket has moved past",
          p.poll() not in (None, 0) and "has written to this object storage bucket" in log,
          f"exit={p.poll()}")
    procs.pop("a", None)
    code, body = req(PORTS["b"], "POST", "/tables/t/records",
                     {"records": [{"id": "late", "vector": vec("late")}]})
    check("b still accepts writes", code == 200, f"{code} {str(body)[:160]}")
    total += 1
    ok = wait_for(lambda: count(PORTS["c"]) == total, 30)
    check("replica c still follows b", ok, f"c={count(PORTS['c'])} want {total}")

    print("=== 5. fail back: stop b, a starts on an empty directory ===")
    stop("b")
    shutil.move(os.path.join(ROOT, "a"), os.path.join(ROOT, "a.superseded"))
    start("a", tag=".fresh")
    check("a recovered from the bucket holds everything", count(PORTS["a"]) == total,
          f"{count(PORTS['a'])} of {total}")
    code, _ = req(PORTS["a"], "POST", "/tables/t/records",
                  {"records": [{"id": "from-a", "vector": vec("from-a")}]})
    ok = code == 200 and wait_for(
        lambda: req(PORTS["c"], "GET", "/tables/t/records/from-a")[0] == 200, 30)
    check("replica c follows a", ok)

    for name in list(procs):
        stop(name)
    print(f"\nRESULT: {'PASS' if all(results) else 'FAIL'} "
          f"({sum(results)}/{len(results)}); logs under {ROOT}")
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
