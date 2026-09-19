#!/usr/bin/env python3
"""End-to-end soak against the real `vectordb serve` process.

The unit and integration suites verify everything in-process, calling snapshot() /
evict_cold_tables() / reevaluate_indexes() directly. In a real server those are three background
timers running concurrently with read and write traffic, and that combination is what this
exercises. It found two bugs nothing in the 35-test recovery suite could see.

The contract under test: every record the server answered 200 for must be retrievable after
`kill -9` and restart, and the server must come back in reasonable time.

    cargo build --release --bin vectordb
    SOAK_SECONDS=120 python3 scripts/soak.py

Exits non-zero on any lost record, failed request, or a restart that does not serve within
ten minutes. Sweep intervals are deliberately mutually prime (5/7/11s) so the three background
tasks drift in and out of phase instead of always firing on the same tick.
"""
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("SOAK_PORT", "9971"))
BASE = f"http://127.0.0.1:{PORT}/api/v1"
DIM = 128
DUR = float(os.environ.get("SOAK_SECONDS", "120"))
WAL = os.environ.get("SOAK_WAL", tempfile.mkdtemp(prefix="dxmt-soak-"))
BIN = os.environ.get("SOAK_BIN", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target/release/vectordb"))


def req(method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400].decode("utf8", "replace")
    except Exception as e:
        return None, repr(e)


def start(tag):
    env = dict(os.environ)
    env.update({
        "DXMT_WAL_DIR": WAL,
        "DXMT_PORT": str(PORT),
        # Mutually prime, so the three sweeps drift in and out of phase with each other
        # instead of always firing on the same tick.
        "DXMT_SNAPSHOT_INTERVAL_SECS": "5",
        "DXMT_EVICTION_INTERVAL_SECS": "7",
        "DXMT_REINDEX_INTERVAL_SECS": "11",
        "RUST_LOG": "info",
    })
    log = open(f"{WAL}.{tag}.log", "w")
    p = subprocess.Popen([BIN, "serve", "--host", "127.0.0.1", "--port", str(PORT)],
                         env=env, stdout=log, stderr=subprocess.STDOUT)
    t_up = time.time()
    for _ in range(3000):
        code, _ = req("GET", "/tables")
        if code == 200:
            print(f"  [{tag}] serving after {time.time()-t_up:.2f}s")
            return p
        if p.poll() is not None:
            print(f"server died during startup, exit={p.returncode}")
            print(open(f"{WAL}.{tag}.log").read()[-3000:])
            sys.exit(1)
        time.sleep(0.2)
    print(f"server never came up within {3000*0.2:.0f}s")
    p.kill()
    p.wait()
    sys.exit(1)


def vec(seed):
    rnd = random.Random(seed)
    return [round(rnd.uniform(-1, 1), 4) for _ in range(DIM)]


acked = set()          # ids the server returned 200 for
acked_lock = threading.Lock()
read_fail = []
read_lat = []
write_fail = []
stop = threading.Event()


def writer(table, start_id, batch):
    n = start_id
    while not stop.is_set():
        ids = [f"{table}-{n + i}" for i in range(batch)]
        recs = [{"id": i, "vector": vec(i)} for i in ids]
        code, body = req("POST", f"/tables/{table}/records", {"records": recs})
        if code == 200:
            with acked_lock:
                acked.update(ids)
        else:
            write_fail.append((code, str(body)[:200]))
        n += batch


def reader(table):
    while not stop.is_set():
        t0 = time.time()
        code, body = req("POST", f"/tables/{table}/search",
                         {"vector": vec(random.randint(0, 10_000)), "top_k": 10})
        dt = time.time() - t0
        read_lat.append(dt)
        if code != 200:
            read_fail.append((code, str(body)[:200]))
        time.sleep(0.05)


def kill_leftovers():
    """A previous run that died in start() leaves its server holding the port. The next run then
    talks to *that* process, on its own old WAL, and every result is contaminated — the tell is a
    409 on CREATE TABLE and an "address already in use" panic in the log."""
    subprocess.run(["pkill", "-9", "-f", f"vectordb serve --host 127.0.0.1 --port {PORT}"],
                   check=False)
    time.sleep(1)


def main():
    kill_leftovers()
    subprocess.run(["rm", "-rf", WAL], check=False)
    os.makedirs(WAL, exist_ok=True)

    print("=== phase 1: start, load, concurrent traffic ===")
    p = start("run1")
    for t in ("hot", "cold"):
        code, body = req("POST", "/tables",
                         {"name": t, "dimension": DIM, "distance_metric": "cosine"})
        print(f"  create {t}: {code}")

    # `cold` is written once and then abandoned, so the eviction sweep has something to
    # actually evict while snapshots and writes are in flight on `hot`. That is the exact
    # combination that lost a whole table before.
    ids = [f"cold-{i}" for i in range(2000)]
    code, _ = req("POST", "/tables/cold/records",
                  {"records": [{"id": i, "vector": vec(i)} for i in ids]}, timeout=120)
    print(f"  seed cold with 2000: {code}")
    if code == 200:
        with acked_lock:
            acked.update(ids)

    threads = [threading.Thread(target=writer, args=("hot", 0, 200), daemon=True),
               threading.Thread(target=writer, args=("hot", 500_000, 200), daemon=True)]
    threads += [threading.Thread(target=reader, args=("hot",), daemon=True) for _ in range(3)]
    for t in threads:
        t.start()

    t0 = time.time()
    while time.time() - t0 < DUR:
        time.sleep(5)
        with acked_lock:
            n = len(acked)
        st = req("GET", "/tables/hot/stats")[1]
        it = st.get("data", {}).get("index_type") if isinstance(st, dict) else "?"
        print(f"  t={int(time.time()-t0)}s acked={n} read_fail={len(read_fail)} "
              f"write_fail={len(write_fail)} p_max_read={max(read_lat or [0]):.2f}s idx={it}")
    stop.set()
    for t in threads:
        t.join(timeout=30)

    with acked_lock:
        expect = sorted(acked)
    print(f"\n  acknowledged {len(expect)} records")
    print(f"  read failures: {len(read_fail)}  write failures: {len(write_fail)}")
    for f in read_fail[:5]:
        print(f"    read: {f}")
    for f in write_fail[:5]:
        print(f"    write: {f}")
    if read_lat:
        s = sorted(read_lat)
        print(f"  read latency: p50={s[len(s)//2]*1000:.0f}ms "
              f"p99={s[int(len(s)*0.99)]*1000:.0f}ms max={s[-1]*1000:.0f}ms  n={len(s)}")

    for t in ("hot", "cold"):
        print(f"  stats {t}: {req('GET', f'/tables/{t}/stats')[1]}")

    print("\n=== phase 2: kill -9, restart, verify every acked record ===")
    p.kill()
    p.wait()
    t0 = time.time()
    p = start("run2")
    print(f"  recovery took {time.time()-t0:.2f}s")

    missing = []
    checked = 0
    # Check all of `cold` plus a sample of `hot` — GET-per-record is slow, and cold is the
    # table eviction touches.
    sample = [i for i in expect if i.startswith("cold-")]
    hot = [i for i in expect if i.startswith("hot-")]
    sample += random.sample(hot, min(1500, len(hot)))
    for i in sample:
        code, _ = req("GET", f"/tables/{i.split('-')[0]}/records/{i}")
        checked += 1
        if code != 200:
            missing.append((i, code))
    print(f"  checked {checked}, missing {len(missing)}")
    for m in missing[:20]:
        print(f"    MISSING {m}")

    for t in ("hot", "cold"):
        print(f"  stats {t}: {req('GET', f'/tables/{t}/stats')[1]}")
    p.kill()
    p.wait()

    print("\n=== log lines from the background sweeps (run1) ===")
    out = open(f"{WAL}.run1.log", errors="replace").read()
    for line in out.splitlines():
        if any(k in line for k in ("Evict", "snapshot", "Snapshot", "Rebuil", "re-eval",
                                   "WARN", "ERROR", "panic")):
            print("  " + line[:200])

    print("\nRESULT:", "FAIL" if (missing or read_fail or write_fail) else "PASS")
    return 1 if (missing or read_fail or write_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
