#!/usr/bin/env python3
"""Is the ~100s freeze still there, or did its trigger just stop occurring?

The soak stopped reproducing it once HNSW construction got 6.4x faster — but that only means the
long BruteForce->HNSW rebuild no longer happens in that workload, not that a long rebuild under
load is safe. So force one: automatic re-evaluation off, load a big table, then POST
/index/rebuild while readers and writers keep going.
"""
import json, os, subprocess, sys, threading, time, urllib.error, urllib.request

import random
import tempfile

PORT = int(os.environ.get("DXMT_PORT", "9978"))
BASE = f"http://127.0.0.1:{PORT}/api/v1"
DIM = 128
# A scratch directory of its own, removed by the OS rather than by this script: a WAL path
# that outlives the run is a few gigabytes nobody goes looking for. `DXMT_SCRATCH` overrides
# it when the run should be inspected afterwards.
SD = os.environ.get("DXMT_SCRATCH") or tempfile.mkdtemp(prefix="dxmt-force-rebuild-")
WAL = f"{SD}/force-wal"
# Relative to this file, so the script works from any checkout. These three used to be
# absolute paths from the machine they were written on -- one of them a temporary directory
# belonging to a tool, complete with its session id.
BIN = os.environ.get("DXMT_BIN") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target/release/vectordb"
)

def req(method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, time.time() - t0
    except Exception:
        return None, time.time() - t0

def vec(seed):
    rnd = random.Random(seed)
    return [round(rnd.uniform(-1, 1), 4) for _ in range(DIM)]

subprocess.run(["pkill", "-9", "-f", f"vectordb serve --host 127.0.0.1 --port {PORT}"], check=False)
subprocess.run(["rm", "-rf", WAL], check=False)
os.makedirs(WAL, exist_ok=True)
env = dict(os.environ, DXMT_WAL_DIR=WAL, DXMT_PORT=str(PORT),
           DXMT_SNAPSHOT_INTERVAL_SECS="5", DXMT_EVICTION_INTERVAL_SECS="7",
           DXMT_REINDEX_INTERVAL_SECS="0",  # no automatic rebuild; we trigger it ourselves
           RUST_LOG="info")
log = open(f"{SD}/force.log", "w")
srv = subprocess.Popen([BIN, "serve", "--host", "127.0.0.1", "--port", str(PORT)],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
for _ in range(300):
    if req("GET", "/tables")[0] == 200:
        break
    time.sleep(0.2)

req("POST", "/tables", {"name": "hot", "dimension": DIM, "distance_metric": "cosine"})
n = 0
while n < 60_000:
    ids = [f"r{n+i}" for i in range(500)]
    req("POST", "/tables/hot/records",
        {"records": [{"id": i, "vector": vec(i)} for i in ids]}, timeout=120)
    n += 500
print(f"loaded {n} records")

stop = threading.Event()
lat = {"read": [], "write": []}
fails = {"read": 0, "write": 0}

def reader():
    while not stop.is_set():
        code, dt = req("POST", "/tables/hot/search", {"vector": vec(random.randint(0, 9999)), "top_k": 10})
        lat["read"].append(dt)
        if code != 200:
            fails["read"] += 1
        time.sleep(0.05)

def writer(start):
    i = start
    while not stop.is_set():
        code, dt = req("POST", "/tables/hot/records",
                       {"records": [{"id": f"w{i+j}", "vector": vec(i + j)} for j in range(100)]})
        lat["write"].append(dt)
        if code != 200:
            fails["write"] += 1
        i += 100

threads = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
threads.append(threading.Thread(target=writer, args=(1_000_000,), daemon=True))
for t in threads:
    t.start()
time.sleep(5)

print("triggering rebuild under load...")
t0 = time.time()
code, dt = req("POST", "/tables/hot/index/rebuild", {}, timeout=600)
print(f"rebuild returned {code} after {dt:.1f}s")

time.sleep(5)
stop.set()
for t in threads:
    t.join(timeout=20)

for k in ("read", "write"):
    s = sorted(lat[k])
    if s:
        print(f"{k}: n={len(s)} p50={s[len(s)//2]*1000:.0f}ms p99={s[int(len(s)*0.99)]*1000:.0f}ms "
              f"max={s[-1]*1000:.0f}ms failures={fails[k]}")
srv.kill(); srv.wait()

out = open(f"{SD}/force.log", errors="replace").read()
for line in out.splitlines():
    if any(w in line for w in ("stalled", "held the table", "waited", "swapped in")):
        print("  " + line.split("dxmt_vectordb::")[-1][:170])
