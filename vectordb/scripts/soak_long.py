#!/usr/bin/env python3
"""A long soak -- hours to a day -- against the real `vectordb serve` process.

`scripts/soak.py` is the two-minute version: it starts a server, drives concurrent traffic
while the three background sweeps fire, then `kill -9`s it and checks that every acknowledged
record survived. It found two bugs the 35-test recovery suite could not see, because those bugs
only exist when snapshotting, eviction and re-indexing run as timers against live traffic.

Two minutes cannot see the other class: anything that only shows up after the hundredth
snapshot. Memory that creeps, a WAL that never quite truncates, a latency floor that rises, a
sweep that stops firing. This script runs the same shape of traffic for as long as you ask and
records enough to tell whether any of those happened.

What it does differently from the short soak, and why:

* **The key space is bounded.** The short soak inserts new ids forever, which is the right
  shape for two minutes and impossible for a day -- at a few hundred rows a second it would be
  tens of millions of rows and hundreds of gigabytes of WAL. Here each writer cycles through a
  fixed range of ids, so after the first pass every write is an upsert and the row count
  plateaus. That is also the more interesting path: upsert has to clean up the sparse postings,
  payload index entries and named vectors the previous version left behind, and that cleanup
  was broken once already.
* **The write rate is capped.** A soak is not a benchmark. Running flat out for a day tells you
  what the machine does under saturation, monopolises it, and makes every latency number a
  measure of the queue rather than of the database. `SOAK_WRITE_RPS` throttles it.
* **There is a watchdog.** It samples the server's resident memory, the size of the WAL
  directory and the free space on its filesystem, and aborts the run -- keeping everything for
  inspection -- before any of them reaches a limit. A soak that fills the disk of the machine
  it runs on has destroyed the evidence it was collecting.
* **Everything is written to a CSV as it goes**, so a run that is killed halfway still leaves a
  usable record, and so "did memory creep?" is answered by a column rather than by two numbers
  at the ends.

Usage:

    cargo build --release --bin vectordb
    SOAK_HOURS=24 python3 scripts/soak_long.py

Knobs, with defaults: `SOAK_HOURS=24`, `SOAK_WRITE_RPS=200`, `SOAK_KEYSPACE=200000` (per
writer, two writers), `SOAK_DIM=768`, `SOAK_CPUS=0-7` (the server is pinned there; empty string
pins nothing), `SOAK_RSS_CAP_MB=12000`, `SOAK_WAL_CAP_MB=20000`, `SOAK_FREE_DISK_MIN_MB=40000`,
`SOAK_DIR`, `SOAK_PORT=9972`, `SOAK_BIN`.

Exits non-zero on a lost record, a failed request, a watchdog trip, or a restart that does not
serve within ten minutes.
"""
import csv
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HOURS = float(os.environ.get("SOAK_HOURS", "24"))
DUR = HOURS * 3600
PORT = int(os.environ.get("SOAK_PORT", "9972"))
BASE = f"http://127.0.0.1:{PORT}/api/v1"
DIM = int(os.environ.get("SOAK_DIM", "768"))
KEYSPACE = int(os.environ.get("SOAK_KEYSPACE", "200000"))
WRITE_RPS = float(os.environ.get("SOAK_WRITE_RPS", "200"))
BATCH = int(os.environ.get("SOAK_BATCH", "100"))
CPUS = os.environ.get("SOAK_CPUS", "0-7")
RSS_CAP_MB = int(os.environ.get("SOAK_RSS_CAP_MB", "12000"))
WAL_CAP_MB = int(os.environ.get("SOAK_WAL_CAP_MB", "20000"))
FREE_DISK_MIN_MB = int(os.environ.get("SOAK_FREE_DISK_MIN_MB", "40000"))
SAMPLE_SECS = int(os.environ.get("SOAK_SAMPLE_SECS", "60"))

DIR = os.environ.get("SOAK_DIR") or os.path.expanduser(
    f"~/dxmt-soak-{time.strftime('%Y%m%d-%H%M%S')}")
WAL = os.path.join(DIR, "wal")
BIN = os.environ.get("SOAK_BIN", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target/release/vectordb"))

# Mutually prime, so the three sweeps drift in and out of phase instead of always firing on the
# same tick. Longer than the short soak's 5/7/11: over a day these still fire hundreds of times
# each, and a snapshot every five seconds would say more about the snapshot path than about the
# database living with one.
SNAPSHOT_SECS, EVICTION_SECS, REINDEX_SECS = 113, 181, 307


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def req(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400].decode("utf8", "replace")
    except Exception as e:  # noqa: BLE001 -- a refused connection and a timeout mean the same
        return None, repr(e)


def vec(seed):
    rnd = random.Random(seed)
    return [round(rnd.uniform(-1, 1), 4) for _ in range(DIM)]


def dir_size_mb(path, allocated=False):
    """Megabytes under `path`: apparent length, or blocks actually allocated."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                st = os.stat(os.path.join(root, f))
                total += st.st_blocks * 512 if allocated else st.st_size
            except OSError:
                pass
    return total / 1e6


def rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def free_disk_mb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e6


def start(tag):
    env = dict(os.environ)
    env.update({
        "DB_WAL_DIR": WAL,
        "DB_SNAPSHOT_INTERVAL_SECS": str(SNAPSHOT_SECS),
        "DB_EVICTION_INTERVAL_SECS": str(EVICTION_SECS),
        "DB_REINDEX_INTERVAL_SECS": str(REINDEX_SECS),
        "RUST_LOG": "info",
    })
    cmd = [BIN, "serve", "--host", "127.0.0.1", "--port", str(PORT)]
    if CPUS:
        # The point of a soak is to run for a day on a machine somebody else is also using.
        # Pinned to a subset of the cores, it cannot starve the rest of the box -- and the
        # numbers stay comparable across runs even when the machine is busy.
        cmd = ["taskset", "-c", CPUS] + cmd
    logfile = open(os.path.join(DIR, f"server.{tag}.log"), "a")
    p = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                         start_new_session=True)
    t_up = time.time()
    while time.time() - t_up < 600:
        code, _ = req("GET", "/tables", timeout=10)
        if code == 200:
            log(f"  [{tag}] serving after {time.time() - t_up:.2f}s (pid {p.pid})")
            return p
        if p.poll() is not None:
            log(f"server died during startup, exit={p.returncode}")
            sys.exit(1)
        time.sleep(0.5)
    log("server never came up within 600s")
    p.kill()
    sys.exit(1)


acked = set()
acked_lock = threading.Lock()
counters = {"writes_ok": 0, "writes_fail": 0, "reads_ok": 0, "reads_fail": 0}
# `counters[k] += 1` from five threads is a read-modify-write, and the verdict at the end is
# read off these numbers: a lost increment on a failure counter turns a FAIL into a PASS.
# Cheap enough at a few hundred operations a second to not think about.
counters_lock = threading.Lock()
failures = []            # bounded; the first few of each kind are what gets diagnosed
read_lat = []            # reset every sample window, so it cannot grow without bound
lat_lock = threading.Lock()
stop = threading.Event()
abort_reason = []


def bump(key):
    with counters_lock:
        counters[key] += 1


def note_failure(kind, code, body):
    if len(failures) < 200:
        failures.append((time.strftime("%H:%M:%S"), kind, code, str(body)[:200]))


def writer(table, base):
    """Cycle through a fixed range of ids, so the table plateaus and every pass after the first
    is an upsert."""
    n = 0
    # Spread the two writers' sleeps so they do not pulse in lockstep.
    per_batch = BATCH / (WRITE_RPS / 2.0)
    while not stop.is_set():
        t0 = time.time()
        ids = [f"{table}-{base + (n + i) % KEYSPACE}" for i in range(BATCH)]
        recs = [{"id": i, "vector": vec(i), "payload": {"pass": n // KEYSPACE, "w": base}}
                for i in ids]
        code, body = req("POST", f"/tables/{table}/records", {"records": recs})
        if code == 200:
            with acked_lock:
                acked.update(ids)
            bump("writes_ok")
        else:
            bump("writes_fail")
            note_failure("write", code, body)
        n += BATCH
        slack = per_batch - (time.time() - t0)
        if slack > 0:
            stop.wait(slack)


def reader(table):
    while not stop.is_set():
        t0 = time.time()
        code, body = req("POST", f"/tables/{table}/search",
                         {"vector": vec(random.randint(0, 10_000)), "top_k": 10})
        dt = time.time() - t0
        with lat_lock:
            read_lat.append(dt)
        if code == 200:
            bump("reads_ok")
        else:
            bump("reads_fail")
            note_failure("read", code, body)
        stop.wait(0.1)


def watchdog(proc):
    """Abort before the machine is in trouble, not after.

    Everything here is a limit on *this* run's footprint, not a health assertion about the
    database: tripping one stops the soak and keeps the directory, so whatever was growing can
    be looked at. A soak that fills the disk destroys its own evidence, and takes the rest of
    the machine with it.
    """
    while not stop.is_set():
        stop.wait(15)
        if stop.is_set():
            return
        rss = rss_mb(proc.pid)
        wal = dir_size_mb(WAL, allocated=True)
        free = free_disk_mb(DIR)
        if rss > RSS_CAP_MB:
            abort_reason.append(f"server RSS {rss:.0f} MB over cap {RSS_CAP_MB} MB")
        elif wal > WAL_CAP_MB:
            abort_reason.append(f"WAL dir {wal:.0f} MB over cap {WAL_CAP_MB} MB")
        elif free < FREE_DISK_MIN_MB:
            abort_reason.append(f"free disk {free:.0f} MB under floor {FREE_DISK_MIN_MB} MB")
        elif proc.poll() is not None:
            abort_reason.append(f"server exited on its own, code {proc.returncode}")
        if abort_reason:
            log(f"WATCHDOG: {abort_reason[-1]} -- stopping")
            stop.set()
            return


def metrics_snapshot():
    code, body = req("GET", "/health/metrics", timeout=15)
    if code != 200 or not isinstance(body, dict):
        return {}
    data = body.get("data", body)
    storage = data.get("storage", {}) if isinstance(data, dict) else {}
    return {
        "wal_next_sequence": storage.get("wal_next_sequence"),
        "last_snapshot_at": storage.get("last_snapshot_at"),
        "snapshot_failures": storage.get("snapshot_failures"),
    }


def main():
    os.makedirs(DIR, exist_ok=True)
    os.makedirs(WAL, exist_ok=True)
    log(f"soak dir  : {DIR}")
    log(f"binary    : {BIN}")
    log(f"duration  : {HOURS} h   write cap: {WRITE_RPS} rows/s   keyspace: {KEYSPACE} x2")
    log(f"dim       : {DIM}   cpus: {CPUS or 'all'}")
    log(f"caps      : RSS {RSS_CAP_MB} MB, WAL {WAL_CAP_MB} MB, free disk floor "
        f"{FREE_DISK_MIN_MB} MB")
    if not os.path.exists(BIN):
        log(f"no binary at {BIN} -- run: cargo build --release --bin vectordb")
        return 1

    csv_path = os.path.join(DIR, "samples.csv")
    csv_file = open(csv_path, "w", newline="")
    writer_csv = csv.writer(csv_file)
    writer_csv.writerow(["elapsed_s", "acked", "rows_hot", "writes_ok", "writes_fail",
                         "reads_ok", "reads_fail", "read_p50_ms", "read_p99_ms",
                         "read_max_ms", "rss_mb", "wal_alloc_mb", "wal_apparent_mb", "free_disk_mb",
                         "wal_next_sequence", "snapshot_failures", "index_type"])
    csv_file.flush()

    log("=== phase 1: start, seed, run ===")
    proc = start("run1")
    for t in ("hot", "cold"):
        code, _ = req("POST", "/tables",
                      {"name": t, "dimension": DIM, "distance_metric": "cosine"})
        log(f"  create {t}: {code}")

    # `cold` is written once and then abandoned, so the eviction sweep has something to evict
    # while snapshots and writes are in flight on `hot`. That combination lost a whole table
    # once.
    cold_ids = [f"cold-{i}" for i in range(2000)]
    code, _ = req("POST", "/tables/cold/records",
                  {"records": [{"id": i, "vector": vec(i)} for i in cold_ids]}, timeout=300)
    log(f"  seed cold with 2000: {code}")
    if code == 200:
        with acked_lock:
            acked.update(cold_ids)

    threads = [threading.Thread(target=writer, args=("hot", 0), daemon=True),
               threading.Thread(target=writer, args=("hot", 1_000_000), daemon=True)]
    threads += [threading.Thread(target=reader, args=("hot",), daemon=True) for _ in range(3)]
    threads.append(threading.Thread(target=watchdog, args=(proc,), daemon=True))
    for t in threads:
        t.start()

    t0 = time.time()
    next_sample = t0 + SAMPLE_SECS
    while time.time() - t0 < DUR and not stop.is_set():
        stop.wait(min(5.0, max(0.0, next_sample - time.time())))
        if time.time() < next_sample or stop.is_set():
            continue
        next_sample += SAMPLE_SECS
        with lat_lock:
            window = sorted(read_lat)
            read_lat.clear()
        with acked_lock:
            n_acked = len(acked)
        st = req("GET", "/tables/hot/stats", timeout=15)[1]
        stats = st.get("data", {}) if isinstance(st, dict) else {}
        m = metrics_snapshot()
        row = [
            int(time.time() - t0), n_acked, stats.get("record_count"),
            counters["writes_ok"], counters["writes_fail"],
            counters["reads_ok"], counters["reads_fail"],
            round(window[len(window) // 2] * 1000, 1) if window else "",
            round(window[int(len(window) * 0.99)] * 1000, 1) if window else "",
            round(window[-1] * 1000, 1) if window else "",
            round(rss_mb(proc.pid)),
            round(dir_size_mb(WAL, allocated=True)),
            round(dir_size_mb(WAL)),
            round(free_disk_mb(DIR)),
            m.get("wal_next_sequence"), m.get("snapshot_failures"),
            stats.get("index_type"),
        ]
        writer_csv.writerow(row)
        csv_file.flush()
        if int(time.time() - t0) % (SAMPLE_SECS * 10) < SAMPLE_SECS:
            log(f"  t={row[0]}s acked={row[1]} rows={row[2]} wfail={row[4]} rfail={row[6]} "
                f"p99={row[8]}ms rss={row[10]}MB wal={row[11]}MB(alloc)/{row[12]}MB(len) "
                f"idx={row[16]}")

    stop.set()
    for t in threads:
        t.join(timeout=60)
    csv_file.close()

    with acked_lock:
        expect = sorted(acked)
    log(f"\n  ran {time.time() - t0:.0f}s, acknowledged {len(expect)} distinct ids")
    log(f"  writes ok/fail: {counters['writes_ok']}/{counters['writes_fail']}   "
        f"reads ok/fail: {counters['reads_ok']}/{counters['reads_fail']}")
    for f in failures[:10]:
        log(f"    {f}")

    log("\n=== phase 2: kill -9, restart, verify ===")
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    t_restart = time.time()
    proc = start("run2")
    recovery = time.time() - t_restart
    log(f"  recovery took {recovery:.2f}s")

    # Every id of `cold` -- the table eviction touches -- plus a sample of `hot`. A
    # GET-per-record over a bounded key space is the only honest check: `record_count` alone
    # cannot tell a lost row from a row that was never written.
    hot = [i for i in expect if i.startswith("hot-")]
    sample = [i for i in expect if i.startswith("cold-")]
    sample += random.sample(hot, min(5000, len(hot)))
    missing = []
    for i in sample:
        code, _ = req("GET", f"/tables/{i.split('-')[0]}/records/{i}", timeout=30)
        if code != 200:
            missing.append((i, code))
    log(f"  checked {len(sample)}, missing {len(missing)}")
    for m in missing[:20]:
        log(f"    MISSING {m}")
    for t in ("hot", "cold"):
        st = req("GET", f"/tables/{t}/stats", timeout=30)[1]
        log(f"  stats {t}: {json.dumps(st)[:300]}")

    proc.send_signal(signal.SIGKILL)
    proc.wait()

    log("\n=== warnings, errors and sweep activity in the run log ===")
    run_log = os.path.join(DIR, "server.run1.log")
    counts = {}
    with open(run_log, errors="replace") as fh:
        for line in fh:
            for key in ("Evict", "napshot", "Rebuil", "re-eval", "WARN", "ERROR", "panic"):
                if key in line:
                    counts[key] = counts.get(key, 0) + 1
                    if key in ("WARN", "ERROR", "panic") and counts[key] <= 10:
                        log("  " + line.rstrip()[:200])
    log(f"  line counts: {counts}")
    log(f"\n  samples: {csv_path}")

    failed = bool(missing or counters["writes_fail"] or counters["reads_fail"] or abort_reason)
    if abort_reason:
        log(f"  ABORTED: {abort_reason[0]}")
    log(f"\nRESULT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
