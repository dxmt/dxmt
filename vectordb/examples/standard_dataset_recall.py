#!/usr/bin/env python3
"""Load a standard benchmark dataset and measure recall against its ground truth.

The functional tests in `tests/api/property/` already assert recall against a brute force they
compute themselves, on generated vectors, and that runs in CI on every change. This is the other
half of the argument: **real embeddings, a published ground truth, and a number you can compare
with somebody else's**.

Defaults to Cohere 100k x 768 -- large enough to be a real distribution, small enough to load in
a couple of minutes. The dataset is the one VectorDBBench uses, so the recall printed here is
directly comparable with the numbers in `docs/architecture/disk-index.md`.

    python3 examples/standard_dataset_recall.py
    DXMT_DATASET_DIR=/data/vectordb_bench/dataset python3 examples/standard_dataset_recall.py
    DXMT_DATASET=openai/openai_small_50k python3 examples/standard_dataset_recall.py
    DXMT_ROWS=20000 python3 examples/standard_dataset_recall.py     # a quicker look

Needs `pyarrow` and a running server. It does not need a key, a model or the network -- the
embeddings are in the file.
"""

import os
import sys
import time
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("this example needs pyarrow:  pip install pyarrow")

try:
    import httpx
except ImportError:
    sys.exit("this example needs httpx:  pip install httpx")

BASE = os.environ.get("DXMT_URL", "http://localhost:9999").rstrip("/") + "/api/v1"
DATASET_DIR = Path(os.environ.get("DXMT_DATASET_DIR", "/tmp/vectordb_bench/dataset"))
DATASET = os.environ.get("DXMT_DATASET", "cohere/cohere_small_100k")
ROWS = int(os.environ.get("DXMT_ROWS", "100000"))
QUERIES = int(os.environ.get("DXMT_QUERIES", "200"))
TOP_K = int(os.environ.get("DXMT_TOP_K", "10"))
TABLE = os.environ.get("DXMT_TABLE", "standard_recall_demo")
BATCH = 1000

HTTP = httpx.Client(timeout=120.0)


def die(message):
    sys.exit(f"\n{message}\n")


def check_dataset():
    """Every file this needs, or a message naming what is missing and where to get it.

    The datasets are tens of gigabytes and are not in the repository. Failing here with an
    explanation beats failing later with a parquet error.
    """
    root = DATASET_DIR / DATASET
    needed = {
        "train": root / "shuffle_train.parquet",
        "test": root / "test.parquet",
        "truth": root / "neighbors.parquet",
    }
    missing = [str(p) for p in needed.values() if not p.exists()]
    if missing:
        die(
            "dataset files not found:\n  "
            + "\n  ".join(missing)
            + "\n\nThese come with VectorDBBench, which downloads them on first use:\n"
            "    pip install vectordb-bench\n"
            "    python -m vectordb_bench   # then run any Cohere case once\n\n"
            f"Point DXMT_DATASET_DIR at wherever they landed (currently {DATASET_DIR}),\n"
            "or DXMT_DATASET at another one, e.g. openai/openai_small_50k."
        )
    return needed


def api(method, path, body=None, allow=()):
    """A request that must succeed, unless its status is in `allow`.

    `allow` exists for exactly one case -- deleting a table that is not there on a first run --
    and is spelled out rather than left to a comment, because a comment saying "ignore the 404"
    above a call that does not ignore it is how the first version of this failed.
    """
    response = HTTP.request(method, BASE + path, json=body)
    if response.status_code >= 300 and response.status_code not in allow:
        die(f"{method} {path} -> {response.status_code}\n{response.text[:500]}")
    return response.json()


def main():
    files = check_dataset()

    train = pq.read_table(files["train"])
    dim = len(train.column("emb")[0])
    rows = min(ROWS, train.num_rows)
    print(f"dataset  {DATASET}: {train.num_rows} rows x {dim} dims, loading {rows}")

    try:
        HTTP.get(BASE + "/health").raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any failure here means the same thing
        die(f"no server at {BASE} ({exc})\n\n    cd vectordb && cargo run --release -- serve")

    api("DELETE", f"/tables/{TABLE}", allow=(404,))  # absent on a first run
    api("POST", "/tables", {"name": TABLE, "dimension": dim, "distance_metric": "cosine"})

    ids = train.column("id").to_pylist()[:rows]
    embs = train.column("emb").to_pylist()[:rows]

    started = time.time()
    for start in range(0, rows, BATCH):
        chunk = [
            {"id": str(ids[i]), "vector": embs[i]}
            for i in range(start, min(start + BATCH, rows))
        ]
        api("POST", f"/tables/{TABLE}/records", {"records": chunk})
        done = min(start + BATCH, rows)
        print(f"\r  inserted {done}/{rows}", end="", flush=True)
    load_seconds = time.time() - started
    print(f"\r  inserted {rows} in {load_seconds:.1f}s ({rows / load_seconds:.0f}/s)")

    test = pq.read_table(files["test"])
    truth = pq.read_table(files["truth"])
    queries = test.column("emb").to_pylist()[:QUERIES]
    # `neighbors_id` is the exact top-k for each test query, over the *whole* training set. When
    # only part of it is loaded, ground-truth ids that were not inserted are dropped rather than
    # counted as misses -- otherwise a smaller DXMT_ROWS would look like worse recall.
    expected = truth.column("neighbors_id").to_pylist()[:QUERIES]
    loaded = set(str(i) for i in ids)

    found = total = 0
    latencies = []
    for query, want in zip(queries, expected):
        wanted = [str(w) for w in want[:TOP_K] if str(w) in loaded]
        if not wanted:
            continue
        t0 = time.time()
        hits = api("POST", f"/tables/{TABLE}/search", {"vector": query, "top_k": TOP_K})
        latencies.append((time.time() - t0) * 1000)
        got = {h["id"] for h in hits["data"]["hits"]}
        found += len(set(wanted) & got)
        total += len(wanted)

    if not total:
        die("no ground-truth neighbours were among the loaded rows; raise DXMT_ROWS")

    latencies.sort()
    print(
        f"\nrecall@{TOP_K}  {found / total:.4f}   "
        f"over {len(latencies)} queries against the shipped ground truth\n"
        f"latency     p50 {latencies[len(latencies) // 2]:.1f} ms   "
        f"p99 {latencies[int(len(latencies) * 0.99)]:.1f} ms   (serial, one connection)\n"
    )
    print(f"the table is still there as '{TABLE}'; DELETE /api/v1/tables/{TABLE} to remove it")


if __name__ == "__main__":
    main()
