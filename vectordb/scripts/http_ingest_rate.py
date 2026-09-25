#!/usr/bin/env python3
"""Measure ingest rate over HTTP, and check the graph it built.

    python3 scripts/http_ingest_rate.py --port 9999 --dim 768 --rows 200000

The repository's other ingest figures come from the in-process benchmark harness. This one goes
through the server the way a client does -- JSON over HTTP, batches of 1,000 -- which is what an
operator's own numbers will look like, and is roughly half the in-process rate.

Three ways this measurement lies if you write it yourself, all of which cost me a wrong
conclusion today:

* **The generator counts.** Building 768 gaussians per row in Python runs at about 1,700 rows/s.
  Generating inside the timed loop made a server doing 3,400 rows/s look like one doing 600.
  Here the client's work is timed separately and reported, and the rate quoted is requests only.
* **Duplicate vectors make recall meaningless.** A load that writes 1,000 distinct vectors a
  thousand times each will answer a query with a thousand records tied at score 1.0; the
  approximate index and the exact scan then pick different tens of them, and recall@10 reads
  0.04 on a perfectly healthy index. Every row here gets its own vector.
* **Quantization finishes after the load does.** An HNSW-PQ table crosses its row threshold
  during the load and trains in the background for minutes afterwards; `quantized` is false and
  the index is at its full size until it lands. Asking once, right after the last batch, says
  nothing. This waits for it.

Needs `numpy` and `orjson`; both are already in the SDK's test environment.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

try:
    import numpy as np
    import orjson
except ImportError:  # pragma: no cover - a message beats a traceback
    sys.exit("this script needs numpy and orjson: pip install numpy orjson")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--table", default="ingest_rate")
    parser.add_argument(
        "--queries", type=int, default=15, help="recall probes after the load; 0 to skip"
    )
    parser.add_argument(
        "--wait-quantize",
        type=int,
        default=900,
        help="seconds to wait for background quantization; 0 to skip",
    )
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}/api/v1"

    def call(method: str, path: str, body=None):
        data = orjson.dumps(body) if body is not None else None
        request = urllib.request.Request(base + path, data, method=method)
        if data is not None:
            request.add_header("content-type", "application/json")
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read() or b"{}")

    # Best effort: a table left behind by a previous run would fail the create below, and a 404
    # here means there was nothing to remove.
    try:
        call("DELETE", f"/tables/{args.table}")
    except urllib.error.HTTPError:
        pass
    return run(args, call)


def run(args, call) -> int:
    try:
        call("POST", "/tables", {"name": args.table, "dimension": args.dim})
    except urllib.error.HTTPError as e:
        sys.exit(f"could not create the table: {e.read()[:300]!r}")

    rng = np.random.default_rng(7)
    centres = rng.normal(size=(64, args.dim))
    build = send = 0.0
    started = time.time()
    for k in range(args.rows // args.batch):
        t = time.time()
        idx = (np.arange(args.batch) + k * args.batch) % 64
        vectors = centres[idx] + rng.normal(scale=0.15, size=(args.batch, args.dim))
        body = {
            "records": [
                {"id": f"r{k * args.batch + i}", "vector": vectors[i].tolist()}
                for i in range(args.batch)
            ]
        }
        build += time.time() - t
        t = time.time()
        call("POST", f"/tables/{args.table}/records", body)
        send += time.time() - t
    wall = time.time() - started

    print(f"{args.rows} rows at {args.dim} dimensions, batches of {args.batch}")
    print(f"  requests only : {args.rows / send:8.0f} rows/s  ({send:.1f}s)")
    print(f"  end to end    : {args.rows / wall:8.0f} rows/s  ({wall:.1f}s, of which the client")
    print(f"                  spent {build:.1f}s building vectors and JSON)")

    if args.queries:
        recall_check(args, call)
    if args.wait_quantize:
        quantize_check(args, call)
    return 0


def recall_check(args, call) -> None:
    """Query with records' own vectors and compare the approximate answer with the exact one.

    Using a record's own vector is the strongest available check on a graph: the record must come
    back first, at the top score, or the graph cannot find a point it contains.
    """
    rng = np.random.default_rng(99)
    ids = [f"r{int(i)}" for i in rng.integers(0, args.rows, args.queries)]
    overlap = self_hit = 0
    latency = []
    for record_id in ids:
        vector = call("GET", f"/tables/{args.table}/records/{record_id}")["data"]["vector"]
        t = time.time()
        approximate = call(
            "POST", f"/tables/{args.table}/search", {"vector": vector, "top_k": 10}
        )["data"]["hits"]
        latency.append((time.time() - t) * 1000)
        exact = call(
            "POST",
            f"/tables/{args.table}/search",
            {"vector": vector, "top_k": 10, "search_mode": "exact"},
        )["data"]["hits"]
        ids_a = [h["id"] for h in approximate]
        if ids_a and ids_a[0] == record_id:
            self_hit += 1
        overlap += len(set(ids_a) & {h["id"] for h in exact})
    latency.sort()
    print(f"  recall@10 against the exact scan: {overlap / (10 * args.queries):.3f}")
    print(f"  the query's own record came first: {self_hit}/{args.queries}")
    print(f"  search p50: {latency[len(latency) // 2]:.1f} ms")


def quantize_check(args, call) -> None:
    """Wait for the background quantization, and say what it cost and what it saved."""
    started = time.time()
    before = call("GET", f"/tables/{args.table}/stats")["data"]
    if before["quantized"]:
        print(f"  already quantized; index {before['index_size_bytes'] / 1e6:.0f} MB")
        return
    while time.time() - started < args.wait_quantize:
        stats = call("GET", f"/tables/{args.table}/stats")["data"]
        if stats["quantized"]:
            print(
                f"  quantized {time.time() - started:.0f}s after the load: index "
                f"{before['index_size_bytes'] / 1e6:.0f} MB -> "
                f"{stats['index_size_bytes'] / 1e6:.0f} MB"
            )
            return
        time.sleep(10)
    print(
        f"  still not quantized after {args.wait_quantize}s "
        f"(below the row threshold, or still training)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
