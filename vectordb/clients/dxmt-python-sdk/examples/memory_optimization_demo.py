"""memory_optimization: what the per-table setting changes, as the server reports it.

`memory_optimization` decides which index family a new table gets. On, the server picks the
compressed one (`hnsw_pq`, product-quantized codes beside the graph); off, it picks the plain
graph (`hnsw`). Unset, the table follows the server's `DXMT_MEMORY_OPTIMIZATION`, which is on
unless set to `false`.

This demo creates one table each way, loads the same vectors into all three, and prints what
the server says it built. It deliberately prints no recall or memory figures: those depend on
the data and on the table's size (below 65,536 rows an `hnsw_pq` table still searches an exact
graph), and a number quoted here would be a number nobody measured on your data. To measure
recall on your own table, use `POST /tables/{t}/recall/calibrate`; `docs/SUPPORT_MATRIX.md`
has the measured figures for the public datasets.

Run a server first (`cargo run -- serve` in `vectordb/`), then `python memory_optimization_demo.py`.
"""

import random
import time

from dxmt import DXMTClient
from environment import client_kwargs

DIMENSION = 128
ROWS = 1_000

# (table name, memory_optimization as sent, what the comment says it means)
TABLES = [
    ("memopt_default", None, "unset: follows DXMT_MEMORY_OPTIMIZATION (on by default)"),
    ("memopt_off", False, "off: the plain graph"),
    ("memopt_on", True, "on: the compressed index"),
]


def random_vector(rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)]


def main() -> None:
    client = DXMTClient(**client_kwargs())
    rng = random.Random(7)
    records = [
        {"id": f"r{i}", "vector": random_vector(rng), "payload": {"n": i}}
        for i in range(ROWS)
    ]
    query = random_vector(rng)

    print("=" * 64)
    print("memory_optimization: what each setting builds")
    print("=" * 64)

    # A previous run that stopped half way leaves these behind; start clean so a create
    # failure below is a real failure and not "already exists".
    for name, _, _ in TABLES:
        try:
            client.delete_table(name)
        except Exception:
            pass

    try:
        for name, setting, meaning in TABLES:
            kwargs = {} if setting is None else {"memory_optimization": setting}
            client.create_table(name=name, dimension=DIMENSION, distance_metric="cosine", **kwargs)
            client.insert_records(name, records)

            start = time.perf_counter()
            hits = client.search(table_name=name, vector=query, k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000

            info = client.get_table(name)
            table = info.get("table", info)
            print(f"\n{name}  ({meaning})")
            print(f"  memory_optimization sent:   {setting}")
            print(f"  memory_optimization stored: {table.get('memory_optimization')}")
            print(f"  index the server built:     {table.get('index_type')}")
            print(f"  records:                    {info.get('record_count')}")
            print(f"  top-10 search:              {len(hits)} hits in {elapsed_ms:.1f} ms")
            if len(hits) != 10:
                raise RuntimeError(f"{name}: expected 10 hits from {ROWS} rows, got {len(hits)}")

        print("\nWhich to choose: leave it on unless a table is small enough that memory does")
        print("not matter and you want the plain graph; measure recall on your own data with")
        print("POST /tables/{t}/recall/calibrate rather than trusting a rule of thumb.")
    finally:
        for name, _, _ in TABLES:
            try:
                client.delete_table(name)
            except Exception as e:
                print(f"cleanup: could not delete {name}: {e}")


if __name__ == "__main__":
    main()
