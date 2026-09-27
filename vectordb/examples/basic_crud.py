"""The five basic operations through the Python SDK: insert, read, update, delete, search.

    cd vectordb && cargo run -- serve                  # standard mode, port 9999
    cd vectordb && python3 examples/basic_crud.py

Needs `dxmt-python-sdk` importable (`pip install -e clients/dxmt-python-sdk`). `DXMT_HOST` / `DXMT_PORT`
point it elsewhere. The table is created at the start and dropped at the end, so this can be
run repeatedly.

The Rust equivalent is `clients/dxmt-rust-sdk/examples/basic_crud.rs`; the two do the same things in
the same order, on purpose, so the clients can be compared side by side.
"""

import os
import sys

from dxmt import Client

TABLE = "python_sdk_demo"
DIM = 8


def vector(i: int) -> list[float]:
    """Leans on one axis, so the nearest neighbour of `vector(i)` is the record built from it."""
    return [1.0 if d == i % DIM else 0.05 for d in range(DIM)]


def main() -> int:
    host = os.environ.get("DXMT_HOST", "localhost")
    port = int(os.environ.get("DXMT_PORT", "9999"))
    db = Client(host=host, port=port)

    try:
        db.health_check()
    except Exception as exc:  # noqa: BLE001 -- unreachable is unreachable, whatever the reason
        print(f"no server at {host}:{port} ({exc})", file=sys.stderr)
        print("start one with: cd vectordb && cargo run -- serve", file=sys.stderr)
        return 1
    print(f"connected to {host}:{port}")

    try:
        db.delete_table(TABLE)
    except Exception:  # noqa: BLE001 -- absent is the normal case
        pass
    db.create_table(name=TABLE, dimension=DIM, distance_metric="cosine")
    print(f"created table '{TABLE}' ({DIM} dimensions)")

    # ── insert ────────────────────────────────────────────────────────────────────────────
    records = [
        {
            "id": f"doc-{i}",
            "vector": vector(i),
            "payload": {"title": f"document {i}", "topic": "even" if i % 2 == 0 else "odd", "rank": i},
        }
        for i in range(6)
    ]
    inserted = db.insert_records(TABLE, records)
    print(f"inserted {inserted.get('count', len(records))} records")

    # ── read one back ─────────────────────────────────────────────────────────────────────
    doc = db.get_records(TABLE, ["doc-3"])[0]
    print(f"read doc-3: payload {doc.get('payload')} vector[0:3] {doc.get('vector', [])[:3]}")

    # ── search ────────────────────────────────────────────────────────────────────────────
    hits = db.search(TABLE, vector=vector(3), k=3)
    print("nearest to vector(3):")
    for hit in hits:
        print(f"    {hit['id']:<8} {hit['score']:.4f}")
    assert hits[0]["id"] == "doc-3", "a record is its own nearest neighbour"

    # ── search with a filter ──────────────────────────────────────────────────────────────
    # A filter is always must/must_not/should: the server ignores unknown fields, so a bare
    # condition does not fail, it matches everything.
    even = db.search(
        TABLE,
        vector=vector(3),
        k=3,
        filter={"must": [{"key": "topic", "operation": {"type": "Equal", "value": "even"}}]},
    )
    print("same query, topic=even only:")
    for hit in even:
        print(f"    {hit['id']:<8} {hit['score']:.4f}")
    assert all(h["id"] != "doc-3" for h in even), "doc-3 is odd and must be filtered out"

    # ── update ────────────────────────────────────────────────────────────────────────────
    # Payload only: the vector is untouched, so the record keeps its place in the graph.
    db.update_record(TABLE, "doc-3", payload={"title": "renamed"}, merge_mode="merge")
    print("after a merge update, doc-3 payload:", db.get_records(TABLE, ["doc-3"])[0].get("payload"))

    db.update_record(TABLE, "doc-3", vector=vector(5))
    moved = db.search(TABLE, vector=vector(5), k=2)
    print("after moving doc-3 onto vector(5), nearest to vector(5):",
          ", ".join(h["id"] for h in moved))

    # ── delete ────────────────────────────────────────────────────────────────────────────
    removed = db.delete_records(TABLE, ["doc-0", "doc-1", "not-a-real-id"])
    print(f"asked to delete 3 ids, server reports: {removed}")

    remaining = db.get_table(TABLE)
    print("table now holds", remaining.get("record_count", remaining.get("table", {}).get("record_count")), "records")

    db.delete_table(TABLE)
    print(f"dropped '{TABLE}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
