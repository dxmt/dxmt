"""Copy Chroma collections into DXMT tables, vectors included, so nothing is re-embedded.

    pip install 'dxmt-python-sdk[chroma]'
    python -m dxmt.migrate.chroma --path ./chroma_db                 # every collection
    python -m dxmt.migrate.chroma --path ./chroma_db --collection notes --table notes

Each collection becomes one table: the same ids, the stored embeddings, the collection's
distance (`l2` -> `euclidean`, `ip` -> `dot`, `cosine`), each metadata dict as the payload and
the document under `text`, with a full-text index on it so text and hybrid search work on the
copy. A record is written by id, so running it again over the same table rewrites the same
records rather than duplicating them -- an interrupted run is finished by running it again.

Chroma's embedding function does not come along: the copy is searched with vectors (or with
text through the full-text index). To search it with text through a model, create the table
with `embedding` naming the same model the collection used and migrate into it with `--table`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from ..client import DXMTClient

_SPACES = {"l2": "euclidean", "cosine": "cosine", "ip": "dot"}


@dataclass
class Migrated:
    collection: str
    table: str
    records: int
    skipped: Optional[str] = None


def table_name_for(collection: str) -> str:
    """A collection name as a table name: DXMT allows letters, digits, `_` and `-`."""
    name = re.sub(r"[^A-Za-z0-9_-]", "_", collection)
    if len(name) > 63:
        raise ValueError(
            f"collection {collection!r} makes a table name over 63 characters; "
            "name the table with --table"
        )
    return name


def distance_of(collection: Any) -> str:
    """The collection's distance as a DXMT metric. Chroma 1.x keeps it in `configuration`,
    older versions in the `hnsw:space` metadata key; the default is `l2`."""
    space = None
    config = getattr(collection, "configuration", None)
    if isinstance(config, dict):
        space = (config.get("hnsw") or {}).get("space") or (config.get("spann") or {}).get(
            "space"
        )
    if space is None:
        space = (collection.metadata or {}).get("hnsw:space", "l2")
    if space not in _SPACES:
        raise ValueError(f"collection {collection.name!r} uses distance {space!r}, which DXMT lacks")
    return _SPACES[space]


def _vector(embedding: Any) -> list[float]:
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


def _existing_dimension(client: DXMTClient, table: str) -> Optional[int]:
    for t in client.list_tables():
        if t.get("name") == table:
            return int(t.get("dimension") or 0)
    return None


def migrate_collection(
    collection: Any,
    client: DXMTClient,
    table: Optional[str] = None,
    *,
    text_field: str = "text",
    analyzer: str = "standard",
    batch_size: int = 500,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> Migrated:
    """Copy one Chroma collection into a DXMT table, creating the table if it does not exist."""
    table = table or table_name_for(collection.name)
    total = collection.count()
    if total == 0:
        return Migrated(collection.name, table, 0, skipped="empty: no vectors to size a table by")

    first = collection.get(limit=1, include=["embeddings"])
    dimension = len(_vector(first["embeddings"][0]))
    existing = _existing_dimension(client, table)
    if existing is None:
        client.create_table(
            name=table,
            dimension=dimension,
            distance_metric=distance_of(collection),
            enable_full_text_search=True,
            text_fields=[
                {
                    "name": "content",
                    "source": {"metadata_field": text_field},
                    "analyzer": analyzer,
                }
            ],
        )
    elif existing and existing != dimension:
        raise ValueError(
            f"table {table!r} holds {existing}-dimension vectors and collection "
            f"{collection.name!r} {dimension}-dimension ones"
        )

    copied = 0
    while copied < total:
        page = collection.get(
            limit=batch_size,
            offset=copied,
            include=["embeddings", "documents", "metadatas"],
        )
        ids = page["ids"]
        if not ids:
            break
        documents = page.get("documents") or [None] * len(ids)
        metadatas = page.get("metadatas") or [None] * len(ids)
        records = []
        for rid, embedding, document, metadata in zip(
            ids, page["embeddings"], documents, metadatas
        ):
            payload = dict(metadata or {})
            if document is not None:
                if text_field in payload and payload[text_field] != document:
                    raise ValueError(
                        f"record {rid!r} in {collection.name!r} has a {text_field!r} metadata "
                        "field as well as a document; choose another field with --text-field"
                    )
                payload[text_field] = document
            records.append({"id": rid, "vector": _vector(embedding), "payload": payload})
        client.insert_records(table, records)
        copied += len(ids)
        if progress:
            progress(collection.name, copied, total)
    return Migrated(collection.name, table, copied)


def migrate(
    chroma: Any,
    client: DXMTClient,
    collections: Optional[Iterable[str]] = None,
    **options: Any,
) -> list[Migrated]:
    """Copy the named collections, or every one, from a Chroma client into DXMT."""
    names = list(collections) if collections else [c if isinstance(c, str) else c.name
                                                   for c in chroma.list_collections()]
    # The embedding function is not needed to read stored vectors, and asking for the
    # collection's own would load its model.
    return [
        migrate_collection(chroma.get_collection(name), client, **options) for name in names
    ]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dxmt.migrate.chroma",
        description="Copy Chroma collections, with their vectors, into DXMT tables.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", help="a Chroma persistent directory")
    source.add_argument("--chroma-host", help="a Chroma server, host[:port]")
    parser.add_argument("--collection", action="append", help="copy only this one (repeatable)")
    parser.add_argument("--table", help="the table to copy into (one --collection only)")
    parser.add_argument("--url", default=os.environ.get("DXMT_URL", "http://localhost:9999"))
    parser.add_argument("--text-field", default="text", help="payload field for the document")
    parser.add_argument("--analyzer", default="standard", help="full-text analyzer, e.g. chinese")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args(argv)

    if args.table and len(args.collection or []) != 1:
        parser.error("--table needs exactly one --collection")
    try:
        import chromadb
    except ImportError:
        print("chromadb is not installed: pip install 'dxmt-python-sdk[chroma]'", file=sys.stderr)
        return 2

    if args.path:
        chroma = chromadb.PersistentClient(path=args.path)
    else:
        host, _, port = args.chroma_host.partition(":")
        chroma = chromadb.HttpClient(host=host, port=int(port or 8000))

    match = re.match(r"^(https?)://([^:/]+)(?::(\d+))?", args.url)
    if not match:
        parser.error(f"--url {args.url!r} is not http(s)://host[:port]")
    scheme, host, port = match.groups()
    client = DXMTClient(
        host=host,
        port=int(port or (443 if scheme == "https" else 9999)),
        ssl=scheme == "https",
        api_key=os.environ.get("DXMT_API_KEY"),
        timeout=120.0,
    )

    def progress(name: str, done: int, total: int) -> None:
        print(f"\r{name}: {done}/{total}", end="", file=sys.stderr, flush=True)

    names = args.collection
    if names is None:
        names = [c if isinstance(c, str) else c.name for c in chroma.list_collections()]
    failed = False
    for name in names:
        try:
            result = migrate_collection(
                chroma.get_collection(name),
                client,
                args.table,
                text_field=args.text_field,
                analyzer=args.analyzer,
                batch_size=args.batch_size,
                progress=progress,
            )
        except Exception as e:  # one collection failing does not stop the others
            print(f"\n{name}: failed: {e}", file=sys.stderr)
            failed = True
            continue
        if result.skipped:
            print(f"{name}: skipped ({result.skipped})", file=sys.stderr)
        else:
            print(f"\n{name} -> {result.table}: {result.records} records", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
