"""Code semantic search — DXMT implementation.

Index a corpus of code snippets (with language, type, and module metadata)
then search semantically. Payload filters let you narrow results to a specific
language, snippet type (function/class/test), or module.

Pattern from the "Code-Documentation-QA-Bot" recipe in LanceDB's vectordb-recipes
(https://github.com/lancedb/vectordb-recipes), reimplemented against DXMT.

This example is self-contained: no external APIs required.
sentence-transformers is used if available; otherwise synthetic vectors.

Usage:
    python code_search.py
    python code_search.py --query "sort a list in Python"
    python code_search.py --query "database connection pool" --lang python
    python code_search.py --query "HTTP client retry" --type function
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Any

from dxmt import Client
from environment import client_kwargs
from common_utils import embed, EMBED_DIM


# ── Code corpus ───────────────────────────────────────────────────────────────
#
# Each entry: (id, language, snippet_type, module, description, code)
# We embed "description + code" for richer semantic matching.

CORPUS: list[dict[str, str]] = [
    {
        "id": "py_sort_list",
        "language": "python",
        "type": "function",
        "module": "utils",
        "description": "Sort a list of integers in ascending or descending order.",
        "code": "def sort_list(items: list[int], reverse: bool = False) -> list[int]:\n    return sorted(items, reverse=reverse)",
    },
    {
        "id": "py_binary_search",
        "language": "python",
        "type": "function",
        "module": "utils",
        "description": "Binary search on a sorted list. Returns index or -1.",
        "code": "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    },
    {
        "id": "py_db_pool",
        "language": "python",
        "type": "class",
        "module": "database",
        "description": "Thread-safe connection pool for SQLite or PostgreSQL databases.",
        "code": "class ConnectionPool:\n    def __init__(self, dsn: str, max_size: int = 10):\n        self._dsn = dsn\n        self._pool: list = []\n        self._lock = threading.Lock()\n        self._max = max_size\n\n    def acquire(self):\n        with self._lock:\n            if self._pool:\n                return self._pool.pop()\n        return create_connection(self._dsn)\n\n    def release(self, conn):\n        with self._lock:\n            if len(self._pool) < self._max:\n                self._pool.append(conn)",
    },
    {
        "id": "py_http_retry",
        "language": "python",
        "type": "function",
        "module": "http",
        "description": "HTTP GET with exponential backoff retry on transient failures.",
        "code": "def get_with_retry(url: str, retries: int = 3, backoff: float = 1.0):\n    for attempt in range(retries):\n        try:\n            resp = requests.get(url, timeout=10)\n            resp.raise_for_status()\n            return resp\n        except requests.RequestException as exc:\n            if attempt == retries - 1:\n                raise\n            time.sleep(backoff * 2 ** attempt)",
    },
    {
        "id": "py_lru_cache",
        "language": "python",
        "type": "class",
        "module": "cache",
        "description": "LRU cache with configurable capacity using an OrderedDict.",
        "code": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self._cap = capacity\n        self._cache: OrderedDict = OrderedDict()\n\n    def get(self, key):\n        if key not in self._cache:\n            return None\n        self._cache.move_to_end(key)\n        return self._cache[key]\n\n    def put(self, key, value):\n        if key in self._cache:\n            self._cache.move_to_end(key)\n        self._cache[key] = value\n        if len(self._cache) > self._cap:\n            self._cache.popitem(last=False)",
    },
    {
        "id": "py_test_sort",
        "language": "python",
        "type": "test",
        "module": "utils",
        "description": "Unit tests for sort_list function — ascending, descending, empty.",
        "code": "def test_sort_ascending():\n    assert sort_list([3, 1, 2]) == [1, 2, 3]\n\ndef test_sort_descending():\n    assert sort_list([3, 1, 2], reverse=True) == [3, 2, 1]\n\ndef test_sort_empty():\n    assert sort_list([]) == []",
    },
    {
        "id": "rs_hashmap_insert",
        "language": "rust",
        "type": "function",
        "module": "collections",
        "description": "Insert a key-value pair into a HashMap and return the old value.",
        "code": "use std::collections::HashMap;\n\nfn insert_and_get_old(map: &mut HashMap<String, i32>, key: String, value: i32) -> Option<i32> {\n    map.insert(key, value)\n}",
    },
    {
        "id": "rs_async_channel",
        "language": "rust",
        "type": "function",
        "module": "concurrency",
        "description": "Spawn a background task communicating over an async mpsc channel.",
        "code": "use tokio::sync::mpsc;\n\nasync fn spawn_worker() -> mpsc::Sender<String> {\n    let (tx, mut rx) = mpsc::channel::<String>(32);\n    tokio::spawn(async move {\n        while let Some(msg) = rx.recv().await {\n            println!(\"worker received: {msg}\");\n        }\n    });\n    tx\n}",
    },
    {
        "id": "rs_rw_lock",
        "language": "rust",
        "type": "class",
        "module": "concurrency",
        "description": "Thread-safe shared state using Arc<RwLock<T>> for concurrent reads.",
        "code": "use std::sync::{Arc, RwLock};\n\n#[derive(Clone)]\nstruct SharedState<T> {\n    inner: Arc<RwLock<T>>,\n}\n\nimpl<T> SharedState<T> {\n    fn new(value: T) -> Self {\n        Self { inner: Arc::new(RwLock::new(value)) }\n    }\n    fn read(&self) -> std::sync::RwLockReadGuard<T> {\n        self.inner.read().unwrap()\n    }\n    fn write(&self) -> std::sync::RwLockWriteGuard<T> {\n        self.inner.write().unwrap()\n    }\n}",
    },
    {
        "id": "rs_retry_macro",
        "language": "rust",
        "type": "function",
        "module": "http",
        "description": "Retry an async HTTP request with exponential backoff using reqwest.",
        "code": "async fn fetch_with_retry(url: &str, max_retries: u32) -> Result<String, reqwest::Error> {\n    let client = reqwest::Client::new();\n    let mut delay = std::time::Duration::from_millis(100);\n    for attempt in 0..max_retries {\n        match client.get(url).send().await {\n            Ok(resp) if resp.status().is_success() => return Ok(resp.text().await?),\n            Ok(_) | Err(_) if attempt < max_retries - 1 => {\n                tokio::time::sleep(delay).await;\n                delay *= 2;\n            }\n            Err(e) => return Err(e),\n            Ok(resp) => return Ok(resp.text().await?),\n        }\n    }\n    unreachable!()\n}",
    },
    {
        "id": "ts_debounce",
        "language": "typescript",
        "type": "function",
        "module": "utils",
        "description": "Debounce a function call by delaying it until after a quiet period.",
        "code": "function debounce<T extends (...args: any[]) => void>(fn: T, delay: number): T {\n    let timer: ReturnType<typeof setTimeout> | null = null;\n    return ((...args: Parameters<T>) => {\n        if (timer) clearTimeout(timer);\n        timer = setTimeout(() => fn(...args), delay);\n    }) as T;\n}",
    },
    {
        "id": "ts_fetch_json",
        "language": "typescript",
        "type": "function",
        "module": "http",
        "description": "Typed fetch wrapper that parses JSON and throws on HTTP errors.",
        "code": "async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {\n    const res = await fetch(url, options);\n    if (!res.ok) {\n        throw new Error(`HTTP ${res.status}: ${await res.text()}`);\n    }\n    return res.json() as Promise<T>;\n}",
    },
    {
        "id": "py_vector_cosine",
        "language": "python",
        "type": "function",
        "module": "math",
        "description": "Compute cosine similarity between two vectors using dot product.",
        "code": "import math\n\ndef cosine_similarity(a: list[float], b: list[float]) -> float:\n    dot = sum(x * y for x, y in zip(a, b))\n    norm_a = math.sqrt(sum(x * x for x in a))\n    norm_b = math.sqrt(sum(x * x for x in b))\n    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0",
    },
    {
        "id": "py_knn",
        "language": "python",
        "type": "function",
        "module": "search",
        "description": "Brute-force k-nearest neighbours search over a list of vectors.",
        "code": "def knn_search(query: list[float], corpus: list[list[float]], k: int) -> list[int]:\n    scores = [(cosine_similarity(query, vec), i) for i, vec in enumerate(corpus)]\n    scores.sort(reverse=True)\n    return [i for _, i in scores[:k]]",
    },
    {
        "id": "rs_test_channel",
        "language": "rust",
        "type": "test",
        "module": "concurrency",
        "description": "Async unit test verifying mpsc channel message delivery.",
        "code": "#[tokio::test]\nasync fn test_channel_delivery() {\n    let (tx, mut rx) = tokio::sync::mpsc::channel::<i32>(4);\n    tx.send(42).await.unwrap();\n    assert_eq!(rx.recv().await, Some(42));\n}",
    },
]

TABLE = "code_snippets"



def embed_snippet(entry: dict[str, str]) -> list[float]:
    """Embed description + code together for richer semantic coverage."""
    combined = f"{entry['description']}\n\n{entry['code']}"
    return embed([combined])[0]


# ── Setup & index ─────────────────────────────────────────────────────────────

def setup_and_index(client: Client) -> None:
    try:
        client.delete_table(TABLE)
    except Exception:
        pass
    client.create_table(name=TABLE, dimension=EMBED_DIM, distance_metric="cosine")

    records = []
    for entry in CORPUS:
        vec = embed_snippet(entry)
        records.append({
            "id": entry["id"],
            "vector": vec,
            "payload": {
                "language": entry["language"],
                "type": entry["type"],
                "module": entry["module"],
                "description": entry["description"],
                "code": entry["code"],
            },
        })
    client.insert_records(table_name=TABLE, records=records)
    langs = sorted({e["language"] for e in CORPUS})
    types = sorted({e["type"] for e in CORPUS})
    print(f"[index] {len(CORPUS)} snippets | languages: {langs} | types: {types}")


# ── Search ────────────────────────────────────────────────────────────────────

def search_code(
    client: Client,
    query: str,
    k: int = 5,
    lang_filter: str | None = None,
    type_filter: str | None = None,
) -> list[dict[str, Any]]:
    q_vec = embed([query])[0]

    # Build DXMT filter
    must_conditions = []
    if lang_filter:
        must_conditions.append({
            "key": "language",
            "operation": {"type": "Equal", "value": lang_filter},
        })
    if type_filter:
        must_conditions.append({
            "key": "type",
            "operation": {"type": "Equal", "value": type_filter},
        })

    filter_dict: dict | None = None
    if must_conditions:
        filter_dict = {"must": must_conditions}

    hits = client.search(table_name=TABLE, vector=q_vec, k=k, filter=filter_dict)
    return hits


# ── Display ───────────────────────────────────────────────────────────────────

def show_results(hits: list[dict[str, Any]], query: str) -> None:
    if not hits:
        print("  (no results)")
        return
    for i, hit in enumerate(hits, 1):
        rec_id = hit.get("id", "?")
        score = hit.get("score", 0.0)
        # Find original entry for display
        entry = next((e for e in CORPUS if e["id"] == rec_id), None)
        if entry is None:
            print(f"  {i}. [{score:.4f}] {rec_id}")
            continue
        desc_short = textwrap.shorten(entry["description"], width=65, placeholder="…")
        code_first = entry["code"].split("\n")[0]
        print(
            f"  {i}. [{score:.4f}] [{entry['language']:10}] [{entry['type']:8}] "
            f"{entry['module']:12} — {desc_short}"
        )
        print(f"       {code_first}")


# ── Demo queries ──────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    {"query": "sort a list", "lang": None, "type": None},
    {"query": "HTTP request with retry logic", "lang": None, "type": None},
    {"query": "thread-safe shared state concurrent reads", "lang": None, "type": None},
    {"query": "compute cosine similarity between vectors", "lang": None, "type": None},
    {"query": "async channel background task", "lang": "rust", "type": None},
    {"query": "HTTP retry", "lang": None, "type": "function"},
    {"query": "unit tests", "lang": None, "type": "test"},
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Code semantic search with DXMT")
    parser.add_argument("--query", type=str, default=None,
                        help="Custom search query (default: run all demo queries)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Filter by language (python | rust | typescript)")
    parser.add_argument("--type", dest="snippet_type", type=str, default=None,
                        help="Filter by type (function | class | test)")
    parser.add_argument("--k", type=int, default=4, help="Top-k results (default 4)")
    args = parser.parse_args()

    from common_utils import _HAS_ST
    embed_mode = "sentence-transformers" if _HAS_ST else "bag-of-words"
    print("=" * 70)
    print("  DXMT Code Semantic Search")
    print(f"  Embedding: {embed_mode}  (dim={EMBED_DIM})")
    print(f"  Corpus: {len(CORPUS)} snippets across Python / Rust / TypeScript")
    print("=" * 70)

    client = Client(**client_kwargs())
    try:
        setup_and_index(client)

        if args.query:
            # Single custom query
            queries = [{"query": args.query, "lang": args.lang, "type": args.snippet_type}]
        else:
            queries = DEMO_QUERIES

        for item in queries:
            q = item["query"]
            lf = item.get("lang")
            tf = item.get("type")

            filter_desc = ""
            if lf:
                filter_desc += f"  lang={lf}"
            if tf:
                filter_desc += f"  type={tf}"

            print(f"\n{'─' * 70}")
            print(f"Query: \"{q}\"{filter_desc}")

            hits = search_code(client, q, k=args.k, lang_filter=lf, type_filter=tf)
            show_results(hits, q)

    finally:
        try:
            client.delete_table(TABLE)
        except Exception:
            pass
        client.close()
        print("\n[done]")


if __name__ == "__main__":
    main()
