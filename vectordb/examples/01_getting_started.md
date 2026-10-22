# Getting Started with DXMT VectorDB

This walks through building the server, running it, and doing the four basic things — create a
table, insert records, search, delete — over HTTP, from Python, and from Rust. The README's
Quick Start covers the same ground more briefly; this adds the client side.

## 1. Build and run the server

Requires [rustup](https://rustup.rs). The Rust version is pinned in `rust-toolchain.toml`, and
rustup installs it the first time you run `cargo` in `vectordb/`.

```bash
git clone https://github.com/dxmt/dxmt.git
cd dxmt/vectordb
cargo build --release

# Listens on 127.0.0.1 only. `--host 0.0.0.0` serves the network; set DXMT_API_KEY first.
./target/release/vectordb serve --port 9999
```

Data goes to `$HOME/.dxmt/store` unless `DXMT_ROOT_DIR` says otherwise. The startup log prints
the data directory, the API address and the embedding model text goes through, if any.

```bash
curl http://localhost:9999/health
curl -f http://localhost:9999/ready    # 503 after a local durability failure
```

## 2. Over HTTP

Every request body is JSON and needs `Content-Type: application/json`.

```bash
# A table: name, width, metric
curl -X POST localhost:9999/api/v1/tables \
  -H 'Content-Type: application/json' \
  -d '{"name": "products", "dimension": 4, "distance_metric": "cosine"}'

# Records: id, vector, optional payload
curl -X POST localhost:9999/api/v1/tables/products/records \
  -H 'Content-Type: application/json' \
  -d '{"records": [
        {"id": "p1", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"name": "Laptop", "category": "electronics"}},
        {"id": "p2", "vector": [0.4, 0.3, 0.2, 0.1], "payload": {"name": "Desk", "category": "furniture"}},
        {"id": "p3", "vector": [0.1, 0.2, 0.3, 0.5], "payload": {"name": "Monitor", "category": "electronics"}}]}'

# Nearest neighbours, optionally filtered on the payload
curl -X POST localhost:9999/api/v1/tables/products/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 2,
       "filter": {"must": [{"key": "category", "operation": {"type": "Equal", "value": "electronics"}}]}}'

# Delete the table
curl -X DELETE localhost:9999/api/v1/tables/products
```

The full API, with a runnable example for every operation, is served at
`http://localhost:9999/api/docs`. `examples/filter_example.md` covers the filter language.

## 3. From Python

The Python SDK is not on PyPI yet; install it from the checkout:

```bash
pip install ./clients/dxmt-python-sdk
```

```python
from dxmt import Client

client = Client(host="localhost", port=9999)   # api_key="..." when DXMT_API_KEY is set

client.create_table(name="products", dimension=4, distance_metric="cosine")
client.insert_records("products", [
    {"id": "p1", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"name": "Laptop"}},
    {"id": "p2", "vector": [0.4, 0.3, 0.2, 0.1], "payload": {"name": "Desk"}},
])

for hit in client.search("products", vector=[0.1, 0.2, 0.3, 0.4], k=2):
    print(hit["id"], hit["score"])

client.delete_records("products", ids=["p2"])
client.delete_table("products")
```

`examples/basic_crud.py` is the same as a runnable script. The JavaScript, Go and Rust clients
are under `clients/`; each README shows how to install it from source.

## 4. In-process from Rust

To embed the engine in a Rust program rather than talk to a server, depend on the crate from
git (it is not on crates.io yet):

```toml
[dependencies]
dxmt-vectordb = { git = "https://github.com/dxmt/dxmt", package = "dxmt-vectordb" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde_json = "1"
```

`examples/quickstart.rs` is a complete program — create an in-memory
`IntelligentIndexedStorage`, create a table, insert, search, list, delete — and is compiled by
every `cargo build --examples`, so it stays in step with the API:

```bash
cargo run --example quickstart
```

An in-process store built with `IntelligentIndexedStorage::new` keeps everything in memory;
the server is what adds the WAL, snapshots and row files under `DXMT_ROOT_DIR`.

## 5. Next

- `examples/02_embeddings.md` — let the server embed text, so records and queries carry text
  instead of vectors.
- `README.md` — configuration, durability, authentication and measured performance.
- `docs/SUPPORT_MATRIX.md` — what is supported, what is experimental, and what each client
  covers.
- `docs/operations/backup-restore.md` — backups and restore.

Questions and bug reports: [GitHub issues](https://github.com/dxmt/dxmt/issues).
