# dxmt-client

Rust client for DXMT VectorDB: insert, read, update, delete and search over the server's HTTP
API.

If you want the engine in-process rather than a server to talk to, depend on `dxmt-vectordb`
itself and use `IntelligentIndexedStorage` — `vectordb/examples/quickstart.rs` shows that. This
crate is for the other case, and is deliberately a separate package so a client's dependencies
never reach the server binary.

## Use

The crate is not on crates.io yet. Until the packages are published, install from source —
from the repository by git, or from a local checkout by path:

```toml
[dependencies]
dxmt-client = { git = "https://github.com/dxmt/dxmt", package = "dxmt-client" }
# or: dxmt-client = { path = "/path/to/dxmt/vectordb/clients/dxmt-rust-sdk" }
serde_json = "1"   # for json!, as below
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use dxmt_client::{eq, Client, CreateTable, DistanceMetric, Filter, Record, SearchQuery, Update};
use serde_json::json;

let db = Client::new("http://127.0.0.1:9999")?;
db.create_table(CreateTable::new("docs", 768).metric(DistanceMetric::Cosine)).await?;

db.insert("docs", vec![
    Record::new("a", embedding).payload(json!({ "topic": "rust" })),
]).await?;

let record = db.get("docs", "a").await?;          // record.payload() parses payload_json
db.update("docs", "a", Update::default().payload(json!({ "topic": "systems" })).merge()).await?;

let hits = db.search("docs", SearchQuery::new(query_vector, 10)
    .filter(Filter::all(vec![eq("topic", json!("systems"))]))).await?;

let removed = db.delete("docs", vec!["a".into()]).await?;   // removed.deleted, not the ask
```

### Authentication

A server started with `DXMT_API_KEY` answers 401 to everything but its health probes until the key is sent. It goes out as the `api-key` header; send it over TLS, or it is readable at every hop.

```rust
let db = dxmt_client::Client::new("https://db.example.com")?.with_api_key("your-key");
```

`{:?}` on the client prints `<redacted>` in place of the key.

## What the API does that the types make explicit

* **A filter is always `must` / `must_not` / `should`.** The server's `Filter` has those three
  fields and serde ignores unknown ones, so a bare condition is not rejected — it deserialises
  to a filter with nothing in it and matches everything. `Filter::all` / `any` / `none` are the
  only ways to build one here.
* **`top_k` and `limit` are both sent.** The server reads both, and a `limit` left at its
  default of 10 truncates a larger `top_k` without saying so. `SearchQuery::new` sets them
  together.
* **Delete reports what it removed**, not what was asked for. `DeleteOutcome::deleted` is 1 when
  you ask for two ids and only one was there.
* **An update with no vector keeps the stored one**, along with the sparse and named vectors.
  That is what makes a payload-only update safe on a table used for hybrid search.
* **Errors keep the server's words.** A refusal arrives as `Error::Api` with the server's own
  `code` and `message`; a body that is not JSON at all (a proxy's 502 page, an empty 404 for a
  path that does not exist) arrives as `Error::Decode` carrying the HTTP status, and a server
  that never answered is `Error::Transport`. The three are separate variants so a caller can
  tell them apart without matching on strings. One route on the server, `POST /query`, still
  answers with a bare `{"error": "..."}`; this client does not call it, and would read it as
  code `UNSTRUCTURED` if it did.
* **Every `DistanceMetric` variant creates a table.** `DotProduct` serialized as
  `"dotproduct"` in 1.0.0, a spelling the server rejects; it is `"dot"` now, and `Manhattan`
  exists. The live test creates one table per variant.

## Run it

```bash
cd vectordb && cargo run -- serve          # port 9999
cd vectordb/clients/dxmt-rust-sdk
cargo run --example basic_crud             # the five operations, end to end
cargo test                                 # tests/live.rs skips when no server answers;
                                           # tests/api_key.rs needs none and always runs
DXMT_URL=http://host:9999 cargo test       # point either elsewhere
```

The tests skip rather than fail when nothing is listening, so `cargo test` is safe in a
checkout with no server. A skip proves nothing, though: watch them actually run before
believing a change to this crate.
