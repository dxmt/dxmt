# dxmt-client

Rust client for DXMT VectorDB: insert, read, update, delete and search over the server's HTTP
API.

If you want the engine in-process rather than a server to talk to, depend on `dxmt-vectordb`
itself and use `IntelligentIndexedStorage` — `vectordb/examples/quickstart.rs` shows that. This
crate is for the other case, and is deliberately a separate package so a client's dependencies
never reach the server binary.

## Use

```toml
[dependencies]
dxmt-client = { path = "../dxmt-rust-sdk" }
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
* **Two error shapes.** Most routes answer inside the envelope with a `{code, message}`; four
  answer with a bare `{"error": "...", "details": "..."}` — batch insert's validation and
  overload paths and two in the DXQL handler. Both arrive as `Error::Api`; the bare ones carry
  the code `UNSTRUCTURED`, because there is no code in the body to pass through.

## Run it

```bash
cd vectordb && cargo run -- serve          # port 9999
cd vectordb/dxmt-rust-sdk
cargo run --example basic_crud             # the five operations, end to end
cargo test                                 # live tests; they skip when no server answers
DXMT_URL=http://host:9999 cargo test       # point either elsewhere
```

The tests skip rather than fail when nothing is listening, so `cargo test` is safe in a
checkout with no server. A skip proves nothing, though: watch them actually run before
believing a change to this crate.
