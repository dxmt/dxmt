//! Talk to the gRPC surface, and measure what it saves.
//!
//!     cd vectordb
//!     cargo run --release --features api-grpc -- serve --grpc-port 9999 &
//!     cargo run --release --features api-grpc --example grpc_client
//!
//! `DXMT_GRPC` overrides the address. The table is created at the start and dropped at the end.
//!
//! The last section is the reason this interface exists. A search request is a vector, and a
//! vector is where JSON is worst: every float becomes decimal text that the sender formats and
//! the receiver parses, where protobuf packs the same array as raw little-endian bytes.

use dxmt_vectordb::api::grpc::proto::vector_db_client::VectorDbClient;
use dxmt_vectordb::api::grpc::proto::*;
use prost::Message;

const DIM: usize = 768;

fn vector(seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("DXMT_GRPC").unwrap_or_else(|_| "http://127.0.0.1:9999".to_string());
    let mut client = match VectorDbClient::connect(addr.clone()).await {
        Ok(client) => client,
        Err(e) => {
            eprintln!("no gRPC server at {addr} ({e})");
            eprintln!("start one with: cargo run --features api-grpc -- serve --grpc-port 9999");
            std::process::exit(1);
        }
    };
    let health = client.health(HealthRequest {}).await?.into_inner();
    println!(
        "connected to {addr}: ready={} version={}",
        health.ready, health.version
    );

    let table = "grpc_demo".to_string();
    let _ = client
        .delete_table(DeleteTableRequest {
            name: table.clone(),
        })
        .await;
    let info = client
        .create_table(CreateTableRequest {
            name: table.clone(),
            dimension: DIM as u32,
            distance_metric: DistanceMetric::Cosine as i32,
            description: "gRPC walkthrough".to_string(),
        })
        .await?
        .into_inner();
    println!("created '{}' ({} dimensions)", info.name, info.dimension);

    // ── insert ───────────────────────────────────────────────────────────────────────────
    let records: Vec<Record> = (0..100)
        .map(|i| Record {
            id: format!("doc-{i}"),
            vector: vector(i as u64 + 1),
            payload_json: format!(
                r#"{{"topic":"{}"}}"#,
                if i % 2 == 0 { "even" } else { "odd" }
            ),
            version: 0,
        })
        .collect();
    let probe = records[7].vector.clone();
    let inserted = client
        .insert(InsertRequest {
            table: table.clone(),
            records,
        })
        .await?
        .into_inner();
    println!("inserted {} records", inserted.count);

    // ── read, and check the floats are the same floats ───────────────────────────────────
    let read = client
        .get(GetRequest {
            table: table.clone(),
            id: "doc-7".to_string(),
        })
        .await?
        .into_inner();
    println!(
        "doc-7 round trip: {} dimensions, identical: {}",
        read.vector.len(),
        read.vector == probe
    );

    // ── search, with and without a filter ────────────────────────────────────────────────
    let hits = client
        .search(SearchRequest {
            table: table.clone(),
            vector: probe.clone(),
            top_k: 3,
            filter: None,
            with_payload: true,
            with_vector: false,
        })
        .await?
        .into_inner();
    println!("nearest to doc-7's vector ({} ms):", hits.took_ms);
    for hit in &hits.hits {
        println!("    {:<8} {:.4}  {}", hit.id, hit.score, hit.payload_json);
    }

    let filtered = client
        .search(SearchRequest {
            table: table.clone(),
            vector: probe.clone(),
            top_k: 3,
            filter: Some(Filter {
                must: vec![Condition {
                    key: "topic".to_string(),
                    operation: Some(condition::Operation::Equal("even".to_string())),
                }],
                must_not: vec![],
                should: vec![],
            }),
            with_payload: false,
            with_vector: false,
        })
        .await?
        .into_inner();
    println!(
        "same query, topic=even: {}",
        filtered
            .hits
            .iter()
            .map(|h| h.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ── update, delete ───────────────────────────────────────────────────────────────────
    client
        .update(UpdateRequest {
            table: table.clone(),
            id: "doc-7".to_string(),
            vector: vec![], // empty means "leave the stored vector alone"
            payload_json: Some(r#"{"seen":true}"#.to_string()),
            merge: true,
            upsert: false,
        })
        .await?;
    let after = client
        .get(GetRequest {
            table: table.clone(),
            id: "doc-7".to_string(),
        })
        .await?
        .into_inner();
    println!("after a payload-only update: {}", after.payload_json);
    println!("  vector untouched: {}", after.vector == probe);

    let deleted = client
        .delete(DeleteRequest {
            table: table.clone(),
            ids: vec!["doc-0".into(), "doc-1".into(), "not-a-real-id".into()],
        })
        .await?
        .into_inner();
    println!(
        "asked to delete {} ids, actually deleted {}",
        deleted.requested, deleted.deleted
    );

    // ── what the wire costs ──────────────────────────────────────────────────────────────
    let request = SearchRequest {
        table: table.clone(),
        vector: probe.clone(),
        top_k: 10,
        filter: None,
        with_payload: true,
        with_vector: false,
    };
    let protobuf = request.encoded_len();
    // The same request as the REST clients send it: a JSON object with the vector as an array
    // of decimal numbers.
    let json = serde_json::json!({
        "vector": probe,
        "top_k": 10,
        "limit": 10,
        "with_payload": true,
        "with_vector": false,
    })
    .to_string()
    .len();
    println!(
        "\none {DIM}-dimension search request on the wire:\n  \
         protobuf {protobuf:>6} bytes\n  \
         JSON     {json:>6} bytes   ({:.1}x)",
        json as f64 / protobuf as f64
    );
    println!(
        "  the vector alone is {} bytes packed; the rest is the field tags",
        DIM * 4
    );

    client
        .delete_table(DeleteTableRequest { name: table })
        .await?;
    println!("\ndropped 'grpc_demo'");
    Ok(())
}
