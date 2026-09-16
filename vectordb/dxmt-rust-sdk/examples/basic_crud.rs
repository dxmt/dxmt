//! Insert, read, update, delete and search, against a running server.
//!
//!     cd vectordb && cargo run -- serve                       # in one terminal
//!     cd vectordb/dxmt-rust-sdk && cargo run --example basic_crud
//!
//! `DXMT_URL` overrides the address. The table is created at the start and dropped at the end,
//! so the example can be run repeatedly without cleaning up after it.

use dxmt_client::{
    eq, Client, CreateTable, DistanceMetric, Error, Filter, Record, SearchQuery, Update,
};
use serde_json::json;

const TABLE: &str = "rust_sdk_demo";
const DIM: usize = 8;

/// A vector that leans towards one axis, so the expected nearest neighbour of `axis(i)` is the
/// record built from `axis(i)` and nothing else. Deterministic, so the output is stable.
fn axis(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| if d == i % DIM { 1.0 } else { 0.05 })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let url = std::env::var("DXMT_URL").unwrap_or_else(|_| "http://127.0.0.1:9999".to_string());
    let db = Client::new(&url)?;

    if !db.ready().await? {
        eprintln!("no server at {url} (start one with: cd vectordb && cargo run -- serve)");
        std::process::exit(1);
    }
    println!("connected to {url}");

    // Start from nothing, even if a previous run left the table behind.
    let _ = db.delete_table(TABLE).await;
    let table = db
        .create_table(CreateTable::new(TABLE, DIM).metric(DistanceMetric::Cosine))
        .await?;
    println!(
        "created table '{}' ({} dimensions)",
        table.name, table.dimension
    );

    // ── insert ───────────────────────────────────────────────────────────────────────────
    let records: Vec<Record> = (0..6)
        .map(|i| {
            Record::new(format!("doc-{i}"), axis(i)).payload(json!({
                "title": format!("document {i}"),
                "topic": if i % 2 == 0 { "even" } else { "odd" },
                "rank": i,
            }))
        })
        .collect();
    let inserted = db.insert(TABLE, records).await?;
    println!("inserted {} records", inserted.count);

    // ── read one back ────────────────────────────────────────────────────────────────────
    let doc = db.get(TABLE, "doc-3").await?;
    println!(
        "read doc-3: payload {} vector[0..3] {:?}",
        doc.payload(),
        &doc.vector[..3.min(doc.vector.len())]
    );

    // ── search ───────────────────────────────────────────────────────────────────────────
    let result = db.search(TABLE, SearchQuery::new(axis(3), 3)).await?;
    println!(
        "nearest to axis(3), {} hits in {} ms:",
        result.total, result.took_ms
    );
    for hit in &result.hits {
        println!("    {:<8} {:.4}", hit.id, hit.score);
    }
    assert_eq!(
        result.hits[0].id, "doc-3",
        "a record is its own nearest neighbour"
    );

    // ── search with a filter ─────────────────────────────────────────────────────────────
    let filtered = db
        .search(
            TABLE,
            SearchQuery::new(axis(3), 3).filter(Filter::all(vec![eq("topic", json!("even"))])),
        )
        .await?;
    println!("same query, topic=even only:");
    for hit in &filtered.hits {
        println!("    {:<8} {:.4}", hit.id, hit.score);
    }
    assert!(
        !filtered.hits.iter().any(|h| h.id == "doc-3"),
        "doc-3 is odd and must be filtered out"
    );

    // ── update ───────────────────────────────────────────────────────────────────────────
    // Payload only: the vector is left alone, so the record keeps its place in the graph.
    db.update(
        TABLE,
        "doc-3",
        Update::default()
            .payload(json!({ "title": "renamed" }))
            .merge(),
    )
    .await?;
    println!(
        "after a merge update, doc-3 payload: {}",
        db.get(TABLE, "doc-3").await?.payload()
    );

    // Vector as well: now it should answer to a different query.
    db.update(TABLE, "doc-3", Update::default().vector(axis(5)))
        .await?;
    let moved = db.search(TABLE, SearchQuery::new(axis(5), 2)).await?;
    println!(
        "after moving doc-3 onto axis(5), nearest to axis(5): {}",
        moved
            .hits
            .iter()
            .map(|h| h.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ── delete ───────────────────────────────────────────────────────────────────────────
    let gone = db
        .delete(
            TABLE,
            vec!["doc-0".into(), "doc-1".into(), "not-a-real-id".into()],
        )
        .await?;
    println!("asked to delete 3 ids, actually deleted {}", gone.deleted);

    match db.get(TABLE, "doc-0").await {
        Err(e) if e.is_not_found() => println!("doc-0 is gone, as expected"),
        Err(e) => return Err(e),
        Ok(_) => panic!("doc-0 should have been deleted"),
    }

    let left = db.get_table(TABLE).await?;
    println!("table now holds {} records", left.record_count);

    db.delete_table(TABLE).await?;
    println!("dropped '{TABLE}'");
    Ok(())
}
