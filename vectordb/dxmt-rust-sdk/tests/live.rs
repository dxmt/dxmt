//! The client against a real server. Skips itself when there is none.
//!
//!     cd vectordb && cargo run -- serve &
//!     cd vectordb/dxmt-rust-sdk && cargo test
//!
//! `DXMT_URL` points it elsewhere. Every test makes its own table and drops it, so they can run
//! against a server that holds other data and in any order.

use dxmt_client::{eq, Client, CreateTable, DistanceMetric, Filter, Record, SearchQuery, Update};
use serde_json::json;

const DIM: usize = 8;

fn url() -> String {
    std::env::var("DXMT_URL").unwrap_or_else(|_| "http://127.0.0.1:9999".to_string())
}

/// `None` when nothing is listening, which is a skip rather than a failure: the unit of this
/// crate is a client, and a client with no server to talk to has nothing to assert.
async fn connect() -> Option<Client> {
    let client = Client::new(url()).expect("the url should parse");
    match client.ready().await {
        Ok(true) => Some(client),
        _ => {
            eprintln!("skipping: no DXMT server at {}", url());
            None
        }
    }
}

fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| if d == i % DIM { 1.0 } else { 0.05 })
        .collect()
}

async fn fresh_table(db: &Client, name: &str) {
    let _ = db.delete_table(name).await;
    db.create_table(CreateTable::new(name, DIM).metric(DistanceMetric::Cosine))
        .await
        .expect("create table");
}

#[tokio::test]
async fn a_record_survives_insert_read_update_and_delete() {
    let Some(db) = connect().await else { return };
    let table = "rust_sdk_crud";
    fresh_table(&db, table).await;

    let inserted = db
        .insert(
            table,
            vec![Record::new("r1", vector(1)).payload(json!({ "colour": "red", "n": 1 }))],
        )
        .await
        .expect("insert");
    assert_eq!(inserted.count, 1);

    let read = db.get(table, "r1").await.expect("get");
    assert_eq!(read.id, "r1");
    assert_eq!(read.payload()["colour"], "red");
    assert_eq!(
        read.vector,
        vector(1),
        "the vector comes back as it went in"
    );

    // Merge keeps the keys it does not mention; replace would drop `n`.
    db.update(
        table,
        "r1",
        Update::default()
            .payload(json!({ "colour": "blue" }))
            .merge(),
    )
    .await
    .expect("update");
    let read = db.get(table, "r1").await.expect("get after update");
    assert_eq!(read.payload()["colour"], "blue");
    assert_eq!(read.payload()["n"], 1, "merge must keep the untouched key");

    let deleted = db
        .delete(table, vec!["r1".into(), "never-existed".into()])
        .await
        .expect("delete");
    assert_eq!(deleted.deleted, 1, "only the id that was there counts");

    let missing = db.get(table, "r1").await.expect_err("should be gone");
    assert!(missing.is_not_found(), "expected a 404, got {missing}");

    db.delete_table(table).await.expect("drop");
}

#[tokio::test]
async fn search_ranks_a_record_first_for_its_own_vector_and_respects_a_filter() {
    let Some(db) = connect().await else { return };
    let table = "rust_sdk_search";
    fresh_table(&db, table).await;

    let records: Vec<Record> = (0..6)
        .map(|i| {
            Record::new(format!("r{i}"), vector(i))
                .payload(json!({ "parity": if i % 2 == 0 { "even" } else { "odd" } }))
        })
        .collect();
    db.insert(table, records).await.expect("insert");

    let hits = db
        .search(table, SearchQuery::new(vector(3), 3))
        .await
        .expect("search");
    assert_eq!(hits.hits.len(), 3);
    assert_eq!(
        hits.hits[0].id, "r3",
        "a record is its own nearest neighbour"
    );
    assert!(
        hits.hits[0].payload.is_some(),
        "payload asked for and returned"
    );

    let filtered = db
        .search(
            table,
            SearchQuery::new(vector(3), 3).filter(Filter::all(vec![eq("parity", json!("even"))])),
        )
        .await
        .expect("filtered search");
    assert!(
        !filtered.hits.is_empty(),
        "the filter matched nothing at all"
    );
    assert!(
        filtered.hits.iter().all(|h| h.id != "r3"),
        "r3 is odd and must not survive the filter"
    );

    let with_vector = db
        .search(table, SearchQuery::new(vector(0), 1).with_vector())
        .await
        .expect("search with vector");
    assert_eq!(
        with_vector.hits[0].vector.as_deref(),
        Some(vector(0).as_slice()),
        "with_vector should return the stored vector"
    );

    db.delete_table(table).await.expect("drop");
}

#[tokio::test]
async fn the_server_error_reaches_the_caller_intact() {
    let Some(db) = connect().await else { return };
    let table = "rust_sdk_errors";
    fresh_table(&db, table).await;

    // Wrong dimension: the server refuses, and the client must pass its reason through rather
    // than flattening it into "request failed".
    let wrong = db
        .insert(table, vec![Record::new("bad", vec![1.0, 2.0])])
        .await
        .expect_err("a 2-dimensional vector in an 8-dimensional table must be refused");
    assert!(
        wrong.code().is_some(),
        "the server's error code should survive: {wrong}"
    );

    let missing_table = db
        .search("no_such_table", SearchQuery::new(vector(0), 1))
        .await
        .expect_err("searching a table that does not exist must fail");
    assert!(
        missing_table.is_not_found(),
        "expected 404, got {missing_table}"
    );

    db.delete_table(table).await.expect("drop");
}

#[tokio::test]
async fn an_id_with_reserved_characters_addresses_the_right_record() {
    let Some(db) = connect().await else { return };
    let table = "rust_sdk_ids";
    fresh_table(&db, table).await;

    // The shapes a RAG corpus actually produces. Unencoded, `#` truncates the path at the
    // fragment, `/` splits one segment into two, and the request lands somewhere else entirely
    // -- usually on a different record, which is worse than an error.
    let awkward = ["README.md#3", "s3://bucket/key.pdf", "page 12", "a+b&c=d", "100%"];
    let records: Vec<Record> = awkward
        .iter()
        .enumerate()
        .map(|(i, id)| Record::new(*id, vector(i)).payload(json!({ "n": i })))
        .collect();
    db.insert(table, records).await.expect("insert");

    for (i, id) in awkward.iter().enumerate() {
        let read = db.get(table, id).await.unwrap_or_else(|e| panic!("get {id:?}: {e}"));
        assert_eq!(&read.id, id, "the id came back changed");
        assert_eq!(read.payload()["n"], i, "{id:?} addressed the wrong record");

        db.update(table, id, Update::default().payload(json!({ "n": i, "seen": true })).merge())
            .await
            .unwrap_or_else(|e| panic!("update {id:?}: {e}"));
        let after = db.get(table, id).await.expect("get after update");
        assert_eq!(after.payload()["seen"], true, "{id:?} updated the wrong record");
    }

    let deleted = db
        .delete(table, awkward.iter().map(|s| s.to_string()).collect())
        .await
        .expect("delete");
    assert_eq!(deleted.deleted, awkward.len(), "every awkward id should delete");

    db.delete_table(table).await.expect("drop");
}
