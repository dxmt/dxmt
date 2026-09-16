//! The gRPC surface, driven by a real client over a real socket.
//!
//! `cargo test --features api-grpc --test grpc_test`
//!
//! The point of this interface is the data path, and the point of this test is that the data
//! survives it: a vector goes out as packed floats and has to come back bit for bit, a filter
//! built from the proto oneof has to mean what the REST filter means, and an update with no
//! vector has to leave the stored one alone -- the same semantics the HTTP handler has, because
//! both now go through `apply_update`.

#![cfg(feature = "api-grpc")]

use dxmt_vectordb::api::grpc::proto::vector_db_client::VectorDbClient;
use dxmt_vectordb::api::grpc::proto::*;
use dxmt_vectordb::api::grpc::GrpcService;
use dxmt_vectordb::config::{MemoryConfig, StorageConfig};
use dxmt_vectordb::storage::IntelligentIndexedStorage;
use std::sync::Arc;
use tokio::net::TcpListener;

const DIM: usize = 16;

fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| if d == i % DIM { 1.0 } else { 0.0625 })
        .collect()
}

/// Start a server on a free port and return a client pointed at it.
async fn serve() -> VectorDbClient<tonic::transport::Channel> {
    let storage = Arc::new(
        IntelligentIndexedStorage::new(&StorageConfig {
            r#type: "memory".to_string(),
            memory: Some(MemoryConfig {
                max_size_mb: 512,
                eviction_policy: "lru".to_string(),
            }),
            ..Default::default()
        })
        .await
        .expect("storage"),
    );
    // Port 0 and then ask what was chosen: a fixed port makes the test fail when anything else
    // on the machine happens to hold it, including a second copy of this test. The listener is
    // dropped before tonic binds the same address, which leaves a gap the retry loop below
    // covers -- narrower than picking a port and hoping.
    let addr = {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        listener.local_addr().expect("addr")
    };
    let service = GrpcService::new(storage).into_server();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service)
            .serve(addr)
            .await
            .expect("grpc server");
    });

    // Retry until the spawned server has the port.
    for _ in 0..50 {
        if let Ok(client) = VectorDbClient::connect(format!("http://{addr}")).await {
            return client;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("gRPC server never accepted a connection on {addr}");
}

async fn table(client: &mut VectorDbClient<tonic::transport::Channel>, name: &str) {
    client
        .create_table(CreateTableRequest {
            name: name.to_string(),
            dimension: DIM as u32,
            distance_metric: DistanceMetric::Cosine as i32,
            description: String::new(),
        })
        .await
        .expect("create table");
}

#[tokio::test]
async fn a_record_round_trips_through_grpc_unchanged() {
    let mut client = serve().await;
    table(&mut client, "t").await;

    let sent = vector(3);
    let inserted = client
        .insert(InsertRequest {
            table: "t".to_string(),
            records: vec![Record {
                id: "r1".to_string(),
                vector: sent.clone(),
                payload_json: r#"{"colour":"red","n":1}"#.to_string(),
                version: 0,
            }],
        })
        .await
        .expect("insert")
        .into_inner();
    assert_eq!(inserted.count, 1);
    assert_eq!(inserted.inserted_ids, vec!["r1"]);

    let read = client
        .get(GetRequest {
            table: "t".to_string(),
            id: "r1".to_string(),
        })
        .await
        .expect("get")
        .into_inner();
    assert_eq!(read.id, "r1");
    // Bit for bit. A float that survives JSON is a float that was printed and re-parsed; this
    // one is the same four bytes that went out, and that is the reason this interface exists.
    assert_eq!(read.vector, sent);
    let payload: serde_json::Value =
        serde_json::from_str(&read.payload_json).expect("payload is JSON");
    assert_eq!(payload["colour"], "red");
}

#[tokio::test]
async fn search_ranks_a_record_first_for_its_own_vector_and_honours_a_filter() {
    let mut client = serve().await;
    table(&mut client, "t").await;

    let records: Vec<Record> = (0..6)
        .map(|i| Record {
            id: format!("r{i}"),
            vector: vector(i),
            payload_json: format!(
                r#"{{"parity":"{}"}}"#,
                if i % 2 == 0 { "even" } else { "odd" }
            ),
            version: 0,
        })
        .collect();
    client
        .insert(InsertRequest {
            table: "t".to_string(),
            records,
        })
        .await
        .expect("insert");

    let hits = client
        .search(SearchRequest {
            table: "t".to_string(),
            vector: vector(3),
            top_k: 3,
            filter: None,
            with_payload: true,
            with_vector: false,
        })
        .await
        .expect("search")
        .into_inner();
    assert_eq!(hits.hits.len(), 3);
    assert_eq!(
        hits.hits[0].id, "r3",
        "a record is its own nearest neighbour"
    );

    // The same filter the REST clients build as must/[{key, operation}], expressed as a oneof
    // so the operator cannot be misspelt.
    let filtered = client
        .search(SearchRequest {
            table: "t".to_string(),
            vector: vector(3),
            top_k: 3,
            filter: Some(Filter {
                must: vec![Condition {
                    key: "parity".to_string(),
                    operation: Some(condition::Operation::Equal("even".to_string())),
                }],
                must_not: vec![],
                should: vec![],
            }),
            with_payload: true,
            with_vector: false,
        })
        .await
        .expect("filtered search")
        .into_inner();
    assert!(
        !filtered.hits.is_empty(),
        "the filter matched nothing at all"
    );
    assert!(
        filtered.hits.iter().all(|h| h.id != "r3"),
        "r3 is odd and must not survive the filter"
    );
}

#[tokio::test]
async fn an_update_without_a_vector_keeps_the_stored_one() {
    let mut client = serve().await;
    table(&mut client, "t").await;

    let original = vector(2);
    client
        .insert(InsertRequest {
            table: "t".to_string(),
            records: vec![Record {
                id: "r1".to_string(),
                vector: original.clone(),
                payload_json: r#"{"colour":"red","n":1}"#.to_string(),
                version: 0,
            }],
        })
        .await
        .expect("insert");

    // Payload only, merging: the vector is untouched and the key not mentioned survives. This
    // is `apply_update`, the same function the HTTP handler calls.
    client
        .update(UpdateRequest {
            table: "t".to_string(),
            id: "r1".to_string(),
            vector: vec![],
            payload_json: Some(r#"{"colour":"blue"}"#.to_string()),
            merge: true,
            upsert: false,
        })
        .await
        .expect("update");

    let read = client
        .get(GetRequest {
            table: "t".to_string(),
            id: "r1".to_string(),
        })
        .await
        .expect("get")
        .into_inner();
    assert_eq!(
        read.vector, original,
        "a payload edit must not move the record"
    );
    let payload: serde_json::Value = serde_json::from_str(&read.payload_json).expect("JSON");
    assert_eq!(payload["colour"], "blue");
    assert_eq!(payload["n"], 1, "merge must keep the untouched key");
}

#[tokio::test]
async fn refusals_arrive_as_the_grpc_code_that_means_the_same_thing() {
    let mut client = serve().await;
    table(&mut client, "t").await;

    let missing = client
        .get(GetRequest {
            table: "t".to_string(),
            id: "never-existed".to_string(),
        })
        .await
        .expect_err("a missing record must be refused");
    assert_eq!(missing.code(), tonic::Code::NotFound, "{missing}");

    let wrong_dimension = client
        .insert(InsertRequest {
            table: "t".to_string(),
            records: vec![Record {
                id: "bad".to_string(),
                vector: vec![1.0, 2.0],
                payload_json: String::new(),
                version: 0,
            }],
        })
        .await
        .expect_err("a 2-dimensional vector in a 16-dimensional table must be refused");
    assert_eq!(
        wrong_dimension.code(),
        tonic::Code::InvalidArgument,
        "{wrong_dimension}"
    );

    let no_table = client
        .search(SearchRequest {
            table: "no_such_table".to_string(),
            vector: vector(0),
            top_k: 1,
            filter: None,
            with_payload: false,
            with_vector: false,
        })
        .await
        .expect_err("searching a table that does not exist must be refused");
    assert_eq!(no_table.code(), tonic::Code::NotFound, "{no_table}");

    // And a delete reports what it removed, not what was asked for.
    let deleted = client
        .delete(DeleteRequest {
            table: "t".to_string(),
            ids: vec!["never-existed".to_string(), "also-not".to_string()],
        })
        .await
        .expect("delete of absent ids is not an error")
        .into_inner();
    assert_eq!(deleted.requested, 2);
    assert_eq!(deleted.deleted, 0);
}
