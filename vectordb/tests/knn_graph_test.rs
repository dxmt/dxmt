//! The k-nearest-neighbour graph over a sample, which exists to move a specific cost off the
//! client.
//!
//! A browser that wants to lay a table out has to know which records are near which. Computing
//! that client-side means downloading the vectors, and at 768 dimensions a thousand records is
//! thirteen megabytes of JSON floats — so the sample size a projection can use was decided by
//! the wire rather than by anything about the data. This endpoint returns the answer instead of
//! the inputs.
//!
//! Three properties make the answer usable, and each is easy to lose:
//!
//! * **Exact.** It is the reference a layout is judged against. An approximate graph would make
//!   "how much structure survived the projection" a measurement of two approximations at once.
//! * **Within the sample.** A layout is drawn from the points on screen; an edge to a point
//!   that is not drawn describes nothing.
//! * **Reproducible.** The same request twice is the same graph, or a shared URL shows someone
//!   else a different picture.
//!
//! Every test here is `#[serial]`. The node cap is read from `DXMT_KNN_GRAPH_MAX_NODES`, and an
//! environment variable is process-global while cargo runs a binary's tests in parallel threads
//! — so without this the cap test's `set_var` reaches whichever tests happen to be running
//! beside it, and they fail for a reason that has nothing to do with them. It cost one
//! confusing failure to find that out, and the same note is already at the top of
//! `wal_recovery_test.rs`.

use dxmt_vectordb::{
    config::{MemoryConfig, StorageConfig},
    storage::IntelligentIndexedStorage,
    types::{
        record::{Condition, CreateRecordRequest, Filter, FilterOperation},
        table::TableConfig,
        vector::DistanceMetric,
    },
};

const DIM: usize = 12;

fn storage_config() -> StorageConfig {
    StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    }
}

async fn open(wal: Option<std::path::PathBuf>) -> IntelligentIndexedStorage {
    IntelligentIndexedStorage::new_with_wal(&storage_config(), wal)
        .await
        .expect("failed to open storage")
}

fn table(name: &str) -> TableConfig {
    TableConfig {
        name: name.to_string(),
        dimension: DIM,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
    }
}

/// Four tight clusters, so "nearest" has an answer anyone can check by eye: a record's
/// neighbours should be in its own cluster and nowhere else.
fn vector(i: usize) -> Vec<f32> {
    let cluster = i % 4;
    let jitter = ((i / 4) % 17) as f32 * 0.002;
    (0..DIM)
        .map(|d| {
            if d / 3 == cluster {
                1.0 + jitter
            } else {
                0.01 + jitter
            }
        })
        .collect()
}

fn record(i: usize) -> CreateRecordRequest {
    CreateRecordRequest {
        id: format!("r{i:04}"),
        vector: vector(i),
        sparse_vector: None,
        named_vectors: None,
        payload: Some(
            serde_json::json!({ "cluster": i % 4, "half": if i < 200 { "a" } else { "b" } }),
        ),
    }
}

async fn filled(rows: usize) -> IntelligentIndexedStorage {
    let storage = open(None).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..rows).map(record).collect())
        .await
        .unwrap();
    storage
}

#[tokio::test]
#[serial_test::serial]
async fn neighbours_are_the_true_nearest_within_the_sample() {
    let storage = filled(400).await;
    // The whole table as the sample, so "within the sample" and "within the table" coincide and
    // the answer can be checked against the construction rather than against another query.
    let graph = storage.knn_graph("t", 400, 5, None, 0, true).await.unwrap();

    assert_eq!(graph.sampled, 400);
    assert_eq!(graph.total, 400);
    assert_eq!(graph.ids.len(), 400);
    assert_eq!(graph.neighbours.len(), 400);

    let cluster_of = |id: &str| id[1..].parse::<usize>().unwrap() % 4;
    let mut wrong = 0;
    for (node, edges) in graph.neighbours.iter().enumerate() {
        assert_eq!(edges.len(), 5, "every node should have its five nearest");
        let mine = cluster_of(&graph.ids[node]);
        for &edge in edges {
            assert_ne!(edge as usize, node, "a node is not its own neighbour");
            if cluster_of(&graph.ids[edge as usize]) != mine {
                wrong += 1;
            }
        }
    }
    assert_eq!(
        wrong, 0,
        "{wrong} edges left their cluster; the graph is not the true nearest neighbours"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn scores_are_similarities_and_ordered_nearest_first() {
    let storage = filled(200).await;
    let graph = storage.knn_graph("t", 200, 8, None, 0, true).await.unwrap();

    for (node, scores) in graph.scores.iter().enumerate() {
        assert_eq!(scores.len(), graph.neighbours[node].len());
        for pair in scores.windows(2) {
            assert!(
                pair[0] >= pair[1] - 1e-6,
                "node {node}'s neighbours are not nearest first: {scores:?}"
            );
        }
        // Higher is nearer, matching what `search` returns, so a client does not have to know
        // which way round this particular metric runs.
        assert!(
            scores[0] > 0.5,
            "a same-cluster neighbour should score high"
        );
    }
}

#[tokio::test]
#[serial_test::serial]
async fn the_sample_is_spread_across_the_table_rather_than_taken_from_one_end() {
    // A table is usually written in some order. A contiguous block samples whenever that block
    // was written, which for a projection means a picture of one afternoon.
    let storage = filled(1_000).await;
    let graph = storage.knn_graph("t", 50, 4, None, 0, true).await.unwrap();

    assert_eq!(graph.sampled, 50);
    assert_eq!(graph.total, 1_000);

    let numbers: Vec<usize> = graph
        .ids
        .iter()
        .map(|id| id[1..].parse::<usize>().unwrap())
        .collect();
    let low = numbers.iter().copied().min().unwrap();
    let high = numbers.iter().copied().max().unwrap();
    assert!(
        high - low > 900,
        "the sample spans only {low}..{high} of 0..1000, so it is a block and not a spread"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn the_same_request_gives_the_same_graph() {
    let storage = filled(300).await;
    let first = storage.knn_graph("t", 60, 6, None, 7, true).await.unwrap();
    let second = storage.knn_graph("t", 60, 6, None, 7, true).await.unwrap();

    assert_eq!(
        first.ids, second.ids,
        "the same seed sampled different records"
    );
    assert_eq!(first.neighbours, second.neighbours);

    // And a different seed moves the sample, or the parameter is decoration.
    let third = storage.knn_graph("t", 60, 6, None, 3, true).await.unwrap();
    assert_ne!(first.ids, third.ids, "the seed does not change the sample");
}

#[tokio::test]
#[serial_test::serial]
async fn a_filter_restricts_the_sample_and_the_edges_with_it() {
    let storage = filled(400).await;
    let mut filter = Filter {
        must: Some(vec![Condition {
            key: "half".to_string(),
            operation: FilterOperation::Equal(serde_json::json!("a")),
        }]),
        must_not: None,
        should: None,
    };
    let graph = storage
        .knn_graph("t", 100, 5, Some(&filter), 0, true)
        .await
        .unwrap();

    assert_eq!(graph.total, 200, "only the first half matches");
    for id in &graph.ids {
        assert!(
            id[1..].parse::<usize>().unwrap() < 200,
            "{id} is outside the filter but in the sample"
        );
    }

    // A filter matching nothing is an empty graph rather than an error: a client asking about a
    // predicate with no rows wants to be told that, not to handle a failure.
    filter.must = Some(vec![Condition {
        key: "half".to_string(),
        operation: FilterOperation::Equal(serde_json::json!("nope")),
    }]);
    let empty = storage
        .knn_graph("t", 100, 5, Some(&filter), 0, true)
        .await
        .unwrap();
    assert_eq!(empty.sampled, 0);
    assert_eq!(empty.total, 0);
    assert!(empty.neighbours.is_empty());
}

#[tokio::test]
#[serial_test::serial]
async fn it_refuses_a_graph_too_big_to_build() {
    // The work is quadratic in the node count, so doubling the request quadruples the wait. A
    // ceiling that refuses beats one that takes a minute, because a caller cannot tell a slow
    // answer from a hung one.
    std::env::set_var("DXMT_KNN_GRAPH_MAX_NODES", "50");
    let storage = filled(200).await;

    let error = storage
        .knn_graph("t", 120, 5, None, 0, true)
        .await
        .expect_err("120 nodes against a 50-node cap must be refused");
    let message = error.to_string();
    assert!(
        message.contains("120") && message.contains("50"),
        "the refusal should say what was asked for and what the limit is: {message}"
    );
    assert!(
        message.contains("quadratic"),
        "and why the limit exists, since it is not obvious: {message}"
    );

    // Under the cap still works, so the guard is a ceiling and not a wall.
    assert_eq!(
        storage
            .knn_graph("t", 50, 5, None, 0, true)
            .await
            .unwrap()
            .sampled,
        50
    );
    std::env::remove_var("DXMT_KNN_GRAPH_MAX_NODES");
}

#[tokio::test]
#[serial_test::serial]
async fn it_works_with_the_vectors_in_a_row_file() {
    // The production shape: the records carry no vector and every one has to be read from
    // `wal/vectors/t.f32`. Same graph, different code path.
    let tmp = tempfile::TempDir::new().unwrap();
    let wal = tmp.path().join("wal");
    let storage = open(Some(wal.clone())).await;
    storage.create_table(table("t")).await.unwrap();
    storage
        .insert_records("t", (0..300).map(record).collect())
        .await
        .unwrap();
    assert!(
        wal.join("vectors").join("t.f32").exists(),
        "this test is about reading the vectors from a row file; there is not one"
    );

    let graph = storage.knn_graph("t", 100, 4, None, 0, true).await.unwrap();
    let cluster_of = |id: &str| id[1..].parse::<usize>().unwrap() % 4;
    for (node, edges) in graph.neighbours.iter().enumerate() {
        for &edge in edges {
            assert_eq!(
                cluster_of(&graph.ids[edge as usize]),
                cluster_of(&graph.ids[node]),
                "an edge left its cluster when the vectors came from disk"
            );
        }
    }
}

#[tokio::test]
#[serial_test::serial]
async fn a_sample_too_small_for_edges_is_answered_rather_than_refused() {
    let storage = filled(1).await;
    let graph = storage.knn_graph("t", 10, 5, None, 0, true).await.unwrap();
    assert_eq!(graph.sampled, 1);
    assert!(
        graph.neighbours.is_empty(),
        "one node has no neighbours to have"
    );

    // And asking for more neighbours than there are other nodes gets what exists.
    let storage = filled(4).await;
    let graph = storage.knn_graph("t", 4, 50, None, 0, true).await.unwrap();
    assert!(graph.neighbours.iter().all(|edges| edges.len() == 3));
}

#[tokio::test]
#[serial_test::serial]
async fn payloads_travel_with_the_nodes_and_can_be_left_out() {
    // A layout needs labels — something to colour by, something to show on hover — and they
    // cannot come from a second request: the sample is scattered across the table and no route
    // fetches a list of ids. They also cost almost nothing beside what this endpoint saves: a
    // payload is tens of bytes where a 768-dimension vector is ten kilobytes of JSON floats.
    let storage = filled(200).await;

    let graph = storage.knn_graph("t", 40, 4, None, 0, true).await.unwrap();
    let payloads = graph.payloads.expect("asked for payloads and got none");
    assert_eq!(
        payloads.len(),
        graph.ids.len(),
        "one payload per node, in node order"
    );

    for (node, raw) in payloads.iter().enumerate() {
        let parsed: serde_json::Value =
            serde_json::from_str(raw.as_deref().expect("a record with a payload sent none"))
                .expect("the payload is not the JSON string records/list sends");
        let expected = graph.ids[node][1..].parse::<usize>().unwrap() % 4;
        assert_eq!(
            parsed["cluster"], expected,
            "{}'s payload belongs to another record",
            graph.ids[node]
        );
    }

    // And omitted entirely when not asked for, rather than sent as a list of nulls.
    let bare = storage.knn_graph("t", 40, 4, None, 0, false).await.unwrap();
    assert!(bare.payloads.is_none());
    assert_eq!(
        bare.ids, graph.ids,
        "asking for payloads should not change the sample"
    );
}
