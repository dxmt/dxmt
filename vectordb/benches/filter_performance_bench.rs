use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dxmt_vectordb::config::{MemoryConfig, StorageConfig};
use dxmt_vectordb::storage::IntelligentIndexedStorage;
use dxmt_vectordb::types::record::{
    Condition, CreateRecordRequest, Filter, FilterOperation, SearchQuery,
};
use dxmt_vectordb::types::table::TableConfig;
use dxmt_vectordb::types::vector::{DenseVector, DistanceMetric};
use tokio::runtime::Runtime;

/// Generate random vector with given dimension
fn random_vector(dim: usize) -> DenseVector {
    (0..dim).map(|_| rand::random::<f32>()).collect()
}

/// Create a table with test data
async fn create_test_table(
    storage: &IntelligentIndexedStorage,
    table_name: &str,
    dimension: usize,
    num_records: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create table
    let config = TableConfig {
        name: table_name.to_string(),
        dimension,
        distance_metric: DistanceMetric::Cosine,
        description: None,
        index_type: None,
        text_fields: None,
        enable_full_text_search: false,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: None,
    };
    storage.create_table(config).await?;

    // Insert records in batches
    let batch_size = 1000;
    for batch_start in (0..num_records).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_records);
        let mut records = Vec::new();

        for i in batch_start..batch_end {
            // Create records with category field for filtering
            // Distribute categories: 1% = category_rare, 10% = category_uncommon,
            // 50% = category_common, 100% = matches all
            let category = if i % 100 == 0 {
                "category_rare" // 1%
            } else if i % 10 == 0 {
                "category_uncommon" // ~10%
            } else if i % 2 == 0 {
                "category_common" // 50%
            } else {
                "category_other" // ~40%
            };

            let record = CreateRecordRequest {
                id: format!("record_{}", i),
                vector: random_vector(dimension),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::json!({
                    "category": category,
                    "index": i,
                    "batch": batch_start / batch_size,
                })),
                text: None,
            };
            records.push(record);
        }

        storage.insert_records(table_name, records).await?;
    }

    Ok(())
}

/// Benchmark filter performance with different selectivity
fn benchmark_filter_selectivity(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Setup storage
    let config = StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };

    let storage = rt.block_on(async { IntelligentIndexedStorage::new(&config).await.unwrap() });

    // Test parameters
    let dimension = 128;
    let dataset_sizes = vec![1_000, 10_000];
    let selectivity_tests = vec![
        ("1%_selectivity", "category_rare", 0.01),
        ("10%_selectivity", "category_uncommon", 0.10),
        ("50%_selectivity", "category_common", 0.50),
    ];

    for dataset_size in dataset_sizes {
        let table_name = format!("bench_table_{}", dataset_size);

        // Create table with test data
        rt.block_on(async {
            create_test_table(&storage, &table_name, dimension, dataset_size)
                .await
                .unwrap();
        });

        for (test_name, category, _expected_selectivity) in &selectivity_tests {
            let bench_name = format!("{}_{}", dataset_size, test_name);

            c.bench_with_input(
                BenchmarkId::new("filter_search", &bench_name),
                &(&storage, &table_name, category, dimension),
                |b, (storage, table_name, category, dimension)| {
                    b.to_async(&rt).iter(|| async move {
                        let query = SearchQuery {
                            vector: random_vector(*dimension),
                            top_k: 10,
                            limit: 10,
                            search_mode: Default::default(),
                            vector_name: None,
                            filter: Some(Filter {
                                must: Some(vec![Condition {
                                    key: "category".to_string(),
                                    operation: FilterOperation::Equal(serde_json::json!(category)),
                                }]),
                                must_not: None,
                                should: None,
                            }),
                            with_payload: true,
                            with_vector: false,
                            ..Default::default()
                        };

                        let result = storage.search(table_name, query).await;
                        black_box(result)
                    });
                },
            );
        }
    }
}

/// Benchmark search without filter (baseline)
fn benchmark_search_no_filter(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = StorageConfig {
        memory: Some(MemoryConfig {
            max_size_mb: 512,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
    };

    let storage = rt.block_on(async { IntelligentIndexedStorage::new(&config).await.unwrap() });

    let dimension = 128;
    let dataset_size = 10_000;
    let table_name = "bench_no_filter";

    rt.block_on(async {
        create_test_table(&storage, table_name, dimension, dataset_size)
            .await
            .unwrap();
    });

    c.bench_function("search_no_filter_10k", |b| {
        b.to_async(&rt).iter(|| async {
            let query = SearchQuery {
                vector: random_vector(dimension),
                top_k: 10,
                limit: 10,
                search_mode: Default::default(),
                vector_name: None,
                filter: None,
                with_payload: true,
                with_vector: false,
                ..Default::default()
            };

            let result = storage.search(table_name, query).await;
            black_box(result)
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = benchmark_filter_selectivity, benchmark_search_no_filter
}
criterion_main!(benches);
