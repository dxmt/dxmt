use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vectordb::storage::intelligent_indexed_storage::IntelligentIndexedStorage;
use vectordb::storage::traits::StorageConfig;
use vectordb::types::{CreateRecordRequest, TableConfig, DistanceMetric};

fn generate_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32) / (dim as f32)).collect()
}

fn generate_requests(count: usize, dim: usize) -> Vec<CreateRecordRequest> {
    (0..count)
        .map(|i| CreateRecordRequest {
            id: format!("vec_{}", i),
            vector: generate_vector(dim),
            sparse_vector: None,
            payload: Some(serde_json::json!({"index": i})),
        })
        .collect()
}

fn bench_insert_batch(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("insert_batch");

    for batch_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    // Create storage
                    let config = StorageConfig::default();
                    let storage = IntelligentIndexedStorage::new(&config)
                        .unwrap();

                    // Create table
                    let table_config = TableConfig {
                        name: "bench_table".to_string(),
                        dimension: 128,
                        distance_metric: DistanceMetric::Cosine,
                        ..Default::default()
                    };
                    storage.create_table(table_config).await.unwrap();

                    // Generate and insert records
                    let requests = generate_requests(size, 128);

                    storage.insert_records("bench_table", black_box(requests))
                        .await
                        .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_inserts(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_inserts");

    for num_tasks in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_tasks),
            num_tasks,
            |b, &tasks| {
                b.to_async(&rt).iter(|| async {
                    // Create storage
                    let config = StorageConfig::default();
                    let storage = std::sync::Arc::new(
                        IntelligentIndexedStorage::new(&config).unwrap()
                    );

                    // Create table
                    let table_config = TableConfig {
                        name: "bench_table".to_string(),
                        dimension: 128,
                        distance_metric: DistanceMetric::Cosine,
                        ..Default::default()
                    };
                    storage.create_table(table_config).await.unwrap();

                    // Spawn concurrent insert tasks
                    let mut handles = vec![];

                    for task_id in 0..tasks {
                        let storage_clone = storage.clone();

                        let handle = tokio::spawn(async move {
                            let requests = generate_requests(100, 128);
                            storage_clone
                                .insert_records("bench_table", requests)
                                .await
                                .unwrap();
                        });

                        handles.push(handle);
                    }

                    // Wait for all tasks
                    for handle in handles {
                        handle.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("search");

    // Pre-populate storage
    let (storage, _) = rt.block_on(async {
        let config = StorageConfig::default();
        let storage = IntelligentIndexedStorage::new(&config).unwrap();

        let table_config = TableConfig {
            name: "bench_table".to_string(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        storage.create_table(table_config).await.unwrap();

        // Insert 10K vectors
        let requests = generate_requests(10000, 128);
        storage.insert_records("bench_table", requests).await.unwrap();

        (storage, ())
    });

    group.bench_function("search_10k", |b| {
        b.to_async(&rt).iter(|| async {
            let query = vectordb::types::SearchQuery {
                vector: generate_vector(128),
                top_k: 10,
                filter: None,
                include_vector: false,
                include_payload: true,
            };

            storage.search("bench_table", black_box(query))
                .await
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_batch,
    bench_concurrent_inserts,
    bench_search
);

criterion_main!(benches);
