// WAL Concurrency Stress Tests
//
// 测试 WAL 在高并发场景下的安全性和正确性

use dxmt_vectordb::{
    config::{StorageConfig, MemoryConfig},
    storage::{IntelligentIndexedStorage, Storage, traits::WalConsistencyMode},
    types::{
        table::TableConfig,
        record::CreateRecordRequest,
        vector::{DistanceMetric, RecordId},
    },
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;
use tokio::time::Duration;

/// 创建测试用的存储配置
async fn create_storage(
    consistency_mode: WalConsistencyMode,
) -> (Arc<IntelligentIndexedStorage>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let wal_dir = temp_dir.path().join("wal");

    let config = StorageConfig {
        r#type: "memory".to_string(),
        memory: Some(MemoryConfig {
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
        }),
        ssd: None,
        object_storage: None,
        enable_wal: true,
        wal_dir: wal_dir.to_str().unwrap().to_string(),
        wal_batch_size: 100,
        wal_sync_interval_ms: 10,
        wal_consistency_mode: consistency_mode,
    };

    let storage = Arc::new(IntelligentIndexedStorage::new(&config).await.unwrap());

    // 创建测试表
    let table_config = TableConfig {
        name: "concurrent_test".to_string(),
        dimension: 128,
        distance_metric: DistanceMetric::Cosine,
        index_type: None,
        description: None,
    };
    storage.create_table(table_config).await.unwrap();

    (storage, temp_dir)
}

#[tokio::test]
async fn test_concurrent_inserts_eventual() {
    println!("\n=== Concurrent Inserts (Eventual Consistency) ===");

    let (storage, _temp_dir) = create_storage(WalConsistencyMode::Eventual).await;
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    // 启动 100 个并发插入任务
    let mut handles = Vec::new();
    for thread_id in 0..100 {
        let storage = Arc::clone(&storage);
        let success_count = Arc::clone(&success_count);
        let error_count = Arc::clone(&error_count);

        let handle = tokio::spawn(async move {
            for i in 0..10 {
                let record_id = format!("thread_{}_record_{}", thread_id, i);
                let records = vec![CreateRecordRequest {
                    id: record_id,
                    vector: vec![(thread_id * 10 + i) as f32; 128],
                    payload: Some(serde_json::json!({
                        "thread_id": thread_id,
                        "index": i,
                    })),
                    sparse_vector: None,
                }];

                match storage.insert_records("concurrent_test", records).await {
                    Ok(_) => {
                        success_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("Insert error: {:?}", e);
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // 等待所有任务完成
    for handle in handles {
        handle.await.unwrap();
    }

    // 等待 WAL 持久化
    tokio::time::sleep(Duration::from_millis(100)).await;

    let success = success_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);

    println!("Total inserts: 1000");
    println!("Successful: {}", success);
    println!("Errors: {}", errors);

    assert_eq!(success, 1000, "All inserts should succeed");
    assert_eq!(errors, 0, "No errors should occur");
}

#[tokio::test]
async fn test_concurrent_inserts_strong() {
    println!("\n=== Concurrent Inserts (Strong Consistency) ===");

    let (storage, _temp_dir) = create_storage(WalConsistencyMode::Strong).await;
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    // 启动 50 个并发插入任务 (Strong 模式较慢，减少并发数)
    let mut handles = Vec::new();
    for thread_id in 0..50 {
        let storage = Arc::clone(&storage);
        let success_count = Arc::clone(&success_count);
        let error_count = Arc::clone(&error_count);

        let handle = tokio::spawn(async move {
            for i in 0..10 {
                let record_id = format!("thread_{}_record_{}", thread_id, i);
                let records = vec![CreateRecordRequest {
                    id: record_id,
                    vector: vec![(thread_id * 10 + i) as f32; 128],
                    payload: Some(serde_json::json!({
                        "thread_id": thread_id,
                        "index": i,
                    })),
                    sparse_vector: None,
                }];

                match storage.insert_records("concurrent_test", records).await {
                    Ok(_) => {
                        success_count.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("Insert error: {:?}", e);
                        error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // 等待所有任务完成
    for handle in handles {
        handle.await.unwrap();
    }

    let success = success_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);

    println!("Total inserts: 500");
    println!("Successful: {}", success);
    println!("Errors: {}", errors);

    assert_eq!(success, 500, "All inserts should succeed");
    assert_eq!(errors, 0, "No errors should occur");
}

#[tokio::test]
async fn test_concurrent_insert_and_query() {
    println!("\n=== Concurrent Insert + Query ===");

    let (storage, _temp_dir) = create_storage(WalConsistencyMode::Eventual).await;

    let insert_count = Arc::new(AtomicUsize::new(0));
    let query_count = Arc::new(AtomicUsize::new(0));

    // 插入线程
    let mut insert_handles = Vec::new();
    for thread_id in 0..20 {
        let storage = Arc::clone(&storage);
        let insert_count = Arc::clone(&insert_count);

        let handle = tokio::spawn(async move {
            for i in 0..50 {
                let record_id = format!("insert_thread_{}_record_{}", thread_id, i);
                let records = vec![CreateRecordRequest {
                    id: record_id,
                    vector: vec![(thread_id * 50 + i) as f32; 128],
                    payload: None,
                    sparse_vector: None,
                }];

                if storage.insert_records("concurrent_test", records).await.is_ok() {
                    insert_count.fetch_add(1, Ordering::Relaxed);
                }

                // 小延迟，模拟真实场景
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        });

        insert_handles.push(handle);
    }

    // 查询线程
    let mut query_handles = Vec::new();
    for _thread_id in 0..10 {
        let storage = Arc::clone(&storage);
        let query_count = Arc::clone(&query_count);

        let handle = tokio::spawn(async move {
            for _ in 0..100 {
                // 尝试获取表信息
                if storage.get_table("concurrent_test").await.is_ok() {
                    query_count.fetch_add(1, Ordering::Relaxed);
                }

                tokio::time::sleep(Duration::from_micros(500)).await;
            }
        });

        query_handles.push(handle);
    }

    // 等待所有任务完成
    for handle in insert_handles {
        handle.await.unwrap();
    }
    for handle in query_handles {
        handle.await.unwrap();
    }

    let inserts = insert_count.load(Ordering::Relaxed);
    let queries = query_count.load(Ordering::Relaxed);

    println!("Successful inserts: {}", inserts);
    println!("Successful queries: {}", queries);

    assert_eq!(inserts, 1000, "All inserts should succeed");
    assert_eq!(queries, 1000, "All queries should succeed");
}

#[tokio::test]
async fn test_concurrent_insert_and_delete() {
    println!("\n=== Concurrent Insert + Delete ===");

    let (storage, _temp_dir) = create_storage(WalConsistencyMode::Strong).await;

    // 先插入一些初始数据
    for i in 0..100 {
        let records = vec![CreateRecordRequest {
            id: format!("initial_record_{}", i),
            vector: vec![i as f32; 128],
            payload: None,
            sparse_vector: None,
        }];
        storage.insert_records("concurrent_test", records).await.unwrap();
    }

    let insert_count = Arc::new(AtomicUsize::new(0));
    let delete_count = Arc::new(AtomicUsize::new(0));

    // 插入线程
    let mut insert_handles = Vec::new();
    for thread_id in 0..10 {
        let storage = Arc::clone(&storage);
        let insert_count = Arc::clone(&insert_count);

        let handle = tokio::spawn(async move {
            for i in 0..20 {
                let record_id = format!("new_record_{}_{}", thread_id, i);
                let records = vec![CreateRecordRequest {
                    id: record_id,
                    vector: vec![(thread_id * 20 + i) as f32; 128],
                    payload: None,
                    sparse_vector: None,
                }];

                if storage.insert_records("concurrent_test", records).await.is_ok() {
                    insert_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        insert_handles.push(handle);
    }

    // 删除线程
    let mut delete_handles = Vec::new();
    for i in 0..10 {
        let storage = Arc::clone(&storage);
        let delete_count = Arc::clone(&delete_count);

        let handle = tokio::spawn(async move {
            let records_to_delete: Vec<RecordId> = (i * 10..(i + 1) * 10)
                .map(|j| format!("initial_record_{}", j))
                .collect();

            if storage.delete_records("concurrent_test", records_to_delete).await.is_ok() {
                delete_count.fetch_add(10, Ordering::Relaxed);
            }
        });

        delete_handles.push(handle);
    }

    // 等待所有任务完成
    for handle in insert_handles {
        handle.await.unwrap();
    }
    for handle in delete_handles {
        handle.await.unwrap();
    }

    let inserts = insert_count.load(Ordering::Relaxed);
    let deletes = delete_count.load(Ordering::Relaxed);

    println!("Successful inserts: {}", inserts);
    println!("Successful deletes: {}", deletes);

    assert_eq!(inserts, 200, "All new inserts should succeed");
    assert_eq!(deletes, 100, "All deletes should succeed");
}

#[tokio::test]
async fn test_high_concurrency_stress() {
    println!("\n=== High Concurrency Stress Test ===");

    let (storage, _temp_dir) = create_storage(WalConsistencyMode::Eventual).await;

    let total_operations = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    let start = std::time::Instant::now();

    // 启动 200 个并发任务
    let mut handles = Vec::new();
    for thread_id in 0..200 {
        let storage = Arc::clone(&storage);
        let total_operations = Arc::clone(&total_operations);
        let errors = Arc::clone(&errors);

        let handle = tokio::spawn(async move {
            for i in 0..5 {
                let record_id = format!("stress_{}_{}", thread_id, i);
                let records = vec![CreateRecordRequest {
                    id: record_id,
                    vector: vec![(thread_id * 5 + i) as f32; 128],
                    payload: None,
                    sparse_vector: None,
                }];

                match storage.insert_records("concurrent_test", records).await {
                    Ok(_) => {
                        total_operations.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("Error in thread {}: {:?}", thread_id, e);
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // 等待所有任务完成
    for handle in handles {
        handle.await.unwrap();
    }

    let duration = start.elapsed();

    // 等待 WAL 持久化
    tokio::time::sleep(Duration::from_millis(100)).await;

    let ops = total_operations.load(Ordering::Relaxed);
    let errs = errors.load(Ordering::Relaxed);
    let throughput = ops as f64 / duration.as_secs_f64();

    println!("Total operations: {}", ops);
    println!("Errors: {}", errs);
    println!("Duration: {:?}", duration);
    println!("Throughput: {:.2} ops/s", throughput);

    assert_eq!(ops, 1000, "All operations should succeed");
    assert_eq!(errs, 0, "No errors should occur");
}

#[tokio::test]
async fn test_wal_ordering_consistency() {
    println!("\n=== WAL Ordering Consistency Test ===");

    let (storage, _temp_dir) = create_storage(WalConsistencyMode::Strong).await;

    // 顺序插入
    for i in 0..100 {
        let records = vec![CreateRecordRequest {
            id: format!("ordered_record_{}", i),
            vector: vec![i as f32; 128],
            payload: Some(serde_json::json!({"sequence": i})),
            sparse_vector: None,
        }];
        storage.insert_records("concurrent_test", records).await.unwrap();
    }

    // 验证: 所有记录都应该可见
    let table = storage.get_table("concurrent_test").await.unwrap();
    println!("Total records in table: {}", table.record_count);

    // Strong 模式下，所有记录应该立即可见且已持久化
    assert!(
        table.record_count >= 100,
        "Strong consistency: all records should be visible immediately"
    );
}
