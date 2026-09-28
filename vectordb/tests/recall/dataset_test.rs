#![cfg(feature = "benchmarks")]

// Dataset infrastructure tests

// Shared module, one directory up now that this test lives in a scenario
// folder. Cargo resolves `mod` relative to the file, so without the path it
// would look for tests/recall/fixtures.rs and fail to compile.
#[path = "../fixtures/mod.rs"]
mod fixtures;

use dxmt_vectordb::benchmarks::datasets::{
    load_fvecs, write_fvecs, Dataset, DatasetCache, DatasetRegistry, DimensionCategory,
    SizeCategory,
};
use std::fs;
use tempfile::{NamedTempFile, TempDir};

#[test]
fn test_synthetic_dataset() {
    let dataset = Dataset::synthetic("test", 64, 100, 10, 10);

    assert_eq!(dataset.name, "test");
    assert_eq!(dataset.dimension, 64);
    assert_eq!(dataset.train_size(), 100);
    assert_eq!(dataset.test_size(), 10);
    assert_eq!(dataset.k(), 10);

    // Verify ground truth structure
    assert_eq!(dataset.ground_truth.len(), 10);
    for neighbors in &dataset.ground_truth {
        assert_eq!(neighbors.len(), 10);
        // Verify all neighbor IDs are valid
        for &neighbor_id in neighbors {
            assert!(neighbor_id < 100);
        }
    }
}

#[test]
fn test_synthetic_dataset_vectors_normalized() {
    let dataset = Dataset::synthetic("test", 32, 10, 5, 5);

    // Check that vectors are normalized (L2 norm ≈ 1.0)
    for vec in &dataset.train_vectors {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Vector not normalized: norm = {}",
            norm
        );
    }

    for vec in &dataset.test_queries {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Vector not normalized: norm = {}",
            norm
        );
    }
}

#[test]
fn test_synthetic_dataset_ground_truth_is_exact_and_unique() {
    let dataset = Dataset::synthetic("ground-truth", 16, 64, 7, 10);

    for (query, neighbors) in dataset.test_queries.iter().zip(dataset.ground_truth.iter()) {
        assert_eq!(neighbors.len(), 10);
        let unique: std::collections::HashSet<usize> = neighbors.iter().copied().collect();
        assert_eq!(
            unique.len(),
            neighbors.len(),
            "ground truth contains duplicates"
        );

        let mut exact: Vec<(usize, f32)> = dataset
            .train_vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                let similarity = query.iter().zip(vector).map(|(a, b)| a * b).sum::<f32>();
                (index, similarity)
            })
            .collect();
        exact.sort_by(|(index_a, score_a), (index_b, score_b)| {
            score_b
                .partial_cmp(score_a)
                .unwrap()
                .then_with(|| index_a.cmp(index_b))
        });
        let expected: Vec<usize> = exact.into_iter().take(10).map(|(index, _)| index).collect();
        assert_eq!(neighbors, &expected);
    }

    // A second invocation must produce byte-for-byte equivalent data.
    let repeat = Dataset::synthetic("ground-truth", 16, 64, 7, 10);
    assert_eq!(dataset.train_vectors, repeat.train_vectors);
    assert_eq!(dataset.test_queries, repeat.test_queries);
    assert_eq!(dataset.ground_truth, repeat.ground_truth);
}

#[test]
fn test_fvecs_loader() {
    // Create test vectors
    let original_vectors = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    // Write to temp file
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_path_buf();

    write_fvecs(&temp_path, &original_vectors).unwrap();

    // Read back
    let loaded_vectors = load_fvecs(&temp_path).unwrap();

    // Verify
    assert_eq!(loaded_vectors.len(), original_vectors.len());
    for (loaded, original) in loaded_vectors.iter().zip(original_vectors.iter()) {
        assert_eq!(loaded.len(), original.len());
        for (l, o) in loaded.iter().zip(original.iter()) {
            assert!((l - o).abs() < 1e-6);
        }
    }
}

#[test]
fn test_dataset_registry() {
    let registry = DatasetRegistry::new();
    let datasets = registry.list_datasets();

    // Should have at least the default datasets
    assert!(datasets.len() >= 6);
    assert!(datasets.contains(&"sift-1k".to_string()));
    assert!(datasets.contains(&"glove-25".to_string()));
    assert!(datasets.contains(&"fashion-mnist".to_string()));
}

#[test]
fn test_registry_get_dataset() {
    let registry = DatasetRegistry::new();
    let sift = registry.get("sift-1k").unwrap();

    assert_eq!(sift.name, "sift-1k");
    assert_eq!(sift.dimension, 128);
    assert_eq!(sift.train_size, 1_000);
    assert_eq!(sift.test_size, 100);
}

#[test]
fn test_registry_filter_by_size() {
    let registry = DatasetRegistry::new();
    let small = registry.by_size(SizeCategory::Small);

    // SIFT-1K should be in small category (< 10K)
    assert!(small.iter().any(|d| d.name == "sift-1k"));

    // Deep-1M should NOT be in small category
    assert!(!small.iter().any(|d| d.name == "deep-1m"));

    // Check very large category
    let very_large = registry.by_size(SizeCategory::VeryLarge);
    assert!(very_large.iter().any(|d| d.name == "deep-1m"));
}

#[test]
fn test_registry_filter_by_dimension() {
    let registry = DatasetRegistry::new();
    let low_dim = registry.by_dimension(DimensionCategory::Low);

    // GloVe-25 should be in low dimension category (< 50)
    assert!(low_dim.iter().any(|d| d.name == "glove-25"));

    // Fashion-MNIST (784D) should NOT be in low dimension
    assert!(!low_dim.iter().any(|d| d.name == "fashion-mnist"));
}

#[test]
fn test_dataset_cache() {
    let temp_dir = TempDir::new().unwrap();
    let cache = DatasetCache::with_directory(temp_dir.path()).unwrap();

    // Create a test file
    let test_data = b"Hello, World! This is test data for caching.";
    let test_file = temp_dir.path().join("test.txt");
    fs::write(&test_file, test_data).unwrap();

    // Cache it
    cache.cache_file(&test_file, "test").unwrap();

    // Verify it's cached
    assert!(cache.is_cached("test"));

    // Read it back
    let cached_data = cache.read_cached("test").unwrap();
    assert_eq!(cached_data, test_data);
}

#[test]
fn test_cache_compression() {
    let temp_dir = TempDir::new().unwrap();
    let cache = DatasetCache::with_directory(temp_dir.path()).unwrap();

    // Create test data (repetitive data compresses well)
    let test_data = "AAAAAAAAAA".repeat(1000);
    let test_file = temp_dir.path().join("test.txt");
    fs::write(&test_file, &test_data).unwrap();

    // Cache it
    cache.cache_file(&test_file, "test-compress").unwrap();

    // Check metadata
    let metadata = cache.get_metadata("test-compress").unwrap();
    println!("Compression ratio: {:.2}x", metadata.compression_ratio);

    // Compression ratio should be > 1 for repetitive data
    assert!(metadata.compression_ratio > 1.0);

    // Read it back and verify
    let cached_data = cache.read_cached("test-compress").unwrap();
    assert_eq!(cached_data, test_data.as_bytes());
}

#[test]
fn test_cache_list() {
    let temp_dir = TempDir::new().unwrap();
    let cache = DatasetCache::with_directory(temp_dir.path()).unwrap();

    // Initially should be empty
    let initial = cache.list_cached().unwrap();
    assert_eq!(initial.len(), 0, "Cache should start empty");

    // Create test files in a separate subdirectory
    let data_dir = temp_dir.path().join("data");
    fs::create_dir(&data_dir).unwrap();

    // Cache multiple files
    for i in 0..3 {
        let test_file = data_dir.join(format!("test{}.txt", i));
        fs::write(&test_file, format!("Test data {}", i)).unwrap();
        cache
            .cache_file(&test_file, &format!("testlist{}", i))
            .unwrap();
    }

    let cached = cache.list_cached().unwrap();
    assert_eq!(cached.len(), 3, "Should have exactly 3 cached files");
    assert!(cached.contains(&"testlist0".to_string()));
    assert!(cached.contains(&"testlist1".to_string()));
    assert!(cached.contains(&"testlist2".to_string()));
}

#[test]
fn test_cache_stats() {
    let temp_dir = TempDir::new().unwrap();
    let cache = DatasetCache::with_directory(temp_dir.path()).unwrap();

    // Initially empty
    let stats = cache.stats().unwrap();
    assert_eq!(stats.file_count, 0, "Cache should start with 0 files");
    assert_eq!(stats.total_size_bytes, 0, "Cache should start with 0 bytes");

    // Create test file in a separate subdirectory to avoid it being counted
    let data_dir = temp_dir.path().join("data");
    fs::create_dir(&data_dir).unwrap();
    let test_file = data_dir.join("test_stats.txt");
    fs::write(&test_file, b"Test data for stats").unwrap();
    cache.cache_file(&test_file, "teststats").unwrap();

    // Check stats
    let stats = cache.stats().unwrap();
    assert_eq!(stats.file_count, 1, "Should have exactly 1 cached file");
    assert!(
        stats.total_size_bytes > 0,
        "Cache should have non-zero size"
    );

    println!("Cache size: {}", stats.format_size());
}

#[test]
fn test_dataset_print_stats() {
    let dataset = Dataset::synthetic("test", 128, 50, 5, 10);
    // Just verify it doesn't panic
    dataset.print_stats();
}
