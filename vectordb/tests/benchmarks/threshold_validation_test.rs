/// Integration tests for performance threshold validation

use dxmt_vectordb::benchmarks::thresholds::{
    PerformanceThreshold, ThresholdValidator, ValidationMetrics,
};

#[test]
fn test_hnsw_meets_threshold() {
    let validator = ThresholdValidator::new();

    // Good HNSW performance
    let metrics = ValidationMetrics {
        recall_at_k: 0.97,
        qps: 15_000.0,
        p99_latency_ms: 7.0,
        precision_at_k: Some(0.97),
        ndcg_at_k: Some(0.96),
    };

    let result = validator.validate("HNSW", metrics);
    assert!(result.passed, "HNSW should meet threshold");
    assert!(result.failures.is_empty());
}

#[test]
fn test_diskann_meets_threshold() {
    let validator = ThresholdValidator::new();

    // Good DiskANN performance
    let metrics = ValidationMetrics {
        recall_at_k: 0.92,
        qps: 6_500.0,
        p99_latency_ms: 45.0,
        precision_at_k: Some(0.91),
        ndcg_at_k: Some(0.90),
    };

    let result = validator.validate("DiskANN", metrics);
    assert!(result.passed, "DiskANN should meet threshold");
}

#[test]
fn test_hnsw_below_recall_threshold() {
    let validator = ThresholdValidator::new();

    let metrics = ValidationMetrics {
        recall_at_k: 0.92,  // Below 0.95 threshold
        qps: 15_000.0,
        p99_latency_ms: 7.0,
        precision_at_k: Some(0.92),
        ndcg_at_k: Some(0.92),
    };

    let result = validator.validate("HNSW", metrics);
    assert!(!result.passed, "HNSW should fail recall threshold");
    assert_eq!(result.failures.len(), 2); // Recall + Precision
    assert!(result.failures[0].contains("Recall"));
}

#[test]
fn test_hnsw_below_qps_threshold() {
    let validator = ThresholdValidator::new();

    let metrics = ValidationMetrics {
        recall_at_k: 0.97,
        qps: 8_000.0,  // Below 10,000 threshold
        p99_latency_ms: 7.0,
        precision_at_k: Some(0.97),
        ndcg_at_k: Some(0.96),
    };

    let result = validator.validate("HNSW", metrics);
    assert!(!result.passed, "HNSW should fail QPS threshold");
    assert!(result.failures.iter().any(|f| f.contains("QPS")));
}

#[test]
fn test_hnsw_exceeds_latency_threshold() {
    let validator = ThresholdValidator::new();

    let metrics = ValidationMetrics {
        recall_at_k: 0.97,
        qps: 15_000.0,
        p99_latency_ms: 15.0,  // Above 10ms threshold
        precision_at_k: Some(0.97),
        ndcg_at_k: Some(0.96),
    };

    let result = validator.validate("HNSW", metrics);
    assert!(!result.passed, "HNSW should fail latency threshold");
    assert!(result.failures.iter().any(|f| f.contains("latency")));
}

#[test]
fn test_batch_validation_mixed_results() {
    let validator = ThresholdValidator::new();

    let results = vec![
        // HNSW passes
        ("HNSW".to_string(), ValidationMetrics {
            recall_at_k: 0.97,
            qps: 15_000.0,
            p99_latency_ms: 7.0,
            precision_at_k: Some(0.97),
            ndcg_at_k: Some(0.96),
        }),
        // DiskANN fails (low recall)
        ("DiskANN".to_string(), ValidationMetrics {
            recall_at_k: 0.85,  // Below 0.90 threshold
            qps: 6_500.0,
            p99_latency_ms: 45.0,
            precision_at_k: None,
            ndcg_at_k: None,
        }),
    ];

    let validations = validator.validate_batch(results);

    assert_eq!(validations.len(), 2);
    assert!(validations[0].passed, "HNSW should pass");
    assert!(!validations[1].passed, "DiskANN should fail");
}

#[test]
fn test_custom_threshold() {
    let mut validator = ThresholdValidator::new();

    // Set stricter threshold for testing
    let custom_threshold = PerformanceThreshold {
        index_type: "CustomIndex".to_string(),
        min_recall_at_k: 0.99,
        recall_k: 10,
        min_qps: 20_000.0,
        max_p99_latency_ms: 5.0,
        min_precision_at_k: Some(0.99),
        min_ndcg_at_k: Some(0.99),
    };

    validator.set_threshold("CustomIndex".to_string(), custom_threshold);

    // Metrics that would pass HNSW but fail custom
    let metrics = ValidationMetrics {
        recall_at_k: 0.97,  // Below 0.99
        qps: 15_000.0,      // Below 20,000
        p99_latency_ms: 7.0, // Above 5.0
        precision_at_k: Some(0.97),
        ndcg_at_k: Some(0.96),
    };

    let result = validator.validate("CustomIndex", metrics);
    assert!(!result.passed, "Should fail custom threshold");
    assert!(result.failures.len() >= 3); // Recall, QPS, Latency
}

#[test]
fn test_brute_force_exact_search() {
    let validator = ThresholdValidator::new();

    // BruteForce should have perfect recall
    let metrics = ValidationMetrics {
        recall_at_k: 1.0,
        qps: 150.0,
        p99_latency_ms: 500.0,
        precision_at_k: Some(1.0),
        ndcg_at_k: Some(1.0),
    };

    let result = validator.validate("BruteForce", metrics);
    assert!(result.passed, "BruteForce should meet threshold");
}

#[test]
fn test_json_serialization_roundtrip() {
    let validator = ThresholdValidator::new();

    // Export to JSON
    let json = validator.to_json().expect("Should serialize to JSON");
    assert!(json.contains("HNSW"));
    assert!(json.contains("min_recall_at_k"));

    // Import from JSON
    let validator2 = ThresholdValidator::from_json(&json)
        .expect("Should deserialize from JSON");

    // Validate same metrics with both
    let metrics = ValidationMetrics {
        recall_at_k: 0.97,
        qps: 15_000.0,
        p99_latency_ms: 7.0,
        precision_at_k: Some(0.97),
        ndcg_at_k: Some(0.96),
    };

    let result1 = validator.validate("HNSW", metrics.clone());
    let result2 = validator2.validate("HNSW", metrics);

    assert_eq!(result1.passed, result2.passed);
}

#[test]
fn test_all_index_types_have_thresholds() {
    let validator = ThresholdValidator::new();

    let index_types = vec!["HNSW", "DiskANN", "IVF-PQ", "HNSW-PQ", "BruteForce"];

    for index_type in index_types {
        assert!(
            validator.get_threshold(index_type).is_some(),
            "Missing threshold for {}",
            index_type
        );
    }
}

#[test]
fn test_validation_report_printing() {
    let validator = ThresholdValidator::new();

    let metrics = ValidationMetrics {
        recall_at_k: 0.92,
        qps: 8_000.0,
        p99_latency_ms: 15.0,
        precision_at_k: Some(0.93),
        ndcg_at_k: Some(0.91),
    };

    let result = validator.validate("HNSW", metrics);

    // Should not panic
    result.print_report();
}
