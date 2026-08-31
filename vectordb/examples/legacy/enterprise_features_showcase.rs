use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use vectordb::{
    // Core types
    AppConfig,
    DataEncryption,
    DataGovernance,
    DataSubject,
    EmbeddingModelConfig,
    EmbeddingModelRegistry,
    EncryptionConfig,
    EntMonitoringConfig,
    EnterpriseMonitoring,
    GdprCompliance,
    GdprConfig,
    GovernanceConfig,
    IndexOptimizer,
    ModelType,
    OptimizationConfig,
    // Enterprise features
    RateLimiter,
    RateLimiterConfig,
    Record,
    RetentionConfig,
    RetentionManager,
    Table,
    Vector,

    VectorOperations,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VectorDB Enterprise Features Showcase");
    println!("==========================================\n");

    // 1. Rate Limiting Demo
    showcase_rate_limiting().await?;

    // 2. ML Integration Demo
    showcase_ml_integration().await?;

    // 3. Security & Encryption Demo
    showcase_security_encryption().await?;

    // 4. Compliance & Governance Demo
    showcase_compliance_governance().await?;

    // 5. Enterprise Monitoring Demo
    showcase_enterprise_monitoring().await?;

    println!("\n✅ Enterprise Features Showcase Completed!");
    Ok(())
}

async fn showcase_rate_limiting() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Rate Limiting & API Protection");
    println!("----------------------------------");

    let config = RateLimiterConfig {
        max_requests: 100,
        window_duration: Duration::from_secs(60),
        burst_size: 20,
        per_ip: true,
        per_user: false,
        global_limit: Some(1000),
    };

    let limiter = Arc::new(RateLimiter::new(config));

    // Simulate API requests
    for i in 1..=25 {
        let allowed = limiter.check_request("192.168.1.100");
        println!(
            "Request {}: {}",
            i,
            if allowed {
                "✅ Allowed"
            } else {
                "❌ Rate Limited"
            }
        );

        if i == 20 {
            println!("   ⚠️  Burst limit reached, subsequent requests will be rate limited");
        }
    }

    println!("✅ Rate limiting demonstration completed\n");
    Ok(())
}

async fn showcase_ml_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 ML Integration & Vector Operations");
    println!("------------------------------------");

    // 1. Embedding Model Registry
    let registry = EmbeddingModelRegistry::new();

    let openai_config = EmbeddingModelConfig {
        name: "text-embedding-3-small".to_string(),
        model_type: ModelType::OpenAI {
            model: "text-embedding-3-small".to_string(),
        },
        dimension: 1536,
        context_length: 8191,
        endpoint: None,
        api_key: Some("your-api-key".to_string()),
        parameters: HashMap::new(),
    };

    let model = registry.create_from_config(openai_config).await?;
    registry
        .register_model("openai-small".to_string(), model)
        .await?;

    let models = registry.list_models().await;
    println!("📋 Registered embedding models: {}", models.len());
    for model in models {
        println!("   • {} ({}D)", model.name, model.dimension);
    }

    // 2. Vector Operations
    println!("\n🧮 Advanced Vector Operations:");

    let vectors = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![2.0, 3.0, 4.0],
    ];

    // Centroid calculation
    let centroid = VectorOperations::centroid(&vectors);
    println!("   • Centroid: {:?}", centroid);

    // Similarity calculations
    let sim = VectorOperations::cosine_similarity(&vectors[0], &vectors[1]);
    println!("   • Cosine similarity: {:.4}", sim);

    // K-means clustering
    let kmeans_result = VectorOperations::kmeans(&vectors, 2, 10);
    println!(
        "   • K-means clusters: {} centroids, {} iterations",
        kmeans_result.centroids.len(),
        kmeans_result.iterations
    );

    // 3. Index Optimization
    let opt_config = OptimizationConfig::default();
    let mut optimizer = IndexOptimizer::new(opt_config);

    // Simulate some performance metrics
    use vectordb::ml::index_optimization::IndexMetrics;
    let metrics = IndexMetrics {
        qps: 800.0, // Below target
        p50_latency_ms: 25.0,
        p95_latency_ms: 120.0, // Above target
        p99_latency_ms: 200.0,
        memory_usage_gb: 6.0,
        recall: 0.92, // Below target
        index_size_mb: 2048.0,
        build_time_ms: 15000.0,
    };

    optimizer.record_metrics(metrics);

    if optimizer.should_optimize() {
        if let Some(recommendations) = optimizer.generate_recommendations() {
            println!(
                "   • Optimization recommended: {}",
                recommendations.reasoning
            );
            println!(
                "   • Confidence: {:.2}%",
                recommendations.confidence_score * 100.0
            );
        }
    }

    println!("✅ ML integration demonstration completed\n");
    Ok(())
}

async fn showcase_security_encryption() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 Security & Encryption");
    println!("------------------------");

    // Data encryption
    let enc_config = EncryptionConfig::default();
    let mut encryption = DataEncryption::new(enc_config)?;

    let sensitive_data = b"Personal information: John Doe, SSN: 123-45-6789";
    println!(
        "📄 Original data: {}",
        String::from_utf8_lossy(sensitive_data)
    );

    // Encrypt data
    let encrypted = encryption.encrypt(sensitive_data)?;
    println!("🔒 Encrypted data size: {} bytes", encrypted.data.len());
    println!("🔑 Key ID: {}", encrypted.key_id);

    // Decrypt data
    let decrypted = encryption.decrypt(&encrypted)?;
    println!("🔓 Decrypted data: {}", String::from_utf8_lossy(&decrypted));

    // Key rotation
    println!("\n🔄 Key Rotation:");
    let old_key_count = encryption.list_keys().len();
    encryption.rotate_keys()?;
    let new_key_count = encryption.list_keys().len();
    println!("   • Keys before rotation: {}", old_key_count);
    println!("   • Keys after rotation: {}", new_key_count);

    // Password security
    use vectordb::security::encryption::PasswordSecurity;
    let password = "secure_password_123!";
    let hash = PasswordSecurity::hash_password(password)?;
    let is_valid = PasswordSecurity::verify_password(password, &hash)?;
    println!(
        "   • Password hashing: {}",
        if is_valid { "✅ Valid" } else { "❌ Invalid" }
    );

    println!("✅ Security & encryption demonstration completed\n");
    Ok(())
}

async fn showcase_compliance_governance() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️ Compliance & Data Governance");
    println!("-------------------------------");

    // 1. GDPR Compliance
    let gdpr_config = GdprConfig::default();
    let mut gdpr = GdprCompliance::new(gdpr_config);

    // Register a data subject
    let subject_id = Uuid::new_v4();
    let subject = DataSubject {
        id: subject_id,
        external_id: "user123".to_string(),
        email: Some("john.doe@example.com".to_string()),
        name: Some("John Doe".to_string()),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        jurisdiction: "EU".to_string(),
    };

    println!("👤 GDPR Compliance:");
    println!(
        "   • Registered data subject: {}",
        subject.name.as_ref().unwrap()
    );

    // Subject access request
    let access_response = gdpr.handle_subject_access_request(subject_id)?;
    println!(
        "   • Subject access request: {} records found",
        access_response.processing_records.len()
    );

    // 2. Data Governance
    let gov_config = GovernanceConfig::default();
    let mut governance = DataGovernance::new(gov_config);

    // Data classification
    use vectordb::compliance::data_governance::{
        ClassificationLevel, DataCategory, DataClassification,
    };
    let table_id = Uuid::new_v4();
    let classification = DataClassification {
        id: Uuid::new_v4(),
        table_id,
        field_path: None,
        classification: ClassificationLevel::Confidential,
        categories: vec![DataCategory::PersonalIdentifiable],
        sensitivity_score: 0.8,
        personal_data: true,
        special_category: false,
        retention_period: Some(365),
        access_restrictions: vec![],
        classified_at: chrono::Utc::now(),
        classified_by: "system".to_string(),
        confidence_score: 0.95,
    };

    governance.classify_data(classification)?;
    println!("   • Data classification: Confidential level assigned");

    let report = governance.generate_governance_report();
    println!(
        "   • Governance score: {:.1}%",
        report.governance_score * 100.0
    );

    // 3. Data Retention
    let retention_config = RetentionConfig::default();
    let retention_manager = RetentionManager::new(retention_config);

    let stats = retention_manager.get_statistics();
    println!(
        "   • Retention policies: {} total, {} active",
        stats.total_policies, stats.active_policies
    );

    println!("✅ Compliance & governance demonstration completed\n");
    Ok(())
}

async fn showcase_enterprise_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Enterprise Monitoring & Alerting");
    println!("-----------------------------------");

    let monitoring_config = EntMonitoringConfig::default();
    let mut monitoring = EnterpriseMonitoring::new(monitoring_config);

    // Start monitoring
    monitoring.start().await?;

    // Get monitoring status
    let status = monitoring.get_status().await;
    println!("📈 Monitoring Status:");
    println!(
        "   • Enabled: {}",
        if status.enabled { "✅ Yes" } else { "❌ No" }
    );
    println!("   • Active alerts: {}", status.active_alerts);
    println!("   • Total metrics: {}", status.total_metrics);
    println!("   • Dashboards: {}", status.dashboards);
    println!("   • SLA compliance: {:.1}%", status.sla_compliance * 100.0);

    // Simulate some metrics collection
    println!("\n📊 Metrics Collection:");

    // You would typically access the metrics collector from the monitoring system
    // For demonstration, we'll show what metrics might look like
    let sample_metrics = vec![
        ("cpu_usage_percent", 75.2),
        ("memory_usage_percent", 68.5),
        ("query_latency_p95_ms", 45.3),
        ("throughput_qps", 1250.0),
        ("error_rate_percent", 0.1),
    ];

    for (metric, value) in sample_metrics {
        println!("   • {}: {}", metric, value);
    }

    println!("✅ Enterprise monitoring demonstration completed\n");
    Ok(())
}

// Helper function to create a sample table
fn create_sample_table() -> Table {
    use vectordb::types::{DistanceMetric, IndexType, TableSchema};

    Table {
        id: Uuid::new_v4(),
        name: "sample_table".to_string(),
        dimension: 384,
        distance_metric: DistanceMetric::Cosine,
        index_type: IndexType::Hnsw,
        schema: TableSchema::Exploratory,
        record_count: 0,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    }
}

// Helper function to create sample records
fn create_sample_records(count: usize, dimension: usize) -> Vec<Record> {
    (0..count)
        .map(|i| Record {
            id: format!("record_{}", i),
            vector: Vector::Dense((0..dimension).map(|_| rand::random::<f32>()).collect()),
            metadata: Some(serde_json::json!({
                "index": i,
                "category": format!("category_{}", i % 3),
                "timestamp": chrono::Utc::now().to_rfc3339()
            })),
        })
        .collect()
}
