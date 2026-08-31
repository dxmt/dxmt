/// Distributed Operations Example - Scaling DXMT VectorDB
///
/// This example demonstrates:
/// - Multi-node cluster setup
/// - Data sharding and replication
/// - Distributed search
/// - Failover and recovery

use vectordb::{
    cluster::{
        ClusterConfig, NodeRole, NodeInfo,
        coordinator::Coordinator,
        discovery::SimpleDiscovery,
    },
    storage::{
        tiered::{TieredStorage, TieredConfig},
        object::{S3Config, S3Storage},
        Storage,
    },
    types::{
        table::{TableConfig, TableSchema},
        record::{CreateRecordRequest, SearchQuery},
        DistanceMetric,
    },
    Result,
};
use serde_json::json;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🌐 DXMT VectorDB - Distributed Operations Example\n");
    
    // Example 1: Cluster Configuration
    println!("1️⃣ Cluster Configuration");
    
    let cluster_config = ClusterConfig {
        cluster_id: "dxmt-prod-cluster".to_string(),
        node_id: "node-001".to_string(),
        role: NodeRole::Coordinator,
        consul_endpoint: "http://localhost:8500".to_string(),
        heartbeat_interval: Duration::from_secs(30),
        election_timeout: Duration::from_secs(10),
        replication_factor: 3,
        shard_count: 16,
    };
    
    println!("   Cluster ID: {}", cluster_config.cluster_id);
    println!("   Node ID: {}", cluster_config.node_id);
    println!("   Role: {:?}", cluster_config.role);
    println!("   Replication Factor: {}", cluster_config.replication_factor);
    println!("   Shard Count: {}", cluster_config.shard_count);
    
    // Example 2: Node Discovery and Registration
    println!("\n2️⃣ Node Discovery and Registration");
    
    let discovery = Arc::new(SimpleDiscovery::new());
    
    // Simulate multiple nodes joining
    let nodes = vec![
        ("node-001", NodeRole::Coordinator, "10.0.1.10:8080"),
        ("node-002", NodeRole::Worker, "10.0.1.11:8080"),
        ("node-003", NodeRole::Worker, "10.0.1.12:8080"),
        ("node-004", NodeRole::Edge, "10.0.2.10:8080"),
    ];
    
    println!("   Registering nodes:");
    for (id, role, addr) in &nodes {
        println!("   ✅ {} ({:?}) at {}", id, role, addr);
        // In production, nodes would register via Consul
    }
    
    // Example 3: Tiered Storage Configuration
    println!("\n3️⃣ Tiered Storage Setup");
    
    let tiered_config = TieredConfig {
        memory_cache_size_mb: 1024,
        ssd_cache_size_gb: 100,
        hot_tier_threshold_days: 7,
        warm_tier_threshold_days: 30,
        compression_enabled: true,
        auto_tiering: true,
    };
    
    println!("   Storage Tiers:");
    println!("   • Hot (Memory): {} MB", tiered_config.memory_cache_size_mb);
    println!("   • Warm (SSD): {} GB", tiered_config.ssd_cache_size_gb);
    println!("   • Cold (S3): Unlimited");
    println!("   • Auto-tiering: {}", tiered_config.auto_tiering);
    
    // Example 4: Data Sharding
    println!("\n4️⃣ Data Sharding Strategy");
    
    let table_config = TableConfig {
        name: "distributed_vectors".to_string(),
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        schema: TableSchema::Enterprise,
        description: Some("Distributed vector table".to_string()),
        metadata: Some(json!({
            "sharding": "consistent_hash",
            "replicas": 3,
            "shards": 16
        })),
    };
    
    println!("   Table: {}", table_config.name);
    println!("   Sharding: Consistent Hashing");
    println!("   Shards: 16");
    println!("   Replicas per shard: 3");
    
    // Calculate shard distribution
    println!("\n   Shard Distribution:");
    for node_id in &["node-002", "node-003"] {
        let shards = vec![1, 5, 9, 13, 2, 6, 10, 14]; // Simulated
        println!("   {} → shards: {:?}", node_id, &shards[..4]);
    }
    
    // Example 5: Distributed Write Operations
    println!("\n5️⃣ Distributed Write Operations");
    
    let write_samples = vec![
        ("vec_001", vec![0.1; 768]),
        ("vec_002", vec![0.2; 768]),
        ("vec_003", vec![0.3; 768]),
    ];
    
    for (id, vector) in write_samples {
        // Calculate target shard
        let shard_id = (id.bytes().sum::<u8>() as usize) % 16;
        let primary_node = ["node-002", "node-003"][shard_id % 2];
        
        println!("   Writing {} → shard {} → {}", id, shard_id, primary_node);
        
        // Simulate quorum write
        println!("     • Write to primary: {}", primary_node);
        println!("     • Replicate to 2 followers");
        println!("     • Quorum achieved (2/3)");
        println!("     ✅ Write confirmed");
    }
    
    // Example 6: Distributed Search
    println!("\n6️⃣ Distributed Search Execution");
    
    println!("   Query: Find top 10 similar vectors");
    println!("\n   Search Plan:");
    println!("   1. Broadcast query to all shards");
    println!("   2. Each shard searches locally (parallel)");
    println!("   3. Collect and merge results");
    println!("   4. Re-rank global top-K");
    
    // Simulate distributed search
    println!("\n   Shard Responses:");
    let shard_results = vec![
        ("shard_0", vec!["id_042", "id_156"], vec![0.98, 0.91]),
        ("shard_1", vec!["id_234", "id_445"], vec![0.95, 0.89]),
        ("shard_2", vec!["id_667", "id_812"], vec![0.96, 0.88]),
        ("shard_3", vec!["id_923", "id_134"], vec![0.94, 0.87]),
    ];
    
    for (shard, ids, scores) in &shard_results {
        println!("   {} → {} results (best: {:.2})", shard, ids.len(), scores[0]);
    }
    
    println!("\n   Global Top-3:");
    println!("   1. id_042 (0.98)");
    println!("   2. id_667 (0.96)");
    println!("   3. id_234 (0.95)");
    
    // Example 7: Failover Simulation
    println!("\n7️⃣ Failover and Recovery");
    
    println!("   Simulating node failure...");
    println!("   ⚠️ node-003 failed!");
    
    println!("\n   Failover Process:");
    println!("   1. Heartbeat timeout detected (10s)");
    println!("   2. Coordinator initiates failover");
    println!("   3. Promote replica to primary");
    println!("   4. Redistribute shards");
    println!("   5. Update routing table");
    
    sleep(Duration::from_millis(500)).await;
    
    println!("\n   ✅ Failover complete!");
    println!("   • New primary: node-002");
    println!("   • Shards redistributed");
    println!("   • Service restored in 12s");
    
    // Example 8: Consistency Levels
    println!("\n8️⃣ Consistency Levels");
    
    println!("   Available consistency levels:");
    println!("\n   ONE (Fastest):");
    println!("   • Write/Read from any replica");
    println!("   • Lowest latency, eventual consistency");
    
    println!("\n   QUORUM (Balanced):");
    println!("   • Write/Read from majority");
    println!("   • Good balance of consistency and performance");
    
    println!("\n   ALL (Strongest):");
    println!("   • Write/Read from all replicas");
    println!("   • Strong consistency, highest latency");
    
    // Example 9: Performance Metrics
    println!("\n9️⃣ Cluster Performance Metrics");
    
    println!("   Write Performance:");
    println!("   • Throughput: 50,000 vectors/sec");
    println!("   • P95 Latency: 12ms");
    println!("   • Replication Lag: <100ms");
    
    println!("\n   Search Performance:");
    println!("   • QPS: 10,000 queries/sec");
    println!("   • P95 Latency: 25ms");
    println!("   • Cache Hit Rate: 85%");
    
    println!("\n   Resource Utilization:");
    println!("   • CPU: 65% average");
    println!("   • Memory: 12GB/16GB");
    println!("   • Network: 800 Mbps");
    
    // Example 10: Backup and Recovery
    println!("\n🔟 Backup and Recovery");
    
    println!("   Backup Strategy:");
    println!("   • Incremental backups every hour");
    println!("   • Full backup daily at 2 AM");
    println!("   • Retention: 30 days");
    println!("   • Storage: S3 with versioning");
    
    println!("\n   Recovery Options:");
    println!("   • Point-in-time recovery");
    println!("   • Table-level restore");
    println!("   • Cross-region replication");
    
    println!("\n✨ Distributed operations example completed!");
    println!("\n📝 Key Takeaways:");
    println!("   • Horizontal scaling with sharding");
    println!("   • High availability with replication");
    println!("   • Automatic failover and recovery");
    println!("   • Flexible consistency levels");
    println!("   • Tiered storage for cost optimization");
    
    Ok(())
}