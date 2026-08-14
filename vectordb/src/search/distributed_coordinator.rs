use futures::stream::{FuturesUnordered, StreamExt};
use crate::types::vector::{RecordId, DenseVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::timeout;

use crate::cluster::{NodeId, NodeInfo, NodeStatus};
use crate::types::{record::SearchHit};

/// Distributed query request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSearchRequest {
    pub query_id: String,
    pub vector: DenseVector,
    pub top_k: usize,
    pub timeout_ms: u64,
    pub partitions: Option<Vec<u32>>,
    pub consistency_level: ConsistencyLevel,
}

/// Consistency levels for distributed search
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Return as soon as any node responds
    One,
    /// Wait for majority of nodes
    Quorum,
    /// Wait for all nodes (highest accuracy)
    All,
}

/// Response from a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSearchResponse {
    pub nodeid: NodeId,
    pub hits: Vec<SearchHit>,
    pub latency_ms: u64,
}

/// Aggregated distributed search result
#[derive(Debug, Clone)]
pub struct DistributedSearchResult {
    pub hits: Vec<SearchHit>,
    pub nodes_queried: usize,
    pub nodes_responded: usize,
    pub total_latency_ms: u64,
    pub node_latencies: HashMap<NodeId, u64>,
}

/// Distributed query coordinator
pub struct DistributedQueryCoordinator {
    /// Cluster nodes
    nodes: Arc<RwLock<HashMap<NodeId, NodeInfo>>>,
    /// Query routing strategy
    router: Arc<QueryRouter>,
    /// Result aggregator
    aggregator: Arc<ResultAggregator>,
    /// Metrics collector
    metrics: Arc<RwLock<QueryMetrics>>,
}

impl DistributedQueryCoordinator {
    pub fn new(nodes: HashMap<NodeId, NodeInfo>) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(nodes)),
            router: Arc::new(QueryRouter::new()),
            aggregator: Arc::new(ResultAggregator::new()),
            metrics: Arc::new(RwLock::new(QueryMetrics::new())),
        }
    }

    /// Execute distributed search
    pub async fn search(
        &self,
        request: DistributedSearchRequest) -> crate::Result<DistributedSearchResult> {
        let start_time = std::time::Instant::now();

        // Select nodes to query
        let target_nodes = self
            .router
            .select_nodes(&request, &*self.nodes.read().await)
            .await?;

        if target_nodes.is_empty() {
            return Err(crate::Error::Internal("No available nodes for search".to_string()));
        }

        // Create futures for parallel node queries
        let futures = FuturesUnordered::new();
        let (tx, mut rx) = tokio::sync::mpsc::channel(target_nodes.len());

        for node_id in &target_nodes {
            let node = self.nodes.read().await.get(node_id).cloned();
            if let Some(node) = node {
                let tx = tx.clone();
                let request = request.clone();
                let node_id = node_id.clone();

                futures.push(tokio::spawn(async move {
                    let result = match Self::query_node(node, request).await {
                        Ok(res) => res,
                        Err(e) => return Err(crate::Error::Internal(format!("Node query failed: {:?}", e))),
                    };
                    match tx.send((node_id, result)).await {
                        Ok(_) => Ok(()),
                        Err(e) => Err(crate::Error::Internal(format!("Channel send failed: {:?}", e))),
                    }
                }))
            }
        }

        drop(tx); // Close sender

        // Collect responses based on consistency level
        let mut responses = Vec::new();
        let mut node_latencies = HashMap::new();
        let requiredresponses =
            self.calculate_requiredresponses(target_nodes.len(), request.consistency_level);

        // Set timeout
        let timeout_duration = Duration::from_millis(request.timeout_ms);

        let timeout_result = timeout(timeout_duration, async {
            while let Some((node_id, result)) = rx.recv().await {
                node_latencies.insert(node_id.clone(), result.latency_ms);
                responses.push(result);

                if responses.len() >= requiredresponses {
                    break;
                }
            }
        }).await;

        match timeout_result {
            Ok(_) => {},
            Err(_) => {
                if responses.is_empty() {
                    return Err(crate::Error::Timeout(
                        "Distributed search timed out".to_string()
                    ))
                }
            }
        }

        // Aggregate results
        let aggregated_hits = self
            .aggregator
            .aggregate_results(&responses, request.top_k)
            .await;

        let total_latency = start_time.elapsed().as_millis() as u64;

        // Update metrics
        self.metrics.write().await.record_query(
            &request.query_id,
            target_nodes.len(),
            responses.len(),
            total_latency,
        );

        Ok(DistributedSearchResult {
            hits: aggregated_hits,
            nodes_queried: target_nodes.len(),
            nodes_responded: responses.len(),
            total_latency_ms: total_latency,
            node_latencies,
        })
    }

    /// Query a single node
    async fn query_node(
        node: NodeInfo,
        request: DistributedSearchRequest) -> crate::Result<NodeSearchResponse> {
        let start_time = std::time::Instant::now();

        // In production, this would make an RPC call to the node
        // For now, simulate with a local search
        let hits = Self::simulate_node_search(&request.vector, request.top_k).await?;

        let latency_ms = start_time.elapsed().as_millis() as u64;

        Ok(NodeSearchResponse {
            nodeid: node.id.clone(),
            hits,
            latency_ms,
        })
    }

    /// Simulate node search (would be RPC in production)
    async fn simulate_node_search(vector: &DenseVector, k: usize) -> crate::Result<Vec<SearchHit>> {
        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Generate mock results
        Ok((0..k)
            .map(|i| SearchHit {
                id: format!("doc_{}", i),
                score: 1.0 / (1.0 + 0.1 * i as f32),
                payload: None,
                vector: None,
            })
            .collect())
    }

    /// Calculate required responses based on consistency level
    fn calculate_requiredresponses(&self, total_nodes: usize, level: ConsistencyLevel) -> usize {
        match level {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => (total_nodes / 2) + 1,
            ConsistencyLevel::All => total_nodes,
        }
    }

    /// Update node availability
    pub async fn update_node(&self, nodeid: NodeId, node: NodeInfo) {
        self.nodes.write().await.insert(nodeid, node);
    }

    /// Remove unavailable node
    pub async fn remove_node(&self, nodeid: &NodeId) {
        self.nodes.write().await.remove(nodeid);
    }

    /// Get query metrics
    pub async fn get_metrics(&self) -> QueryMetrics {
        self.metrics.read().await.clone()
    }
}

/// Query routing strategy
pub struct QueryRouter {
    strategy: RoutingStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum RoutingStrategy {
    /// Route to all nodes
    Broadcast,
    /// Route based on data partitioning
    Partitioned,
    /// Route to nearest nodes
    Proximity,
    /// Route based on load
    LoadBalanced,
}

impl Default for QueryRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryRouter {
    pub fn new() -> Self {
        Self {
            strategy: RoutingStrategy::Broadcast,
        }
    }

    pub fn with_strategy(strategy: RoutingStrategy) -> Self {
        Self { strategy }
    }

    /// Select nodes for query
    async fn select_nodes(
        &self,
        request: &DistributedSearchRequest,
        nodes: &HashMap<NodeId, NodeInfo>,
    ) -> crate::Result<Vec<NodeId>> {
        let available_nodes: Vec<_> = nodes
            .iter()
            .filter(|(_, node)| node.status == NodeStatus::Active)
            .map(|(id, _)| id.clone())
            .collect();

        if available_nodes.is_empty() {
            return Err(crate::Error::Internal("No healthy nodes available".to_string()));
        }

        match self.strategy {
            RoutingStrategy::Broadcast => Ok(available_nodes),
            RoutingStrategy::Partitioned => {
                // Select nodes based on partitions
                if let Some(partitions) = &request.partitions {
                    Ok(self.select_by_partitions(&available_nodes, partitions))
                } else {
                    Ok(available_nodes)
                }
            }
            RoutingStrategy::Proximity => {
                // Select nearest nodes (simplified)
                Ok(available_nodes.into_iter().take(3).collect())
            }
            RoutingStrategy::LoadBalanced => {
                // Select least loaded nodes (simplified)
                Ok(available_nodes.into_iter().take(5).collect())
            }
        }
    }

    fn select_by_partitions(&self, nodes: &[NodeId], partitions: &[u32]) -> Vec<NodeId> {
        // Simple partition mapping
        nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| partitions.contains(&(*i as u32)))
            .map(|(_, id)| id.clone())
            .collect()
    }
}

/// Result aggregation strategies
pub struct ResultAggregator {
    strategy: AggregationStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregationStrategy {
    /// Simple merge and sort
    MergeSort,
    /// Weighted by node reliability
    WeightedMerge,
    /// Reciprocal rank fusion
    ReciprocalRankFusion,
}

impl Default for ResultAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultAggregator {
    pub fn new() -> Self {
        Self {
            strategy: AggregationStrategy::MergeSort,
        }
    }

    /// Aggregate results from multiple nodes
    async fn aggregate_results(
        &self,
        responses: &[NodeSearchResponse],
        top_k: usize) -> Vec<SearchHit> {
        match self.strategy {
            AggregationStrategy::MergeSort => self.merge_sort_aggregate(responses, top_k),
            AggregationStrategy::WeightedMerge => self.weighted_merge_aggregate(responses, top_k),
            AggregationStrategy::ReciprocalRankFusion => self.rrf_aggregate(responses, top_k),
        }
    }

    fn merge_sort_aggregate(
        &self,
        responses: &[NodeSearchResponse],
        top_k: usize) -> Vec<SearchHit> {
        let mut all_hits = Vec::new();

        for response in responses {
            all_hits.extend(response.hits.clone())
        }

        // Remove duplicates and sort
        let mut seen = std::collections::HashSet::new();
        all_hits.retain(|hit| seen.insert(hit.id.clone()));
        all_hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_hits.truncate(top_k);

        all_hits
    }

    fn weighted_merge_aggregate(
        &self,
        responses: &[NodeSearchResponse],
        top_k: usize) -> Vec<SearchHit> {
        // Weight by inverse latency (faster nodes get higher weight)
        let total_latency: f32 = responses.iter().map(|r| r.latency_ms as f32).sum();

        let mut weighted_scores: HashMap<String, f32> = HashMap::new();

        for response in responses {
            let weight = 1.0 - (response.latency_ms as f32 / total_latency);

            for hit in &response.hits {
                *weighted_scores.entry(hit.id.clone()).or_insert(0.0) += hit.score * weight
            }
        }

        // Convert to hits and sort
        let mut hits: Vec<SearchHit> = weighted_scores
            .into_iter()
            .map(|(id, score)| SearchHit {
                id: RecordId::from(id),
                score,
                payload: None,
                vector: None,
            }) 
                    .collect();

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(top_k);

        hits
    }

    fn rrf_aggregate(&self, responses: &[NodeSearchResponse], top_k: usize) -> Vec<SearchHit> {
        let k = 60.0; // RRF constant
        let mut rrf_scores: HashMap<String, f32> = HashMap::new();

        for response in responses {
            for (rank, hit) in response.hits.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                *rrf_scores.entry(hit.id.clone()).or_insert(0.0) += rrf_score
            }
        }

        // Convert to hits
        let mut hits: Vec<SearchHit> = rrf_scores
            .into_iter()
            .map(|(id, score)| SearchHit {
                id: RecordId::from(id),
                score,
                payload: None,
                vector: None,
            }) 
                    .collect();

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(top_k);

        hits
    }
}

/// Query metrics collection
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    pub total_queries: u64,
    pub total_latency_ms: u64,
    pub node_query_counts: HashMap<NodeId, u64>,
    pub failed_queries: u64,
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryMetrics {
    pub fn new() -> Self {
        Self {
            total_queries: 0,
            total_latency_ms: 0,
            node_query_counts: HashMap::new(),
            failed_queries: 0,
        }
    }

    pub fn record_query(
        &mut self,
        query_id: &str,
        nodes_queried: usize,
        nodes_responded: usize,
        latency_ms: u64) {
        self.total_queries += 1;
        self.total_latency_ms += latency_ms;

        if nodes_responded < nodes_queried {
            self.failed_queries += (nodes_queried - nodes_responded) as u64
        }
    }

    pub fn average_latency_ms(&self) -> f64 {
        if self.total_queries > 0 {
            self.total_latency_ms as f64 / self.total_queries as f64
        } else {
            0.0
        }
    }
}

/// Adaptive load balancer
pub struct AdaptiveLoadBalancer {
    node_loads: Arc<RwLock<HashMap<NodeId, NodeLoad>>>,
    window_size: Duration,
}

#[derive(Debug, Clone)]
struct NodeLoad {
    queries_per_second: f64,
    average_latency_ms: f64,
    error_rate: f64,
    last_updated: std::time::Instant,
}

impl Default for AdaptiveLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveLoadBalancer {
    pub fn new() -> Self {
        Self {
            node_loads: Arc::new(RwLock::new(HashMap::new())),
            window_size: Duration::from_secs(60),
        }
    }

    /// Select nodes based on current load
    pub async fn select_nodes(&self, n: usize) -> Vec<NodeId> {
        let loads = self.node_loads.read().await;

        // Score nodes (lower is better)
        let mut scored_nodes: Vec<(NodeId, f64)> = loads
            .iter()
            .map(|(id, load)| {
                let score = load.queries_per_second * 0.3
                    + load.average_latency_ms * 0.5
                    + load.error_rate * 100.0;
                (id.clone(), score)
            }) 
                    .collect();

        // Sort by score
        scored_nodes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top n nodes
        scored_nodes.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Update node load information
    pub async fn update_load(&self, nodeid: NodeId, load: NodeLoad) {
        self.node_loads.write().await.insert(nodeid, load);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::RecordId;


    #[tokio::test]
    async fn test_distributed_coordinator() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "node1".to_string(),
            NodeInfo {
                id: "node1".to_string(),
                address: "127.0.0.1:8001".to_string(),
                is_healthy: true,
                role: crate::cluster::NodeRole::Worker,
            },
        );
        nodes.insert(
            "node2".to_string(),
            NodeInfo {
                id: "node2".to_string(),
                address: "127.0.0.1:8002".to_string(),
                is_healthy: true,
                role: crate::cluster::NodeRole::Worker,
            },
        );

        let coordinator = DistributedQueryCoordinator::new(nodes);

        let request = DistributedSearchRequest {
            query_id: "test_query".to_string(),
            vector: vec![1.0; 128],
            top_k: 10,
            timeout_ms: 1000,
            partitions: None,
            consistency_level: _ConsistencyLevel::Quorum,
        };

        let result = coordinator.search(request)?.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        assert!(!result.hits.is_empty());
        assert!(result.nodes_responded > 0);
        assert!(result.nodes_responded <= result.nodes_queried);
    }

    #[test]
    fn test_result_aggregator() {
        let aggregator = ResultAggregator::new();

        let responses = vec![
            NodeSearchResponse {
                nodeid: "node1".to_string(),
                hits: vec![
                    SearchHit {
                        id: _RecordId::from("a".to_string()),
                        score: 0.9,
                        payload: None,
                        vector: None,
                    },
                    SearchHit {
                        id: _RecordId::from("b".to_string()),
                        score: 0.8,
                        payload: None,
                        vector: None,
                    },
                ],
                latency_ms: 10,
            },
            NodeSearchResponse {
                nodeid: "node2".to_string(),
                hits: vec![
                    SearchHit {
                        id: _RecordId::from("a".to_string()),
                        score: 0.85,
                        payload: None,
                        vector: None,
                    },
                    SearchHit {
                        id: _RecordId::from("c".to_string()),
                        score: 0.75,
                        payload: None,
                        vector: None,
                    },
                ],
                latency_ms: 15,
            },
        ];

        let aggregated = aggregator.merge_sort_aggregate(&responses, 3);

        assert_eq!(aggregated.len(), 3);
        assert_eq!(aggregated[0].id, "a");
        assert!(aggregated[0].distance <= aggregated[1].distance);
    }
}
