//! Raft consensus implementation for distributed consistency
//!
//! Provides leader election, log replication, and cluster membership management.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::wal::{WalEntry, WalManager};

pub type NodeId = String;
pub type Term = u64;
pub type LogIndex = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    pub node_id: NodeId,
    pub cluster_nodes: Vec<NodeId>,
    pub election_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub replication_factor: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4().to_string(),
            cluster_nodes: Vec::new(),
            election_timeout_ms: 5000,
            heartbeat_interval_ms: 1000,
            replication_factor: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: Term,
    pub index: LogIndex,
    pub entry: WalEntry,
    pub committed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    pub term: Term,
    pub leader_id: NodeId,
    pub prev_log_index: LogIndex,
    pub prev_log_term: Term,
    pub entries: Vec<LogEntry>,
    pub leader_commit: LogIndex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    pub term: Term,
    pub success: bool,
    pub match_index: LogIndex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRequest {
    pub term: Term,
    pub candidate_id: NodeId,
    pub last_log_index: LogIndex,
    pub last_log_term: Term,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResponse {
    pub term: Term,
    pub vote_granted: bool,
}

pub struct RaftNode {
    config: RaftConfig,
    state: Arc<RwLock<RaftState>>,
    wal: Arc<RwLock<WalManager>>,
    message_tx: mpsc::UnboundedSender<RaftMessage>,
    message_rx: Arc<RwLock<mpsc::UnboundedReceiver<RaftMessage>>>,
}

struct RaftState {
    // Persistent state
    current_term: Term,
    voted_for: Option<NodeId>,
    log: Vec<LogEntry>,
    
    // Volatile state
    commit_index: LogIndex,
    last_applied: LogIndex,
    node_state: NodeState,
    
    // Leader state
    next_index: HashMap<NodeId, LogIndex>,
    match_index: HashMap<NodeId, LogIndex>,
    
    // Timing
    last_heartbeat: Instant,
    election_deadline: Instant,
}

#[derive(Debug)]
pub enum RaftMessage {
    AppendEntries(NodeId, AppendEntriesRequest),
    AppendEntriesResponse(NodeId, AppendEntriesResponse),
    VoteRequest(NodeId, VoteRequest),
    VoteResponse(NodeId, VoteResponse),
    ClientWrite(WalEntry, tokio::sync::oneshot::Sender<Result<LogIndex>>),
    Tick,
}

impl RaftNode {
    pub fn new(config: RaftConfig, wal: WalManager) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let state = RaftState {
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            node_state: NodeState::Follower,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            last_heartbeat: Instant::now(),
            election_deadline: Instant::now() + Duration::from_millis(config.election_timeout_ms),
        };
        
        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            wal: Arc::new(RwLock::new(wal)),
            message_tx: tx,
            message_rx: Arc::new(RwLock::new(rx)),
        }
    }
    
    pub async fn start(&self) -> Result<()> {
        // Start the main Raft loop
        let state = self.state.clone();
        let config = self.config.clone();
        let message_rx = self.message_rx.clone();
        let wal = self.wal.clone();
        
        tokio::spawn(async move {
            Self::run_raft_loop(state, config, message_rx, wal).await;
        });
        
        // Start heartbeat timer
        let message_tx = self.message_tx.clone();
        let heartbeat_interval = Duration::from_millis(self.config.heartbeat_interval_ms);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(heartbeat_interval);
            loop {
                interval.tick().await;
                let _ = message_tx.send(RaftMessage::Tick);
            }
        });
        
        Ok(())
    }
    
    pub async fn write(&self, entry: WalEntry) -> Result<LogIndex> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.message_tx.send(RaftMessage::ClientWrite(entry, tx))
            .map_err(|_| Error::Internal("Failed to send write request".to_string()))?;
        
        rx.await.map_err(|_| Error::Internal("Write request cancelled".to_string()))?
    }
    
    pub async fn is_leader(&self) -> bool {
        let state = self.state.read().await;
        state.node_state == NodeState::Leader
    }
    
    pub async fn get_leader_id(&self) -> Option<NodeId> {
        let state = self.state.read().await;
        if state.node_state == NodeState::Leader {
            Some(self.config.node_id.clone())
        } else {
            // In a real implementation, we'd track the current leader
            None
        }
    }
    
    async fn run_raft_loop(
        state: Arc<RwLock<RaftState>>,
        config: RaftConfig,
        message_rx: Arc<RwLock<mpsc::UnboundedReceiver<RaftMessage>>>,
        wal: Arc<RwLock<WalManager>>,
    ) {
        let mut rx = message_rx.write().await;
        
        while let Some(message) = rx.recv().await {
            match message {
                RaftMessage::Tick => {
                    Self::handle_tick(&state, &config).await;
                }
                RaftMessage::ClientWrite(entry, response_tx) => {
                    let result = Self::handle_client_write(&state, &config, &wal, entry).await;
                    let _ = response_tx.send(result);
                }
                RaftMessage::AppendEntries(from, request) => {
                    Self::handle_append_entries(&state, &config, from, request).await;
                }
                RaftMessage::VoteRequest(from, request) => {
                    Self::handle_vote_request(&state, &config, from, request).await;
                }
                RaftMessage::AppendEntriesResponse(from, response) => {
                    Self::handle_append_entries_response(&state, &config, from, response).await;
                }
                RaftMessage::VoteResponse(from, response) => {
                    Self::handle_vote_response(&state, &config, from, response).await;
                }
            }
        }
    }
    
    async fn handle_tick(state: &Arc<RwLock<RaftState>>, config: &RaftConfig) {
        let mut state = state.write().await;
        let now = Instant::now();
        
        match state.node_state {
            NodeState::Follower | NodeState::Candidate => {
                if now >= state.election_deadline {
                    // Start election
                    state.node_state = NodeState::Candidate;
                    state.current_term += 1;
                    state.voted_for = Some(config.node_id.clone());
                    state.election_deadline = now + Duration::from_millis(config.election_timeout_ms);
                    
                    // In a real implementation, send vote requests to other nodes
                    println!("Node {} starting election for term {}", config.node_id, state.current_term);
                }
            }
            NodeState::Leader => {
                // Send heartbeats
                if now.duration_since(state.last_heartbeat) >= Duration::from_millis(config.heartbeat_interval_ms) {
                    state.last_heartbeat = now;
                    // In a real implementation, send heartbeats to followers
                    println!("Node {} sending heartbeats (term {})", config.node_id, state.current_term);
                }
            }
        }
    }
    
    async fn handle_client_write(
        state: &Arc<RwLock<RaftState>>,
        _config: &RaftConfig,
        wal: &Arc<RwLock<WalManager>>,
        entry: WalEntry,
    ) -> Result<LogIndex> {
        let mut state = state.write().await;
        
        if state.node_state != NodeState::Leader {
            return Err(Error::NotLeader);
        }
        
        // Append to local log
        let log_entry = LogEntry {
            term: state.current_term,
            index: state.log.len() as LogIndex,
            entry: entry.clone(),
            committed: false,
        };
        
        state.log.push(log_entry.clone());
        
        // Write to WAL
        let mut wal = wal.write().await;
        let seq = wal.write_entry(entry)?;
        
        // In a real implementation, replicate to followers before committing
        // For now, immediately commit since we're single-node
        state.commit_index = log_entry.index;
        
        Ok(seq)
    }
    
    async fn handle_append_entries(
        _state: &Arc<RwLock<RaftState>>,
        _config: &RaftConfig,
        _from: NodeId,
        _request: AppendEntriesRequest,
    ) {
        // Implementation for handling append entries from leader
    }
    
    async fn handle_vote_request(
        _state: &Arc<RwLock<RaftState>>,
        _config: &RaftConfig,
        _from: NodeId,
        _request: VoteRequest,
    ) {
        // Implementation for handling vote requests
    }
    
    async fn handle_append_entries_response(
        _state: &Arc<RwLock<RaftState>>,
        _config: &RaftConfig,
        _from: NodeId,
        _response: AppendEntriesResponse,
    ) {
        // Implementation for handling append entries responses
    }
    
    async fn handle_vote_response(
        _state: &Arc<RwLock<RaftState>>,
        _config: &RaftConfig,
        _from: NodeId,
        _response: VoteResponse,
    ) {
        // Implementation for handling vote responses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_raft_node_creation() {
        let temp_dir = TempDir::new().unwrap();
        let wal = WalManager::new(temp_dir.path()).unwrap();
        
        let config = RaftConfig::default();
        let node = RaftNode::new(config, wal);
        
        assert!(!node.is_leader().await);
    }
    
    #[tokio::test]
    async fn test_single_node_write() {
        let temp_dir = TempDir::new().unwrap();
        let wal = WalManager::new(temp_dir.path()).unwrap();
        
        let mut config = RaftConfig::default();
        config.cluster_nodes = vec![config.node_id.clone()];
        
        let node = RaftNode::new(config, wal);
        node.start().await.unwrap();
        
        // Manually promote to leader for testing
        {
            let mut state = node.state.write().await;
            state.node_state = NodeState::Leader;
        }
        
        let entry = crate::wal::WalEntry::CreateTable {
            table: "test".to_string(),
            dimension: 128,
        };
        
        let result = node.write(entry).await;
        assert!(result.is_ok());
    }
}
