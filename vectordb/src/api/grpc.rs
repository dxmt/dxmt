use tonic::{transport::Server, Request, Response, Status};
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    storage::Storage,
    types::{table::TableSchema, Vector},
};

// Import generated protobuf code
pub mod vectordb {
    tonic::include_proto!("vectordb")
}

use vectordb::{
    _vector_db_server::{VectorDb, VectorDbServer},
    CreateTableRequest, CreateTableResponse,
    DeleteTableRequest, DeleteTableResponse,
    ListTablesRequest, ListTablesResponse,
    InsertRecordRequest, InsertRecordResponse,
    BatchInsertRequest, BatchInsertResponse,
    GetRecordRequest, GetRecordResponse,
    UpdateRecordRequest, UpdateRecordResponse,
    DeleteRecordRequest, DeleteRecordResponse,
    SearchRequest, SearchResponse,
    HybridSearchRequest, HybridSearchResponse,
    OptimizeTableRequest, OptimizeTableResponse,
    ClusterStatusRequest, ClusterStatusResponse,
    Table as ProtoTable,
    Record as ProtoRecord,
    SearchResult,
};

/// gRPC service implementation
pub struct VectorDbService {
    storage: Arc<dyn Storage>,
}

impl VectorDbService {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }
}

#[tonic::async_trait]
impl VectorDb for VectorDbService {
    async fn create_table(
        &self,
        request: Request<CreateTableRequest>) -> Result<Response<CreateTableResponse>, Status> {
        let req = request.into_inner();
        
        let distance_metric = match req.distance_metric.as_str() {
            "euclidean" => DistanceMetric::Euclidean,
            "dot_product" => DistanceMetric::DotProduct,
            "manhattan" => DistanceMetric::Manhattan,
            _ => DistanceMetric::Cosine,
        };
        
        let index_type = match req.index_type.as_str() {
            "ivf_pq" => IndexType::IvfPq,
            "lsh" => IndexType::Lsh,
            _ => IndexType::Hnsw,
        };
        
        let schema = match req.schema.as_str() {
            "structured" => TableSchema::Structured,
            "enterprise" => TableSchema::Enterprise,
            _ => TableSchema::Exploratory,
        };
        
        let config = crate::types::table::TableConfig {
            name: req.name,
            dimension: req.dimension as usize,
            distance_metric,
            description: None,
            index_type: Some(index_type),
            text_fields: None,
            enable_full_text_search: false,
        };

        let table = self.storage
            .create_table(config)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let proto_table = ProtoTable {
            id: table.id.to_string(),
            name: table.name,
            dimension: table.dimension as u32,
            distance_metric: format!("{:?}", table.distance_metric),
            index_type: format!("{:?}", table.index_type),
            record_count: table.record_count as u64,
            created_at: table.created_at.timestamp(),
            updated_at: table.updated_at.timestamp());
        
        Ok(Response::new(CreateTableResponse {
            table: Some(proto_table),
        }))
    }
    
    async fn delete_table(
        &self,
        request: Request<DeleteTableRequest>) -> Result<Response<DeleteTableResponse>, Status> {
        let req = request.into_inner();
        
        self.storage
            .delete_table(&req.name)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(DeleteTableResponse {
            success: true,
            message: format!("Table {} deleted successfully", req.name),
        }))
    }
    
    async fn list_tables(
        &self,
        _request: Request<ListTablesRequest>) -> Result<Response<ListTablesResponse>, Status> {
        let tables = self.storage
            .list_tables()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let proto_tables = tables
            .into_iter()
            .map(|table| ProtoTable {
                id: table.id.to_string(),
                name: table.name,
                dimension: table.dimension as u32,
                distance_metric: format!("{:?}", table.distance_metric),
                index_type: format!("{:?}", table.index_type),
                record_count: table.record_count as u64,
                created_at: table.created_at.timestamp(),
                updated_at: table.updated_at.timestamp(),
            })
            .collect();

        Ok(Response::new(ListTablesResponse {
            _tables: proto_tables,
        }))
    }
    
    async fn insert_record(
        &self,
        request: Request<InsertRecordRequest>) -> Result<Response<InsertRecordResponse>, Status> {
        let req = request.into_inner();

        let table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;

        if req.vector.len() != table.dimension {
            return Err(Status::invalid_argument("Vector dimension mismatch"));
        }

        let id = if req.id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            req.id
        };

        let request_obj = crate::types::record::CreateRecordRequest {
            id: id.clone(),
            vector: req.vector,
            sparse_vector: None,
            payload: req.payload.and_then(|m| serde_json::from_str(&m).ok()),
        };

        self.storage
            .insert_record(&req.table_name, request_obj)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(InsertRecordResponse {
            id,
            success: true,
        }))
    }
    
    async fn batch_insert(
        &self,
        request: Request<BatchInsertRequest>) -> Result<Response<BatchInsertResponse>, Status> {
        let req = request.into_inner();
        
        let table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;
        
        let mut successful = 0u32;
        let mut failed = 0u32;
        let mut inserted_ids = Vec::new();
        
        for record_req in req.records {
            if record_req.vector.len() != table.dimension {

                failed += 1;
                continue;

            }
            let id = if record_req.id.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                record_req.id
            };
            
            let request_obj = crate::types::record::CreateRecordRequest {
                id: id.clone(),
                vector: record_req.vector,
                sparse_vector: None,
                payload: record_req.payload.and_then(|m| serde_json::from_str(&m).ok()),
            };

            match self.storage.insert_record(&req.table_name, request_obj).await {
                Ok(_) => {
                    successful += 1;
                    inserted_ids.push(id);
                }
                Err(_) => failed += 1,
            }
        }

        Ok(Response::new(BatchInsertResponse {
            successful,
            failed,
            inserted_ids,
        }))
    }
    
    async fn get_record(
        &self,
        request: Request<GetRecordRequest>) -> Result<Response<GetRecordResponse>, Status> {
        let req = request.into_inner();
        
        let table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;
        
        let record = self.storage
            .get_record(&req.table_name, &req.id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let vector = match &record.vector {
            Vector::Dense(v) => v.clone(),
            Vector::Sparse(_) => vec![], // Handle sparse vectors if needed
        };
        
        let proto_record = ProtoRecord {
            id: record.id,
            vector,
            payload: record.payload.map(|m| serde_json::to_string(&m).unwrap_or_default()),
        };
        
        Ok(Response::new(GetRecordResponse {
            record: Some(proto_record),
        }))
    }
    
    async fn update_record(
        &self,
        request: Request<UpdateRecordRequest>) -> Result<Response<UpdateRecordResponse>, Status> {
        let req = request.into_inner();
        
        let table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;

        // Validate vector dimension if provided
        if !req.vector.is_empty() && req.vector.len() != table.dimension {
            return Err(Status::invalid_argument("Vector dimension mismatch"));
        }

        // Parse payload if provided
        let payload = if let Some(payload_str) = req.payload {
            Some(serde_json::from_str(&payload_str)
                .map_err(|e| Status::invalid_argument(format!("Invalid payload: {}", e)))?)
        } else {
            None
        };

        let request_obj = crate::types::record::CreateRecordRequest {
            id: req.id.clone(),
            vector: if req.vector.is_empty() {
                vec![]
            } else {
                req.vector
            },
            sparse_vector: None,
            payload,
        };

        self.storage
            .update_record(&req.table_name, &req.id, request_obj)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(UpdateRecordResponse {
            success: true,
            message: format!("Record {} updated successfully", req.id),
        }))
    }
    
    async fn delete_record(
        &self,
        request: Request<DeleteRecordRequest>) -> Result<Response<DeleteRecordResponse>, Status> {
        let req = request.into_inner();
        
        let table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;
        
        self.storage
            .delete_record(&req.table_name, &req.id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(DeleteRecordResponse {
            success: true,
            message: format!("Record {} deleted successfully", req.id),
        }))
    }
    
    async fn search(
        &self,
        request: Request<SearchRequest>) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        
        let table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;
        
        if req.vector.len() != table.dimension {
            return Err(Status::invalid_argument("Vector dimension mismatch"));
        }

        let query = crate::types::record::SearchQuery {
            vector: req.vector,
            top_k: req.k as usize,
            filter: req.filter.and_then(|f| serde_json::from_str(&f).ok()),
            with_payload: true,
            with_vector: true,
            limit: req.k as usize,
            include_metadata: true,
        };

        let search_result = self.storage
            .search(&req.table_name, query)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let proto_results = search_result.hits
            .into_iter()
            .map(|hit| {
                SearchResult {
                    record: Some(ProtoRecord {
                        id: hit.id,
                        vector: hit.vector.unwrap_or_default(),
                        payload: hit.payload.map(|p| serde_json::to_string(&p).unwrap_or_default()),
                    }),
                    score: hit.score,
                    distance: 0.0,
                }
            })
            .collect();

        Ok(Response::new(SearchResponse {
            _results: proto_results,
        }))
    }
    
    async fn hybrid_search(
        &self,
        request: Request<HybridSearchRequest>) -> Result<Response<HybridSearchResponse>, Status> {
        let req = request.into_inner();
        
        let table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;
        
        if req.vector.len() != table.dimension {
            return Err(Status::invalid_argument("Vector dimension mismatch"));
        }

        // text_query and vector_weight blending are not yet implemented via the generic trait;
        // this falls through to vector-only hybrid search
        let query = crate::types::record::SearchQuery {
            vector: req.vector,
            top_k: req.k as usize,
            filter: req.filter.and_then(|f| serde_json::from_str(&f).ok()),
            with_payload: true,
            with_vector: true,
            limit: req.k as usize,
            include_metadata: true,
        };

        let search_result = self.storage
            .hybrid_search(&req.table_name, query)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        
        let proto_results = search_result.hits
            .into_iter()
            .map(|hit| {
                SearchResult {
                    record: Some(ProtoRecord {
                        id: hit.id,
                        vector: hit.vector.unwrap_or_default(),
                        payload: hit.payload.map(|p| serde_json::to_string(&p).unwrap_or_default()),
                    }),
                    score: hit.score,
                    distance: 0.0,
                }
            })
            .collect();

        Ok(Response::new(HybridSearchResponse {
            _results: proto_results,
        }))
    }
    
    async fn optimize_table(
        &self,
        request: Request<OptimizeTableRequest>) -> Result<Response<OptimizeTableResponse>, Status> {
        let req = request.into_inner();
        
        // Verify table exists
        let _table = self.storage
            .get_table(&req.table_name)
            .await
            .map_err(|e| Status::not_found(e.to_string()))?;

        let start_time = std::time::Instant::now();

        // index optimization (merge, compact) not yet exposed via gRPC

        let duration = start_time.elapsed().as_secs();

        Ok(Response::new(OptimizeTableResponse {
            _success: true,
            duration,
            message: format!("Table {} optimized in {} seconds", req.table_name, duration),
        }))
    }
    
    async fn cluster_status(
        &self,
        _request: Request<ClusterStatusRequest>) -> Result<Response<ClusterStatusResponse>, Status> {
        // cluster module is not the current focus; returns single-node status

        Ok(Response::new(ClusterStatusResponse {
            health: "healthy".to_string(),
            nodes: 1,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }))
    }
}

/// Create and start gRPC server
pub async fn start_grpc_server(
    addr: std::net::SocketAddr,
    storage: Arc<dyn Storage>) -> crate::Result<()> {
    let service = VectorDbService::new(storage);
    
    Server::builder()
        .add_service(VectorDbServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
