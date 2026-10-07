//! The gRPC surface, behind `api-grpc`.
//!
//! Carries the data path -- search, insert, the record routes -- and the table lifecycle around
//! it. Not the whole HTTP API: hybrid search, full-text search, DXQL, snapshots and backups stay
//! on REST, where an extra millisecond of JSON does not matter and a second schema would only
//! have to be kept in step for no gain. `proto/dxmt.proto` says the same thing and is the
//! authority on the wire shape.
//!
//! Every method here goes through the same storage calls the HTTP handlers use, including
//! `apply_update`, which is where the "keep what was not given" semantics live. Two interfaces
//! that each implement their own idea of an update is how they come to disagree.

// `tonic::Status` is 176 bytes, over clippy's 128-byte threshold for an error type, and every
// method of the generated `VectorDb` trait returns `Result<Response<T>, Status>`. Boxing it in
// the helpers below while the generated signatures keep it unboxed would move bytes around and
// change nothing. The size is tonic's to fix.
#![allow(clippy::result_large_err)]

use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::storage::IntelligentIndexedStorage;
use crate::types::record::{CreateRecordRequest, SearchQuery};
use crate::types::table::TableConfig;
use crate::types::vector::DistanceMetric as InternalMetric;

pub mod proto {
    tonic::include_proto!("dxmt.v1");
}

use proto::vector_db_server::{VectorDb, VectorDbServer};
use proto::*;

pub struct GrpcService {
    storage: Arc<IntelligentIndexedStorage>,
}

impl GrpcService {
    pub fn new(storage: Arc<IntelligentIndexedStorage>) -> Self {
        Self { storage }
    }

    /// Wrap this service in a tonic server, ready to serve.
    pub fn into_server(self) -> VectorDbServer<Self> {
        VectorDbServer::new(self)
    }
}

/// Map a storage error onto the gRPC code that means the same thing.
///
/// The HTTP surface maps the same errors onto status codes; this is that table in gRPC's
/// vocabulary, and it is written out rather than defaulting everything to `internal` because a
/// caller retrying a `not_found` forever is the kind of thing a wrong code causes.
fn status(error: crate::Error) -> Status {
    use crate::Error::*;
    let message = error.to_string();
    match error {
        TableNotFound(_) | RecordNotFound(_) | NotFound(_) => Status::not_found(message),
        TableAlreadyExists(_) | RecordAlreadyExists(_) => Status::already_exists(message),
        DimensionMismatch { .. }
        | InvalidInput(_)
        | ValidationError { .. }
        | InvalidVector(_)
        | EmptyVector => Status::invalid_argument(message),
        NotImplemented(_) => Status::unimplemented(message),
        Timeout(_) => Status::deadline_exceeded(message),
        // The read-only state a failed fsync leaves behind: the request was understood and the
        // server cannot serve it right now.
        ServiceUnavailable { .. } => Status::failed_precondition(message),
        _ => Status::internal(message),
    }
}

/// Payloads travel as JSON text. Empty means none, which is what the field's default gives a
/// caller who does not set it.
fn payload_json(payload: &serde_json::Value) -> String {
    if payload.is_null() {
        String::new()
    } else {
        payload.to_string()
    }
}

fn parse_payload(text: &str) -> Result<Option<serde_json::Value>, Status> {
    if text.is_empty() {
        return Ok(None);
    }
    serde_json::from_str(text)
        .map(Some)
        .map_err(|e| Status::invalid_argument(format!("payload_json is not JSON: {e}")))
}

fn record_to_proto(record: crate::types::record::Record) -> Record {
    Record {
        id: record.id.to_string(),
        vector: record.vector,
        payload_json: payload_json(&record.payload),
        version: record.version,
    }
}

impl From<DistanceMetric> for InternalMetric {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Euclidean => InternalMetric::Euclidean,
            DistanceMetric::DotProduct => InternalMetric::Dot,
            // Unspecified is the server's default rather than an error: a caller who does not
            // care should not have to name one.
            DistanceMetric::Unspecified | DistanceMetric::Cosine => InternalMetric::Cosine,
        }
    }
}

fn table_to_proto(table: crate::types::table::Table) -> TableInfo {
    TableInfo {
        name: table.name,
        dimension: table.dimension as u32,
        record_count: table.record_count,
        index_type: format!("{:?}", table.index_type).to_lowercase(),
    }
}

/// Rebuild a filter from the proto oneof.
///
/// The REST shape spells the operator as a string, so a misspelt one is a runtime error; here
/// the oneof means it cannot be spelt at all. What it costs is this translation.
fn condition_from_proto(condition: Condition) -> Result<crate::types::record::Condition, Status> {
    use crate::types::record::FilterOperation as Op;
    use proto::condition::Operation as P;
    let operation = match condition.operation {
        Some(P::Equal(v)) => Op::Equal(serde_json::Value::String(v)),
        Some(P::EqualBool(v)) => Op::Equal(serde_json::Value::Bool(v)),
        Some(P::EqualNumber(v)) => Op::Equal(serde_json::json!(v)),
        Some(P::NotEqual(v)) => Op::NotEqual(serde_json::Value::String(v)),
        Some(P::GreaterThan(v)) => Op::GreaterThan(serde_json::json!(v)),
        Some(P::GreaterThanOrEqual(v)) => Op::GreaterThanOrEqual(serde_json::json!(v)),
        Some(P::LessThan(v)) => Op::LessThan(serde_json::json!(v)),
        Some(P::LessThanOrEqual(v)) => Op::LessThanOrEqual(serde_json::json!(v)),
        Some(P::In(list)) => Op::In(
            list.values
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        ),
        Some(P::NotIn(list)) => Op::NotIn(
            list.values
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        ),
        Some(P::Contains(v)) => Op::Contains(v),
        Some(P::StartsWith(v)) => Op::StartsWith(v),
        Some(P::EndsWith(v)) => Op::EndsWith(v),
        None => {
            return Err(Status::invalid_argument(format!(
                "condition on '{}' names no operation",
                condition.key
            )))
        }
    };
    Ok(crate::types::record::Condition {
        key: condition.key,
        operation,
    })
}

fn filter_from_proto(filter: Filter) -> Result<crate::types::Filter, Status> {
    fn convert(
        conditions: Vec<Condition>,
    ) -> Result<Option<Vec<crate::types::record::Condition>>, Status> {
        if conditions.is_empty() {
            return Ok(None);
        }
        conditions
            .into_iter()
            .map(condition_from_proto)
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
    Ok(crate::types::Filter {
        must: convert(filter.must)?,
        must_not: convert(filter.must_not)?,
        should: convert(filter.should)?,
    })
}

#[tonic::async_trait]
impl VectorDb for GrpcService {
    async fn health(&self, _: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            // The same question `GET /ready` answers: not merely alive, but able to make a
            // write durable. A failed fsync puts the process in a read-only state, and a
            // caller needs to know that before it tries to write.
            ready: self.storage.health_snapshot().await.ready,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }))
    }

    async fn create_table(
        &self,
        request: Request<CreateTableRequest>,
    ) -> Result<Response<TableInfo>, Status> {
        let request = request.into_inner();
        let table = self
            .storage
            .create_table(TableConfig {
                name: request.name,
                dimension: request.dimension as usize,
                distance_metric: DistanceMetric::try_from(request.distance_metric)
                    .unwrap_or(DistanceMetric::Unspecified)
                    .into(),
                description: Some(request.description).filter(|d| !d.is_empty()),
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
                rerank: None,
            })
            .await
            .map_err(status)?;
        Ok(Response::new(table_to_proto(table)))
    }

    async fn list_tables(
        &self,
        _: Request<ListTablesRequest>,
    ) -> Result<Response<ListTablesResponse>, Status> {
        let tables = self.storage.list_tables().await.map_err(status)?;
        Ok(Response::new(ListTablesResponse {
            tables: tables.into_iter().map(table_to_proto).collect(),
        }))
    }

    async fn get_table(
        &self,
        request: Request<GetTableRequest>,
    ) -> Result<Response<TableInfo>, Status> {
        let table = self
            .storage
            .get_table(&request.into_inner().name)
            .await
            .map_err(status)?;
        Ok(Response::new(table_to_proto(table)))
    }

    async fn delete_table(
        &self,
        request: Request<DeleteTableRequest>,
    ) -> Result<Response<DeleteTableResponse>, Status> {
        self.storage
            .delete_table(&request.into_inner().name)
            .await
            .map_err(status)?;
        Ok(Response::new(DeleteTableResponse {}))
    }

    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let request = request.into_inner();
        let mut records = Vec::with_capacity(request.records.len());
        for record in request.records {
            records.push(CreateRecordRequest {
                id: record.id,
                vector: record.vector,
                sparse_vector: None,
                named_vectors: None,
                payload: parse_payload(&record.payload_json)?,
                text: None,
            });
        }
        let inserted = self
            .storage
            .insert_records(&request.table, records)
            .await
            .map_err(status)?;
        Ok(Response::new(InsertResponse {
            count: inserted.len() as u32,
            inserted_ids: inserted.into_iter().map(|r| r.id.to_string()).collect(),
        }))
    }

    async fn get(&self, request: Request<GetRequest>) -> Result<Response<Record>, Status> {
        let request = request.into_inner();
        let record = self
            .storage
            .get_record(&request.table, &request.id)
            .await
            .map_err(status)?;
        Ok(Response::new(record_to_proto(record)))
    }

    async fn update(
        &self,
        request: Request<UpdateRequest>,
    ) -> Result<Response<UpdateResponse>, Status> {
        let request = request.into_inner();
        let payload = match request.payload_json.as_deref() {
            Some(text) => parse_payload(text)?,
            None => None,
        };
        let record = self
            .storage
            .apply_update(
                &request.table,
                &request.id,
                crate::storage::RecordUpdate {
                    // An empty vector is the field's default, which is how a caller says "leave
                    // it alone"; proto3 gives no other way to omit a repeated field.
                    vector: Some(request.vector).filter(|v| !v.is_empty()),
                    payload,
                    merge: request.merge,
                    upsert: request.upsert,
                    // gRPC has no text field on UpdateRequest: proto3 would need it added to
                    // dxmt.proto, and the data path there is deliberately the small one.
                    ..Default::default()
                },
            )
            .await
            .map_err(status)?;
        Ok(Response::new(UpdateResponse {
            version: record.version,
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let request = request.into_inner();
        let requested = request.ids.len() as u32;
        let deleted = self
            .storage
            .delete_records(&request.table, request.ids)
            .await
            .map_err(status)?;
        Ok(Response::new(DeleteResponse {
            deleted: deleted as u32,
            requested,
        }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let request = request.into_inner();
        let top_k = request.top_k.max(1) as usize;
        let result = self
            .storage
            .search(
                &request.table,
                SearchQuery {
                    vector: request.vector,
                    top_k,
                    // Sent together deliberately: the storage layer reads both, and a `limit`
                    // left at its default truncates a larger `top_k` without saying so. The
                    // REST clients each have to remember this; here there is one field.
                    limit: top_k,
                    filter: request.filter.map(filter_from_proto).transpose()?,
                    with_payload: request.with_payload,
                    with_vector: request.with_vector,
                    ..Default::default()
                },
            )
            .await
            .map_err(status)?;
        Ok(Response::new(SearchResponse {
            hits: result
                .hits
                .into_iter()
                .map(|hit| SearchHit {
                    id: hit.id.to_string(),
                    score: hit.score,
                    payload_json: hit.payload.as_ref().map(payload_json).unwrap_or_default(),
                    vector: hit.vector.unwrap_or_default(),
                })
                .collect(),
            total: result.total as u32,
            took_ms: result.tookms,
        }))
    }
}
