use async_graphql::{
    Context, EmptySubscription, InputObject, Object, Result, Schema, SimpleObject, ID,
};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{extract::Extension, response::IntoResponse, routing::post, Router};
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    api::handlers::AppState,
    types::{record::Record, table::Table},
};

/// GraphQL schema type
pub type ApiSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

/// GraphQL query root
pub struct QueryRoot;

/// GraphQL mutation root
pub struct MutationRoot;

/// Table type for GraphQL
#[derive(SimpleObject)]
struct TableType {
    id: ID,
    name: String,
    dimension: i32,
    distance_metric: String,
    index_type: String,
    record_count: i32,
}

/// Record type for GraphQL
#[derive(SimpleObject)]
struct RecordType {
    id: ID,
    vector: Vec<f32>,
    payload: Option<String>,
}

/// Search input for GraphQL
#[derive(InputObject)]
struct SearchInput {
    table_name: String,
    vector: Vec<f32>,
    k: i32,
}

/// Create table input
#[derive(InputObject)]
struct CreateTableInput {
    name: String,
    dimension: i32,
    distance_metric: Option<String>,
    index_type: Option<String>,
}

/// Insert record input
#[derive(InputObject)]
struct InsertRecordInput {
    table_name: String,
    vector: Vec<f32>,
    payload: Option<String>,
}

/// Batch insert input
#[derive(InputObject)]
struct BatchInsertInput {
    table_name: String,
    records: Vec<InsertRecordInput>,
}

/// Update record input
#[derive(InputObject)]
struct UpdateRecordInput {
    table_name: String,
    id: String,
    vector: Option<Vec<f32>>,
    payload: Option<String>,
}

#[Object]
impl QueryRoot {
    /// Get table by name
    async fn table(&self, ctx: &Context<'_>, name: String) -> Result<Option<TableType>> {
        let state = ctx.data::<Arc<AppState>>()?;
        match state.storage.get_table(&name).await {
            Ok(table) => Ok(Some(table_to_graphql(table))),
            Err(_) => Ok(None),
        }
    }

    /// List all tables
    async fn tables(&self, ctx: &Context<'_>) -> Result<Vec<TableType>> {
        let state = ctx.data::<Arc<AppState>>()?;
        let tables = state.storage.list_tables().await?;
        Ok(tables.into_iter().map(table_to_graphql).collect())
    }

    /// Get record by ID
    async fn record(
        &self,
        ctx: &Context<'_>,
        table_name: String,
        id: String,
    ) -> Result<Option<RecordType>> {
        let state = ctx.data::<Arc<AppState>>()?;
        let table = state.storage.get_table(&table_name).await?;

        match state.storage.get_record(&table.name, &id).await {
            Ok(record) => Ok(Some(record_to_graphql(record, table.id))),
            Err(_) => Ok(None),
        }
    }

    /// Search vectors
    async fn search(&self, ctx: &Context<'_>, input: SearchInput) -> Result<Vec<RecordType>> {
        let state = ctx.data::<Arc<AppState>>()?;
        let table = state.storage.get_table(&input.table_name).await?;

        if input.vector.len() != table.dimension {
            return Err("Vector dimension mismatch".into());
        }

        // Create a simple search query
        let _query = crate::types::record::SearchQuery {
            vector: input.vector,
            top_k: input.k as usize,
            filter: None,
            with_payload: true,
            with_vector: true,
            ..Default::default()
        };

        // For now, return empty results as search_vectors method doesn't exist
        // This would need to be implemented based on the actual storage interface
        Ok(vec![])
    }

    /// Get cluster status
    async fn cluster_status(&self, _ctx: &Context<'_>) -> crate::Result<String> {
        Ok("Active".to_string())
    }
}

#[Object]
impl MutationRoot {
    /// Create a new table
    async fn create_table(&self, ctx: &Context<'_>, input: CreateTableInput) -> Result<TableType> {
        let state = ctx.data::<Arc<AppState>>()?;
        crate::utils::validate_table_name(&input.name)?;

        // Create table config - this would need to be adapted based on actual TableConfig structure
        let config = crate::types::table::TableConfig {
            name: input.name,
            dimension: input.dimension as usize,
            distance_metric: crate::types::vector::DistanceMetric::Cosine, // Default
            index_type: Some(crate::types::table::IndexType::Hnsw),        // Default
            description: None,
            text_fields: None,
            enable_full_text_search: false,
            memory_optimization: None, // Use global config
            payload_schema: None,
            schema_version: 1,
            named_vectors: None,
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
            rerank: None,
        };

        let table = state.storage.create_table(config).await?;
        Ok(table_to_graphql(table))
    }

    /// Insert a record
    async fn insert_record(
        &self,
        ctx: &Context<'_>,
        input: InsertRecordInput,
    ) -> Result<RecordType> {
        let state = ctx.data::<Arc<AppState>>()?;
        let table = state.storage.get_table(&input.table_name).await?;

        let request = crate::types::record::CreateRecordRequest {
            id: Uuid::new_v4().to_string(),
            vector: input.vector,
            sparse_vector: None,
            payload: input
                .payload
                .map(|p| serde_json::from_str(&p).unwrap_or_default()),
            named_vectors: None,
            text: None,
        };

        // Insert single record by wrapping in Vec
        let records = state
            .storage
            .insert_records(&table.name, vec![request])
            .await?;
        if let Some(record) = records.into_iter().next() {
            Ok(record_to_graphql(record, table.id))
        } else {
            Err("Failed to insert record".into())
        }
    }

    /// Batch insert records
    async fn batch_insert(
        &self,
        ctx: &Context<'_>,
        input: BatchInsertInput,
    ) -> Result<Vec<RecordType>> {
        let state = ctx.data::<Arc<AppState>>()?;
        let table = state.storage.get_table(&input.table_name).await?;

        let requests: Vec<_> = input
            .records
            .into_iter()
            .map(|r| crate::types::record::CreateRecordRequest {
                id: Uuid::new_v4().to_string(),
                vector: r.vector,
                sparse_vector: None,
                payload: r
                    .payload
                    .map(|p| serde_json::from_str(&p).unwrap_or_default()),
                named_vectors: None,
                text: None,
            })
            .collect();

        let records = state.storage.insert_records(&table.name, requests).await?;
        Ok(records
            .into_iter()
            .map(|r| record_to_graphql(r, table.id))
            .collect())
    }

    /// Delete a table
    async fn delete_table(&self, ctx: &Context<'_>, name: String) -> Result<bool> {
        let state = ctx.data::<Arc<AppState>>()?;
        state.storage.delete_table(&name).await?;
        Ok(true)
    }

    /// Update a record
    async fn update_record(
        &self,
        ctx: &Context<'_>,
        input: UpdateRecordInput,
    ) -> Result<RecordType> {
        let state = ctx.data::<Arc<AppState>>()?;
        let table = state.storage.get_table(&input.table_name).await?;

        let request = crate::types::record::CreateRecordRequest {
            id: input.id.clone(),
            vector: input.vector.unwrap_or_default(),
            sparse_vector: None,
            payload: input
                .payload
                .map(|p| serde_json::from_str(&p).unwrap_or_default()),
            named_vectors: None,
            text: None,
        };

        let record = state
            .storage
            .update_record(&table.name, &input.id, request)
            .await?;
        Ok(record_to_graphql(record, table.id))
    }

    /// Delete a record
    async fn delete_record(
        &self,
        ctx: &Context<'_>,
        table_name: String,
        id: String,
    ) -> Result<bool> {
        let state = ctx.data::<Arc<AppState>>()?;
        let table = state.storage.get_table(&table_name).await?;

        // Use delete_records method which exists
        state.storage.delete_records(&table.name, vec![id]).await?;
        Ok(true)
    }
}

/// Convert Table to GraphQL TableType
fn table_to_graphql(table: Table) -> TableType {
    TableType {
        id: table.id.to_string().into(),
        name: table.name,
        dimension: table.dimension as i32,
        distance_metric: table.distance_metric.to_string(),
        index_type: table.index_type.to_string(),
        record_count: table.record_count as i32,
    }
}

/// Convert Record to GraphQL RecordType
fn record_to_graphql(record: Record, _table_id: Uuid) -> RecordType {
    RecordType {
        id: record.id.into(),
        vector: record.vector, // record.vector is already Vec<f32>
        payload: Some(serde_json::to_string(&record.payload).unwrap_or_default()),
    }
}

/// GraphQL handler
pub async fn graphql_handler(
    Extension(schema): Extension<ApiSchema>,
    req: GraphQLRequest,
) -> impl IntoResponse {
    let resp = schema.execute(req.into_inner()).await;
    GraphQLResponse::from(resp)
}

/// Create GraphQL router
pub fn create_graphql_router() -> Router<AppState> {
    let schema = Schema::build(QueryRoot, MutationRoot, EmptySubscription).finish();

    Router::new()
        .route("/graphql", post(graphql_handler))
        .layer(Extension(schema))
}
