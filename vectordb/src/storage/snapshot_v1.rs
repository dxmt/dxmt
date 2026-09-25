//! The snapshot metadata layout that builds up to 2026-09-24 wrote, so their data still opens.
//!
//! **Nothing here may be edited to follow the live types.** These describe bytes that already
//! exist on disk. A snapshot's `tN.meta` was a bincode encoding of `TableSnapshot`, and bincode
//! is positional: it stores no field names, so adding a field to `Table` or a value to an
//! `IndexAlgorithm` variant changes where every later byte is read from. Two fields added to
//! `Table` and one to `IndexAlgorithm::HnswPQ` turned every existing snapshot into
//! `tag for enum is not valid, found 16`, and the server refuses to start on a snapshot it
//! cannot read — by design, because the alternative is coming up with an empty database.
//! `tests/compat/` caught it, which is what that suite is for.
//!
//! So a `table-snapshot` file is read with the definitions below and converted, while what
//! gets written is a `table-snapshot-v2`: the same bincode records, behind a head encoded in
//! MessagePack, which carries field names. From there on a new field with `#[serde(default)]`
//! costs nothing and this file never needs another entry.

use serde::Deserialize;
use std::sync::Arc;

use crate::types::record::Record;
use crate::types::table::{FieldDef, IndexType, Table, TextFieldConfig, VectorSpaceConfig};
use crate::types::vector::DistanceMetric;

/// `Table` as it was before per-table `hnsw` and `quantization` parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct TableV1 {
    pub id: uuid::Uuid,
    pub name: String,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub description: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub record_count: u64,
    pub index_type: IndexType,
    pub text_fields: Option<Vec<TextFieldConfig>>,
    pub enable_full_text_search: bool,
    pub schema_version: u64,
    pub payload_schema: Option<Vec<FieldDef>>,
    pub memory_optimization: Option<bool>,
    pub named_vectors: Option<Vec<VectorSpaceConfig>>,
}

impl From<TableV1> for Table {
    fn from(old: TableV1) -> Self {
        Table {
            id: old.id,
            name: old.name,
            dimension: old.dimension,
            distance_metric: old.distance_metric,
            description: old.description,
            created_at: old.created_at,
            updated_at: old.updated_at,
            record_count: old.record_count,
            index_type: old.index_type,
            text_fields: old.text_fields,
            enable_full_text_search: old.enable_full_text_search,
            schema_version: old.schema_version,
            payload_schema: old.payload_schema,
            memory_optimization: old.memory_optimization,
            named_vectors: old.named_vectors,
            // A table written before these existed asked for nothing, so it gets the server's
            // defaults -- which is what it was built with.
            hnsw: None,
            quantization: None,
            payload_index: None,
            embedding: None,
        }
    }
}

/// `IndexAlgorithm` as it was before `HnswPQ` carried `quantize_at_rows`.
#[derive(Debug, Clone, Deserialize)]
pub enum IndexAlgorithmV1 {
    HNSW {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    },
    HnswPQ {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        num_subvectors: usize,
        num_centroids: usize,
        enable_reranking: bool,
    },
    QuantizedIVF {
        nlist: usize,
        nprobe: usize,
    },
    BruteForce {
        enable_parallel: bool,
    },
    DiskANN {
        max_degree: usize,
        search_list_size: usize,
    },
    EnhancedDiskANN {
        max_degree: usize,
        search_list_size: usize,
        use_pq: bool,
        num_pq_subvectors: usize,
        cache_size: usize,
    },
    PartitionedIvf {
        nlist: usize,
        nprobe: usize,
        max_resident_partitions: usize,
    },
    ScaNN {
        num_leaves: usize,
        num_leaves_to_search: usize,
    },
}

impl From<IndexAlgorithmV1> for crate::storage::index_selector::IndexAlgorithm {
    fn from(old: IndexAlgorithmV1) -> Self {
        use crate::storage::index_selector::IndexAlgorithm as New;
        match old {
            IndexAlgorithmV1::HNSW {
                m,
                ef_construction,
                ef_search,
            } => New::HNSW {
                m,
                ef_construction,
                ef_search,
            },
            IndexAlgorithmV1::HnswPQ {
                m,
                ef_construction,
                ef_search,
                num_subvectors,
                num_centroids,
                enable_reranking,
            } => New::HnswPQ {
                m,
                ef_construction,
                ef_search,
                num_subvectors,
                num_centroids,
                enable_reranking,
                // Not stated, so the index's own default (65,536) -- the threshold every
                // table written before this field had.
                quantize_at_rows: None,
            },
            IndexAlgorithmV1::QuantizedIVF { nlist, nprobe } => New::QuantizedIVF { nlist, nprobe },
            IndexAlgorithmV1::BruteForce { enable_parallel } => New::BruteForce { enable_parallel },
            IndexAlgorithmV1::DiskANN {
                max_degree,
                search_list_size,
            } => New::DiskANN {
                max_degree,
                search_list_size,
            },
            IndexAlgorithmV1::EnhancedDiskANN {
                max_degree,
                search_list_size,
                use_pq,
                num_pq_subvectors,
                cache_size,
            } => New::EnhancedDiskANN {
                max_degree,
                search_list_size,
                use_pq,
                num_pq_subvectors,
                cache_size,
            },
            IndexAlgorithmV1::PartitionedIvf {
                nlist,
                nprobe,
                max_resident_partitions,
            } => New::PartitionedIvf {
                nlist,
                nprobe,
                max_resident_partitions,
            },
            IndexAlgorithmV1::ScaNN {
                num_leaves,
                num_leaves_to_search,
            } => New::ScaNN {
                num_leaves,
                num_leaves_to_search,
            },
        }
    }
}

/// A `tN.meta` from before the migration.
#[derive(Debug, Deserialize)]
pub struct TableSnapshotV1 {
    pub table: TableV1,
    pub algorithm: IndexAlgorithmV1,
    pub records: Vec<Arc<Record>>,
    pub has_index: bool,
}
