use crate::types::vector::{DenseVector, RecordId};
use std::any::Any;

/// Synchronous vector index interface (legacy; prefer `index_traits::VectorIndex` for new code).
pub trait SyncVectorIndex: Send + Sync + std::fmt::Debug {
    /// 插入向量到索引
    fn insert(&mut self, id: RecordId, vector: DenseVector) -> crate::Result<()>;

    /// 搜索最近邻
    fn search(&self, query: &DenseVector, k: usize) -> crate::Result<Vec<(RecordId, f32)>>;

    /// 从索引中删除向量
    fn remove(&mut self, id: &RecordId) -> crate::Result<bool>;

    /// 获取索引大小
    fn len(&self) -> usize;

    /// 检查索引是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 清空索引
    fn clear(&mut self) -> crate::Result<()>;

    /// Allows downcasting trait objects
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// 索引统计信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexStats {
    pub total_vectors: usize,
    pub memory_usagebytes: usize,
    pub index_type: String,
}

/// 可序列化的索引接口
pub trait SerializableIndex {
    /// 序列化索引到字节数组
    fn serialize(&self) -> crate::Result<Vec<u8>>;

    /// 从字节数组反序列化索引
    fn deserialize(&mut self, data: &[u8]) -> crate::Result<()>;

    /// 获取索引统计信息
    fn get_stats(&self) -> IndexStats;
}

// New trait that combines SyncVectorIndex and SerializableIndex
pub trait VectorIndexSerializable: SyncVectorIndex + SerializableIndex {}

/// Back-compat alias — points to the canonical async trait in index_traits.
pub use crate::search::index_traits::VectorIndex;
