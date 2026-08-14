use crossbeam_queue::ArrayQueue;
use std::sync::Arc;

/// 搜索操作的内存池
/// 减少频繁的内存分配，提高查询性能
pub struct SearchMemoryPool {
    /// 候选集缓冲池
    _candidate_buffers: Arc<ArrayQueue<Vec<SearchCandidate>>>,
    /// 访问标记缓冲池
    visited_buffers: Arc<ArrayQueue<Vec<bool>>>,
    /// 距离缓冲池
    distance_buffers: Arc<ArrayQueue<Vec<f32>>>,
    /// 默认容量
    default_capacity: usize,
}

#[derive(Clone, Copy)]
pub struct SearchCandidate {
    pub distance: f32,
    pub idx: usize,
}

impl SearchMemoryPool {
    pub fn new(pool_size: usize, default_capacity: usize) -> Self {
        let candidate_buffers = Arc::new(ArrayQueue::new(pool_size));
        let visited_buffers = Arc::new(ArrayQueue::new(pool_size));
        let distance_buffers = Arc::new(ArrayQueue::new(pool_size));

        // 预填充池
        for _ in 0..pool_size {
            let _ = candidate_buffers.push(Vec::with_capacity(default_capacity));
            let _ = visited_buffers.push(vec![false; default_capacity]);
            let _ = distance_buffers.push(Vec::with_capacity(default_capacity));
        }

        Self {
            _candidate_buffers: candidate_buffers,
            visited_buffers,
            distance_buffers,
            default_capacity,
        }
    }

    /// 获取候选集缓冲区
    pub fn get_candidate_buffer(&self) -> CandidateBuffer {
        let buffer = self
            ._candidate_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.default_capacity));

        CandidateBuffer {
            buffer,
            pool: Arc::clone(&self._candidate_buffers),
        }
    }

    /// 获取访问标记缓冲区
    pub fn get_visited_buffer(&self, size: usize) -> VisitedBuffer {
        let mut buffer = self
            .visited_buffers
            .pop()
            .unwrap_or_else(|| vec![false; size]);

        // 调整大小并重置
        buffer.resize(size, false);
        buffer.fill(false);

        VisitedBuffer {
            buffer,
            pool: Arc::clone(&self.visited_buffers),
        }
    }

    /// 获取距离缓冲区
    pub fn get_distance_buffer(&self) -> DistanceBuffer {
        let buffer = self
            .distance_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.default_capacity));

        DistanceBuffer {
            buffer,
            pool: Arc::clone(&self.distance_buffers),
        }
    }
}

/// RAII包装器：候选集缓冲区
pub struct CandidateBuffer {
    pub buffer: Vec<SearchCandidate>,
    pool: Arc<ArrayQueue<Vec<SearchCandidate>>>,
}

impl Drop for CandidateBuffer {
    fn drop(&mut self) {
        self.buffer.clear();
        let _ = self.pool.push(std::mem::take(&mut self.buffer));
    }
}

impl std::ops::Deref for CandidateBuffer {
    type Target = Vec<SearchCandidate>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for CandidateBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

/// RAII包装器：访问标记缓冲区
pub struct VisitedBuffer {
    pub buffer: Vec<bool>,
    pool: Arc<ArrayQueue<Vec<bool>>>,
}

impl Drop for VisitedBuffer {
    fn drop(&mut self) {
        let _ = self.pool.push(std::mem::take(&mut self.buffer));
    }
}

impl std::ops::Deref for VisitedBuffer {
    type Target = Vec<bool>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for VisitedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

/// RAII包装器：距离缓冲区
pub struct DistanceBuffer {
    pub buffer: Vec<f32>,
    pool: Arc<ArrayQueue<Vec<f32>>>,
}

impl Drop for DistanceBuffer {
    fn drop(&mut self) {
        self.buffer.clear();
        let _ = self.pool.push(std::mem::take(&mut self.buffer));
    }
}

impl std::ops::Deref for DistanceBuffer {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for DistanceBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

lazy_static::lazy_static! {
    pub static ref _SEARCH_MEMORY_POOL: SearchMemoryPool = {
        let pool_size = num_cpus::get() * 4; // 每个CPU核心4个缓冲区
        let default_capacity = 1024;
        SearchMemoryPool::new(pool_size, default_capacity)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = SearchMemoryPool::new(4, 100);

        // 获取缓冲区
        let mut candidate_buf = pool.get_candidate_buffer();
        candidate_buf.push(SearchCandidate {
            distance: 1.0,
            idx: 0,
        });

        let visited_buf = pool.get_visited_buffer(100);
        assert_eq!(visited_buf.len(), 100);

        let mut distance_buf = pool.get_distance_buffer();
        distance_buf.push(1.0);

        // 缓冲区会在drop时自动返回池中
    }

    #[test]
    fn test_buffer_reuse() {
        let pool = SearchMemoryPool::new(1, 100);

        {
            let mut buf1 = pool.get_candidate_buffer();
            buf1.push(SearchCandidate {
                distance: 1.0,
                idx: 0,
            });
        } // buf1 返回池中

        {
            let buf2 = pool.get_candidate_buffer();
            assert_eq!(buf2.len(), 0); // 缓冲区已被清空
        }
    }
}
