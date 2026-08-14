//! Parallel execution framework for DXMT VectorDB

#![allow(clippy::arc_with_non_send_sync)]

use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use futures::future::join_all;
use crossbeam::channel::{bounded, Sender, Receiver};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::types::DenseVector;
use crate::embeddings::EmbeddingError;

/// Parallel executor for CPU-bound tasks
#[allow(dead_code)]
pub struct ParallelExecutor {
    thread_pool: Arc<ThreadPool>,
    work_stealing: Arc<WorkStealing>,
    semaphore: Arc<Semaphore>,
}

impl ParallelExecutor {
    pub fn new(num_threads: usize) -> Self {
        let thread_pool = Arc::new(ThreadPool::new(num_threads));
        let work_stealing = Arc::new(WorkStealing::new(num_threads));
        let semaphore = Arc::new(Semaphore::new(num_threads * 2));
        
        Self {
            thread_pool,
            work_stealing,
            semaphore,
        }
    }
    
    pub async fn map_parallel<T, R, F>(
        &self,
        items: Vec<T>,
        f: F,
    ) -> Result<Vec<R>, EmbeddingError>
    where
        T: Send + Sync + Clone + 'static,
        R: Send + Sync + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
    {
        let chunk_size = (items.len() / self.thread_pool.num_threads()).max(1);
        let chunks: Vec<_> = items.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut handles = Vec::new();
        for chunk in chunks {
            let f = f.clone();
            let permit = self.semaphore.clone().acquire_owned().await
                .map_err(|e| EmbeddingError::InternalError(e.to_string()))?;
            
            let handle = tokio::spawn(async move {
                let results: Vec<R> = chunk.into_iter().map(f).collect();
                drop(permit);
                results
            });
            
            handles.push(handle);
        }
        
        let results = join_all(handles).await;
        let mut all_results = Vec::new();
        
        for result in results {
            let chunk_results = result
                .map_err(|e| EmbeddingError::InternalError(e.to_string()))?;
            all_results.extend(chunk_results);
        }
        
        Ok(all_results)
    }
    
    pub async fn reduce_parallel<T, F>(
        &self,
        items: Vec<T>,
        identity: T,
        f: F,
    ) -> Result<T, EmbeddingError>
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        if items.is_empty() {
            return Ok(identity);
        }

        let chunk_size = (items.len() / self.thread_pool.num_threads()).max(1);
        let chunks: Vec<_> = items.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut handles = Vec::new();
        for chunk in chunks {
            let f = f.clone();
            let identity = identity.clone();
            let permit = self.semaphore.clone().acquire_owned().await
                .map_err(|e| EmbeddingError::InternalError(e.to_string()))?;

            let handle = tokio::spawn(async move {
                let result = chunk.into_iter().fold(identity, &f);
                drop(permit);
                result
            });

            handles.push(handle);
        }

        let results = join_all(handles).await;
        let mut final_result = identity;

        for result in results {
            let chunk_result = result
                .map_err(|e| EmbeddingError::InternalError(e.to_string()))?;
            final_result = f(final_result, chunk_result);
        }

        Ok(final_result)
    }
    
    pub fn parallel_cosine_similarity(
        &self,
        query: &[f32],
        vectors: &[DenseVector],
    ) -> Vec<f32> {
        vectors.par_iter()
            .map(|v| cosine_similarity(query, v))
            .collect()
    }
    
    pub fn parallel_euclidean_distance(
        &self,
        query: &[f32],
        vectors: &[DenseVector],
    ) -> Vec<f32> {
        vectors.par_iter()
            .map(|v| euclidean_distance(query, v))
            .collect()
    }
    
    pub async fn batch_process<T, R, F>(
        &self,
        items: Vec<T>,
        batch_size: usize,
        f: F,
    ) -> Result<Vec<R>, EmbeddingError>
    where
        T: Send + Sync + Clone + 'static,
        R: Send + Sync + 'static,
        F: Fn(Vec<T>) -> Vec<R> + Send + Sync + Clone + 'static,
    {
        let batches: Vec<_> = items.chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut handles = Vec::new();
        for batch in batches {
            let f = f.clone();
            let permit = self.semaphore.clone().acquire_owned().await
                .map_err(|e| EmbeddingError::InternalError(e.to_string()))?;
            
            let handle = tokio::spawn(async move {
                let results = f(batch);
                drop(permit);
                results
            });
            
            handles.push(handle);
        }
        
        let results = join_all(handles).await;
        let mut all_results = Vec::new();
        
        for result in results {
            let batch_results = result
                .map_err(|e| EmbeddingError::InternalError(e.to_string()))?;
            all_results.extend(batch_results);
        }
        
        Ok(all_results)
    }
}

/// Thread pool for CPU-bound tasks
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let (sender, receiver) = bounded(size * 2);
        let receiver = Arc::new(RwLock::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            workers.push(Worker::new(id, receiver.clone()));
        }
        
        Self { workers, sender }
    }
    
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
    
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }
}

#[allow(dead_code)]
struct Worker {
    id: usize,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<RwLock<Receiver<Job>>>) -> Self {
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                loop {
                    let receiver = receiver.read().await;
                    let job = receiver.recv();
                    drop(receiver);
                    
                    match job {
                        Ok(job) => job(),
                        Err(_) => break,
                    }
                }
            });
        });
        
        Self {
            id,
            thread: Some(thread),
        }
    }
}

/// Work-stealing scheduler
pub struct WorkStealing {
    queues: Vec<Arc<RwLock<VecDeque<Task>>>>,
    num_workers: usize,
}

type Task = Box<dyn FnOnce() + Send>;

impl WorkStealing {
    pub fn new(num_workers: usize) -> Self {
        let mut queues = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            queues.push(Arc::new(RwLock::new(VecDeque::new())));
        }
        
        Self {
            queues,
            num_workers,
        }
    }
    
    pub async fn submit<F>(&self, worker_id: usize, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let queue = &self.queues[worker_id % self.num_workers];
        let mut q = queue.write().await;
        q.push_back(Box::new(task));
    }
    
    pub async fn steal_task(&self, worker_id: usize) -> Option<Task> {
        // Try to get task from own queue first
        {
            let mut queue = self.queues[worker_id].write().await;
            if let Some(task) = queue.pop_front() {
                return Some(task);
            }
        }
        
        // Try to steal from other queues
        for i in 1..self.num_workers {
            let target = (worker_id + i) % self.num_workers;
            let mut queue = self.queues[target].write().await;
            if let Some(task) = queue.pop_back() {
                return Some(task);
            }
        }
        
        None
    }
    
    pub async fn run_worker(&self, worker_id: usize) {
        loop {
            if let Some(task) = self.steal_task(worker_id).await {
                task();
            } else {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            }
        }
    }
}

/// Parallel matrix operations
pub struct ParallelMatrixOps;

impl ParallelMatrixOps {
    pub fn matrix_multiply_parallel(
        a: &[Vec<f32>],
        b: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let m = a.len();
        let n = b[0].len();
        let k = b.len();
        
        (0..m).into_par_iter()
            .map(|i| {
                (0..n).map(|j| {
                    (0..k).map(|l| a[i][l] * b[l][j]).sum()
                }).collect()
            })
            .collect()
    }
    
    pub fn batch_cosine_similarity_parallel(
        queries: &[DenseVector],
        vectors: &[DenseVector],
    ) -> Vec<Vec<f32>> {
        queries.par_iter()
            .map(|query| {
                vectors.iter()
                    .map(|vector| cosine_similarity(query, vector))
                    .collect()
            })
            .collect()
    }
    
    pub fn batch_normalize_parallel(
        vectors: &[DenseVector],
    ) -> Vec<DenseVector> {
        vectors.par_iter()
            .map(|vector| {
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vector.iter().map(|x| x / norm).collect()
                } else {
                    vector.clone()
                }
            })
            .collect()
    }
}

// Helper functions
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_parallel_map() {
        let executor = ParallelExecutor::new(4);
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8];
        
        let results = executor.map_parallel(items, |x| x * 2).await.unwrap();
        assert_eq!(results, vec![2, 4, 6, 8, 10, 12, 14, 16]);
    }
    
    #[tokio::test]
    async fn test_parallel_reduce() {
        let executor = ParallelExecutor::new(4);
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8];
        
        let sum = executor.reduce_parallel(items, 0, |acc, x| acc + x).await.unwrap();
        assert_eq!(sum, 36);
    }
    
    #[test]
    fn test_parallel_matrix_multiply() {
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        
        let result = ParallelMatrixOps::matrix_multiply_parallel(&a, &b);
        assert_eq!(result[0][0], 19.0); // 1*5 + 2*7
        assert_eq!(result[0][1], 22.0); // 1*6 + 2*8
        assert_eq!(result[1][0], 43.0); // 3*5 + 4*7
        assert_eq!(result[1][1], 50.0); // 3*6 + 4*8
    }
}