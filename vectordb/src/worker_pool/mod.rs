#![allow(dead_code)]

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
    pub cpu_workers: usize,
    pub io_workers: usize,
    pub queue_size: usize,
    pub enable_cpu_affinity: bool,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        let num_cpus = num_cpus::get();
        Self {
            cpu_workers: num_cpus / 2,
            io_workers: num_cpus / 2,
            queue_size: 1000,
            enable_cpu_affinity: false,
        }
    }
}

pub struct OptimizedWorkerPool {
    cpu_pool: CpuWorkerPool,
    io_runtime: Arc<Runtime>,
    continuation_tx: Sender<ContinuationTask>,
    continuation_rx: Arc<Mutex<Receiver<ContinuationTask>>>,
    _continuation_workers: Vec<JoinHandle<()>>,
}

impl OptimizedWorkerPool {
    pub fn new(config: WorkerPoolConfig) -> Self {
        let (continuation_tx, continuation_rx) = bounded(config.queue_size);

        let io_runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(config.io_workers)
                .thread_name("io-worker")
                .enable_all()
                .build()
                .expect("Failed to create IO runtime"),
        );

        let cpu_pool = CpuWorkerPool::new(
            config.cpu_workers,
            config.queue_size,
            config.enable_cpu_affinity,
        );

        let continuation_rx = Arc::new(Mutex::new(continuation_rx));

        let mut continuation_workers = Vec::new();
        for i in 0..2 {
            let rx = Arc::clone(&continuation_rx);
            let rt = Arc::clone(&io_runtime);

            let handle = thread::Builder::new()
                .name(format!("continuation-worker-{}", i))
                .spawn(move || loop {
                    let task = {
                        let rx = rx.lock();
                        rx.recv()
                    };

                    match task {
                        Ok(ContinuationTask::IoToCompute { task, response_tx }) => {
                            let result = task();
                            let _ = response_tx.send(result);
                        }
                        Ok(ContinuationTask::ComputeToIo {
                            future,
                            response_tx,
                        }) => {
                            rt.spawn(async move {
                                let result = future.await;
                                let _ = response_tx.send(result);
                            });
                        }
                        Ok(ContinuationTask::Shutdown) => break,
                        Err(_) => break,
                    }
                })
                .expect("Failed to spawn continuation worker");

            continuation_workers.push(handle);
        }

        Self {
            cpu_pool,
            io_runtime,
            continuation_tx,
            continuation_rx,
            _continuation_workers: continuation_workers,
        }
    }

    pub async fn execute_cpu_task<F, R>(&self, task: F) -> Result<R, WorkerError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.cpu_pool.execute(task).await
    }

    pub async fn execute_io_task<F>(&self, future: F) -> F::Output
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.io_runtime.spawn(future).await.unwrap()
    }

    pub async fn execute_hybrid_task<F, R>(&self, task: HybridTask<F, R>) -> Result<R, WorkerError>
    where
        F: std::future::Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        match task {
            HybridTask::CpuBound(f) => self.cpu_pool.execute(f).await,
            HybridTask::IoBound(future) => Ok(self.io_runtime.spawn(future).await.unwrap()),
            HybridTask::Mixed { cpu_part, io_part } => {
                let cpu_result = self.cpu_pool.execute(cpu_part).await?;

                let (tx, rx) = oneshot::channel();
                self.continuation_tx
                    .send(ContinuationTask::ComputeToIo {
                        future: Box::pin(async move {
                            let result = io_part(cpu_result).await;
                            Box::new(result) as Box<dyn std::any::Any + Send>
                        }),
                        response_tx: tx,
                    })
                    .map_err(|_| WorkerError::QueueFull)?;

                let boxed_result = rx.await.map_err(|_| WorkerError::TaskCancelled)?;
                let result = *boxed_result
                    .downcast::<R>()
                    .map_err(|_| WorkerError::TypeMismatch)?;
                Ok(result)
            }
        }
    }

    pub fn shutdown(self) {
        for _ in 0..self._continuation_workers.len() {
            let _ = self.continuation_tx.send(ContinuationTask::Shutdown);
        }

        self.cpu_pool.shutdown();
    }
}

struct CpuWorkerPool {
    tx: Sender<CpuTask>,
    _workers: Vec<JoinHandle<()>>,
}

impl CpuWorkerPool {
    fn new(num_workers: usize, queue_size: usize, enable_affinity: bool) -> Self {
        let (tx, rx) = bounded(queue_size);

        let mut workers = Vec::new();

        for i in 0..num_workers {
            let rx = rx.clone();

            let handle = thread::Builder::new()
                .name(format!("cpu-worker-{}", i))
                .spawn(move || {
                    if enable_affinity {
                        if let Some(core_ids) = core_affinity::get_core_ids() {
                            if let Some(core_id) = core_ids.get(i % core_ids.len()) {
                                let _ = core_affinity::set_for_current(*core_id);
                            }
                        }
                    }

                    loop {
                        match rx.recv() {
                            Ok(CpuTask::Execute { task, response_tx }) => {
                                let result = task();
                                let _ = response_tx.send(result);
                            }
                            Ok(CpuTask::Shutdown) => break,
                            Err(_) => break,
                        }
                    }
                })
                .expect("Failed to spawn CPU worker");

            workers.push(handle);
        }

        Self {
            tx,
            _workers: workers,
        }
    }

    async fn execute<F, R>(&self, task: F) -> Result<R, WorkerError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        self.tx
            .send(CpuTask::Execute {
                task: Box::new(move || Box::new(task()) as Box<dyn std::any::Any + Send>),
                response_tx: tx,
            })
            .map_err(|_| WorkerError::QueueFull)?;

        let result = rx.await.map_err(|_| WorkerError::TaskCancelled)?;

        result
            .downcast::<R>()
            .map(|boxed| *boxed)
            .map_err(|_| WorkerError::TypeMismatch)
    }

    fn shutdown(self) {
        for _ in &self._workers {
            let _ = self.tx.send(CpuTask::Shutdown);
        }
    }
}

enum CpuTask {
    Execute {
        task: Box<dyn FnOnce() -> Box<dyn std::any::Any + Send> + Send>,
        response_tx: oneshot::Sender<Box<dyn std::any::Any + Send>>,
    },
    Shutdown,
}

enum ContinuationTask {
    IoToCompute {
        task: Box<dyn FnOnce() -> Box<dyn std::any::Any + Send> + Send>,
        response_tx: oneshot::Sender<Box<dyn std::any::Any + Send>>,
    },
    ComputeToIo {
        future: std::pin::Pin<
            Box<dyn std::future::Future<Output = Box<dyn std::any::Any + Send>> + Send>,
        >,
        response_tx: oneshot::Sender<Box<dyn std::any::Any + Send>>,
    },
    Shutdown,
}

pub enum HybridTask<F, R> {
    CpuBound(Box<dyn FnOnce() -> R + Send>),
    IoBound(F),
    Mixed {
        cpu_part: Box<dyn FnOnce() -> Box<dyn std::any::Any + Send> + Send>,
        io_part: Box<dyn FnOnce(Box<dyn std::any::Any + Send>) -> F + Send>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStats {
    pub cpu_tasks_completed: u64,
    pub io_tasks_completed: u64,
    pub continuation_tasks: u64,
    pub queue_depth: usize,
    pub active_workers: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Worker queue is full")]
    QueueFull,

    #[error("Task was cancelled")]
    TaskCancelled,

    #[error("Type mismatch in task result")]
    TypeMismatch,

    #[error("Worker pool is shutting down")]
    ShuttingDown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cpu_task_execution() {
        let pool = OptimizedWorkerPool::new(WorkerPoolConfig::default());

        let result = pool
            .execute_cpu_task(|| {
                let mut sum = 0;
                for i in 1..=100 {
                    sum += i;
                }
                sum
            })
            .await
            .unwrap();

        assert_eq!(result, 5050);
    }

    #[tokio::test]
    async fn test_io_task_execution() {
        let pool = OptimizedWorkerPool::new(WorkerPoolConfig::default());

        let result = pool
            .execute_io_task(async {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                "io task completed"
            })
            .await;

        assert_eq!(result, "io task completed");
    }

    #[tokio::test]
    async fn test_hybrid_task() {
        let pool = OptimizedWorkerPool::new(WorkerPoolConfig::default());

        let task = HybridTask::<std::future::Ready<i32>, i32>::CpuBound(Box::new(|| {
            [1, 2, 3, 4, 5].iter().sum::<i32>()
        }));

        let result = pool.execute_hybrid_task(task).await.unwrap();
        assert_eq!(result, 15);
    }
}
