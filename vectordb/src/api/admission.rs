//! Query admission: a bound on searches in flight, a bounded queue in front of it, and a 429
//! for whatever does not fit.
//!
//! Every search runs on the request's own task and uses the CPU until it is done, so nothing
//! stopped a burst of them from taking every core and every byte of memory the candidate lists
//! need: under overload all of them slowed down together, the client timed out and retried,
//! and the retries were more load. A database under more load than it can serve should say so
//! early and cheaply. Weaviate added the same shape in 1.37/1.39 (a node-wide limit with a FIFO
//! queue shedding as 429) and Qdrant has rate limits in strict mode; this is the first half.
//!
//! Only reads are admitted here -- the search routes, `/query`, facets, `records/list` and
//! `records/count`. Writes already go through the single insertion processor, which is their
//! own queue.
//!
//! - `DXMT_MAX_CONCURRENT_QUERIES`: queries running at once. Default twice the cores; `0` turns
//!   admission off.
//! - `DXMT_QUERY_QUEUE`: queries allowed to wait for a slot. Default four times the limit.
//! - `DXMT_QUERY_QUEUE_TIMEOUT_MS`: how long one may wait. Default 5,000.
//!
//! A query that cannot run or wait gets `429 TOO_MANY_REQUESTS` with `Retry-After: 1`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tokio::sync::Semaphore;

pub struct Admission {
    permits: Arc<Semaphore>,
    limit: usize,
    queue: usize,
    wait: Duration,
    waiting: AtomicUsize,
    rejected: AtomicU64,
}

static ADMISSION: OnceLock<Option<Admission>> = OnceLock::new();

fn env_usize(name: &str) -> Option<usize> {
    crate::config::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
}

impl Admission {
    pub fn new(limit: usize, queue: usize, wait: Duration) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(limit)),
            limit,
            queue,
            wait,
            waiting: AtomicUsize::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    fn from_env() -> Option<Self> {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let limit = env_usize("DXMT_MAX_CONCURRENT_QUERIES").unwrap_or(cores * 2);
        if limit == 0 {
            return None;
        }
        let queue = env_usize("DXMT_QUERY_QUEUE").unwrap_or(limit * 4);
        let wait =
            Duration::from_millis(env_usize("DXMT_QUERY_QUEUE_TIMEOUT_MS").unwrap_or(5_000) as u64);
        Some(Self::new(limit, queue, wait))
    }

    /// The process's admission, read from the environment once; `None` when it is off.
    pub fn global() -> Option<&'static Admission> {
        ADMISSION.get_or_init(Self::from_env).as_ref()
    }

    /// A slot to run in, or `None` when there is no room to run or to wait.
    pub async fn admit(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        if let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() {
            return Some(permit);
        }
        // Reserve a place in the queue first, so the queue's bound is exact under contention.
        if self.waiting.fetch_add(1, Ordering::SeqCst) >= self.queue {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let permit = tokio::time::timeout(self.wait, Arc::clone(&self.permits).acquire_owned())
            .await
            .ok()
            .and_then(Result::ok);
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        if permit.is_none() {
            self.rejected.fetch_add(1, Ordering::Relaxed);
        }
        permit
    }

    /// `(running, waiting, rejected since start, limit)`, for `/metrics`.
    pub fn snapshot(&self) -> (usize, usize, u64, usize) {
        (
            self.limit - self.permits.available_permits(),
            self.waiting.load(Ordering::SeqCst),
            self.rejected.load(Ordering::Relaxed),
            self.limit,
        )
    }
}

/// Whether a request is a read this layer admits.
pub(crate) fn is_query(method: &Method, path: &str) -> bool {
    *method == Method::POST
        && (path.contains("/search")
            || path.ends_with("/hybrid_search")
            || path.ends_with("/batch_search")
            || path.ends_with("/query")
            || path.ends_with("/facets")
            || path.ends_with("/records/list")
            || path.ends_with("/records/count"))
}

pub async fn admit_queries(request: Request<Body>, next: Next<Body>) -> Response {
    let Some(admission) = Admission::global() else {
        return next.run(request).await;
    };
    if !is_query(request.method(), request.uri().path()) {
        return next.run(request).await;
    }
    match admission.admit().await {
        Some(_permit) => next.run(request).await,
        None => {
            let body = serde_json::json!({
                "success": false,
                "data": null,
                "error": {
                    "code": "TOO_MANY_REQUESTS",
                    "message": "the server is running as many queries as it is configured to \
                                (DXMT_MAX_CONCURRENT_QUERIES) and its queue is full or the wait \
                                ran out; retry shortly",
                },
            });
            let mut response = (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
            response
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn which_requests_are_queries() {
        for path in [
            "/api/v1/tables/t/search",
            "/api/v1/tables/t/search/hybrid",
            "/api/v1/tables/t/records/list",
            "/api/v1/query",
            "/api/v1/tables/t/facets",
        ] {
            assert!(is_query(&Method::POST, path), "{path}");
        }
        assert!(!is_query(&Method::POST, "/api/v1/tables/t/records"));
        assert!(!is_query(&Method::GET, "/api/v1/tables/t"));
        assert!(!is_query(&Method::PUT, "/api/v1/tables/t/documents/d"));
    }

    #[tokio::test]
    async fn the_limit_runs_the_queue_waits_and_the_rest_are_turned_away() {
        let a = Arc::new(Admission::new(2, 1, Duration::from_millis(200)));
        let p1 = a.admit().await.expect("first runs");
        let _p2 = a.admit().await.expect("second runs");
        // Third waits; free a slot before its wait runs out and it gets one.
        let waiter = {
            let a = Arc::clone(&a);
            tokio::spawn(async move { a.admit().await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(a.snapshot().1, 1, "one waiting");
        // Fourth finds the queue full: refused at once.
        assert!(a.admit().await.is_none());
        drop(p1);
        let _p3 = waiter
            .await
            .unwrap()
            .expect("the waiter got the freed slot");
        // A waiter whose wait runs out is refused too.
        assert!(a.admit().await.is_none(), "no slot within 200 ms");
        let (running, waiting, rejected, limit) = a.snapshot();
        assert_eq!((running, waiting, rejected, limit), (2, 0, 2, 2));
    }
}
