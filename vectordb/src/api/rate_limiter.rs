use axum::{
    extract::State,
    http::Request,
    middleware::Next,
    response::Response,
    body::Body,
};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    pub maxrequests: u64,
    pub windowduration: Duration,
    pub burstsize: u64,
    pub globallimit: Option<u64>,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            maxrequests: 100,
            windowduration: Duration::from_secs(60),
            burstsize: 10,
            globallimit: None,
        }
    }
}

// Rate limiter implementation
pub struct RateLimiter {
    requests: Arc<DashMap<String, Vec<Instant>>>,
    max_requests: usize,
    window: Duration,
    #[allow(dead_code)]
    semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            requests: Arc::new(DashMap::new()),
            max_requests,
            window,
            semaphore: Arc::new(Semaphore::new(1000)), // Default concurrent limit
        }
    }

    pub async fn check_rate_limit(&self, identifier: &str) -> bool {
        let now = Instant::now();
        let mut entry = self.requests.entry(identifier.to_string()).or_default();
        
        // Clean old requests
        entry.retain(|&time| now.duration_since(time) < self.window);
        
        // Check if under limit
        if entry.len() < self.max_requests {
            entry.push(now);
            true
        } else {
            false
        }
    }
}

// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(_rate_limiter): State<Arc<RateLimiter>>,
    req: Request<Body>,
    next: Next<Body>,
) -> Response {
    // For now, just pass through - implement rate limiting logic here
    next.run(req).await
}

#[allow(dead_code)]
fn extract_identifier(req: &Request<Body>) -> String {
    // Extract client identifier (IP, API key, etc.)
    req.headers()
        .get("x-forwarded-for")
        .or_else(|| req.headers().get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string()
}

// Adaptive rate limiter
pub struct AdaptiveRateLimiter {
    baselimiter: Arc<RateLimiter>,
    #[allow(dead_code)]
    cputhreshold: f64,
    #[allow(dead_code)]
    memorythreshold: f64,
}

impl AdaptiveRateLimiter {
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            baselimiter: Arc::new(RateLimiter::new(config.maxrequests as usize, config.windowduration)),
            cputhreshold: 0.8,
            memorythreshold: 0.85,
        }
    }

    pub async fn check_request(&self, identifier: &str) -> bool {
        self.baselimiter.check_rate_limit(identifier).await
    }
}

// Tiered rate limiter
pub struct TieredRateLimiter {
    limiters: Vec<(u8, Arc<RateLimiter>)>,
}

impl Default for TieredRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl TieredRateLimiter {
    pub fn new() -> Self {
        let limiters = vec![
            (0, Arc::new(RateLimiter::new(10, Duration::from_secs(60)))),
            (1, Arc::new(RateLimiter::new(50, Duration::from_secs(60)))),
            (2, Arc::new(RateLimiter::new(200, Duration::from_secs(60)))),
            (3, Arc::new(RateLimiter::new(1000, Duration::from_secs(60)))),
        ];

        Self { limiters }
    }

    pub async fn check_request(&self, identifier: &str, priority: u8) -> bool {
        for (p, limiter) in &self.limiters {
            if *p == priority {
                return limiter.check_rate_limit(identifier).await;
            }
        }
        false
    }
}
