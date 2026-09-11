//! A floor under available memory, checked before a write is accepted.
//!
//! Loading a 768-dimension dataset into a 62 GB box, the server allocated straight through
//! the end of memory: at roughly 7M records it had 54.9 GB of anonymous memory and the
//! machine stopped responding. ICMP still answered, so the kernel was alive, but sshd could
//! no longer fork, no OOM kill was ever recorded (the box had no swap), and it took a hard
//! reboot. Every write was accepted right up to that point.
//!
//! Refusing a write with 503 once memory is nearly gone is strictly better than that: the
//! caller learns the write did not happen, and whatever else runs on the machine keeps
//! running. This is the same contract `ensure_writable` already applies when local
//! durability has failed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// How long one reading is reused.
///
/// `/proc/meminfo` is a single small read, but it is on the write path and there is no
/// reason to read it twice in the same tick. At the fastest load this project has measured
/// (about 7,500 records/s, 8.5 KB each) half a second is under 40 MB of allocation between
/// checks, which is why the floor is measured in hundreds of megabytes rather than tens.
const CACHE_MILLIS: u64 = 500;

/// Default floor as a fraction of total memory, overridable with `DB_MIN_FREE_MEMORY_MB`.
///
/// It was a flat 1 GB, and 1 GB is not a floor on a large machine: a single batch insert into
/// a 10M-node graph allocates a `Visited` buffer of one `u32` per node per rayon thread --
/// 40 MB each, 1.3 GB across 32 threads -- so an operation can step over a 1 GB gap between
/// two samples. It did: a 10M load with a payload was OOM-killed at 122.6 GB resident on a
/// 125 GB box without the guard ever refusing a write. A twentieth of total is 6 GB there and
/// 25 MB on a 512 MB container, which is the right shape for a number meant to leave room for
/// one operation's burst.
const DEFAULT_FLOOR_FRACTION: u64 = 20;

/// Floor when total memory cannot be read at all.
const FALLBACK_FLOOR_MB: u64 = 1024;

pub(crate) struct MemoryGuard {
    floor_bytes: u64,
    cached_available: AtomicU64,
    cached_at_millis: AtomicU64,
    start: Instant,
}

impl MemoryGuard {
    pub(crate) fn from_env() -> Self {
        let configured = std::env::var("DB_MIN_FREE_MEMORY_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok());
        let floor_bytes = match configured {
            // An operator who names a number knows their machine; it is used as given.
            Some(mb) => mb * 1024 * 1024,
            None => match total_bytes() {
                Some(total) => total / DEFAULT_FLOOR_FRACTION,
                None => FALLBACK_FLOOR_MB * 1024 * 1024,
            },
        };
        if floor_bytes == 0 {
            tracing::info!("Write-side memory floor disabled");
        } else {
            tracing::info!(
                "Writes are refused below {} MB of available memory (DB_MIN_FREE_MEMORY_MB)",
                floor_bytes / 1024 / 1024
            );
        }
        Self::with_floor(floor_bytes)
    }

    /// A guard whose floor is already decided. `from_env` is the production path; the split
    /// lets a test ask for a floor no machine can meet without writing it into the process
    /// environment, where every other test that constructs a storage in parallel would read it
    /// too and have its writes refused.
    pub(crate) fn with_floor(floor_bytes: u64) -> Self {
        // Read once here rather than leaving the cache empty. A guard whose first reading is
        // "plenty" is off for its first CACHE_MILLIS, and the cheapest way to find that out
        // is a test that writes through it -- which is how this was found.
        let start = Instant::now();
        Self {
            floor_bytes,
            cached_available: AtomicU64::new(available_bytes().unwrap_or(u64::MAX)),
            cached_at_millis: AtomicU64::new(start.elapsed().as_millis() as u64),
            start,
        }
    }

    /// `Err` when accepting more data would take the machine towards the state described
    /// above. `Ok` when the check is disabled or unavailable — a guard that cannot read the
    /// system must not refuse writes.
    pub(crate) fn check(&self) -> crate::Result<()> {
        if self.floor_bytes == 0 {
            return Ok(());
        }
        let available = self.available_cached();
        if available >= self.floor_bytes {
            return Ok(());
        }
        Err(crate::Error::ServiceUnavailable {
            message: format!(
                "{} MB of memory available, under the {} MB floor; writes are refused until \
                 memory frees up (set DB_MIN_FREE_MEMORY_MB to change or 0 to disable)",
                available / 1024 / 1024,
                self.floor_bytes / 1024 / 1024
            ),
        })
    }

    fn available_cached(&self) -> u64 {
        let now = self.start.elapsed().as_millis() as u64;
        let last = self.cached_at_millis.load(Ordering::Relaxed);
        if now.saturating_sub(last) < CACHE_MILLIS {
            return self.cached_available.load(Ordering::Relaxed);
        }
        // An unreadable source reports "plenty": the guard exists to catch exhaustion, not
        // to become a new way for writes to fail.
        let available = available_bytes().unwrap_or(u64::MAX);
        self.cached_available.store(available, Ordering::Relaxed);
        self.cached_at_millis.store(now, Ordering::Relaxed);
        available
    }
}

/// Memory available to this process, in bytes.
///
/// Inside a container `/proc/meminfo` describes the host, not the limit the process is
/// actually held to, so the cgroup's own headroom wins when it is smaller.
fn available_bytes() -> Option<u64> {
    let host = meminfo_field("MemAvailable:");
    match cgroup_available() {
        Some(cg) => Some(host.map_or(cg, |h| h.min(cg))),
        None => host,
    }
}

/// Total memory the process may use: the host's, capped by a cgroup limit when there is one.
///
/// `None` where neither is readable (not Linux).
pub fn total_bytes() -> Option<u64> {
    let host = meminfo_field("MemTotal:");
    match cgroup_limit() {
        Some(limit) => Some(host.map_or(limit, |h| h.min(limit))),
        None => host,
    }
}

/// `/proc/meminfo` reports kB, whatever the field.
fn meminfo_field(prefix: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find(|line| line.starts_with(prefix))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

fn cgroup_limit() -> Option<u64> {
    read_u64("/sys/fs/cgroup/memory.max")
        .or_else(|| read_u64("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        // An unlimited v1 cgroup reports a number near u64::MAX rather than a word.
        .filter(|&limit| limit < u64::MAX / 2)
}

fn cgroup_available() -> Option<u64> {
    let limit = cgroup_limit()?;
    let used = read_u64("/sys/fs/cgroup/memory.current")
        .or_else(|| read_u64("/sys/fs/cgroup/memory/memory.usage_in_bytes"))?;
    Some(limit.saturating_sub(used))
}

/// `memory.max` holds the literal `max` when the cgroup is unlimited, which is not a number.
fn read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_guard_never_refuses() {
        let guard = MemoryGuard {
            floor_bytes: 0,
            cached_available: AtomicU64::new(0),
            cached_at_millis: AtomicU64::new(0),
            start: Instant::now(),
        };
        assert!(guard.check().is_ok(), "floor 0 means the check is off");
    }

    #[test]
    fn a_floor_above_what_is_available_refuses_with_both_numbers() {
        let guard = MemoryGuard {
            floor_bytes: 8 * 1024 * 1024 * 1024,
            cached_available: AtomicU64::new(1024 * 1024 * 1024),
            // Far enough in the future that the cached reading is used as-is.
            cached_at_millis: AtomicU64::new(u64::MAX / 2),
            start: Instant::now(),
        };
        let err = guard.check().expect_err("1 GB is under an 8 GB floor");
        let message = err.to_string();
        assert!(
            message.contains("1024 MB"),
            "says what is available: {message}"
        );
        assert!(
            message.contains("8192 MB"),
            "says what the floor is: {message}"
        );
    }

    #[test]
    fn an_unreadable_system_does_not_refuse_writes() {
        let guard = MemoryGuard {
            floor_bytes: 1024 * 1024 * 1024,
            cached_available: AtomicU64::new(u64::MAX),
            cached_at_millis: AtomicU64::new(u64::MAX / 2),
            start: Instant::now(),
        };
        assert!(
            guard.check().is_ok(),
            "a guard that cannot read the system must not become a new way for writes to fail"
        );
    }

    #[test]
    fn the_default_floor_scales_with_the_machine() {
        // A fraction rather than a constant, in both directions: big enough on a large box to
        // survive one operation's allocation burst, small enough on a container not to refuse
        // every write.
        assert_eq!(
            (125 * 1024 * 1024 * 1024_u64) / DEFAULT_FLOOR_FRACTION,
            6_710_886_400,
            "a 125 GB box gets a 6.25 GB floor, not the 1 GB it was OOM-killed under"
        );
        assert_eq!(
            (512 * 1024 * 1024_u64) / DEFAULT_FLOOR_FRACTION,
            26_843_545,
            "a 512 MB container gets 25 MB, not a floor that refuses everything"
        );
    }

    #[test]
    fn a_new_guard_is_active_immediately() {
        // The cache starts populated. With it empty, the first CACHE_MILLIS of the guard's
        // life reported u64::MAX and every write in that window sailed through.
        //
        // Not through `from_env` with `DB_MIN_FREE_MEMORY_MB` set. The environment is
        // process-wide and the unit tests share one process: while this test held the variable
        // at a billion MB, any test constructing an `IntelligentIndexedStorage` in parallel got
        // that floor and had its inserts refused -- a different victim each run. Reading the
        // variable is covered by `tests/memory_backpressure_test.rs`, alone in its binary.
        let guard = MemoryGuard::with_floor(1_000_000_000 * 1024 * 1024);
        assert!(
            guard.check().is_err(),
            "a floor no machine can meet must refuse from the first call, not after 500 ms"
        );
    }

    #[test]
    fn this_machine_reports_a_plausible_amount_of_memory() {
        let total = total_bytes().expect("Linux exposes MemTotal");
        let available = available_bytes().expect("Linux exposes MemAvailable");
        assert!(total > 0 && available <= total, "{available} of {total}");
    }
}
