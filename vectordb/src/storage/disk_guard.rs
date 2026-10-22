//! A floor under free disk space on the data directory, checked before a write is accepted.
//!
//! Records, row files, the WAL and snapshots all live under one directory, and nothing
//! stopped a write when that disk was nearly full: the WAL append that finally failed made the
//! store read-only until restart (`ensure_writable`), and a snapshot that ran out of room
//! midway left less for the next one. Refusing writes while there is still room -- so that
//! deletes, compaction and a snapshot can still run -- is the same trade `memory_guard` makes
//! for memory. Weaviate does the same (`DISK_USE_READONLY_PERCENTAGE`).
//!
//! `DXMT_MIN_FREE_DISK_MB` sets the floor; `0` disables it. Unset, it is a twentieth of the
//! disk, at least 256 MB and at most 10 GB. A disk whose space cannot be read is treated as
//! having room: the guard must not become a new way for writes to fail.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const CACHE: Duration = Duration::from_secs(1);
const MIN_DEFAULT: u64 = 256 * 1024 * 1024;
const MAX_DEFAULT: u64 = 10 * 1024 * 1024 * 1024;

/// `(available, total)` bytes, or `None` when the disk could not be read.
type Space = Option<(u64, u64)>;

pub(crate) struct DiskGuard {
    path: Option<PathBuf>,
    /// `None` until the first reading, when the default floor needs the disk's size.
    floor: Mutex<Option<u64>>,
    configured: Option<u64>,
    cached: Mutex<Option<(Instant, Space)>>,
}

/// `(available, total)` bytes on the disk holding `path`: the mount point that is the
/// longest prefix of it.
fn space(path: &Path) -> Option<(u64, u64)> {
    let path = std::fs::canonicalize(path).ok()?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| (d.available_space(), d.total_space()))
}

impl DiskGuard {
    pub(crate) fn from_env(path: Option<&Path>) -> Self {
        let configured = crate::config::env::var("DXMT_MIN_FREE_DISK_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024);
        Self::with_floor(path, configured)
    }

    /// `floor` overrides the default; tests pass one rather than set the environment, which
    /// every other storage built in the same process would read.
    pub(crate) fn with_floor(path: Option<&Path>, floor: Option<u64>) -> Self {
        Self {
            path: path.map(Path::to_path_buf),
            floor: Mutex::new(floor),
            configured: floor,
            cached: Mutex::new(None),
        }
    }

    fn reading(&self) -> Space {
        let path = self.path.as_ref()?;
        let mut cached = self.cached.lock().ok()?;
        match *cached {
            Some((at, value)) if at.elapsed() < CACHE => value,
            _ => {
                let value = space(path);
                *cached = Some((Instant::now(), value));
                value
            }
        }
    }

    /// `Err` (503) when the data directory's disk has less free space than the floor.
    pub(crate) fn check(&self) -> crate::Result<()> {
        if self.configured == Some(0) {
            return Ok(());
        }
        let Some((available, total)) = self.reading() else {
            return Ok(());
        };
        let floor = {
            let mut floor = match self.floor.lock() {
                Ok(f) => f,
                Err(_) => return Ok(()),
            };
            *floor.get_or_insert_with(|| (total / 20).clamp(MIN_DEFAULT, MAX_DEFAULT))
        };
        if available >= floor {
            return Ok(());
        }
        Err(crate::Error::ServiceUnavailable {
            message: format!(
                "{} MB free on the data directory's disk, under the {} MB floor; writes are \
                 refused until space frees up -- deletes and snapshots still run (set \
                 DXMT_MIN_FREE_DISK_MB to change it, or 0 to disable)",
                available / 1024 / 1024,
                floor / 1024 / 1024
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_floor_no_disk_meets_refuses_and_zero_disables() {
        let dir = tempfile::tempdir().unwrap();
        let impossible = DiskGuard::with_floor(Some(dir.path()), Some(u64::MAX));
        let err = impossible.check().unwrap_err();
        assert!(err.to_string().contains("DXMT_MIN_FREE_DISK_MB"), "{err}");
        assert!(DiskGuard::with_floor(Some(dir.path()), Some(0))
            .check()
            .is_ok());
        assert!(DiskGuard::with_floor(Some(dir.path()), Some(1))
            .check()
            .is_ok());
        assert!(
            DiskGuard::with_floor(None, Some(u64::MAX)).check().is_ok(),
            "no directory, no guard"
        );
    }

    #[test]
    fn the_disk_holding_a_directory_is_found() {
        let dir = tempfile::tempdir().unwrap();
        let (available, total) = space(dir.path()).expect("the temp directory is on some disk");
        assert!(total > 0 && available <= total);
    }
}
