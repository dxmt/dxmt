//! Where everything this server writes goes, decided in one place.
//!
//! It used to be decided in five, with no relationship between them: `DB_WAL_DIR` (which held
//! the whole database despite its name), `DB_BACKUP_ROOT`, `DB_SSD_CACHE_DIR` and
//! `DB_PROCESSES_DATA_DIR` — three of those relative to the *working directory*, so starting
//! the server from elsewhere silently used different ones — plus `DB_STORAGE_PATH`, and a
//! `VECTORDB_ROOT_DIR` in the Dockerfile that no code has ever read.
//!
//! One root now, and everything hangs off it:
//!
//! ```text
//! ~/.dxmt/                DB_ROOT_DIR
//! ├── store/              the database. Losing this loses the data
//! │   ├── wal.log
//! │   ├── records/
//! │   ├── vectors/
//! │   ├── snapshot/
//! │   └── fulltext/
//! ├── backups/            DB_BACKUP_ROOT — the only place POST /backup may write
//! └── cache/              derived. Deleting it costs time, not data
//!     └── ssd/
//! ```
//!
//! The three children have three different lifetimes, and the layout says which is which.
//! That distinction has cost this project before: `snapshot/` was treated as a cache until the
//! WAL started being compacted behind it, at which point it became the only copy of everything
//! it covered.
//!
//! Each child can still be set on its own. Setting only `DB_ROOT_DIR` sets all of them, which
//! is the point — one variable for "put this somewhere else".
//!
//! # Why `store/` and not `wal/`
//!
//! A directory called `wal` that holds the records, the vectors, the snapshots and the
//! full-text indexes is the kind of name this codebase keeps paying for. The rename costs
//! nothing today — `~/.dxmt/wal` has existed for one commit and was never released — and the
//! previous names are still honoured, with a warning, so nobody starts empty next to their own
//! data.

use std::path::{Path, PathBuf};

/// A path that was chosen for the operator, and what to tell them about it.
pub struct Resolved {
    pub path: PathBuf,
    /// Said once at startup. `None` when there is nothing surprising to report.
    pub note: Option<String>,
}

/// The directory layout, rooted at `DB_ROOT_DIR`.
pub struct Layout {
    root: PathBuf,
    /// Whether the operator named the root, as opposed to it being defaulted for them. It
    /// decides one thing: whether a store moved out on its own drags the backups with it.
    root_explicit: bool,
    /// Set when the root itself could not be determined from the environment.
    root_note: Option<String>,
}

/// What `DB_ROOT_DIR` defaults to under the user's home.
const ROOT_UNDER_HOME: &str = ".dxmt";

fn from_env(name: &str) -> Option<PathBuf> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => Some(PathBuf::from(value)),
        _ => None,
    }
}

impl Layout {
    /// Read the root from the environment, or put it under the user's home.
    pub fn from_env() -> Self {
        if let Some(root) = from_env("DB_ROOT_DIR") {
            return Self {
                root,
                root_explicit: true,
                root_note: None,
            };
        }
        match std::env::var_os("HOME").map(PathBuf::from) {
            Some(home) => Self {
                root: home.join(ROOT_UNDER_HOME),
                root_explicit: false,
                root_note: None,
            },
            // `/nonexistent` under systemd's `DynamicUser=`, `/` in some images. Falling back to
            // the working directory keeps the server usable and says why that is a poor place
            // for it, rather than failing on a path nobody chose.
            None => Self {
                root: PathBuf::from("."),
                root_explicit: false,
                root_note: Some(
                    "no home directory, so data goes under the working directory — starting the \
                     server elsewhere will open a different database. Set DB_ROOT_DIR."
                        .to_string(),
                ),
            },
        }
    }

    /// A layout rooted wherever the caller says. For tests, and for callers that already know.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            root_explicit: true,
            root_note: None,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The database: the WAL and everything the WAL protects.
    ///
    /// `DB_WAL_DIR` still names it, because that is what it has always been called from the
    /// outside and what every existing deployment sets.
    pub fn store(&self) -> Resolved {
        match from_env("DB_WAL_DIR") {
            Some(path) => Resolved { path, note: None },
            None => Resolved {
                path: self.root.join("store"),
                note: self.root_note.clone(),
            },
        }
    }

    /// Where `POST /backup` may write, and nowhere else.
    ///
    /// Beside the store rather than under the root when the store was moved out on its own
    /// *and nobody named a root*: `DB_WAL_DIR=/fast/nvme/store` with no `DB_ROOT_DIR` says
    /// where the database is and nothing about `~/.dxmt`, so backups belong near the database
    /// rather than under a default root that would then hold nothing else. An operator who
    /// did name a root meant it, and gets `$DB_ROOT_DIR/backups` wherever the store went.
    ///
    /// A backup may not go *inside* the store — the copy would include itself — so the
    /// store's parent is the nearest place that is allowed.
    pub fn backups(&self) -> PathBuf {
        self.backups_for(&self.store().path)
    }

    /// The same rule, for a store this layout did not choose.
    ///
    /// `IntelligentIndexedStorage` may hold a WAL directory that came from a constructor
    /// rather than the environment — every test does. It has to reach the same answer as
    /// `prepare` does, or the directory checked at startup is not the directory written to.
    pub fn backups_for(&self, store: &Path) -> PathBuf {
        if let Some(explicit) = from_env("DB_BACKUP_ROOT") {
            return explicit;
        }
        if self.root_explicit || store.starts_with(&self.root) {
            return self.root.join("backups");
        }
        store
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Derived state that can be deleted. Nothing here is the only copy of anything.
    pub fn cache(&self) -> PathBuf {
        from_env("DB_CACHE_DIR").unwrap_or_else(|| self.root.join("cache"))
    }

    /// The tiered storage's warm tier.
    pub fn ssd_cache(&self) -> PathBuf {
        from_env("DB_SSD_CACHE_DIR").unwrap_or_else(|| self.cache().join("ssd"))
    }

    /// Create everything this layout names, and prove each one can be written to.
    ///
    /// Done once at startup, because the alternative is finding out on the first write — and
    /// the first write to `backups/` may be weeks after the server came up. `create_dir_all`
    /// succeeding is not the same as being able to write: a directory that exists with `r-x`
    /// passes it and fails everything after, so each one takes a probe file and removes it.
    ///
    /// The error names the path, the reason, and the variable that moves it. A message that
    /// says only `Permission denied` is worth very little when the path was chosen by a
    /// default nobody typed.
    pub fn prepare(&self) -> Result<(), String> {
        for (path, variable) in [
            (self.root.to_path_buf(), "DB_ROOT_DIR"),
            (self.store().path, "DB_WAL_DIR"),
            (self.backups(), "DB_BACKUP_ROOT"),
            (self.cache(), "DB_CACHE_DIR"),
        ] {
            if let Err(reason) = Self::usable(&path) {
                return Err(format!(
                    "cannot use {}: {reason}. Set {variable} to a writable directory\
                     {}",
                    path.display(),
                    if variable == "DB_ROOT_DIR" {
                        String::new()
                    } else {
                        ", or DB_ROOT_DIR to move all of them at once".to_string()
                    }
                ));
            }
        }
        Ok(())
    }

    /// Exists (or can be made) and accepts a file.
    fn usable(path: &Path) -> Result<(), String> {
        if let Err(e) = std::fs::create_dir_all(path) {
            return Err(format!("{e}"));
        }
        let probe = path.join(".dxmt-write-test");
        match std::fs::write(&probe, b"") {
            Ok(()) => {
                let _ = std::fs::remove_file(&probe);
                Ok(())
            }
            Err(e) => Err(format!("the directory exists but is not writable ({e})")),
        }
    }

    /// The process supervisor's state, when that is enabled at all.
    pub fn processes(&self) -> PathBuf {
        from_env("DB_PROCESSES_DATA_DIR").unwrap_or_else(|| self.root.join("processes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Every child hangs off the root, so one variable moves the lot.
    ///
    /// `#[serial]` on all of these: they set process-wide environment variables, and cargo runs
    /// tests in one process on many threads. `tests/knn_graph_test.rs` learned that the noisy
    /// way — a neighbouring test reading a variable this one had just changed.
    #[test]
    #[serial]
    fn one_root_places_everything() {
        let guard = Env::clear();
        std::env::set_var("DB_ROOT_DIR", "/srv/dxmt");

        let layout = Layout::from_env();
        assert_eq!(layout.root(), Path::new("/srv/dxmt"));
        assert_eq!(layout.store().path, Path::new("/srv/dxmt/store"));
        assert_eq!(layout.backups(), Path::new("/srv/dxmt/backups"));
        assert_eq!(layout.cache(), Path::new("/srv/dxmt/cache"));
        assert_eq!(layout.ssd_cache(), Path::new("/srv/dxmt/cache/ssd"));
        assert_eq!(layout.processes(), Path::new("/srv/dxmt/processes"));
        assert!(layout.store().note.is_none());
        drop(guard);
    }

    /// Under the user's home, not the working directory. That was the whole point: `./data/wal`
    /// meant `cd elsewhere && serve` opened a different, empty database.
    #[test]
    #[serial]
    fn the_default_root_is_under_home() {
        let guard = Env::clear();
        std::env::set_var("HOME", "/home/someone");

        let layout = Layout::from_env();
        assert_eq!(layout.root(), Path::new("/home/someone/.dxmt"));
        assert_eq!(layout.store().path, Path::new("/home/someone/.dxmt/store"));
        drop(guard);
    }

    /// Each child can still be moved on its own, and does not drag the others with it.
    #[test]
    #[serial]
    fn a_child_can_be_moved_without_moving_the_rest() {
        let guard = Env::clear();
        std::env::set_var("DB_ROOT_DIR", "/srv/dxmt");
        std::env::set_var("DB_WAL_DIR", "/fast/nvme/store");
        std::env::set_var("DB_SSD_CACHE_DIR", "/scratch");

        let layout = Layout::from_env();
        assert_eq!(layout.store().path, Path::new("/fast/nvme/store"));
        assert_eq!(layout.ssd_cache(), Path::new("/scratch"));
        // Untouched children stay where the root puts them.
        assert_eq!(layout.backups(), Path::new("/srv/dxmt/backups"));
        assert_eq!(layout.cache(), Path::new("/srv/dxmt/cache"));
        drop(guard);
    }

    /// A store moved out on its own takes the backups with it — but only when nobody named a
    /// root. Two places used to decide this independently, `Layout` and
    /// `IntelligentIndexedStorage::backup_root`, which is how the directory checked at startup
    /// could differ from the directory written to.
    #[test]
    #[serial]
    fn backups_follow_a_store_that_was_moved_out_alone() {
        let guard = Env::clear();
        std::env::set_var("HOME", "/home/someone");
        std::env::set_var("DB_WAL_DIR", "/fast/nvme/store");

        let layout = Layout::from_env();
        assert_eq!(
            layout.backups(),
            Path::new("/fast/nvme"),
            "with only the default root, backups belong beside the database"
        );
        // And the storage layer reaches the same answer for the same store.
        assert_eq!(
            layout.backups_for(Path::new("/fast/nvme/store")),
            layout.backups()
        );

        // Naming a root means it: the backups stay there.
        std::env::set_var("DB_ROOT_DIR", "/srv/dxmt");
        assert_eq!(Layout::from_env().backups(), Path::new("/srv/dxmt/backups"));
        drop(guard);
    }

    /// No home and no `DB_ROOT_DIR` — `/nonexistent` under systemd's `DynamicUser=`, `/` in
    /// some images. Usable, and loud about why this is a poor place for a database.
    #[test]
    #[serial]
    fn without_a_home_it_says_so() {
        let guard = Env::clear();
        std::env::remove_var("HOME");

        let layout = Layout::from_env();
        assert_eq!(layout.root(), Path::new("."));
        let note = layout
            .store()
            .note
            .expect("a fallback root has to explain itself");
        assert!(note.contains("DB_ROOT_DIR"), "{note}");
        drop(guard);
    }

    /// An empty value is not a value: `DB_ROOT_DIR=` falls through to the default rather than
    /// rooting the database at the empty path.
    #[test]
    #[serial]
    fn an_empty_variable_is_ignored() {
        let guard = Env::clear();
        std::env::set_var("HOME", "/home/someone");
        std::env::set_var("DB_ROOT_DIR", "   ");

        assert_eq!(Layout::from_env().root(), Path::new("/home/someone/.dxmt"));
        drop(guard);
    }

    /// Creating the directories is not the check; being able to write to them is.
    ///
    /// A directory that exists with `r-x` passes `create_dir_all` and fails every write after
    /// it, which is why `prepare` puts a file in each one. The startup path refuses to serve
    /// when this fails, so the discovery happens before anything depends on it — an unwritable
    /// `backups/` would otherwise be found by whoever takes the first backup.
    #[test]
    #[serial]
    fn prepare_makes_the_layout_and_proves_it_is_writable() {
        let guard = Env::clear();
        let root = tempfile::TempDir::new().unwrap();
        std::env::set_var("DB_ROOT_DIR", root.path());

        let layout = Layout::from_env();
        layout.prepare().expect("a fresh writable root");
        for child in ["store", "backups", "cache"] {
            assert!(root.path().join(child).is_dir(), "{child}/ was not created");
        }
        // And nothing is left behind by the probe.
        assert!(!root.path().join(".dxmt-write-test").exists());
        drop(guard);
    }

    #[test]
    #[serial]
    fn prepare_refuses_a_directory_it_cannot_write_to() {
        use std::os::unix::fs::PermissionsExt;

        let guard = Env::clear();
        let root = tempfile::TempDir::new().unwrap();
        let readonly = root.path().join("readonly");
        std::fs::create_dir(&readonly).unwrap();
        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o555)).unwrap();
        std::env::set_var("DB_ROOT_DIR", &readonly);

        let error = Layout::from_env()
            .prepare()
            .expect_err("a read-only root is not usable");
        assert!(error.contains(&readonly.display().to_string()), "{error}");
        assert!(error.contains("not writable"), "{error}");
        assert!(error.contains("DB_ROOT_DIR"), "{error}");

        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o755)).unwrap();
        drop(guard);
    }

    /// The message has to say which variable moves the thing that failed, not just the root.
    #[test]
    #[serial]
    fn a_child_that_fails_names_its_own_variable() {
        use std::os::unix::fs::PermissionsExt;

        let guard = Env::clear();
        let root = tempfile::TempDir::new().unwrap();
        let readonly = root.path().join("readonly");
        std::fs::create_dir(&readonly).unwrap();
        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o555)).unwrap();
        std::env::set_var("DB_ROOT_DIR", root.path());
        std::env::set_var("DB_BACKUP_ROOT", readonly.join("backups"));

        let error = Layout::from_env()
            .prepare()
            .expect_err("an unwritable backup root");
        assert!(error.contains("DB_BACKUP_ROOT"), "{error}");
        assert!(
            error.contains("DB_ROOT_DIR"),
            "it should also offer to move everything: {error}"
        );

        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o755)).unwrap();
        drop(guard);
    }

    /// Saves and restores every variable this module reads, so one test cannot leave the
    /// environment changed under the next.
    struct Env(Vec<(&'static str, Option<std::ffi::OsString>)>);

    impl Env {
        const NAMES: [&'static str; 7] = [
            "DB_ROOT_DIR",
            "DB_WAL_DIR",
            "DB_BACKUP_ROOT",
            "DB_CACHE_DIR",
            "DB_SSD_CACHE_DIR",
            "DB_PROCESSES_DATA_DIR",
            "HOME",
        ];

        fn clear() -> Self {
            let saved = Self::NAMES
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect();
            for name in Self::NAMES {
                std::env::remove_var(name);
            }
            Env(saved)
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            for (name, value) in &self.0 {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}
