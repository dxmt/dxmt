//! Where everything this server writes goes, decided in one place.
//!
//! It used to be decided in five, with no relationship between them: `DXMT_WAL_DIR` (which held
//! the whole database despite its name), `DXMT_BACKUP_ROOT`, `DXMT_SSD_CACHE_DIR` and
//! `DXMT_PROCESSES_DATA_DIR` — three of those relative to the *working directory*, so starting
//! the server from elsewhere silently used different ones — plus `DXMT_STORAGE_PATH`, and a
//! `VECTORDB_ROOT_DIR` in the Dockerfile that no code has ever read.
//!
//! One root now, and everything hangs off it:
//!
//! ```text
//! ~/.dxmt/                DXMT_ROOT_DIR
//! ├── store/              the database. Losing this loses the data
//! │   ├── wal.log
//! │   ├── records/
//! │   ├── vectors/
//! │   ├── snapshot/
//! │   └── fulltext/
//! ├── backups/            DXMT_BACKUP_ROOT — the only place POST /backup may write
//! └── cache/              derived. Deleting it costs time, not data
//!     └── ssd/
//! ```
//!
//! The three children have three different lifetimes, and the layout says which is which.
//! That distinction has cost this project before: `snapshot/` was treated as a cache until the
//! WAL started being compacted behind it, at which point it became the only copy of everything
//! it covered.
//!
//! Each child can still be set on its own. Setting only `DXMT_ROOT_DIR` sets all of them, which
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

/// The directory layout, rooted at `DXMT_ROOT_DIR`.
pub struct Layout {
    root: PathBuf,
    /// Whether the operator named the root, as opposed to it being defaulted for them. It
    /// decides one thing: whether a store moved out on its own drags the backups with it.
    root_explicit: bool,
    /// Set when the root itself could not be determined from the environment.
    root_note: Option<String>,
    /// The per-child overrides, read once.
    store: Option<PathBuf>,
    backups: Option<PathBuf>,
    cache: Option<PathBuf>,
    ssd_cache: Option<PathBuf>,
    processes: Option<PathBuf>,
}

/// What `DXMT_ROOT_DIR` defaults to under the user's home.
const ROOT_UNDER_HOME: &str = ".dxmt";

impl Layout {
    /// Read the whole layout from the environment, once.
    pub fn from_env() -> Self {
        Self::resolve(|name| crate::config::env::var(name).ok())
    }

    /// The same, from any source of values.
    ///
    /// Everything is resolved here rather than looked up when each path is asked for, which is
    /// what makes a `Layout` a snapshot of the configuration instead of a live view of the
    /// process environment. The tests below construct one from a map and mutate nothing: an
    /// earlier version set real environment variables and restored them afterwards, and other
    /// tests running on other threads read them in between. `cargo test --lib` passed alone and
    /// three `config` tests failed when the whole suite ran together.
    pub fn resolve(get: impl Fn(&str) -> Option<String>) -> Self {
        let value = |name: &str| -> Option<PathBuf> {
            get(name)
                .filter(|v| !v.trim().is_empty())
                .map(PathBuf::from)
        };

        let (root, root_explicit, root_note) = match value("DXMT_ROOT_DIR") {
            Some(root) => (root, true, None),
            None => match value("HOME") {
                Some(home) => (home.join(ROOT_UNDER_HOME), false, None),
                // `/nonexistent` under systemd's `DynamicUser=`, `/` in some images. Falling
                // back to the working directory keeps the server usable and says why that is a
                // poor place for it, rather than failing on a path nobody chose.
                None => (
                    PathBuf::from("."),
                    false,
                    Some(
                        "no home directory, so data goes under the working directory — starting \
                         the server elsewhere will open a different database. Set DXMT_ROOT_DIR."
                            .to_string(),
                    ),
                ),
            },
        };

        Self {
            root,
            root_explicit,
            root_note,
            store: value("DXMT_WAL_DIR"),
            backups: value("DXMT_BACKUP_ROOT"),
            cache: value("DXMT_CACHE_DIR"),
            ssd_cache: value("DXMT_SSD_CACHE_DIR"),
            processes: value("DXMT_PROCESSES_DATA_DIR"),
        }
    }

    /// A layout rooted wherever the caller says. For tests, and for callers that already know.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            root_explicit: true,
            root_note: None,
            store: None,
            backups: None,
            cache: None,
            ssd_cache: None,
            processes: None,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The database: the WAL and everything the WAL protects.
    ///
    /// `DXMT_WAL_DIR` still names it, because that is what it has always been called from the
    /// outside and what every existing deployment sets.
    pub fn store(&self) -> Resolved {
        match self.store.clone() {
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
    /// *and nobody named a root*: `DXMT_WAL_DIR=/fast/nvme/store` with no `DXMT_ROOT_DIR` says
    /// where the database is and nothing about `~/.dxmt`, so backups belong near the database
    /// rather than under a default root that would then hold nothing else. An operator who
    /// did name a root meant it, and gets `$DXMT_ROOT_DIR/backups` wherever the store went.
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
        if let Some(explicit) = self.backups.clone() {
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
        self.cache
            .clone()
            .unwrap_or_else(|| self.root.join("cache"))
    }

    /// The tiered storage's warm tier.
    pub fn ssd_cache(&self) -> PathBuf {
        self.ssd_cache
            .clone()
            .unwrap_or_else(|| self.cache().join("ssd"))
    }

    /// Create the directories this server will use, and prove each one can be written to.
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
        // Exactly the directories this server will use, which is not the same as every
        // directory this layout can name. A server started with `DXMT_WAL_DIR` pointing outside
        // the root does not touch the root — and creating `~/.dxmt/cache` for it anyway leaves
        // a directory nobody asked for and proves the writability of something that will never
        // be written. `cache/` is absent for its own reason: it is made by whatever fills it,
        // if anything ever does.
        //
        // The root is not listed either. When the children are under it, creating them creates
        // it; when they are not, it is not this server's business.
        for (path, variable) in [
            (self.store().path, "DXMT_WAL_DIR"),
            (self.backups(), "DXMT_BACKUP_ROOT"),
        ] {
            if let Err(reason) = Self::usable(&path) {
                return Err(format!(
                    "cannot use {}: {reason}. Set {variable} to a writable directory\
                     {}",
                    path.display(),
                    if variable == "DXMT_ROOT_DIR" {
                        String::new()
                    } else {
                        ", or DXMT_ROOT_DIR to move all of them at once".to_string()
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
        self.processes
            .clone()
            .unwrap_or_else(|| self.root.join("processes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout built from named values instead of the real environment.
    ///
    /// These tests used to `set_var`/`remove_var` and restore afterwards under `#[serial]`.
    /// That is not enough: cargo runs one process on many threads, and the tests in other
    /// modules that read these same variables are not in the serial group, so they saw `HOME`
    /// cleared mid-flight. `cargo test --lib` passed on its own and three `config` tests failed
    /// when the whole suite ran. Resolving from a map removes the shared mutable state rather
    /// than scheduling around it.
    fn layout(vars: &[(&str, &str)]) -> Layout {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Layout::resolve(move |name| {
            vars.iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        })
    }

    /// Every child hangs off the root, so one variable moves the lot.
    #[test]
    fn one_root_places_everything() {
        let layout = layout(&[("DXMT_ROOT_DIR", "/srv/dxmt")]);

        assert_eq!(layout.root(), Path::new("/srv/dxmt"));
        assert_eq!(layout.store().path, Path::new("/srv/dxmt/store"));
        assert_eq!(layout.backups(), Path::new("/srv/dxmt/backups"));
        assert_eq!(layout.cache(), Path::new("/srv/dxmt/cache"));
        assert_eq!(layout.ssd_cache(), Path::new("/srv/dxmt/cache/ssd"));
        assert_eq!(layout.processes(), Path::new("/srv/dxmt/processes"));
        assert!(layout.store().note.is_none());
    }

    /// Under the user's home, not the working directory. That was the whole point: `./data/wal`
    /// meant `cd elsewhere && serve` opened a different, empty database.
    #[test]
    fn the_default_root_is_under_home() {
        let layout = layout(&[("HOME", "/home/someone")]);

        assert_eq!(layout.root(), Path::new("/home/someone/.dxmt"));
        assert_eq!(layout.store().path, Path::new("/home/someone/.dxmt/store"));
    }

    /// Each child can still be moved on its own, and does not drag the others with it.
    #[test]
    fn a_child_can_be_moved_without_moving_the_rest() {
        let layout = layout(&[
            ("DXMT_ROOT_DIR", "/srv/dxmt"),
            ("DXMT_WAL_DIR", "/fast/nvme/store"),
            ("DXMT_SSD_CACHE_DIR", "/scratch"),
        ]);

        assert_eq!(layout.store().path, Path::new("/fast/nvme/store"));
        assert_eq!(layout.ssd_cache(), Path::new("/scratch"));
        // Untouched children stay where the root puts them.
        assert_eq!(layout.backups(), Path::new("/srv/dxmt/backups"));
        assert_eq!(layout.cache(), Path::new("/srv/dxmt/cache"));
    }

    /// A store moved out on its own takes the backups with it — but only when nobody named a
    /// root. Two places used to decide this independently, `Layout` and
    /// `IntelligentIndexedStorage::backup_root`, which is how the directory checked at startup
    /// could differ from the directory written to.
    #[test]
    fn backups_follow_a_store_that_was_moved_out_alone() {
        let moved = layout(&[
            ("HOME", "/home/someone"),
            ("DXMT_WAL_DIR", "/fast/nvme/store"),
        ]);
        assert_eq!(
            moved.backups(),
            Path::new("/fast/nvme"),
            "with only the default root, backups belong beside the database"
        );
        // And the storage layer reaches the same answer for the same store.
        assert_eq!(
            moved.backups_for(Path::new("/fast/nvme/store")),
            moved.backups()
        );

        // Naming a root means it: the backups stay there.
        let rooted = layout(&[
            ("HOME", "/home/someone"),
            ("DXMT_ROOT_DIR", "/srv/dxmt"),
            ("DXMT_WAL_DIR", "/fast/nvme/store"),
        ]);
        assert_eq!(rooted.backups(), Path::new("/srv/dxmt/backups"));
    }

    /// No home and no `DXMT_ROOT_DIR` — `/nonexistent` under systemd's `DynamicUser=`, `/` in
    /// some images. Usable, and loud about why this is a poor place for a database.
    #[test]
    fn without_a_home_it_says_so() {
        let layout = layout(&[]);

        assert_eq!(layout.root(), Path::new("."));
        let note = layout
            .store()
            .note
            .expect("a fallback root has to explain itself");
        assert!(note.contains("DXMT_ROOT_DIR"), "{note}");
    }

    /// An empty value is not a value: `DXMT_ROOT_DIR=` falls through to the default rather than
    /// rooting the database at the empty path.
    #[test]
    fn an_empty_variable_is_ignored() {
        let layout = layout(&[("HOME", "/home/someone"), ("DXMT_ROOT_DIR", "   ")]);

        assert_eq!(layout.root(), Path::new("/home/someone/.dxmt"));
    }

    /// Creating the directories is not the check; being able to write to them is.
    ///
    /// A directory that exists with `r-x` passes `create_dir_all` and fails every write after
    /// it, which is why `prepare` puts a file in each one. The startup path refuses to serve
    /// when this fails, so the discovery happens before anything depends on it — an unwritable
    /// `backups/` would otherwise be found by whoever takes the first backup.
    #[test]
    fn prepare_makes_the_layout_and_proves_it_is_writable() {
        let root = tempfile::TempDir::new().unwrap();

        let layout = layout(&[("DXMT_ROOT_DIR", &root.path().display().to_string())]);
        layout.prepare().expect("a fresh writable root");

        for child in ["store", "backups"] {
            assert!(root.path().join(child).is_dir(), "{child}/ was not created");
        }
        // Not `cache/`: it is derived, it is made by whatever fills it, and creating it here
        // would mean a server that never caches anything still leaves the directory behind.
        assert!(!root.path().join("cache").exists());
        // And nothing is left behind by the probe.
        assert!(!root.path().join(".dxmt-write-test").exists());
    }

    /// A store outside the root means the root is not this server's business.
    ///
    /// The demo runs this way — an explicit `DXMT_WAL_DIR` and no `DXMT_ROOT_DIR` — and starting it
    /// created an empty `~/.dxmt/` with a `cache/` in it, for a server that would never look
    /// there. Checking a directory that will not be used is not a check, it is litter.
    #[test]
    fn prepare_leaves_the_default_root_alone_when_the_store_is_elsewhere() {
        let home = tempfile::TempDir::new().unwrap();
        let elsewhere = tempfile::TempDir::new().unwrap();

        layout(&[
            ("HOME", &home.path().display().to_string()),
            (
                "DXMT_WAL_DIR",
                &elsewhere.path().join("store").display().to_string(),
            ),
        ])
        .prepare()
        .expect("a writable store elsewhere");

        assert!(elsewhere.path().join("store").is_dir());
        assert!(
            !home.path().join(".dxmt").exists(),
            "the default root was created for a server that will never use it"
        );
    }

    /// Whether this process is stopped by a directory's permission bits at all. Root is not:
    /// a 0o555 directory takes new entries anyway, so a test of what happens when it refuses
    /// one has nothing to observe there (as in a container's default user, or the Dockerfile's
    /// `test` stage). Asked of the directory rather than inferred from the uid, so it also
    /// covers CAP_DAC_OVERRIDE and filesystems that ignore the bits.
    fn permissions_bind(readonly: &Path) -> bool {
        let probe = readonly.join(".probe");
        match std::fs::File::create(&probe) {
            Ok(_) => {
                let _ = std::fs::remove_file(&probe);
                eprintln!("permission bits are not enforced for this process; nothing to test");
                false
            }
            Err(_) => true,
        }
    }

    /// Two distinct failures, and the message has to tell them apart.
    ///
    /// `create_dir_all` failing and a directory that exists but will not take a file are
    /// different problems with different fixes, and the second is the one a permissions check
    /// that only creates would miss entirely.
    #[test]
    fn prepare_refuses_what_it_cannot_create_and_what_it_cannot_write() {
        use std::os::unix::fs::PermissionsExt;

        let outer = tempfile::TempDir::new().unwrap();
        let readonly = outer.path().join("readonly");
        std::fs::create_dir(&readonly).unwrap();
        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o555)).unwrap();
        if !permissions_bind(&readonly) {
            return;
        }

        // Cannot be created: the store would go under a directory that takes no new entries.
        let error = layout(&[("DXMT_ROOT_DIR", &readonly.display().to_string())])
            .prepare()
            .expect_err("an unwritable root");
        assert!(
            error.contains(&readonly.join("store").display().to_string()),
            "{error}"
        );
        assert!(error.contains("DXMT_WAL_DIR"), "{error}");
        assert!(
            error.contains("DXMT_ROOT_DIR"),
            "it should offer to move everything: {error}"
        );

        // Exists and will not take a file. `create_dir_all` returns Ok for this, which is the
        // whole reason `usable` writes a probe.
        let error = layout(&[
            ("HOME", &outer.path().display().to_string()),
            ("DXMT_WAL_DIR", &readonly.display().to_string()),
        ])
        .prepare()
        .expect_err("a read-only store");
        assert!(error.contains("not writable"), "{error}");
        assert!(error.contains(&readonly.display().to_string()), "{error}");

        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// The message has to say which variable moves the thing that failed, not just the root.
    #[test]
    fn a_child_that_fails_names_its_own_variable() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::TempDir::new().unwrap();
        let readonly = root.path().join("readonly");
        std::fs::create_dir(&readonly).unwrap();
        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o555)).unwrap();
        if !permissions_bind(&readonly) {
            return;
        }

        let error = layout(&[
            ("DXMT_ROOT_DIR", &root.path().display().to_string()),
            (
                "DXMT_BACKUP_ROOT",
                &readonly.join("backups").display().to_string(),
            ),
        ])
        .prepare()
        .expect_err("an unwritable backup root");
        assert!(error.contains("DXMT_BACKUP_ROOT"), "{error}");
        assert!(
            error.contains("DXMT_ROOT_DIR"),
            "it should also offer to move everything: {error}"
        );

        std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// The real environment still reaches the same place — `from_env` is the only caller that
    /// touches it, and this is the one test that proves the wiring is not inverted.
    #[test]
    fn from_env_reads_the_real_environment() {
        let expected = match crate::config::env::var("DXMT_ROOT_DIR") {
            Ok(root) if !root.trim().is_empty() => PathBuf::from(root),
            _ => match std::env::var("HOME") {
                Ok(home) if !home.trim().is_empty() => PathBuf::from(home).join(ROOT_UNDER_HOME),
                _ => PathBuf::from("."),
            },
        };
        assert_eq!(Layout::from_env().root(), expected);
    }
}
