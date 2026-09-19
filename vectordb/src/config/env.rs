//! One place to read a configuration variable, and one place that knows the old names.
//!
//! Every server-side variable is `DXMT_*`. It used to be `DB_*`, which is not a namespace —
//! it is the word "database", and everyone uses it. The collision is not hypothetical:
//! `DB_HOST` is this server's bind address, and `DB_HOST=postgres` is the most ordinary line
//! in any compose file that also runs Postgres, which would have sent us off to bind a
//! hostname nobody meant. `DB_PORT` and `DB_PASSWORD` are the same story. Qdrant namespaces
//! its variables (`QDRANT__SERVICE__API_KEY`) and so does Weaviate; this is the convention.
//!
//! The old name still works and says so once, so an existing deployment does not break on an
//! upgrade. Renaming was only cheap before there were users of the published name.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// The prefix every server-side variable carries.
pub const PREFIX: &str = "DXMT_";

/// The prefix they carried before, accepted with a warning.
pub const LEGACY_PREFIX: &str = "DB_";

/// Read `name`, falling back to its pre-rename spelling.
///
/// Returns `Result` rather than `Option` so it substitutes for `std::env::var` at a call site
/// without disturbing the `unwrap_or_else`/`if let Ok` around it.
///
/// A variable that is *set and empty* is a value, not an absence — `DXMT_WAL_DIR=` means "no
/// persistence at all" — so an empty `DXMT_` value is returned as-is and never falls through
/// to the legacy name.
pub fn var(name: &str) -> Result<String, std::env::VarError> {
    match std::env::var(name) {
        Ok(value) => Ok(value),
        Err(missing) => match legacy_name(name) {
            // Not one of ours — `HOME`, say. There is nothing to fall back to and no
            // deprecation to announce; it is read exactly as `std::env::var` would.
            None => Err(missing),
            Some(legacy) => match std::env::var(legacy) {
                Ok(value) => {
                    warn_once(name);
                    Ok(value)
                }
                Err(_) => Err(missing),
            },
        },
    }
}

/// `DXMT_WAL_DIR` -> `DB_WAL_DIR`; `None` for a variable this server does not own.
pub fn legacy_name(name: &str) -> Option<String> {
    name.strip_prefix(PREFIX)
        .map(|rest| format!("{LEGACY_PREFIX}{rest}"))
}

/// Copy every legacy `DB_*` variable onto its `DXMT_*` name, and return what was found.
///
/// Called once at the top of `main`, before the CLI is parsed and before any thread exists —
/// which is the one place mutating the process environment is defensible, and the only way to
/// cover the variables this process does not read through [`var`]. `--grpc-port` is the case
/// in point: clap reads `DXMT_GRPC_PORT` itself, takes exactly one name, and would otherwise
/// have been the single flag where the compatibility promise quietly did not hold.
///
/// It also means an operator gets the whole list at startup instead of discovering the names
/// one warning at a time. A `DXMT_*` value already set always wins and is never overwritten.
pub fn adopt_legacy_names() -> Vec<(String, String)> {
    let mut adopted: Vec<(String, String)> = std::env::vars()
        .filter_map(|(name, value)| {
            let rest = name.strip_prefix(LEGACY_PREFIX)?;
            let current = format!("{PREFIX}{rest}");
            // Set and empty counts as set: `DXMT_WAL_DIR=` is an instruction, not an absence.
            if std::env::var(&current).is_ok() {
                return None;
            }
            Some((name, current, value))
        })
        .map(|(legacy, current, value)| {
            std::env::set_var(&current, value);
            (legacy, current)
        })
        .collect();
    adopted.sort();
    adopted
}

/// Once per variable, not once per read: several of these are read on every request path.
fn warn_once(name: &str) {
    static WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let warned = WARNED.get_or_init(|| Mutex::new(HashSet::new()));
    let Ok(mut warned) = warned.lock() else {
        return;
    };
    if warned.insert(name.to_string()) {
        tracing::warn!(
            "{} is deprecated and will be removed; use {} instead",
            legacy_name(name).unwrap_or_else(|| name.to_string()),
            name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// These four touch the real process environment — there is no other way to test a reader
    /// of it — so they are serial *and* they use names no other test or module reads.
    const NEW: &str = "DXMT_ENV_HELPER_PROBE";
    const OLD: &str = "DB_ENV_HELPER_PROBE";

    fn clear() {
        std::env::remove_var(NEW);
        std::env::remove_var(OLD);
    }

    #[test]
    #[serial]
    fn the_new_name_wins() {
        clear();
        std::env::set_var(NEW, "new");
        std::env::set_var(OLD, "old");
        assert_eq!(var(NEW).unwrap(), "new");
        clear();
    }

    /// The point of the whole exercise: an existing deployment keeps working.
    #[test]
    #[serial]
    fn the_old_name_still_works() {
        clear();
        std::env::set_var(OLD, "old");
        assert_eq!(var(NEW).unwrap(), "old");
        clear();
    }

    /// Set-and-empty is a value. `DXMT_WAL_DIR=` means no persistence, and falling through to
    /// `DB_WAL_DIR` there would quietly turn durability back on against the operator's
    /// instruction.
    #[test]
    #[serial]
    fn an_empty_new_value_does_not_fall_through() {
        clear();
        std::env::set_var(NEW, "");
        std::env::set_var(OLD, "old");
        assert_eq!(var(NEW).unwrap(), "", "an empty value is a value");
        clear();
    }

    #[test]
    #[serial]
    fn neither_set_is_still_absent() {
        clear();
        assert!(var(NEW).is_err());
    }

    #[test]
    fn the_legacy_name_is_the_prefix_swapped() {
        assert_eq!(legacy_name("DXMT_WAL_DIR").as_deref(), Some("DB_WAL_DIR"));
        assert_eq!(legacy_name("DXMT_API_KEY").as_deref(), Some("DB_API_KEY"));
    }

    /// A variable this server does not own is read straight through. `Layout` asks for `HOME`
    /// by the same route, and there is no `DB_HOME` to look for.
    #[test]
    #[serial]
    fn a_variable_we_do_not_own_is_read_as_is() {
        assert_eq!(legacy_name("HOME"), None);
        std::env::set_var("DXMT_ENV_HELPER_UNOWNED", "x");
        assert_eq!(var("DXMT_ENV_HELPER_UNOWNED").unwrap(), "x");
        std::env::remove_var("DXMT_ENV_HELPER_UNOWNED");
        assert!(
            var("PATH").is_ok(),
            "PATH is not ours and is still readable"
        );
    }
}
