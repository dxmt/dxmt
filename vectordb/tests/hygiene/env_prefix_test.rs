//! Every server-side variable is `DXMT_*`, and every read goes through one helper.
//!
//! `DB_` is not a namespace — it is the word "database". `DB_HOST` was this server's bind
//! address and is also what practically every compose file calls its Postgres, so an
//! unrelated service in the same environment could have sent us off to bind a hostname
//! nobody meant. The rename was only cheap before the names were published.
//!
//! Two things rot from here: a new `std::env::var` that skips the compatibility fallback, and
//! a new variable that picks a generic prefix because the file it lands in has no example of
//! the convention. These fail on both.

use std::path::{Path, PathBuf};

/// Names that belong to somebody else and must keep their conventional spelling.
const NOT_OURS: [&str; 4] = ["HOME", "PATH", "OPENAI_API_KEY", "COHERE_API_KEY"];

/// The one file allowed to name the old prefix and to call `std::env::var` on a bare name.
const THE_HELPER: &str = "src/config/env.rs";

fn sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    walk(&root.join("src"), &mut paths);
    assert!(paths.len() > 50, "the walk found almost nothing");
    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path).unwrap();
            let relative = path.strip_prefix(&root).unwrap().to_path_buf();
            (relative, text)
        })
        .collect()
}

/// Reading the environment directly skips the `DB_*` fallback, so the variable silently stops
/// honouring the name an existing deployment already has in its compose file.
#[test]
fn every_read_goes_through_the_helper() {
    let mut direct = Vec::new();
    for (path, text) in sources() {
        if path == Path::new(THE_HELPER) {
            continue;
        }
        for (number, line) in text.lines().enumerate() {
            if !line.contains("std::env::var(") {
                continue;
            }
            // A name that is not ours has no legacy spelling and nothing to fall back to.
            if NOT_OURS
                .iter()
                .any(|name| line.contains(&format!("\"{name}\"")))
            {
                continue;
            }
            direct.push(format!(
                "{}:{}: {}",
                path.display(),
                number + 1,
                line.trim()
            ));
        }
    }
    assert!(
        direct.is_empty(),
        "these bypass crate::config::env::var and so lose the DB_* fallback:\n  {}",
        direct.join("\n  ")
    );
}

/// A variable that is ours wears the prefix. Anything else is a name some other tool may
/// already be using for something entirely different.
#[test]
fn every_variable_we_own_is_prefixed() {
    let pattern = regex_lite_upper_names();
    let mut unprefixed = Vec::new();
    for (path, text) in sources() {
        if path == Path::new(THE_HELPER) {
            continue;
        }
        for (number, line) in text.lines().enumerate() {
            if !line.contains("env::var(") && !line.contains("env = \"") {
                continue;
            }
            for name in pattern(line) {
                if name.starts_with("DXMT_") || NOT_OURS.contains(&name.as_str()) {
                    continue;
                }
                unprefixed.push(format!("{}:{}: {name}", path.display(), number + 1));
            }
        }
    }
    assert!(
        unprefixed.is_empty(),
        "these name a variable without the DXMT_ prefix:\n  {}",
        unprefixed.join("\n  ")
    );
}

/// The old prefix survives in exactly one place: the helper that translates it.
#[test]
fn the_old_prefix_lives_in_one_file() {
    let mut mentions = Vec::new();
    for (path, text) in sources() {
        if path == Path::new(THE_HELPER) || path == Path::new("src/main.rs") {
            continue;
        }
        for (number, line) in text.lines().enumerate() {
            if line.contains("\"DB_") {
                mentions.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        mentions.is_empty(),
        "the DB_ prefix should only appear in {THE_HELPER}:\n  {}",
        mentions.join("\n  ")
    );
}

/// The compatibility promise itself, end to end through the public helper.
#[test]
fn the_old_name_still_reaches_the_new_one() {
    // A name nothing else reads, so this does not race a test in another module.
    const NEW: &str = "DXMT_PREFIX_TEST_PROBE";
    const OLD: &str = "DB_PREFIX_TEST_PROBE";
    std::env::set_var(OLD, "from-the-old-name");
    assert_eq!(
        dxmt_vectordb::config::env::var(NEW).unwrap(),
        "from-the-old-name"
    );
    std::env::remove_var(OLD);
}

/// `adopt_legacy_names` is what covers the variables clap reads for itself, and clap takes
/// exactly one name per argument.
#[test]
fn the_startup_bridge_copies_the_old_names_forward() {
    const NEW: &str = "DXMT_PREFIX_BRIDGE_PROBE";
    const OLD: &str = "DB_PREFIX_BRIDGE_PROBE";
    const KEPT_NEW: &str = "DXMT_PREFIX_BRIDGE_KEPT";
    const KEPT_OLD: &str = "DB_PREFIX_BRIDGE_KEPT";

    std::env::set_var(OLD, "carried");
    std::env::set_var(KEPT_OLD, "old");
    std::env::set_var(KEPT_NEW, "new");

    let adopted = dxmt_vectordb::config::env::adopt_legacy_names();

    assert_eq!(std::env::var(NEW).unwrap(), "carried");
    assert!(adopted.iter().any(|(was, _)| was == OLD), "{adopted:?}");
    assert_eq!(
        std::env::var(KEPT_NEW).unwrap(),
        "new",
        "a value already set under the new name must not be overwritten"
    );

    for name in [NEW, OLD, KEPT_NEW, KEPT_OLD] {
        std::env::remove_var(name);
    }
}

/// Pull the `SCREAMING_CASE` names out of a line, without a regex dependency.
fn regex_lite_upper_names() -> impl Fn(&str) -> Vec<String> {
    |line: &str| {
        let mut names = Vec::new();
        let mut rest = line;
        while let Some(open) = rest.find('"') {
            let after = &rest[open + 1..];
            let Some(close) = after.find('"') else { break };
            let candidate = &after[..close];
            if candidate.len() >= 3
                && candidate
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                && candidate
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic())
            {
                names.push(candidate.to_string());
            }
            rest = &after[close + 1..];
        }
        names
    }
}

/// The places that *set* variables, outside the source: CI, containers, scripts, the sample
/// env file, the contributor guide.
///
/// The rename to `DXMT_*` was applied with a script run inside `vectordb/`, and the CI workflow
/// at the repository root kept starting its server with `DB_WAL_DIR` for several commits. The
/// alias kept it working, which is exactly why nothing noticed: a deprecation warning in a CI
/// log is read by nobody. Every test above scans `src/` only.
#[test]
fn nothing_that_sets_variables_uses_the_old_names() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in ["../.github/workflows", "scripts"] {
        if let Ok(entries) = std::fs::read_dir(crate_root.join(dir)) {
            files.extend(entries.flatten().map(|e| e.path()).filter(|p| p.is_file()));
        }
    }
    for file in [
        "../CONTRIBUTING.md",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        ".env.example",
        "Makefile",
    ] {
        let path = crate_root.join(file);
        if path.is_file() {
            files.push(path);
        }
    }
    // Outside a git checkout (a crate tarball) the repository-root files are absent; the ones in
    // the crate are not, so an empty list means the paths are wrong, not that all is well.
    assert!(files.len() >= 5, "found only {files:?}");

    let mut found = Vec::new();
    for path in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue; // binary or unreadable
        };
        for (number, line) in text.lines().enumerate() {
            let mut rest = line;
            while let Some(at) = rest.find("DB_") {
                let before_ok = at == 0
                    || !rest.as_bytes()[at - 1].is_ascii_alphanumeric()
                        && rest.as_bytes()[at - 1] != b'_';
                let name: String = rest[at..]
                    .chars()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                    .collect();
                let after = rest[at + name.len()..].trim_start();
                if before_ok && name.len() > 4 && (after.starts_with('=') || after.starts_with(':'))
                {
                    found.push(format!("{}:{}: {name}", path.display(), number + 1));
                }
                rest = &rest[at + 3..];
            }
        }
    }
    assert!(
        found.is_empty(),
        "these set a variable by its pre-rename name; use DXMT_*:\n  {}",
        found.join("\n  ")
    );
}
