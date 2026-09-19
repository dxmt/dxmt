//! The directory layout is documented in three places and defined in one.
//!
//! `src/config/layout.rs` decides where everything goes; `README.md` tells an operator, and
//! `.env.example` is what they copy. Those two drift silently — the README carried
//! `DB_WAL_DIR=./data/wal` for a while after the default had moved to `~/.dxmt/store`, and
//! `.env.example` still *assigned* `DB_SSD_CACHE_DIR=./cache`, so copying the file as
//! instructed re-introduced the working-directory-relative path the layout exists to avoid.
//!
//! These read the source rather than restate it, so adding a variable to `layout.rs` and not
//! the docs is a failing test rather than a discovery six months later.

use std::path::{Path, PathBuf};

fn crate_file(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every variable the layout reads, taken from the layout itself.
fn variables_layout_reads() -> Vec<String> {
    let source = crate_file("src/config/layout.rs");
    let mut found: Vec<String> = source
        .match_indices("\"DB_")
        .filter_map(|(at, _)| {
            let rest = &source[at + 1..];
            rest.find('"').map(|end| rest[..end].to_string())
        })
        .filter(|name| name.chars().all(|c| c.is_ascii_uppercase() || c == '_'))
        .collect();
    found.sort();
    found.dedup();
    assert!(
        found.len() >= 5,
        "the extraction found almost nothing, so it is measuring itself: {found:?}"
    );
    found
}

/// A variable an operator cannot find is a variable they will not use.
#[test]
fn the_readme_names_every_layout_variable() {
    let readme = crate_file("README.md");
    let undocumented: Vec<String> = variables_layout_reads()
        .into_iter()
        .filter(|name| !readme.contains(name))
        .collect();
    assert!(
        undocumented.is_empty(),
        "layout.rs reads these and README.md never mentions them: {undocumented:?}"
    );
}

/// The README must not print a default the code does not hold.
///
/// Checked as an exact string rather than by parsing: the pair of paths below *is* the claim,
/// and it was wrong in the README for three commits.
#[test]
fn the_readme_prints_the_defaults_the_code_uses() {
    let readme = crate_file("README.md");
    for claim in ["~/.dxmt", "$DB_ROOT_DIR/store", "$DB_ROOT_DIR/backups"] {
        assert!(readme.contains(claim), "README.md no longer says {claim:?}");
    }
    assert!(
        !readme.contains("DB_WAL_DIR=./data/wal"),
        "the README still advertises the pre-~/.dxmt default"
    );
}

/// `.env.example` is copied verbatim, so anything it assigns becomes that operator's layout.
///
/// Leaving the layout variables commented is what keeps a copied file from pinning the cache —
/// or the database — to whatever directory the shell happened to be in.
#[test]
fn the_sample_env_does_not_assign_a_layout_variable() {
    let sample = crate_file(".env.example");
    let assignments: Vec<&str> = sample
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .filter(|line| {
            variables_layout_reads()
                .iter()
                .any(|name| line.starts_with(&format!("{name}=")))
        })
        .collect();
    assert!(
        assignments.is_empty(),
        ".env.example is copied as-is, so these would become a real layout — comment them out \
         and show the default instead: {assignments:?}"
    );
}

/// The one thing the layout cannot tolerate: a default that moves with the shell.
#[test]
fn nothing_documented_points_at_the_working_directory() {
    for file in [".env.example", "README.md"] {
        let text = crate_file(file);
        for (number, line) in text.lines().enumerate() {
            let line = line.trim_start_matches(['#', '>', ' ']);
            for name in variables_layout_reads() {
                let Some(value) = line.strip_prefix(&format!("{name}=")) else {
                    continue;
                };
                let value = value.split_whitespace().next().unwrap_or("");
                assert!(
                    !value.starts_with("./") && !value.starts_with("../"),
                    "{file}:{} documents {name} as {value:?}. A relative path means `cd \
                     elsewhere && serve` opens a different database.",
                    number + 1
                );
            }
        }
    }
}

/// And the root itself is the user's, not the process's.
#[test]
fn the_default_root_is_absolute() {
    let layout = dxmt_vectordb::config::layout::Layout::resolve(|name| match name {
        "HOME" => Some("/home/someone".to_string()),
        _ => None,
    });
    assert!(
        layout.root().is_absolute(),
        "{} is not absolute",
        layout.root().display()
    );
    assert!(layout.store().path.starts_with(Path::new("/home/someone")));
}
