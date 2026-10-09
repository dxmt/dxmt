//! Every test file is a declared target, and every declaration points at a file.
//!
//! The integration tests live in scenario folders (`tests/filters/`, `tests/durability/`, …),
//! and Cargo only auto-discovers `tests/*.rs`. Each one therefore has a `[[test]]` entry in
//! `Cargo.toml`. That is the whole reason for the layout: **a file that is moved and not
//! declared fails to build**, loudly, at the moment it happens.
//!
//! The alternative was aggregating each folder behind one `tests/<folder>.rs` that `include!`s
//! its members, which is what `tests/unit.rs` and its three siblings already do. It needs no
//! Cargo entries and has the failure mode this repository keeps paying for: a member added to
//! the folder and not to the aggregator is **silently never run**. `tests/unit.rs` carries the
//! comment `// Add more unit test modules here as they are created`, which is a note asking
//! somebody to remember. The same shape has cost this project twice already -- CI naming four
//! test targets and quietly skipping twenty-four, and a feature-gated file reporting
//! `0 passed` while looking green.
//!
//! Declaring them is not free of that risk either, only louder about it: a file with no entry
//! compiles nowhere, so nothing breaks and nothing runs it. That is what this test is for.
//! It reads `Cargo.toml` and the directory tree and insists they describe each other.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Folders under `tests/` that hold something other than test targets.
///
/// `common`, `fixtures` and `fake_provider` are modules other tests pull in with `#[path]`.
/// `unit`, `integration`, `property` and `benchmarks` are the four aggregated suites, whose
/// members are reached through `include!` from a top-level file of the same name -- those top
/// level files are auto-discovered, so they need no entry.
const NOT_TARGETS: &[&str] = &[
    "common",
    "fixtures",
    "fake_provider",
    "unit",
    "integration",
    "property",
    "benchmarks",
    // Python suites, run by `scripts/api_tests.sh` rather than by cargo. `api` in particular
    // is a pytest *package* -- it has `__init__.py` and `conftest.py` -- so a `.rs` file put
    // there is both undiscovered by cargo and in pytest's way. The Rust tests that exercise
    // the HTTP, gRPC and MCP surfaces live in `tests/interfaces/`.
    "api",
    "stateful",
    "compat",
    "__pycache__",
];

/// `(name, path)` for every `[[test]]` in `Cargo.toml`.
///
/// Parsed by hand rather than with a TOML crate: this runs in the default build, and adding a
/// dependency to the dev-dependencies so a test can read the manifest that lists it is a
/// circularity worth avoiding. The shape it accepts is the shape the generator writes, and
/// anything else makes the counts disagree below.
fn declared() -> Vec<(String, String)> {
    let manifest = std::fs::read_to_string(root().join("Cargo.toml")).expect("Cargo.toml");
    let mut out = Vec::new();
    let mut lines = manifest.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() != "[[test]]" {
            continue;
        }
        let (mut name, mut path) = (None, None);
        for next in lines.by_ref() {
            let t = next.trim();
            if t.starts_with('[') || (t.is_empty() && name.is_some() && path.is_some()) {
                break;
            }
            if let Some(v) = t.strip_prefix("name = ") {
                name = Some(v.trim_matches('"').to_string());
            } else if let Some(v) = t.strip_prefix("path = ") {
                path = Some(v.trim_matches('"').to_string());
            }
        }
        match (name, path) {
            (Some(n), Some(p)) => out.push((n, p)),
            (Some(n), None) => panic!("[[test]] {n} has no `path`"),
            _ => panic!("a [[test]] entry with no `name`"),
        }
    }
    out
}

/// Every `.rs` under `tests/`, excluding the shared-module and aggregated folders.
fn test_files() -> BTreeSet<String> {
    fn walk(dir: &Path, base: &Path, out: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).expect("read tests/") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                if !NOT_TARGETS.contains(&name.as_str()) {
                    walk(&path, base, out);
                }
            } else if name.ends_with(".rs") {
                out.insert(
                    path.strip_prefix(base)
                        .expect("under the crate root")
                        .to_string_lossy()
                        // Cargo.toml writes `/` whatever the platform.
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(&root().join("tests"), &root(), &mut out);
    // The four aggregated suites are auto-discovered; they are files, not folders, so the
    // exclusion above does not reach them.
    out.retain(|p| {
        !matches!(
            p.as_str(),
            "tests/unit.rs" | "tests/integration.rs" | "tests/property.rs" | "tests/benchmarks.rs"
        )
    });
    out
}

#[test]
fn every_test_file_is_declared_and_every_declaration_has_a_file() {
    let declared = declared();
    let on_disk = test_files();
    let declared_paths: BTreeSet<String> = declared.iter().map(|(_, p)| p.clone()).collect();

    let undeclared: Vec<&String> = on_disk.difference(&declared_paths).collect();
    assert!(
        undeclared.is_empty(),
        "these files are under tests/ and have no `[[test]]` entry, so cargo does not build \
         them and nothing runs them:\n  {}\n\nAdd to Cargo.toml:\n{}",
        undeclared
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  "),
        undeclared
            .iter()
            .map(|p| {
                let stem = p.rsplit('/').next().unwrap().trim_end_matches(".rs");
                format!("\n[[test]]\nname = \"{stem}\"\npath = \"{p}\"\n")
            })
            .collect::<String>()
    );

    let missing: Vec<&String> = declared_paths.difference(&on_disk).collect();
    assert!(
        missing.is_empty(),
        "these `[[test]]` entries point at files that do not exist:\n  {}",
        missing
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

#[test]
fn a_declared_name_is_the_file_stem() {
    // `cargo test --test wal_recovery_test`, the Makefile and the CI workflow all address a
    // test by its name. Letting the name drift from the file would mean grepping for a file
    // that does not answer to the name in the error message.
    let wrong: Vec<String> = declared()
        .into_iter()
        .filter(|(name, path)| path.rsplit('/').next().unwrap().trim_end_matches(".rs") != name)
        .map(|(name, path)| format!("{name} <- {path}"))
        .collect();
    assert!(
        wrong.is_empty(),
        "a test's name must be its file stem:\n  {}",
        wrong.join("\n  ")
    );
}

#[test]
fn no_test_file_sits_loose_at_the_top_of_tests() {
    // The four aggregated suites are the exception, named in `NOT_TARGETS`'s sibling list
    // above. Anything else at the top level is a file that skipped the scenario folders, and
    // the next person will not find it where its subject lives.
    let loose: Vec<String> = std::fs::read_dir(root().join("tests"))
        .expect("tests/")
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().to_string();
            (e.path().is_file()
                && name.ends_with(".rs")
                && !matches!(
                    name.as_str(),
                    "unit.rs" | "integration.rs" | "property.rs" | "benchmarks.rs"
                ))
            .then_some(name)
        })
        .collect();
    assert!(
        loose.is_empty(),
        "these are at the top of tests/ rather than in a scenario folder: {loose:?}\n\
         The folders are durability, filters, hygiene, interfaces, recall, search, storage."
    );
}
