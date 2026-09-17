//! Every `npm ci` in CI has a lockfile in the tree to install from.
//!
//! `cargo test --test ci_lockfiles_test`
//!
//! `npm ci` is a hard error without a `package-lock.json`, and `dxmt-js-sdk/.gitignore` ignored
//! its own — so the step that runs this SDK's unit tests, its build and its live contract test
//! against a real server could never have run. It is the last step of the `sdk-live` job, so
//! everything before it ran and the job then failed on a missing file. Nothing in the SDK was
//! broken; the file was.
//!
//! The list is read out of `.github/workflows/ci.yml` rather than written here, for the same
//! reason `tests/api/test_request_fields.py` reads the OpenAPI document: a hand-kept list does
//! not know when someone adds a third npm package to CI, and this is exactly the kind of thing
//! nobody thinks to update. Add an `npm ci` to the workflow and this test starts checking it.
//!
//! Deliberately a Rust test on the default feature set: it runs in `cargo test --tests`, with
//! no Node and no network, which is the common path. A check that only runs in the job it is
//! checking would be the same mistake again.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository root: this crate sits one level below it.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate has a parent directory")
        .to_path_buf()
}

/// Every `working-directory` whose step runs `npm ci`, as paths relative to the repo root.
///
/// Parsed by step, not by proximity. The first version of this walked outwards from the
/// `npm ci` line looking for the nearest `working-directory:`, and found the job-level
/// `defaults.run.working-directory` instead — reporting `vectordb`, a directory with no
/// package.json, for both steps. A step begins at a `- ` list item, so the block between two
/// of those is the unit a `run:` and its `working-directory:` share.
fn npm_ci_directories(workflow: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut block: Vec<&str> = Vec::new();

    let flush = |block: &Vec<&str>, found: &mut Vec<String>| {
        if !block.iter().any(|line| line.contains("npm ci")) {
            return;
        }
        let directory = block.iter().find_map(|line| {
            line.trim()
                .strip_prefix("working-directory:")
                .map(|rest| rest.trim().trim_matches(['"', '\'']).to_string())
        });
        // A step with no `working-directory` of its own runs in the job's default, which for
        // this workflow is the crate directory and has no package.json — so an `npm ci` there
        // is a defect this test should report rather than skip.
        found.push(directory.unwrap_or_else(|| "<the job default>".to_string()));
    };

    for line in workflow.lines() {
        let is_step =
            line.trim_start().starts_with("- ") && line.len() - line.trim_start().len() >= 4;
        if is_step {
            flush(&block, &mut found);
            block.clear();
        }
        block.push(line);
    }
    flush(&block, &mut found);

    found.sort();
    found.dedup();
    found
}

fn tracked(root: &Path, relative: &str) -> bool {
    Command::new("git")
        .current_dir(root)
        .args(["ls-files", "--error-unmatch", relative])
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

#[test]
fn every_npm_ci_in_ci_has_a_tracked_lockfile() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/ci.yml");
    let workflow = match std::fs::read_to_string(&workflow_path) {
        Ok(text) => text,
        // A checkout without the workflow (a vendored copy, a tarball) has nothing to check.
        Err(_) => return,
    };

    let directories = npm_ci_directories(&workflow);
    assert!(
        !directories.is_empty(),
        "no `npm ci` found in {}; if the workflow stopped installing node dependencies this \
         test is checking nothing and should be removed with them",
        workflow_path.display()
    );

    for directory in directories {
        let lockfile = format!("{}/package-lock.json", directory.trim_end_matches('/'));
        assert!(
            root.join(&lockfile).exists(),
            "CI runs `npm ci` in {directory} and there is no {lockfile}; that step cannot run"
        );
        assert!(
            tracked(&root, &lockfile),
            "CI runs `npm ci` in {directory} and {lockfile} is not tracked by git, so the \
             checkout CI installs from will not have it and the step fails before it starts. \
             Check the directory's .gitignore."
        );
    }
}

#[test]
fn the_parser_finds_both_shapes_of_step() {
    // The single-line `run:` and the block `run: |`, which is how the two npm steps in this
    // workflow are actually written. A parser that only understood one of them would report
    // an empty list, and the test above would then pass by checking nothing -- so this pins
    // the parser rather than trusting it.
    let sample = "\
jobs:
  a:
    steps:
      - name: single line
        run: npm ci && npm test
        working-directory: pkg/one
      - name: a block
        run: |
          npm ci
          npm test
        working-directory: pkg/two
      - name: no npm here
        run: cargo test
        working-directory: pkg/three
";
    assert_eq!(npm_ci_directories(sample), vec!["pkg/one", "pkg/two"]);
}
