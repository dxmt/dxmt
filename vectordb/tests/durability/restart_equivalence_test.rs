//! The server answers the same questions the same way after a restart.
//!
//! `cargo test --features text-bm25 --test restart_equivalence_test`
//!
//! Four bugs found in one day shared a shape: state built while records are written, never
//! written down, and never rebuilt when the server comes back. The Tantivy index, the payload
//! index, the sparse postings — each was silently empty after a restart, and each answered
//! `200 OK` with nothing in it, which a caller cannot tell from a query that genuinely matches
//! nothing. The whole existing suite was green throughout, because `tests/api/` inserts and
//! queries inside one process and so never asks a *restored* table anything.
//!
//! Enumerating the fields that might not be rebuilt is how those four were found, and it only
//! finds what someone thought to look for. This test does the opposite: it asks one table every
//! kind of question the API has, restarts the binary, asks again, and requires the answers to be
//! identical. Derived state that is not rebuilt shows up as a differing answer, whether or not
//! anyone knew that state existed.
//!
//! Adding a feature to the server means adding a question here. A question that cannot tell a
//! rebuilt index from an empty one is worse than none: each is written so that an empty answer
//! fails, which is why the fixture makes sure every one of them matches something.

#![cfg(feature = "text-bm25")]

use std::collections::BTreeMap;
use std::process::{Child, Command, Stdio};

#[path = "../common/http.rs"]
mod http;

struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn start(dir: &std::path::Path, port: u16) -> Server {
    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DXMT_WAL_DIR", dir.join("wal"))
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");

    let server = Server { child, port };
    for _ in 0..200 {
        if call(port, "GET", "/api/v1/health", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {port}");
}

fn call(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<serde_json::Value> {
    let (_, text) = http::call(port, method, path, &[], body)?;
    serde_json::from_str(&text).ok()
}

/// Strip what is allowed to differ between two runs of the same question.
///
/// Three things legitimately change and would drown the comparison: the envelope's `timestamp`,
/// the server-reported duration (`tookms` on the dense paths, `took_ms` on the text one — the
/// wire really does spell them differently), and floating-point scores, which are compared to
/// six places rather than exactly because a restored index may sum in a different order.
///
/// Nothing else is normalised. A field that changes because it was rebuilt differently is
/// exactly what this test exists to catch, so the temptation to widen this function is the
/// temptation to delete the test.
fn normalise(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(fields) => serde_json::Value::Object(
            fields
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        // The envelope's own clock, and the two spellings of the server-reported
                        // duration.
                        "timestamp" | "tookms" | "took_ms"
                            // Sizes of structures that are legitimately rebuilt rather than
                            // restored byte for byte -- a named vector space, for instance, is
                            // built from the records. These are measurements, not answers; what
                            // the index *says* is compared everywhere else in this suite.
                            | "index_size_bytes"
                            | "memory_usage_bytes"
                    )
                })
                .map(|(key, inner)| (key.clone(), normalise(inner)))
                .collect(),
        ),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(normalise).collect())
        }
        serde_json::Value::Number(number) => match number.as_f64() {
            Some(float) if !number.is_i64() && !number.is_u64() => {
                serde_json::json!(format!("{float:.6}"))
            }
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

const TABLE: &str = "everything";
const VECTOR: &str = "[1.0,0.0,0.0,0.0]";

/// A table using every feature that has any state behind it, and records that exercise the
/// update and delete paths as well as the insert one.
fn seed(port: u16) {
    call(
        port,
        "POST",
        "/api/v1/tables",
        Some(
            r#"{"name":"everything","dimension":4,"distance_metric":"cosine",
                "description":"restart equivalence",
                "enable_full_text_search":true,
                "text_fields":[
                  {"name":"content","source":{"metadata_field":"content"},"weight":1.0},
                  {"name":"title","source":{"metadata_field":"title"},"weight":2.0}],
                "named_vectors":[{"name":"title","dimension":4,"distance_metric":"cosine"}],
                "payload_schema":[{"name":"topic","field_type":"string","default":"none"}]}"#,
        ),
    );

    let vectors = [
        "[1,0,0,0]",
        "[0.9,0.4,0,0]",
        "[0.7,0.7,0,0]",
        "[0.4,0.9,0,0]",
        "[0,1,0,0]",
        "[0,0,1,0]",
    ];
    let mut records = Vec::new();
    for (i, vector) in vectors.iter().enumerate() {
        // The named space is deliberately the reverse of the default one, so a query that
        // silently falls back to the default vector returns a different order.
        let named = vectors[vectors.len() - 1 - i];
        records.push(format!(
            r#"{{"id":"r{i}","vector":{vector},"named_vectors":{{"title":{named}}},
                "sparse_vector":{{"indices":[{i},9],"values":[1.0,0.5],"dimension":16}},
                "payload":{{"topic":"{}","n":{i},"flag":{},
                   "title":"{}","content":"document number {i} about widgets and things"}}}}"#,
            if i < 3 { "alpha" } else { "beta" },
            i % 2 == 0,
            if i == 0 {
                "widgets in the title"
            } else {
                "an ordinary title"
            },
        ));
    }
    call(
        port,
        "POST",
        &format!("/api/v1/tables/{TABLE}/records"),
        Some(&format!(r#"{{"records":[{}]}}"#, records.join(","))),
    );

    // An update: the record keeps its `created_at` and moves to the end of the row file, which
    // is the case row order cannot stand in for.
    call(
        port,
        "PUT",
        &format!("/api/v1/tables/{TABLE}/records/r2"),
        Some(r#"{"payload":{"topic":"gamma","updated":true},"merge_mode":"merge"}"#),
    );
    // A delete: a tombstoned row, and an id every index has to stop naming.
    call(
        port,
        "DELETE",
        &format!("/api/v1/tables/{TABLE}/records"),
        Some(r#"{"ids":["r5"]}"#),
    );
}

/// One question, and what makes its answer non-empty.
///
/// The second half is not decoration. A question whose answer is empty either way cannot tell a
/// rebuilt index from one that was never built — which is precisely the failure being hunted —
/// so every question carries a pointer into the answer that must be non-empty, checked before
/// the two runs are compared at all.
struct Question {
    name: &'static str,
    method: &'static str,
    path: String,
    body: Option<String>,
    /// A JSON pointer that must resolve to a non-empty array, or `None` for answers that are
    /// documents rather than result sets.
    non_empty: Option<&'static str>,
}

fn q(
    name: &'static str,
    method: &'static str,
    path: String,
    body: Option<&str>,
    non_empty: Option<&'static str>,
) -> Question {
    Question {
        name,
        method,
        path,
        body: body.map(str::to_string),
        non_empty,
    }
}

fn questions() -> Vec<Question> {
    let t = TABLE;
    let filter_alpha =
        r#""filter":{"must":[{"key":"topic","operation":{"type":"Equal","value":"alpha"}}]}"#;
    vec![
        // --- what the table says about itself -------------------------------------------
        q(
            "table metadata",
            "GET",
            format!("/api/v1/tables/{t}"),
            None,
            None,
        ),
        q(
            "table stats",
            "GET",
            format!("/api/v1/tables/{t}/stats"),
            None,
            None,
        ),
        q(
            "tables list",
            "GET",
            "/api/v1/tables".into(),
            None,
            Some("/data/tables"),
        ),
        // --- records ---------------------------------------------------------------------
        q(
            "one record",
            "GET",
            format!("/api/v1/tables/{t}/records/r0"),
            None,
            None,
        ),
        q(
            "the updated record",
            "GET",
            format!("/api/v1/tables/{t}/records/r2"),
            None,
            None,
        ),
        q(
            "list page 1",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(r#"{"limit":3,"offset":0}"#),
            Some("/data/records"),
        ),
        q(
            "list page 2",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(r#"{"limit":3,"offset":3}"#),
            Some("/data/records"),
        ),
        q(
            "list with vectors",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(r#"{"limit":2,"with_vector":true}"#),
            Some("/data/records"),
        ),
        // --- filtering, one per operation the payload index answers ----------------------
        q(
            "list filtered equal",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(&format!(r#"{{"limit":9,{filter_alpha}}}"#)),
            Some("/data/records"),
        ),
        q(
            "list filtered range",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(
                r#"{"limit":9,"filter":{"must":[{"key":"n","operation":{"type":"GreaterThanOrEqual","value":2}}]}}"#,
            ),
            Some("/data/records"),
        ),
        q(
            "list filtered in",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(
                r#"{"limit":9,"filter":{"must":[{"key":"topic","operation":{"type":"In","value":["alpha","gamma"]}}]}}"#,
            ),
            Some("/data/records"),
        ),
        q(
            "list filtered bool",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(
                r#"{"limit":9,"filter":{"must":[{"key":"flag","operation":{"type":"Equal","value":true}}]}}"#,
            ),
            Some("/data/records"),
        ),
        q(
            "list filtered not",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(
                r#"{"limit":9,"filter":{"must_not":[{"key":"topic","operation":{"type":"Equal","value":"beta"}}]}}"#,
            ),
            Some("/data/records"),
        ),
        q(
            "list filtered on the updated field",
            "POST",
            format!("/api/v1/tables/{t}/records/list"),
            Some(
                r#"{"limit":9,"filter":{"must":[{"key":"topic","operation":{"type":"Equal","value":"gamma"}}]}}"#,
            ),
            Some("/data/records"),
        ),
        // --- vector search, every mode plus the filtered and named variants ---------------
        q(
            "search default",
            "POST",
            format!("/api/v1/tables/{t}/search"),
            Some(&format!(r#"{{"vector":{VECTOR},"top_k":5,"limit":5}}"#)),
            Some("/data/hits"),
        ),
        q(
            "search fast",
            "POST",
            format!("/api/v1/tables/{t}/search"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"top_k":5,"limit":5,"search_mode":"fast"}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "search accurate",
            "POST",
            format!("/api/v1/tables/{t}/search"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"top_k":5,"limit":5,"search_mode":"accurate"}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "search exact",
            "POST",
            format!("/api/v1/tables/{t}/search"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"top_k":5,"limit":5,"search_mode":"exact"}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "search filtered",
            "POST",
            format!("/api/v1/tables/{t}/search"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"top_k":9,"limit":9,{filter_alpha}}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "search by named vector",
            "POST",
            format!("/api/v1/tables/{t}/search"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"top_k":5,"limit":5,"vector_name":"title"}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "search with payload and vector",
            "POST",
            format!("/api/v1/tables/{t}/search"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"top_k":3,"limit":3,"with_vector":true}}"#
            )),
            Some("/data/hits"),
        ),
        // --- full text and fusion ---------------------------------------------------------
        q(
            "text search",
            "POST",
            format!("/api/v1/tables/{t}/search/text"),
            Some(r#"{"query":"widgets","limit":9}"#),
            Some("/data/hits"),
        ),
        q(
            "text search on one field",
            "POST",
            format!("/api/v1/tables/{t}/search/text"),
            Some(r#"{"query":"widgets","limit":9,"fields":["title"]}"#),
            Some("/data/hits"),
        ),
        q(
            "text search fuzzy",
            "POST",
            format!("/api/v1/tables/{t}/search/text"),
            Some(r#"{"query":"widgts","limit":9,"fuzzy":{"max_distance":2,"prefix_length":0}}"#),
            Some("/data/hits"),
        ),
        q(
            "text search filtered",
            "POST",
            format!("/api/v1/tables/{t}/search/text"),
            Some(&format!(
                r#"{{"query":"widgets","limit":9,{filter_alpha}}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "hybrid dense and text",
            "POST",
            format!("/api/v1/tables/{t}/search/hybrid"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"text":"widgets","limit":5,"fusion_method":"dbsf"}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "hybrid with sparse",
            "POST",
            format!("/api/v1/tables/{t}/search/hybrid"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"text":"widgets","limit":5,"fusion_method":"dbsf","sparse_vector":{{"indices":[4],"values":[1.0],"dimension":16}}}}"#
            )),
            Some("/data/hits"),
        ),
        q(
            "hybrid weighted",
            "POST",
            format!("/api/v1/tables/{t}/search/hybrid"),
            Some(&format!(
                r#"{{"vector":{VECTOR},"text":"widgets","limit":5,"fusion_method":"reciprocal_rank_fusion","vector_weight":0.2}}"#
            )),
            Some("/data/hits"),
        ),
        // --- the graph the console draws from ---------------------------------------------
        q(
            "knn graph",
            "POST",
            format!("/api/v1/tables/{t}/graph"),
            Some(r#"{"sample":5,"neighbours":2,"seed":1}"#),
            Some("/data/ids"),
        ),
        q(
            "knn graph filtered",
            "POST",
            format!("/api/v1/tables/{t}/graph"),
            Some(&format!(
                r#"{{"sample":5,"neighbours":2,"seed":1,{filter_alpha}}}"#
            )),
            Some("/data/ids"),
        ),
        // --- a deleted record stays deleted, everywhere -----------------------------------
        q(
            "the deleted record",
            "GET",
            format!("/api/v1/tables/{t}/records/r5"),
            None,
            None,
        ),
    ]
}

fn ask_all(port: u16) -> BTreeMap<&'static str, serde_json::Value> {
    let mut answers = BTreeMap::new();
    for question in questions() {
        let answer = call(
            port,
            question.method,
            &question.path,
            question.body.as_deref(),
        )
        .unwrap_or_else(|| panic!("no answer for {:?}", question.name));
        if let Some(pointer) = question.non_empty {
            let found = answer.pointer(pointer).and_then(|v| v.as_array());
            assert!(
                found.is_some_and(|items| !items.is_empty()),
                "{:?} answered with nothing at {pointer}, so it cannot tell a rebuilt index from \
                 an empty one and would pass whatever the server did: {answer}",
                question.name
            );
        }
        answers.insert(question.name, normalise(&answer));
    }
    answers
}

#[test]
fn every_answer_survives_a_restart() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    let before = {
        let server = start(dir.path(), port);
        seed(server.port);
        let answers = ask_all(server.port);
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
        answers
    };

    let server = start(dir.path(), port);
    let after = ask_all(server.port);

    let mut differed = Vec::new();
    for (name, expected) in &before {
        let actual = after.get(name).expect("the same questions are asked twice");
        if actual != expected {
            differed.push(format!(
                "  {name}\n    before: {}\n    after:  {}",
                truncate(&expected.to_string()),
                truncate(&actual.to_string())
            ));
        }
    }

    assert!(
        differed.is_empty(),
        "{} of {} answers changed across a restart. State that is built while records are \
         written and not rebuilt when the server comes back looks exactly like this — the \
         records are all present and something derived from them is not:\n{}",
        differed.len(),
        before.len(),
        differed.join("\n")
    );
}

fn truncate(text: &str) -> String {
    if text.len() <= 400 {
        return text.to_string();
    }
    format!("{}… ({} bytes)", &text[..400], text.len())
}

/// The suite is only as good as its coverage of what has state behind it, and coverage is easy
/// to lose by accident: a question whose answer becomes empty stops discriminating without
/// failing. `ask_all` already refuses an empty answer; this refuses a suite that has quietly
/// shrunk.
#[test]
fn the_questions_cover_what_has_state_behind_it() {
    let asked = questions();
    let names: Vec<&str> = asked.iter().map(|q| q.name).collect();
    for subject in [
        "filtered", // the payload index
        "named",    // the named vector indexes
        "text",     // the Tantivy index
        "sparse",   // the sparse postings
        "graph",    // the sampling and its filter
        "exact",    // the scan that bypasses the graph
    ] {
        assert!(
            names.iter().any(|name| name.contains(subject)),
            "no question mentions {subject:?}; every subsystem that keeps state derived from \
             the records needs one, or a restart that loses it goes unnoticed"
        );
    }
    assert!(
        asked.len() >= 25,
        "the question list has shrunk to {}",
        asked.len()
    );
}

/// The three the sweep above found, each named so a failure says which one came back.
///
/// `every_answer_survives_a_restart` would catch all of them again, but it reports "these
/// answers differ" — useful for finding something nobody knew about, useless for saying what
/// broke. These say it.
#[test]
fn a_named_vector_space_is_rebuilt_and_a_deleted_record_leaves_it() {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    let named = |port: u16| -> Vec<String> {
        let body = format!(r#"{{"vector":{VECTOR},"top_k":5,"limit":5,"vector_name":"title"}}"#);
        call(
            port,
            "POST",
            &format!("/api/v1/tables/{TABLE}/search"),
            Some(&body),
        )
        .and_then(|answer| {
            answer["data"]["hits"].as_array().map(|hits| {
                hits.iter()
                    .filter_map(|hit| hit["id"].as_str().map(str::to_string))
                    .collect()
            })
        })
        .unwrap_or_default()
    };

    let before = {
        let server = start(dir.path(), port);
        seed(server.port);
        let hits = named(server.port);
        // The seed deletes r5, whose named vector is the query itself. A delete used to leave
        // the named indexes alone, so it came back first, at the score of an exact match.
        assert!(
            !hits.contains(&"r5".to_string()),
            "a deleted record answered a named vector search: {hits:?}"
        );
        assert!(
            !hits.is_empty(),
            "the named space answered nothing before any restart"
        );
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
        hits
    };

    let server = start(dir.path(), port);
    let after = named(server.port);
    assert_eq!(
        after, before,
        "the named vector space was not rebuilt: the snapshot carries the default index and not \
         this one, and the collection is created empty"
    );
}

#[test]
fn a_table_keeps_the_time_it_was_created() {
    // `restore_table` rebuilds a table through `TableConfig`, which carries no timestamps, so
    // both were stamped afresh on every start — the same gap that once made a table's id change
    // on every restart. `GET /tables/{t}` reported a table created seconds ago holding records
    // from months back.
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    let created = |port: u16| -> String {
        call(port, "GET", &format!("/api/v1/tables/{TABLE}"), None).unwrap()["data"]["table"]
            ["created_at"]
            .as_str()
            .expect("a table reports when it was created")
            .to_string()
    };

    let before = {
        let server = start(dir.path(), port);
        seed(server.port);
        let at = created(server.port);
        call(server.port, "POST", "/api/v1/snapshot", Some("{}"));
        at
    };

    let server = start(dir.path(), port);
    assert_eq!(
        created(server.port),
        before,
        "created_at moved across a restart"
    );
}
