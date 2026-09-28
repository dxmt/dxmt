//! The gRPC wire surface, extracted and held against a committed snapshot.
//!
//! A `.proto` change is a change to a contract other people's generated code was built from,
//! and most of the ways to break one are invisible in review: renumber a field, widen `int32`
//! to `int64`, drop `repeated`, delete a field and reuse its number later. Nothing in this
//! repository looked at `proto/dxmt.proto` except `prost-build`, which compiles whatever it is
//! given without an opinion about what it used to be.
//!
//! Weaviate runs `buf breaking` against the base branch for this. `buf` is the right tool and
//! the wrong dependency here: it is a separate binary, it would live only in a CI that has
//! **never run in this repository**, and a gate nobody has seen fire is a gate on paper. This
//! is the same idea inside `cargo test`, where it runs today: extract the wire-relevant
//! surface, compare with `tests/proto_surface.txt`, and on a difference print both sides and
//! say which lines are breaking.
//!
//! **What counts as breaking**, and the list is the point:
//!
//! * a field or method that disappears -- an old client still sends it
//! * a field number reused for a different name -- the worst one, because old and new clients
//!   both "work" and silently exchange the wrong value
//! * a type or label (`repeated`, `optional`) that changes under a stable number
//! * a method whose request or response type changes
//!
//! Adding a field or a method is not breaking; the snapshot still has to be updated, which is
//! the point at which somebody looks.
//!
//! The extractor is deliberately strict. A line inside a message or service that it does not
//! recognise **fails the test** rather than being skipped, because a parser that quietly
//! ignores what it has not been taught reports a clean surface for a file it did not read.
//! `map<...>` and `reserved` are the two constructs it would meet next; neither is in the file
//! today and both would stop it.
//!
//! Regenerate with `DXMT_BLESS_PROTO=1 cargo test --test proto_surface_test`, and read the
//! diff in the commit rather than the one the test printed.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn proto_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("proto/dxmt.proto")
}

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/proto_surface.txt")
}

/// One line of the surface: what a client's generated code is built from.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Item {
    /// `service.method(request) -> response`
    Method {
        service: String,
        name: String,
        request: String,
        response: String,
    },
    /// `message.field = number : label type`
    Field {
        message: String,
        number: u32,
        name: String,
        kind: String,
    },
    /// `enum.variant = number`
    EnumValue {
        enumeration: String,
        number: i64,
        name: String,
    },
}

impl Item {
    fn render(&self) -> String {
        match self {
            Item::Method {
                service,
                name,
                request,
                response,
            } => format!("rpc {service}.{name}({request}) -> {response}"),
            Item::Field {
                message,
                number,
                name,
                kind,
            } => format!("field {message}.{name} = {number} : {kind}"),
            Item::EnumValue {
                enumeration,
                number,
                name,
            } => format!("enum {enumeration}.{name} = {number}"),
        }
    }
}

/// Strip a trailing `// ...` that is not inside a string. No string literals appear in field
/// declarations, so finding `//` is enough here; if one ever does, the strict parser below
/// will reject the mangled line rather than accept a wrong one.
fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(at) => &line[..at],
        None => line,
    }
}

fn extract(source: &str) -> Vec<Item> {
    let mut items = Vec::new();
    // What block we are in, and its name. `oneof` is transparent: its fields belong to the
    // enclosing message, which is what the wire format says too.
    let mut block: Option<(&'static str, String)> = None;
    let mut in_oneof = false;

    for (lineno, raw) in source.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let at = || format!("proto/dxmt.proto:{}", lineno + 1);

        // Leaving a block.
        if line == "}" {
            if in_oneof {
                in_oneof = false;
            } else {
                block = None;
            }
            continue;
        }

        // File-level declarations carry nothing a client is built from.
        if line.starts_with("syntax ")
            || line.starts_with("package ")
            || line.starts_with("import ")
            || line.starts_with("option ")
        {
            continue;
        }

        // Entering a block. `message Foo {}` on one line is a message with no fields.
        for (keyword, tag) in [
            ("service ", "service"),
            ("message ", "message"),
            ("enum ", "enum"),
        ] {
            if let Some(rest) = line.strip_prefix(keyword) {
                let name = rest
                    .split_whitespace()
                    .next()
                    .unwrap_or_else(|| panic!("{}: `{keyword}` with no name", at()))
                    .to_string();
                block = if line.ends_with("{}") {
                    None
                } else {
                    Some((tag, name))
                };
            }
        }
        if line.starts_with("service ") || line.starts_with("message ") || line.starts_with("enum ")
        {
            continue;
        }

        let Some((tag, ref owner)) = block else {
            panic!(
                "{}: `{line}` is outside any block, and this extractor does not know what \
                    it means. Teach it rather than letting it skip a line of the contract.",
                at()
            );
        };

        if line.starts_with("oneof ") {
            assert_eq!(tag, "message", "{}: a oneof outside a message", at());
            in_oneof = true;
            continue;
        }

        let body = line
            .strip_suffix(';')
            .unwrap_or_else(|| panic!("{}: `{line}` does not end in `;`", at()));

        match tag {
            "service" => {
                // rpc Name(Request) returns (Response)
                let rest = body
                    .strip_prefix("rpc ")
                    .unwrap_or_else(|| panic!("{}: `{body}` in a service is not an rpc", at()));
                let (name, rest) = rest
                    .split_once('(')
                    .unwrap_or_else(|| panic!("{}: no `(` in `{rest}`", at()));
                let (request, rest) = rest
                    .split_once(')')
                    .unwrap_or_else(|| panic!("{}: no `)` in `{rest}`", at()));
                let response = rest
                    .trim()
                    .strip_prefix("returns")
                    .and_then(|r| r.trim().strip_prefix('('))
                    .and_then(|r| r.strip_suffix(')'))
                    .unwrap_or_else(|| panic!("{}: no `returns (…)` in `{rest}`", at()));
                items.push(Item::Method {
                    service: owner.clone(),
                    name: name.trim().to_string(),
                    request: request.trim().to_string(),
                    response: response.trim().to_string(),
                });
            }
            "enum" => {
                let (name, number) = body
                    .split_once('=')
                    .unwrap_or_else(|| panic!("{}: no `=` in enum value `{body}`", at()));
                items.push(Item::EnumValue {
                    enumeration: owner.clone(),
                    name: name.trim().to_string(),
                    number: number.trim().parse().unwrap_or_else(|e| {
                        panic!("{}: enum value `{number}` is not a number: {e}", at())
                    }),
                });
            }
            "message" => {
                let (decl, number) = body
                    .split_once('=')
                    .unwrap_or_else(|| panic!("{}: no `=` in field `{body}`", at()));
                let words: Vec<&str> = decl.split_whitespace().collect();
                // `[label] type name`
                let (kind, name) = match words.as_slice() {
                    [ty, name] => (ty.to_string(), *name),
                    [label @ ("repeated" | "optional"), ty, name] => {
                        (format!("{label} {ty}"), *name)
                    }
                    _ => panic!(
                        "{}: `{decl}` is not `[repeated|optional] type name`. `map<…>` and \
                         `reserved` would land here; both change the wire surface and this \
                         extractor must be taught them, not allowed to skip them.",
                        at()
                    ),
                };
                items.push(Item::Field {
                    message: owner.clone(),
                    name: name.to_string(),
                    number: number.trim().parse().unwrap_or_else(|e| {
                        panic!("{}: field number `{number}` is not a number: {e}", at())
                    }),
                    kind,
                });
            }
            other => panic!("{}: unknown block kind {other}", at()),
        }
    }
    items.sort();
    items
}

fn render(items: &[Item]) -> String {
    let mut out = String::from(
        "# The gRPC wire surface of proto/dxmt.proto, generated by tests/proto_surface_test.rs.\n\
         # Do not edit by hand. `DXMT_BLESS_PROTO=1 cargo test --test proto_surface_test`.\n",
    );
    for item in items {
        out.push_str(&item.render());
        out.push('\n');
    }
    out
}

/// Which differences are breaking, described in the words a reviewer needs.
fn breaking(before: &str, after: &str) -> Vec<String> {
    fn lines(text: &str) -> Vec<&str> {
        text.lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .collect()
    }
    let (old, new) = (lines(before), lines(after));

    // A field line is `field Message.name = N : kind`; key it by (Message, N) so a reused
    // number shows up as the same key with a different name.
    fn by_slot(ls: &[&str]) -> BTreeMap<(String, String), String> {
        ls.iter()
            .filter_map(|l| {
                let rest = l.strip_prefix("field ")?;
                let (lhs, kind) = rest.split_once(" : ")?;
                let (path, number) = lhs.split_once(" = ")?;
                let message = path.split('.').next()?.to_string();
                let name = path.split('.').nth(1)?.to_string();
                Some(((message, number.to_string()), format!("{name} : {kind}")))
            })
            .collect()
    }
    fn methods(ls: &[&str]) -> BTreeMap<String, String> {
        ls.iter()
            .filter_map(|l| {
                let rest = l.strip_prefix("rpc ")?;
                let (name, signature) = rest.split_once('(')?;
                Some((name.to_string(), signature.to_string()))
            })
            .collect()
    }

    let (old_slots, new_slots) = (by_slot(&old), by_slot(&new));
    let (old_methods, new_methods) = (methods(&old), methods(&new));
    let mut out = Vec::new();

    for (slot, was) in &old_slots {
        match new_slots.get(slot) {
            None => out.push(format!(
                "  {}.{} is gone: an old client still sends field {}",
                slot.0, was, slot.1
            )),
            Some(now) if now != was => {
                let old_name = was.split(" : ").next().unwrap_or(was);
                let new_name = now.split(" : ").next().unwrap_or(now);
                if old_name != new_name {
                    out.push(format!(
                        "  {}: field number {} was `{was}` and is now `{now}`. A number may \
                         never be reused -- old and new clients both parse it and exchange \
                         the wrong value with no error.",
                        slot.0, slot.1
                    ));
                } else {
                    out.push(format!(
                        "  {}.{old_name}: the type changed, `{was}` -> `{now}`, under the same \
                         field number",
                        slot.0
                    ));
                }
            }
            Some(_) => {}
        }
    }
    for (name, signature) in &old_methods {
        match new_methods.get(name) {
            None => out.push(format!("  rpc {name} is gone")),
            Some(now) if now != signature => out.push(format!(
                "  rpc {name}: the signature changed, ({signature} -> ({now}"
            )),
            Some(_) => {}
        }
    }
    out
}

#[test]
fn the_grpc_wire_surface_matches_its_snapshot() {
    let source = std::fs::read_to_string(proto_path()).expect("proto/dxmt.proto");
    let current = render(&extract(&source));

    if std::env::var("DXMT_BLESS_PROTO").is_ok() {
        std::fs::write(snapshot_path(), &current).expect("write the snapshot");
        eprintln!("wrote {}", snapshot_path().display());
        return;
    }

    let recorded = std::fs::read_to_string(snapshot_path()).unwrap_or_else(|_| {
        panic!(
            "{} does not exist. Create it with \
             `DXMT_BLESS_PROTO=1 cargo test --test proto_surface_test`.",
            snapshot_path().display()
        )
    });

    if recorded == current {
        return;
    }

    let breaks = breaking(&recorded, &current);
    let mut message = String::from("proto/dxmt.proto's wire surface changed.\n\n");
    if breaks.is_empty() {
        message.push_str(
            "Nothing in the change is breaking -- additions only, as far as this test can \
             tell. Bless the snapshot and put the diff in the commit:\n\n  \
             DXMT_BLESS_PROTO=1 cargo test --test proto_surface_test\n\n",
        );
    } else {
        message.push_str(&format!(
            "{} of the changes break clients built from the old contract:\n\n{}\n\n\
             A generated client in the wild does not recompile. If this is deliberate, bless \
             the snapshot and say why in the commit message.\n\n",
            breaks.len(),
            breaks.join("\n")
        ));
    }

    let recorded_set: std::collections::BTreeSet<&str> = recorded.lines().collect();
    let current_set: std::collections::BTreeSet<&str> = current.lines().collect();
    for line in current_set.difference(&recorded_set) {
        message.push_str(&format!("  + {line}\n"));
    }
    for line in recorded_set.difference(&current_set) {
        message.push_str(&format!("  - {line}\n"));
    }
    panic!("{message}");
}

#[test]
fn the_extractor_refuses_what_it_has_not_been_taught() {
    // The guard that makes the snapshot mean anything: a construct this parser does not
    // understand must stop it, because a surface extracted from a file it only half read is
    // worse than no surface at all. `map<string, string>` is the realistic next one.
    let with_a_map =
        "syntax = \"proto3\";\npackage p;\nmessage M {\n  map<string, string> m = 1;\n}\n";
    let panicked = std::panic::catch_unwind(|| extract(with_a_map)).is_err();
    assert!(
        panicked,
        "the extractor accepted a `map<…>` field it cannot represent, so the snapshot would \
         have silently lost it"
    );
}

#[test]
fn a_reused_field_number_is_reported_as_breaking() {
    // The check has to fail on the one change that is invisible at run time: the number stays,
    // the meaning does not. Verified here rather than assumed, because every other assertion
    // in this file is about a diff that a human would also notice.
    let before = "field Foo.alpha = 1 : string\n";
    let after = "field Foo.beta = 1 : string\n";
    let found = breaking(before, after);
    assert_eq!(
        found.len(),
        1,
        "expected one breaking change, got {found:?}"
    );
    assert!(
        found[0].contains("may never be reused"),
        "a reused number must be named as such, not reported as an ordinary edit: {found:?}"
    );

    // And the other three shapes.
    assert!(
        breaking("field Foo.a = 1 : string\n", "").len() == 1,
        "a removed field"
    );
    assert!(
        breaking("field Foo.a = 1 : int32\n", "field Foo.a = 1 : int64\n").len() == 1,
        "a widened type under a stable number"
    );
    assert!(
        breaking("rpc S.M(A) -> B\n", "rpc S.M(C) -> B\n").len() == 1,
        "a changed request type"
    );
    // Adding is not breaking.
    assert!(
        breaking(
            "field Foo.a = 1 : string\n",
            "field Foo.a = 1 : string\nfield Foo.b = 2 : string\n"
        )
        .is_empty(),
        "adding a field must not be reported as breaking"
    );
}
