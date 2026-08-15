//! Checks that every route the client SDKs call actually exists.
//!
//! The SDKs are separately packaged and nothing cross-validates them against the Rust API, so a
//! route renamed or removed here surfaces as a 404 in someone's application rather than as a
//! failing build. This test is that missing link: it lists the method/path pairs each SDK
//! issues, hits them against a real server, and fails if the router does not recognise one.
//!
//! It deliberately does not assert on response bodies. The contract being protected is "this
//! endpoint exists and accepts this method" — anything stricter would duplicate the handler
//! tests and break on every legitimate change to a payload.

use std::process::{Child, Command, Stdio};

/// A route an SDK depends on, and where that dependency lives.
struct Contract {
    method: &'static str,
    path: &'static str,
    caller: &'static str,
}

/// Extracted from dxmt-python-sdk/dxmt/client.py and dxmt-js-sdk/src/client.ts.
///
/// langchain-dxmt and llama-index-vector-stores-dxmt both go through the Python client, so they
/// add no paths of their own.
const CONTRACTS: &[Contract] = &[
    // Python SDK — also the transitive surface of the LangChain and LlamaIndex packages.
    Contract {
        method: "GET",
        path: "/tables",
        caller: "python",
    },
    Contract {
        method: "POST",
        path: "/tables",
        caller: "python",
    },
    Contract {
        method: "GET",
        path: "/tables/{t}",
        caller: "python",
    },
    Contract {
        method: "DELETE",
        path: "/tables/{t}",
        caller: "python",
    },
    Contract {
        method: "POST",
        path: "/tables/{t}/records",
        caller: "python",
    },
    Contract {
        method: "GET",
        path: "/tables/{t}/records/{r}",
        caller: "python",
    },
    Contract {
        method: "PUT",
        path: "/tables/{t}/records/{r}",
        caller: "python",
    },
    Contract {
        method: "POST",
        path: "/tables/{t}/search",
        caller: "python",
    },
    Contract {
        method: "POST",
        path: "/tables/{t}/search/text",
        caller: "python",
    },
    Contract {
        method: "POST",
        path: "/tables/{t}/search/hybrid",
        caller: "python",
    },
    Contract {
        method: "POST",
        path: "/query",
        caller: "python",
    },
    // JS SDK.
    Contract {
        method: "GET",
        path: "/health",
        caller: "js",
    },
    Contract {
        method: "DELETE",
        path: "/tables/{t}/records",
        caller: "js",
    },
    Contract {
        method: "GET",
        path: "/tables/{t}/stats",
        caller: "js",
    },
];

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

struct Server {
    child: Child,
    port: u16,
    _dir: tempfile::TempDir,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_server() -> Server {
    let dir = tempfile::TempDir::new().unwrap();
    let port = free_port();

    let child = Command::new(env!("CARGO_BIN_EXE_vectordb"))
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .env("DB_WAL_DIR", dir.path().join("wal"))
        // Background tasks would only add noise to a routing check.
        .env("DB_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DB_EVICTION_INTERVAL_SECS", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the vectordb binary");

    let server = Server {
        child,
        port,
        _dir: dir,
    };

    for _ in 0..120 {
        if curl(server.port, "GET", "/health", None).is_some() {
            return server;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    panic!("server did not become ready on port {}", server.port);
}

/// Issue a request and return (status, body), or None if the connection failed.
fn curl(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let url = if path == "/health" {
        // The JS SDK's health check is the one path served at the root, not under /api/v1.
        format!("http://127.0.0.1:{}/health", port)
    } else {
        format!("http://127.0.0.1:{}/api/v1{}", port, path)
    };

    let mut cmd = Command::new("curl");
    cmd.arg("-s")
        .arg("-w")
        .arg("\n%{http_code}")
        .arg("-m")
        .arg("10")
        .arg("-X")
        .arg(method)
        .arg(&url);
    if let Some(body) = body {
        cmd.arg("-H")
            .arg("Content-Type: application/json")
            .arg("-d")
            .arg(body);
    }

    let out = cmd.output().expect("curl is required for this test");
    let text = String::from_utf8_lossy(&out.stdout);
    let (body, code) = text.rsplit_once('\n')?;
    let code: u16 = code.trim().parse().ok()?;
    if code == 0 {
        None
    } else {
        Some((code, body.to_string()))
    }
}

/// True when the router did not recognise the request at all.
///
/// A 404 is ambiguous on its own: axum returns one with an empty body for an unmatched route,
/// and the handlers return one with a JSON error for a table that does not exist. Only the
/// former is a broken contract, so the body is what separates them.
fn route_is_missing(status: u16, body: &str) -> bool {
    status == 405 || (status == 404 && !body.contains("\"error\""))
}

#[test]
fn every_route_the_sdks_call_exists() {
    let server = start_server();

    // Give the parameterised paths something real to address, so a 404 can only mean the route
    // is missing rather than the table being absent.
    let created = curl(
        server.port,
        "POST",
        "/tables",
        Some(r#"{"name":"contract","dimension":4,"distance_metric":"cosine"}"#),
    );
    assert_eq!(
        created.as_ref().map(|(c, _)| *c),
        Some(200),
        "could not create the fixture table"
    );
    // DELETE /tables/{t} gets its own table so it cannot pull the fixture out from under the
    // contracts that run after it.
    curl(
        server.port,
        "POST",
        "/tables",
        Some(r#"{"name":"disposable","dimension":4,"distance_metric":"cosine"}"#),
    );
    curl(
        server.port,
        "POST",
        "/tables/contract/records",
        Some(r#"{"records":[{"id":"r1","vector":[1,0,0,0]}]}"#),
    );

    let body_for = |method: &str, path: &str| -> Option<&'static str> {
        if method == "GET" {
            return None;
        }
        Some(match path {
            "/tables" => r#"{"name":"contract2","dimension":4,"distance_metric":"cosine"}"#,
            p if p.ends_with("/records") && method == "POST" => {
                r#"{"records":[{"id":"r2","vector":[0,1,0,0]}]}"#
            }
            p if p.ends_with("/records") && method == "DELETE" => r#"{"ids":["r2"]}"#,
            p if p.contains("/records/") => r#"{"vector":[0,0,1,0]}"#,
            p if p.ends_with("/search") => r#"{"vector":[1,0,0,0],"top_k":1}"#,
            p if p.ends_with("/search/text") => r#"{"query":"x","top_k":1}"#,
            p if p.ends_with("/search/hybrid") => r#"{"vector":[1,0,0,0],"top_k":1}"#,
            "/query" => r#"{"query":"SELECT * FROM contract LIMIT 1"}"#,
            _ => "{}",
        })
    };

    let mut missing = Vec::new();
    for contract in CONTRACTS {
        let table = if contract.method == "DELETE" && contract.path == "/tables/{t}" {
            "disposable"
        } else {
            "contract"
        };
        let path = contract.path.replace("{t}", table).replace("{r}", "r1");
        let result = curl(
            server.port,
            contract.method,
            &path,
            body_for(contract.method, &path),
        );

        match result {
            Some((status, body)) if route_is_missing(status, &body) => missing.push(format!(
                "{} {} (used by the {} SDK) -> {}",
                contract.method, contract.path, contract.caller, status
            )),
            // Everything else, 400 and 500 included, proves the route is wired — which is the
            // only thing this test is about.
            Some(_) => {}
            None => panic!("connection failed for {} {}", contract.method, path),
        }
    }

    assert!(
        missing.is_empty(),
        "the server does not serve {} route(s) an SDK depends on:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
