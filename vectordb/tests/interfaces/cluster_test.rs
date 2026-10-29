//! A cluster, as processes: two shards, a router, then a replica and a second router, all on one
//! filesystem-backed "bucket". Clients talk only to the router; what is checked is that every
//! request lands on the shard that holds its table, and that the router's view of the cluster
//! is the bucket's -- a second router sees what the first one did.
#![cfg(feature = "cluster")]

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[path = "../common/http.rs"]
mod http;

struct Proc {
    child: Child,
    port: u16,
}

impl Drop for Proc {
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

fn base(cmd: &mut Command, bucket_root: &Path) {
    cmd.env("DXMT_SSD_CACHE_DIR", bucket_root)
        .env("DXMT_OBJECT_STORAGE_PROVIDER", "filesystem")
        .env("DXMT_S3_BUCKET", "bucket")
        .env("DXMT_EMBEDDING_AUTODETECT", "0")
        .env("DXMT_SNAPSHOT_INTERVAL_SECS", "0")
        .env("DXMT_HEARTBEAT_SECS", "1")
        .env("DXMT_REPLICA_POLL_MS", "100")
        .env("DXMT_ROUTER_REFRESH_MS", "200")
        .env_remove("DXMT_API_KEY")
        .env_remove("DXMT_WAL_DIR")
        .env_remove("DXMT_OBJECT_STORAGE_PREFIX")
        .env_remove("DXMT_SHARD")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
}

fn wait_live(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(40);
    while Instant::now() < deadline {
        if http::call(port, "GET", "/live", &[], None).is_some_and(|(s, _)| s == 200) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("nothing answered on {port}");
}

fn node(data: &Path, bucket_root: &Path, shard: &str, replica: bool) -> Proc {
    let port = free_port();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordb"));
    cmd.args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
        .args([
            "--shard",
            shard,
            "--advertise",
            &format!("http://127.0.0.1:{port}"),
        ]);
    if replica {
        cmd.arg("--replica");
    }
    base(&mut cmd, bucket_root);
    cmd.env("DXMT_ROOT_DIR", data);
    let child = cmd.spawn().expect("spawn a node");
    wait_live(port);
    Proc { child, port }
}

fn router(bucket_root: &Path) -> Proc {
    let port = free_port();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordb"));
    cmd.args(["router", "--host", "127.0.0.1", "--port", &port.to_string()]);
    base(&mut cmd, bucket_root);
    let child = cmd.spawn().expect("spawn a router");
    wait_live(port);
    Proc { child, port }
}

fn json(
    port: u16,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> (u16, serde_json::Value) {
    let text = body.map(|b| b.to_string());
    let (status, answer) = http::call(port, method, path, &[], text.as_deref()).expect("answer");
    (
        status,
        serde_json::from_str(&answer).unwrap_or(serde_json::Value::Null),
    )
}

/// Wait until the router sees `want` (shard, has-primary, replicas) for every shard named.
fn wait_for_shards(port: u16, want: &[(&str, usize)]) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (_, status) = json(port, "GET", "/cluster/status", None);
        let ok = want.iter().all(|(shard, replicas)| {
            status["data"]["shards"].as_array().is_some_and(|shards| {
                shards.iter().any(|s| {
                    s["shard"] == *shard
                        && !s["primary"].is_null()
                        && s["replicas"].as_array().map(Vec::len) == Some(*replicas)
                })
            })
        });
        if ok {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "router never saw {want:?}: {status}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
fn a_router_spreads_tables_over_shards_and_sends_each_request_to_its_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let bucket = tmp.path().join("bucket-root");
    let a = node(&tmp.path().join("a"), &bucket, "a", false);
    let b = node(&tmp.path().join("b"), &bucket, "b", false);
    let r = router(&bucket);
    wait_for_shards(r.port, &[("a", 0), ("b", 0)]);

    // Six tables through the router land three and three.
    for i in 0..6 {
        let (status, body) = json(
            r.port,
            "POST",
            "/api/v1/tables",
            Some(
                serde_json::json!({"name": format!("t{i}"), "dimension": 4, "distance_metric": "euclidean"}),
            ),
        );
        assert_eq!(status, 200, "create t{i}: {body}");
    }
    let (_, status) = json(r.port, "GET", "/cluster/status", None);
    let per_shard: Vec<u64> = status["data"]["shards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["tables"].as_u64().unwrap())
        .collect();
    assert_eq!(per_shard, vec![3, 3], "{status}");

    // Every table takes a write and answers a search through the router, from its shard.
    let placement: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bucket.join("bucket/cluster/PLACEMENT")).unwrap())
            .unwrap();
    for i in 0..6 {
        let t = format!("t{i}");
        let (status, body) = json(
            r.port,
            "POST",
            &format!("/api/v1/tables/{t}/records"),
            Some(
                serde_json::json!({"records": [{"id": format!("{t}-a"), "vector": [1, 0, 0, i]}]}),
            ),
        );
        assert_eq!(status, 200, "{body}");
        let reply = http::request(
            r.port,
            "POST",
            &format!("/api/v1/tables/{t}/search"),
            &[],
            Some(br#"{"vector":[1,0,0,0],"top_k":1}"#),
            Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(reply.status, 200, "{}", reply.text());
        assert_eq!(
            reply.header("x-dxmt-shard"),
            placement["tables"][&t]["shard"].as_str(),
            "served by the shard the placement names"
        );
        let hits: serde_json::Value = serde_json::from_slice(&reply.body).unwrap();
        assert_eq!(hits["data"]["hits"][0]["id"], format!("{t}-a"));
    }
    // Each shard holds only its own three.
    for (p, shard) in [(a.port, "a"), (b.port, "b")] {
        let (_, list) = json(p, "GET", "/api/v1/tables", None);
        assert_eq!(list["data"]["total"], 3, "shard {shard}: {list}");
    }

    // Listing asks every shard.
    let (_, list) = json(r.port, "GET", "/api/v1/tables", None);
    assert_eq!(list["data"]["total"], 6, "{list}");

    // A name is one table across the cluster.
    let (status, body) = json(
        r.port,
        "POST",
        "/api/v1/tables",
        Some(serde_json::json!({"name": "t3", "dimension": 4})),
    );
    assert_eq!(
        (status, body["error"]["code"].as_str()),
        (409, Some("TABLE_ALREADY_EXISTS"))
    );

    // A node will not create a table the router does not know about.
    let (status, body) = json(
        a.port,
        "POST",
        "/api/v1/tables",
        Some(serde_json::json!({"name": "rogue", "dimension": 4})),
    );
    assert_eq!(
        (status, body["error"]["code"].as_str()),
        (403, Some("USE_THE_ROUTER"))
    );

    // An unknown table is a 404, as on one server.
    let (status, _) = json(
        r.port,
        "POST",
        "/api/v1/tables/nope/search",
        Some(serde_json::json!({"vector": [1,0,0,0]})),
    );
    assert_eq!(status, 404);

    // Dropping removes it from its shard and from the placement.
    let (status, body) = json(r.port, "DELETE", "/api/v1/tables/t0", None);
    assert_eq!(status, 200, "{body}");
    let (_, list) = json(r.port, "GET", "/api/v1/tables", None);
    assert_eq!(list["data"]["total"], 5);
    let (status, _) = json(
        r.port,
        "POST",
        "/api/v1/tables/t0/search",
        Some(serde_json::json!({"vector": [1,0,0,0]})),
    );
    assert_eq!(status, 404);

    // The knowledge-base table is placed on first use and the request reaches a node, which
    // here has no model and says so.
    let (status, body) = json(
        r.port,
        "POST",
        "/api/v1/tables/kb/records",
        Some(serde_json::json!({"records": [{"id": "n", "text": "a note"}]})),
    );
    assert_eq!(status, 400, "{body}");
    assert!(body.to_string().contains("ollama pull"), "{body}");

    // A replica of shard a serves the reads that ask for one, once it has caught up.
    let replica = node(&tmp.path().join("a-replica"), &bucket, "a", true);
    wait_for_shards(r.port, &[("a", 1), ("b", 0)]);
    let on_a = (0..6)
        .map(|i| format!("t{i}"))
        .find(|t| t != "t0" && placement["tables"][t]["shard"] == "a")
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let reply = http::request(
            r.port,
            "POST",
            &format!("/api/v1/tables/{on_a}/search"),
            &[("x-dxmt-read", "replica")],
            Some(br#"{"vector":[1,0,0,0],"top_k":1}"#),
            Duration::from_secs(30),
        )
        .unwrap();
        let from_replica =
            reply.header("x-dxmt-node") == Some(format!("127-0-0-1-{}", replica.port).as_str());
        let hits: serde_json::Value = serde_json::from_slice(&reply.body).unwrap_or_default();
        if reply.status == 200
            && from_replica
            && hits["data"]["hits"][0]["id"] == format!("{on_a}-a")
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "replica never served {on_a}: {}",
            reply.text()
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    // A write asking for a replica still goes to the primary.
    let reply = http::request(
        r.port,
        "POST",
        &format!("/api/v1/tables/{on_a}/records"),
        &[("x-dxmt-read", "replica")],
        Some(br#"{"records":[{"id":"w","vector":[0,1,0,0]}]}"#),
        Duration::from_secs(30),
    )
    .unwrap();
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(
        reply.header("x-dxmt-node"),
        Some(format!("127-0-0-1-{}", a.port).as_str())
    );

    // A second router reads the same cluster from the bucket.
    let r2 = router(&bucket);
    wait_for_shards(r2.port, &[("a", 1), ("b", 0)]);
    let (_, list) = json(r2.port, "GET", "/api/v1/tables", None);
    assert_eq!(list["data"]["total"], 5, "{list}");
    drop((a, b, r, r2, replica));
}

/// A shard's primary dies; its replica sees the lease run out, takes it, promotes itself, and
/// the router sends writes to it -- with nothing restarted by hand, and nothing the old primary
/// acknowledged missing.
#[test]
fn a_replica_takes_over_its_shard_when_the_primary_dies() {
    let tmp = tempfile::tempdir().unwrap();
    let bucket = tmp.path().join("bucket-root");
    let fast = |cmd: &mut Command| {
        cmd.env("DXMT_NODE_TTL_SECS", "3")
            .env("DXMT_LEASE_GRACE_SECS", "1");
    };
    let start = |name: &str, replica: bool| {
        let port = free_port();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordb"));
        cmd.args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()])
            .args([
                "--shard",
                "a",
                "--advertise",
                &format!("http://127.0.0.1:{port}"),
            ]);
        if replica {
            cmd.arg("--replica");
        }
        base(&mut cmd, &bucket);
        fast(&mut cmd);
        cmd.env("DXMT_ROOT_DIR", tmp.path().join(name));
        let child = cmd.spawn().unwrap();
        wait_live(port);
        Proc { child, port }
    };
    let primary = start("p", false);
    let replica = start("r", true);
    let r = {
        let port = free_port();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordb"));
        cmd.args(["router", "--host", "127.0.0.1", "--port", &port.to_string()]);
        base(&mut cmd, &bucket);
        fast(&mut cmd);
        let child = cmd.spawn().unwrap();
        wait_live(port);
        Proc { child, port }
    };
    wait_for_shards(r.port, &[("a", 1)]);

    let (status, body) = json(
        r.port,
        "POST",
        "/api/v1/tables",
        Some(serde_json::json!({"name": "t", "dimension": 4, "distance_metric": "euclidean"})),
    );
    assert_eq!(status, 200, "{body}");
    // Writes go through until the primary holds its lease (the first heartbeat).
    let mut acked = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    while acked.len() < 20 {
        let id = format!("before-{}", acked.len());
        let (status, _) = json(
            r.port,
            "POST",
            "/api/v1/tables/t/records",
            Some(serde_json::json!({"records": [{"id": id, "vector": [1, 0, 0, 0]}]})),
        );
        if status == 200 {
            acked.push(id);
        } else {
            assert!(
                Instant::now() < deadline,
                "the primary never took writes: {status}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    let killed = Instant::now();
    drop(primary); // SIGKILL
    let deadline = Instant::now() + Duration::from_secs(30);
    let served_by = loop {
        let reply = http::request(
            r.port,
            "POST",
            "/api/v1/tables/t/records",
            &[],
            Some(br#"{"records":[{"id":"after","vector":[0,1,0,0]}]}"#),
            Duration::from_secs(30),
        )
        .unwrap();
        if reply.status == 200 {
            break reply.header("x-dxmt-node").map(str::to_string);
        }
        assert!(
            Instant::now() < deadline,
            "no failover within 30 s: {} {}",
            reply.status,
            reply.text()
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    eprintln!("failover took {:.1} s", killed.elapsed().as_secs_f64());
    assert_eq!(
        served_by.as_deref(),
        Some(format!("127-0-0-1-{}", replica.port).as_str()),
        "the replica is the primary now"
    );
    acked.push("after".to_string());
    for id in &acked {
        let (status, _) = json(
            r.port,
            "GET",
            &format!("/api/v1/tables/t/records/{id}"),
            None,
        );
        assert_eq!(status, 200, "{id} was acknowledged and is gone");
    }
}
