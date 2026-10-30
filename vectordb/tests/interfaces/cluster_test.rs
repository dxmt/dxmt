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

fn records_of(port: u16, table: &str) -> u64 {
    let (status, body) = json(
        port,
        "POST",
        &format!("/api/v1/tables/{table}/records/count"),
        Some(serde_json::json!({})),
    );
    assert_eq!(status, 200, "{body}");
    body["data"]["count"].as_u64().unwrap()
}

/// A table moves to the other shard while it is being written to: writes during the move are
/// refused and retried, not lost; afterwards the table is on the new shard only, and every
/// acknowledged write is in it.
#[test]
fn a_table_moves_between_shards_under_writes_without_losing_any() {
    let tmp = tempfile::tempdir().unwrap();
    let bucket = tmp.path().join("bucket-root");
    let a = node(&tmp.path().join("a"), &bucket, "a", false);
    let b = node(&tmp.path().join("b"), &bucket, "b", false);
    let r = router(&bucket);
    wait_for_shards(r.port, &[("a", 0), ("b", 0)]);
    let (status, body) = json(
        r.port,
        "POST",
        "/api/v1/tables",
        Some(serde_json::json!({"name": "mover", "dimension": 4, "distance_metric": "euclidean"})),
    );
    assert_eq!(status, 200, "{body}");
    let records: Vec<_> = (0..50)
        .map(|i| serde_json::json!({"id": format!("r{i}"), "vector": [i, 1, 0, 0], "payload": {"i": i}}))
        .collect();
    let (status, _) = json(
        r.port,
        "POST",
        "/api/v1/tables/mover/records",
        Some(serde_json::json!({ "records": records })),
    );
    assert_eq!(status, 200);
    let placement = |bucket: &Path| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(bucket.join("bucket/cluster/PLACEMENT")).unwrap())
            .unwrap()
    };
    let from = placement(&bucket)["tables"]["mover"]["shard"]
        .as_str()
        .unwrap()
        .to_string();
    let to = if from == "a" { "b" } else { "a" };

    // A writer that keeps going through the move, retrying what is refused.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer = {
        let stop = stop.clone();
        let port = r.port;
        std::thread::spawn(move || {
            let mut acked = Vec::new();
            let mut refused = 0;
            let mut n = 0;
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                let id = format!("during-{n}");
                let (status, _) = json(
                    port,
                    "POST",
                    "/api/v1/tables/mover/records",
                    Some(serde_json::json!({"records": [{"id": id, "vector": [0, 0, 1, n]}]})),
                );
                if status == 200 {
                    acked.push(id);
                    n += 1;
                } else {
                    refused += 1;
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
            (acked, refused)
        })
    };
    std::thread::sleep(Duration::from_millis(300));
    let (status, body) = json(
        r.port,
        "POST",
        "/cluster/tables/mover/move",
        Some(serde_json::json!({ "to": to })),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["data"]["moved"], true, "{body}");
    std::thread::sleep(Duration::from_millis(300));
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    let (acked, refused) = writer.join().unwrap();
    eprintln!(
        "moved in {:.2} s; {} writes acknowledged during it, {refused} refused and retried",
        body["data"]["seconds"].as_f64().unwrap(),
        acked.len()
    );

    assert_eq!(placement(&bucket)["tables"]["mover"]["shard"], to);
    assert_eq!(records_of(r.port, "mover"), 50 + acked.len() as u64);
    for id in acked
        .iter()
        .chain(["r0".to_string(), "r49".to_string()].iter())
    {
        let reply = http::request(
            r.port,
            "GET",
            &format!("/api/v1/tables/mover/records/{id}"),
            &[],
            None,
            Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(reply.status, 200, "{id}: {}", reply.text());
        assert_eq!(reply.header("x-dxmt-shard"), Some(to));
    }
    // The payload came along, and the old shard no longer has the table.
    let (_, rec) = json(r.port, "GET", "/api/v1/tables/mover/records/r7", None);
    let payload: serde_json::Value =
        serde_json::from_str(rec["data"]["payload_json"].as_str().unwrap()).unwrap();
    assert_eq!(payload["i"], 7, "{rec}");
    let old = if from == "a" { a.port } else { b.port };
    let (status, _) = json(
        old,
        "POST",
        "/api/v1/tables/mover/records/count",
        Some(serde_json::json!({})),
    );
    assert_eq!(status, 404, "the original was dropped");
}

/// Tables made while there was one shard spread out when a second joins and the cluster is
/// rebalanced; nothing in them changes.
#[test]
fn rebalancing_spreads_tables_onto_a_new_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let bucket = tmp.path().join("bucket-root");
    let _a = node(&tmp.path().join("a"), &bucket, "a", false);
    let r = router(&bucket);
    wait_for_shards(r.port, &[("a", 0)]);
    for i in 0..6 {
        let t = format!("rb{i}");
        let (status, _) = json(
            r.port,
            "POST",
            "/api/v1/tables",
            Some(serde_json::json!({"name": t, "dimension": 4})),
        );
        assert_eq!(status, 200);
        let records: Vec<_> = (0..=i)
            .map(|j| serde_json::json!({"id": format!("x{j}"), "vector": [j, 0, 0, 1]}))
            .collect();
        let (status, _) = json(
            r.port,
            "POST",
            &format!("/api/v1/tables/{t}/records"),
            Some(serde_json::json!({ "records": records })),
        );
        assert_eq!(status, 200);
    }
    let _b = node(&tmp.path().join("b"), &bucket, "b", false);
    wait_for_shards(r.port, &[("a", 0), ("b", 0)]);

    let (status, plan) = json(
        r.port,
        "POST",
        "/cluster/rebalance",
        Some(serde_json::json!({"dry_run": true})),
    );
    assert_eq!(status, 200);
    assert_eq!(plan["data"]["moves"].as_array().unwrap().len(), 3, "{plan}");
    let (status, done) = json(
        r.port,
        "POST",
        "/cluster/rebalance",
        Some(serde_json::json!({})),
    );
    assert_eq!(status, 200, "{done}");
    let (_, st) = json(r.port, "GET", "/cluster/status", None);
    let per_shard: Vec<u64> = st["data"]["shards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["tables"].as_u64().unwrap())
        .collect();
    assert_eq!(per_shard, vec![3, 3], "{st}");
    for i in 0..6 {
        assert_eq!(records_of(r.port, &format!("rb{i}")), i + 1);
    }
    let (_, again) = json(
        r.port,
        "POST",
        "/cluster/rebalance",
        Some(serde_json::json!({"dry_run": true})),
    );
    assert_eq!(
        again["data"]["moves"].as_array().unwrap().len(),
        0,
        "balanced: {again}"
    );
}

/// A whole shard goes away -- primary and replicas. Its tables answer 503 with Retry-After
/// rather than wrong answers or hangs; every other shard is untouched; a listing says which
/// shard is missing; new tables go elsewhere. A node started for the shard on an empty
/// directory, on any machine, recovers it from the bucket with everything acknowledged.
#[test]
fn a_shard_that_goes_away_is_reported_and_comes_back_from_the_bucket() {
    let tmp = tempfile::tempdir().unwrap();
    let bucket = tmp.path().join("bucket-root");
    let _a = node(&tmp.path().join("a"), &bucket, "a", false);
    let b = node(&tmp.path().join("b"), &bucket, "b", false);
    let r = router(&bucket);
    wait_for_shards(r.port, &[("a", 0), ("b", 0)]);
    for t in ["x", "y"] {
        let (status, _) = json(
            r.port,
            "POST",
            "/api/v1/tables",
            Some(serde_json::json!({"name": t, "dimension": 4})),
        );
        assert_eq!(status, 200);
        let records: Vec<_> = (0..30)
            .map(|i| serde_json::json!({"id": format!("{t}{i}"), "vector": [i, 1, 0, 0]}))
            .collect();
        let (status, _) = json(
            r.port,
            "POST",
            &format!("/api/v1/tables/{t}/records"),
            Some(serde_json::json!({ "records": records })),
        );
        assert_eq!(status, 200);
    }
    let placement: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bucket.join("bucket/cluster/PLACEMENT")).unwrap())
            .unwrap();
    let on_b = ["x", "y"]
        .into_iter()
        .find(|t| placement["tables"][*t]["shard"] == "b")
        .unwrap();
    let on_a = if on_b == "x" { "y" } else { "x" };

    drop(b); // the whole of shard b
             // Once its record has expired (TTL 10 s by default here), the router knows it is gone.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (_, st) = json(r.port, "GET", "/cluster/status", None);
        if st["data"]["unavailable_shards"]
            .as_array()
            .is_some_and(|u| u.iter().any(|s| s == "b"))
        {
            break;
        }
        assert!(Instant::now() < deadline, "never noticed: {st}");
        std::thread::sleep(Duration::from_millis(250));
    }

    let reply = http::request(
        r.port,
        "POST",
        &format!("/api/v1/tables/{on_b}/search"),
        &[],
        Some(br#"{"vector":[1,1,0,0],"top_k":1}"#),
        Duration::from_secs(30),
    )
    .unwrap();
    assert_eq!(reply.status, 503, "{}", reply.text());
    assert!(reply.text().contains("SHARD_UNAVAILABLE"));
    assert_eq!(reply.header("retry-after"), Some("2"));
    assert_eq!(records_of(r.port, on_a), 30, "the other shard is untouched");
    let (status, list) = json(r.port, "GET", "/api/v1/tables", None);
    assert_eq!(status, 200);
    assert_eq!(list["data"]["total"], 1, "{list}");
    assert_eq!(list["data"]["unavailable_shards"], serde_json::json!(["b"]));
    let (status, body) = json(
        r.port,
        "POST",
        "/api/v1/tables",
        Some(serde_json::json!({"name": "z", "dimension": 4})),
    );
    assert_eq!(status, 200, "{body}");
    let placement: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bucket.join("bucket/cluster/PLACEMENT")).unwrap())
            .unwrap();
    assert_eq!(
        placement["tables"]["z"]["shard"], "a",
        "new tables avoid the missing shard"
    );

    // A replacement, on an empty directory: everything comes back from the bucket.
    let _b2 = node(&tmp.path().join("b-replacement"), &bucket, "b", false);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (status, body) = json(
            r.port,
            "POST",
            &format!("/api/v1/tables/{on_b}/records/count"),
            Some(serde_json::json!({})),
        );
        if status == 200 {
            assert_eq!(body["data"]["count"], 30, "{body}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "shard b never came back: {status} {body}"
        );
        std::thread::sleep(Duration::from_millis(250));
    }
    let (_, list) = json(r.port, "GET", "/api/v1/tables", None);
    assert_eq!(list["data"]["total"], 3, "{list}");
}

/// A shard with no replica loses its only node; a standby, which served nothing, takes the
/// shard over from the bucket. It leaves a healthy shard alone.
#[test]
fn a_standby_takes_over_a_shard_that_lost_every_node() {
    let tmp = tempfile::tempdir().unwrap();
    let bucket = tmp.path().join("bucket-root");
    let fast = |cmd: &mut Command| {
        cmd.env("DXMT_NODE_TTL_SECS", "3")
            .env("DXMT_LEASE_GRACE_SECS", "1")
            .env("DXMT_STANDBY_DELAY_SECS", "1");
    };
    let spawn = |args: Vec<String>, root: Option<&Path>| {
        let port = free_port();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vectordb"));
        let args: Vec<String> = args
            .into_iter()
            .map(|a| a.replace("{port}", &port.to_string()))
            .collect();
        cmd.args(&args);
        base(&mut cmd, &bucket);
        fast(&mut cmd);
        if let Some(root) = root {
            cmd.env("DXMT_ROOT_DIR", root);
        }
        let child = cmd.spawn().unwrap();
        (child, port)
    };
    let words = |s: &str| s.split(' ').map(str::to_string).collect::<Vec<_>>();
    let (child, port) = spawn(
        words("serve --host 127.0.0.1 --port {port} --shard a --advertise http://127.0.0.1:{port}"),
        Some(&tmp.path().join("a")),
    );
    wait_live(port);
    let primary = Proc { child, port };
    // A standby does not listen until it has a shard, so it is not waited for.
    let (child, standby_port) = spawn(
        words("serve --host 127.0.0.1 --port {port} --standby --advertise http://127.0.0.1:{port}"),
        Some(&tmp.path().join("standby")),
    );
    let _standby = Proc {
        child,
        port: standby_port,
    };
    let (child, port) = spawn(words("router --host 127.0.0.1 --port {port}"), None);
    wait_live(port);
    let r = Proc { child, port };
    wait_for_shards(r.port, &[("a", 0)]);

    let (status, _) = json(
        r.port,
        "POST",
        "/api/v1/tables",
        Some(serde_json::json!({"name": "t", "dimension": 4})),
    );
    assert_eq!(status, 200);
    let records: Vec<_> = (0..25)
        .map(|i| serde_json::json!({"id": format!("k{i}"), "vector": [i, 0, 1, 0]}))
        .collect();
    let (status, _) = json(
        r.port,
        "POST",
        "/api/v1/tables/t/records",
        Some(serde_json::json!({ "records": records })),
    );
    assert_eq!(status, 200);
    // Healthy: the standby stays out of it.
    std::thread::sleep(Duration::from_secs(6));
    assert!(
        http::call(standby_port, "GET", "/live", &[], None).is_none(),
        "a standby took a healthy shard"
    );

    let killed = Instant::now();
    drop(primary);
    let deadline = Instant::now() + Duration::from_secs(60);
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
            "never taken over: {} {}",
            reply.status,
            reply.text()
        );
        std::thread::sleep(Duration::from_millis(250));
    };
    eprintln!(
        "standby took over in {:.1} s",
        killed.elapsed().as_secs_f64()
    );
    assert_eq!(
        served_by.as_deref(),
        Some(format!("127-0-0-1-{standby_port}").as_str())
    );
    assert_eq!(records_of(r.port, "t"), 26);
}
