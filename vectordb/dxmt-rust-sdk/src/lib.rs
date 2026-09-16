//! A Rust client for DXMT VectorDB.
//!
//! The server speaks HTTP and the repository ships clients for Python and JavaScript; this is
//! the one for Rust callers who are talking to a server rather than embedding the engine. If
//! you want the engine in-process instead, depend on `dxmt-vectordb` itself and use
//! `IntelligentIndexedStorage` -- `vectordb/examples/quickstart.rs` shows that shape.
//!
//! ```no_run
//! use dxmt_client::{Client, CreateTable, Record, SearchQuery};
//! # async fn run() -> Result<(), dxmt_client::Error> {
//! let db = Client::new("http://127.0.0.1:9999")?;
//! db.create_table(CreateTable::new("docs", 4)).await?;
//! db.insert("docs", vec![Record::new("a", vec![1.0, 0.0, 0.0, 0.0])]).await?;
//! let hits = db.search("docs", SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0], 5)).await?;
//! # Ok(()) }
//! ```
//!
//! Everything is `async`. The server is tokio-based and so is anything likely to be calling
//! this; a blocking wrapper is a `Runtime::block_on` away and is not worth a feature flag until
//! someone asks.

mod types;

use types::Envelope;
pub use types::{
    eq, ApiError, Condition, CreateTable, DeleteOutcome, DistanceMetric, Filter, InsertOutcome,
    Operation, Record, SearchHit, SearchQuery, SearchResult, StoredRecord, TableInfo, Update,
};

use serde::de::DeserializeOwned;
use serde::Serialize;

/// What can go wrong, kept apart so a caller can tell a refused request from an unreachable
/// server without matching on strings.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The server answered, and said no. Carries its code and message.
    #[error("{} ({}): {}", .status, .error.code, .error.message)]
    Api { status: u16, error: ApiError },
    /// The server answered with something this client could not read.
    #[error("unexpected response from {path}: {source}")]
    Decode {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    /// The request never got an answer.
    #[error("request failed: {0}")]
    Transport(#[from] reqwest::Error),
    /// The base URL was not a URL.
    #[error("bad base url: {0}")]
    BadUrl(String),
}

impl Error {
    /// The server's error code, for the cases where a caller wants to branch on one --
    /// `"NOT_FOUND"`, `"VALIDATION_ERROR"`, `"NOT_IMPLEMENTED"`.
    pub fn code(&self) -> Option<&str> {
        match self {
            Error::Api { error, .. } => Some(&error.code),
            _ => None,
        }
    }

    /// Whether this is the server saying the thing does not exist.
    pub fn is_not_found(&self) -> bool {
        matches!(self, Error::Api { status: 404, .. })
    }
}

/// A handle to one server. Cheap to clone -- the inner HTTP client pools connections, so one
/// `Client` shared across tasks is better than one per task.
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
}

impl Client {
    /// `base` is the server's root, e.g. `http://127.0.0.1:9999`. The `/api/v1` prefix is added
    /// here, so callers pass the same URL they would put in a browser.
    pub fn new(base: impl Into<String>) -> Result<Self, Error> {
        let base = base.into();
        let base = base.trim_end_matches('/').to_string();
        if !base.starts_with("http://") && !base.starts_with("https://") {
            return Err(Error::BadUrl(base));
        }
        Ok(Self {
            http: reqwest::Client::new(),
            base,
        })
    }

    /// Build on an HTTP client you already have -- to set timeouts, proxies or a connection
    /// pool size without this crate having to grow an option for each.
    pub fn with_http_client(base: impl Into<String>, http: reqwest::Client) -> Result<Self, Error> {
        let mut client = Self::new(base)?;
        client.http = http;
        Ok(client)
    }

    /// `true` when the server is up and its local durability is healthy. `/ready` is the
    /// deployment probe: it answers 503 once a WAL write has failed, where `/health` is a
    /// diagnostic document that still answers 200.
    pub async fn ready(&self) -> Result<bool, Error> {
        let response = self.http.get(format!("{}/ready", self.base)).send().await?;
        Ok(response.status().is_success())
    }

    // ── tables ───────────────────────────────────────────────────────────────────────────

    pub async fn create_table(&self, table: CreateTable) -> Result<TableInfo, Error> {
        self.send("POST", "/tables", Some(&table)).await
    }

    pub async fn list_tables(&self) -> Result<Vec<TableInfo>, Error> {
        #[derive(serde::Deserialize)]
        struct Listing {
            #[serde(default)]
            tables: Vec<TableInfo>,
        }
        let listing: Listing = self.send::<(), _>("GET", "/tables", None).await?;
        Ok(listing.tables)
    }

    /// Describe one table.
    ///
    /// `POST /tables` answers with the table flat and `GET /tables/{t}` answers with it nested
    /// under `table`, alongside a `record_count` that is the live one -- the copy inside
    /// `table` is whatever was stamped when the table was described. Both are unwrapped to the
    /// same [`TableInfo`] here so a caller does not have to know which route it came from.
    pub async fn get_table(&self, name: &str) -> Result<TableInfo, Error> {
        #[derive(serde::Deserialize)]
        struct Described {
            #[serde(default)]
            record_count: u64,
            table: TableInfo,
        }
        let described: Described = self
            .send::<(), _>("GET", &format!("/tables/{}", Self::segment(name)), None)
            .await?;
        Ok(TableInfo {
            record_count: described.record_count,
            ..described.table
        })
    }

    pub async fn delete_table(&self, name: &str) -> Result<(), Error> {
        self.send_ignoring_body::<()>("DELETE", &format!("/tables/{}", Self::segment(name)), None)
            .await
    }

    // ── records ──────────────────────────────────────────────────────────────────────────

    /// Insert or replace. An id that is already there is overwritten, which is what makes this
    /// an upsert rather than an insert. The server caps one request at 10,000 records.
    pub async fn insert(&self, table: &str, records: Vec<Record>) -> Result<InsertOutcome, Error> {
        #[derive(Serialize)]
        struct Body {
            records: Vec<Record>,
        }
        self.send(
            "POST",
            &format!("/tables/{}/records", Self::segment(table)),
            Some(&Body { records }),
        )
        .await
    }

    /// One record by id. `Err(e)` with `e.is_not_found()` when there is none.
    pub async fn get(&self, table: &str, id: &str) -> Result<StoredRecord, Error> {
        self.send::<(), _>(
            "GET",
            &format!(
                "/tables/{}/records/{}",
                Self::segment(table),
                Self::segment(id)
            ),
            None,
        )
        .await
    }

    /// Change a record's vector, its payload, or both. See [`Update`] for what `None` means.
    pub async fn update(&self, table: &str, id: &str, update: Update) -> Result<(), Error> {
        self.send_ignoring_body(
            "PUT",
            &format!(
                "/tables/{}/records/{}",
                Self::segment(table),
                Self::segment(id)
            ),
            Some(&update),
        )
        .await
    }

    /// Delete by id, returning how many were actually removed -- which is not the number asked
    /// for when some of the ids were not there.
    pub async fn delete(&self, table: &str, ids: Vec<String>) -> Result<DeleteOutcome, Error> {
        #[derive(Serialize)]
        struct Body {
            ids: Vec<String>,
        }
        self.send(
            "DELETE",
            &format!("/tables/{}/records", Self::segment(table)),
            Some(&Body { ids }),
        )
        .await
    }

    // ── search ───────────────────────────────────────────────────────────────────────────

    pub async fn search(&self, table: &str, query: SearchQuery) -> Result<SearchResult, Error> {
        self.send(
            "POST",
            &format!("/tables/{}/search", Self::segment(table)),
            Some(&query),
        )
        .await
    }

    // ── plumbing ─────────────────────────────────────────────────────────────────────────

    /// Percent-encode one path segment.
    ///
    /// A record id goes into the URL path, and ids in a RAG corpus routinely carry characters
    /// the path reserves: `doc.md#3` stops at the `#`, which is a fragment delimiter, so the
    /// request reaches `/records/doc.md` and the server answers about a record nobody asked
    /// for. `s3://bucket/key` turns one segment into three. Everything outside the unreserved
    /// set goes through, so this is safe to apply to ids that need no encoding at all.
    fn segment(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        for byte in value.as_bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                    out.push(*byte as char)
                }
                _ => out.push_str(&format!("%{byte:02X}")),
            }
        }
        out
    }

    async fn request<B: Serialize>(
        &self,
        method: &str,
        path: &str,
        body: Option<&B>,
    ) -> Result<(u16, String), Error> {
        let url = format!("{}/api/v1{}", self.base, path);
        let method = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|_| Error::BadUrl(method.to_string()))?;
        let mut request = self.http.request(method, &url);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().await?;
        let status = response.status().as_u16();
        Ok((status, response.text().await?))
    }

    /// Send, then unwrap the server's envelope. A body that says `success: false`, or an error
    /// status, becomes [`Error::Api`] carrying the server's own code and message -- the client
    /// never invents one.
    async fn send<B: Serialize, T: DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        body: Option<&B>,
    ) -> Result<T, Error> {
        let (status, text) = self.request(method, path, body).await?;
        let envelope: Envelope<T> =
            serde_json::from_str(&text).map_err(|source| Error::Decode {
                path: path.to_string(),
                source,
            })?;
        match (envelope.success, envelope.data, envelope.error) {
            (true, Some(data), _) => Ok(data),
            (_, _, Some(error)) => {
                let mut error: ApiError = error.into();
                error.details = error.details.or(envelope.details);
                Err(Error::Api { status, error })
            }
            // `success: true` with no data, or `false` with no error: the envelope is the
            // contract, so say which half was missing rather than guessing at a default.
            (success, _, _) => Err(Error::Api {
                status,
                error: ApiError {
                    code: "MALFORMED_RESPONSE".to_string(),
                    message: format!("success={success} with neither data nor error"),
                    details: None,
                },
            }),
        }
    }

    /// For the routes whose answer is only "it worked": delete, update.
    async fn send_ignoring_body<B: Serialize>(
        &self,
        method: &str,
        path: &str,
        body: Option<&B>,
    ) -> Result<(), Error> {
        let (status, text) = self.request(method, path, body).await?;
        let envelope: Envelope<serde_json::Value> =
            serde_json::from_str(&text).map_err(|source| Error::Decode {
                path: path.to_string(),
                source,
            })?;
        if envelope.success {
            return Ok(());
        }
        Err(Error::Api {
            status,
            error: envelope.error.map(ApiError::from).unwrap_or(ApiError {
                code: "MALFORMED_RESPONSE".to_string(),
                message: "success=false with no error".to_string(),
                details: None,
            }),
        })
    }
}
