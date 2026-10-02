//! The wire shapes, mirroring what `src/api/handlers.rs` accepts and returns.
//!
//! Deliberately not shared with the server crate. A client that depends on the server's types
//! compiles the server to talk to it, and the two would drift together silently rather than
//! failing where a user can see it. `tests/api_contract_test.rs` in the server is what keeps
//! the route list honest; these structs keep the bodies honest.

use serde::{Deserialize, Serialize};

/// Every response the server sends is wrapped in this.
#[derive(Debug, Deserialize)]
pub(crate) struct Envelope<T> {
    /// Absent on the un-enveloped error bodies, which is what `default` covers: no `success`
    /// field means this was not a success.
    #[serde(default)]
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ErrorBody>,
    /// Carried only by the bare bodies; the enveloped ones put details inside `error`.
    #[serde(default)]
    pub details: Option<serde_json::Value>,
}

/// The server's error body: a stable code, a message, and optional details.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub details: Option<serde_json::Value>,
}

/// One route still answers outside the envelope, with a bare `{"error": "..."}` and no
/// `success` field: `POST /query`, the DXQL handler, which this client never calls. Batch
/// insert's validation and overload paths used to as well -- the wrong-dimension refusal, the
/// most common first mistake anyone makes, arrived bare -- and were moved onto the envelope on
/// 2026-09-27. The bare shape is still read, because a client that rejected it would turn a
/// future regression on the server into an unreadable error here instead of a readable one.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ErrorBody {
    Structured(ApiError),
    Bare(String),
}

impl From<ErrorBody> for ApiError {
    fn from(body: ErrorBody) -> Self {
        match body {
            ErrorBody::Structured(error) => error,
            // No code to pass through, so say where the absence comes from rather than
            // inventing one that a caller might match on.
            ErrorBody::Bare(message) => ApiError {
                code: "UNSTRUCTURED".to_string(),
                message,
                details: None,
            },
        }
    }
}

/// How distance is measured in a table, spelled the way the server's `DistanceMetric`
/// parses it: `cosine`, `euclidean`, `dot`, `manhattan`.
///
/// `DotProduct` serialized as `"dotproduct"` -- `rename_all = "lowercase"` on the variant
/// name -- and the server has never accepted that spelling, so every table ever created with
/// it was a 400 `INVALID_DISTANCE_METRIC`. Nothing caught it because the live test builds its
/// table with `Cosine`. `Manhattan` was missing outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    #[serde(rename = "dot")]
    DotProduct,
    Manhattan,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateTable {
    pub name: String,
    pub dimension: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance_metric: Option<DistanceMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_type: Option<String>,
}

impl CreateTable {
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            distance_metric: None,
            description: None,
            index_type: None,
        }
    }

    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = Some(metric);
        self
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = Some(text.into());
        self
    }
}

/// A record on its way in. `payload` is free-form JSON.
#[derive(Debug, Clone, Serialize)]
pub struct Record {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl Record {
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            payload: None,
        }
    }

    pub fn payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }
}

/// A record on its way out. The server names the payload field `payload_json` and sends it as
/// a string; [`StoredRecord::payload`] parses it so callers do not have to know that.
#[derive(Debug, Clone, Deserialize)]
pub struct StoredRecord {
    pub id: String,
    #[serde(default)]
    pub vector: Vec<f32>,
    #[serde(default)]
    pub payload_json: Option<String>,
    #[serde(default)]
    pub version: Option<u64>,
}

impl StoredRecord {
    /// The payload as JSON, or `Null` when the record has none or it does not parse.
    pub fn payload(&self) -> serde_json::Value {
        self.payload_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null)
    }
}

/// What an update changes. A `None` vector keeps the stored dense, sparse and named vectors,
/// which is what makes a payload-only update safe for a table used with hybrid search.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Update {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    /// `"replace"` (the server's default) or `"merge"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge_mode: Option<String>,
    /// Create the record when it is not there. Off by default, like the server.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub upsert: bool,
}

impl Update {
    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    pub fn payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Merge the given payload keys into the stored payload instead of replacing it.
    pub fn merge(mut self) -> Self {
        self.merge_mode = Some("merge".to_string());
        self
    }

    pub fn upsert(mut self) -> Self {
        self.upsert = true;
        self
    }
}

/// One condition of a filter: a payload key and what must be true of it.
#[derive(Debug, Clone, Serialize)]
pub struct Condition {
    pub key: String,
    pub operation: Operation,
}

/// The payload comparisons the server understands, tagged the way it expects
/// (`{"type": "Equal", "value": ...}`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "value")]
pub enum Operation {
    Equal(serde_json::Value),
    NotEqual(serde_json::Value),
    GreaterThan(serde_json::Value),
    GreaterThanOrEqual(serde_json::Value),
    LessThan(serde_json::Value),
    LessThanOrEqual(serde_json::Value),
    In(Vec<serde_json::Value>),
    NotIn(Vec<serde_json::Value>),
    Contains(String),
    StartsWith(String),
    EndsWith(String),
}

/// A filter is always `must` / `must_not` / `should`, never a bare condition: the server's
/// `Filter` has those three fields and serde ignores anything else, so an unwrapped condition
/// does not fail -- it quietly matches everything.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Filter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub must: Option<Vec<Condition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub must_not: Option<Vec<Condition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub should: Option<Vec<Condition>>,
}

impl Filter {
    /// Every condition has to hold.
    pub fn all(conditions: Vec<Condition>) -> Self {
        Self {
            must: Some(conditions),
            ..Default::default()
        }
    }

    /// At least one condition has to hold.
    pub fn any(conditions: Vec<Condition>) -> Self {
        Self {
            should: Some(conditions),
            ..Default::default()
        }
    }

    /// None of the conditions may hold.
    pub fn none(conditions: Vec<Condition>) -> Self {
        Self {
            must_not: Some(conditions),
            ..Default::default()
        }
    }
}

/// Shorthand for the common case: `eq("colour", json!("red"))`.
pub fn eq(key: impl Into<String>, value: serde_json::Value) -> Condition {
    Condition {
        key: key.into(),
        operation: Operation::Equal(value),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    pub top_k: usize,
    /// Sent as well as `top_k`: the server reads both, and leaving `limit` at its default of
    /// 10 silently truncates a larger `top_k`.
    pub limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    pub with_payload: bool,
    pub with_vector: bool,
}

impl SearchQuery {
    pub fn new(vector: Vec<f32>, top_k: usize) -> Self {
        Self {
            vector,
            top_k,
            limit: top_k,
            filter: None,
            with_payload: true,
            with_vector: false,
        }
    }

    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Return each hit's vector as well as its payload.
    pub fn with_vector(mut self) -> Self {
        self.with_vector = true;
        self
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    #[serde(default)]
    pub total: usize,
    #[serde(default, rename = "tookms")]
    pub took_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TableInfo {
    pub name: String,
    pub dimension: usize,
    #[serde(default)]
    pub record_count: u64,
    #[serde(default)]
    pub index_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InsertOutcome {
    #[serde(default)]
    pub count: usize,
    #[serde(default)]
    pub inserted_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeleteOutcome {
    /// How many records the server actually removed, which is not always how many were asked
    /// for: deleting ids that are not there removes none of them.
    #[serde(default, alias = "deleted_count")]
    pub deleted: usize,
}
