//! Memory for agents: what an agent learns about its user and its work, kept so that the next
//! session -- or another agent -- can recall it.
//!
//! A memory is an ordinary record in one table (`DXMT_MEMORY_TABLE`, default `memory`), made on
//! first use with the server's embedding model and a keyword index over the same text, so
//! recall is a hybrid search: by meaning and by the words themselves. What makes it memory
//! rather than a vector store is what each record carries and how recall treats it:
//!
//! - `kind`: what sort of thing it is -- `fact`, `preference`, `event`, `instruction`, `note`
//!   (the default), or anything an agent finds useful. Recall can ask for one kind.
//! - `tags`, `source` (which agent or conversation it came from), and `at`, when it was
//!   remembered (seconds since the epoch), which recall can bound with `since`.
//! - **Facts change.** A memory may `supersede` others: they are kept, marked `superseded_by`
//!   the new one, and left out of recall unless asked for. "I moved to Berlin" does not erase
//!   "I live in Shanghai"; it ends it, and the history stays inspectable.
//! - **Saying it twice keeps it once.** A memory whose meaning is already held -- cosine
//!   similarity at or above `DXMT_MEMORY_DUPLICATE_SIMILARITY` (default 0.97) to a live memory
//!   of the same kind -- is not added: the one already there is refreshed (`at` moves to now,
//!   tags are merged) and its id returned with `duplicate: true`. Events are exempt, because
//!   an event that happens twice happened twice; so is a call naming an `id` or `supersedes`,
//!   which already says what it means. The threshold is measured, on bge-m3: restatements of
//!   one sentence scored 0.981-0.995, and contradicting facts -- Berlin and Lisbon, March 3
//!   and March 4, Chinese and English -- 0.81-0.93. A paraphrase in other words (0.81-0.87)
//!   is kept as its own memory; that costs a near-duplicate, where merging at that distance
//!   would cost a fact.
//! - **The user owns it.** Every memory can be listed, read, corrected (remember again with the
//!   same id) and deleted (`forget`); the table exports like any other.
//!
//! The same calls serve MCP (`remember`, `recall`, `forget`, `list_memories`) and REST
//! (`/api/v1/memories`), so every agent -- Claude, ChatGPT, a script -- shares one memory.

use serde::{Deserialize, Serialize};

use crate::storage::IntelligentIndexedStorage;
use crate::types::record::{Condition, CreateRecordRequest, Filter, FilterOperation};

/// What `remember` takes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct RememberRequest {
    /// The memory, in words: "Prefers answers in Chinese", "Deadline for the grant is 12 May".
    pub text: String,
    /// `fact`, `preference`, `event`, `instruction`, `note` (the default), or your own.
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Which agent or conversation it came from.
    #[serde(default)]
    pub source: Option<String>,
    /// Memories this one replaces: kept, marked superseded, left out of recall.
    #[serde(default)]
    pub supersedes: Option<Vec<String>>,
    /// To correct a memory in place, its id; a new one is made otherwise.
    #[serde(default)]
    pub id: Option<String>,
    /// Anything else to keep with it.
    #[serde(default)]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

/// What `recall` takes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct RecallRequest {
    /// What to remember: a question, a topic, a few words.
    pub query: String,
    /// How many, default 10.
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub kind: Option<String>,
    /// Memories carrying any of these tags.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Only memories from this moment on: seconds since the epoch, or RFC 3339.
    #[serde(default)]
    pub since: Option<serde_json::Value>,
    /// Superseded memories too, for history. Default false.
    #[serde(default)]
    pub include_superseded: bool,
}

/// Payload keys a memory owns; `metadata` may not use them.
const RESERVED: [&str; 6] = ["text", "kind", "tags", "source", "at", "superseded_by"];

fn table(storage: &IntelligentIndexedStorage) -> crate::Result<String> {
    storage.memory_table().map(str::to_string).ok_or_else(|| {
        crate::Error::InvalidConfig(
            "memory is turned off on this server (DXMT_MEMORY_TABLE=off)".to_string(),
        )
    })
}

fn condition(key: &str, operation: FilterOperation) -> Condition {
    Condition {
        key: key.to_string(),
        operation,
    }
}

fn seconds(since: &serde_json::Value) -> crate::Result<f64> {
    if let Some(n) = since.as_f64() {
        return Ok(n);
    }
    since
        .as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|t| t.timestamp() as f64)
        .ok_or_else(|| {
            crate::Error::InvalidInput(
                "`since` is seconds since the epoch or an RFC 3339 time".to_string(),
            )
        })
}

/// The similarity at or above which a new memory is the same as one already held;
/// `DXMT_MEMORY_DUPLICATE_SIMILARITY`, `off` (or anything above 1) to keep every memory.
fn duplicate_similarity() -> Option<f32> {
    match crate::config::env::var("DXMT_MEMORY_DUPLICATE_SIMILARITY") {
        Ok(v) if v.trim().eq_ignore_ascii_case("off") => None,
        Ok(v) => v.trim().parse::<f32>().ok().filter(|s| *s <= 1.0),
        Err(_) => Some(0.97),
    }
}

/// A memory as agents see it: the record's payload with its id and, from a search, its score.
fn view(id: &str, payload: &serde_json::Value, score: Option<f32>) -> serde_json::Value {
    let mut out = serde_json::json!({ "id": id });
    if let Some(fields) = payload.as_object() {
        for (k, v) in fields {
            out[k] = v.clone();
        }
    }
    if let Some(score) = score {
        out["score"] = serde_json::json!(score);
    }
    out
}

impl IntelligentIndexedStorage {
    /// Keep a memory; mark the ones it supersedes. The memory's id, and those it superseded.
    pub async fn remember(&self, request: RememberRequest) -> crate::Result<serde_json::Value> {
        let table = table(self)?;
        if request.text.trim().is_empty() {
            return Err(crate::Error::InvalidInput(
                "a memory needs `text`".to_string(),
            ));
        }
        let mut payload = serde_json::Map::new();
        if let Some(extra) = request.metadata {
            if let Some(key) = extra.keys().find(|k| RESERVED.contains(&k.as_str())) {
                return Err(crate::Error::InvalidInput(format!(
                    "metadata may not set `{key}`: it is one of the memory's own fields"
                )));
            }
            payload.extend(extra);
        }
        payload.insert(
            "kind".to_string(),
            serde_json::json!(request.kind.as_deref().unwrap_or("note")),
        );
        if let Some(tags) = request.tags {
            payload.insert("tags".to_string(), serde_json::json!(tags));
        }
        if let Some(source) = request.source {
            payload.insert("source".to_string(), serde_json::json!(source));
        }
        payload.insert(
            "at".to_string(),
            serde_json::json!(chrono::Utc::now().timestamp()),
        );
        let kind = request.kind.as_deref().unwrap_or("note").to_string();
        let checks_duplicates = request.id.is_none()
            && request.supersedes.as_ref().is_none_or(|s| s.is_empty())
            && kind != "event";
        let mut vector = Vec::new();
        if let (true, Some(threshold)) = (checks_duplicates, duplicate_similarity()) {
            self.ensure_kb_table(&table).await?;
            vector = self.embed_for_table(&table, &request.text).await?;
            if let Some(held) = self
                .same_memory(&table, &kind, vector.clone(), threshold)
                .await?
            {
                self.refresh_memory(&table, &held, payload.get("tags"))
                    .await?;
                return Ok(serde_json::json!({
                    "id": held,
                    "superseded": [],
                    "duplicate": true,
                }));
            }
        }
        let id = request
            .id
            .unwrap_or_else(|| format!("mem-{}", uuid::Uuid::new_v4()));
        self.insert_records(
            &table,
            vec![CreateRecordRequest {
                id: id.clone(),
                vector,
                text: Some(request.text),
                sparse_vector: None,
                named_vectors: None,
                payload: Some(serde_json::Value::Object(payload)),
            }],
        )
        .await?;

        let mut superseded = Vec::new();
        for old in request.supersedes.unwrap_or_default() {
            if old == id {
                continue;
            }
            let record = self.get_record(&table, &old).await?;
            let mut payload = record.payload.clone();
            if let Some(fields) = payload.as_object_mut() {
                fields.insert("superseded_by".to_string(), serde_json::json!(id));
            }
            self.update_record(
                &table,
                &old,
                CreateRecordRequest {
                    id: old.clone(),
                    vector: record.vector.clone(),
                    text: None,
                    sparse_vector: None,
                    named_vectors: None,
                    payload: Some(payload),
                },
            )
            .await?;
            superseded.push(old);
        }
        Ok(serde_json::json!({ "id": id, "superseded": superseded }))
    }

    /// The live memory of `kind` nearest `vector`, if it is at least `threshold` similar.
    async fn same_memory(
        &self,
        table: &str,
        kind: &str,
        vector: Vec<f32>,
        threshold: f32,
    ) -> crate::Result<Option<String>> {
        let mut query: crate::types::record::SearchQuery =
            serde_json::from_value(serde_json::json!({ "top_k": 1, "search_mode": "exact" }))
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
        query.vector = vector;
        query.filter = Some(Filter {
            must: Some(vec![
                condition("kind", FilterOperation::Equal(serde_json::json!(kind))),
                condition("superseded_by", FilterOperation::IsEmpty),
            ]),
            must_not: None,
            should: None,
        });
        let found = self.search(table, query).await?;
        Ok(found
            .hits
            .first()
            .filter(|hit| hit.score >= threshold)
            .map(|hit| hit.id.clone()))
    }

    /// Remembered again: `at` moves to now and the new tags join the old.
    async fn refresh_memory(
        &self,
        table: &str,
        id: &str,
        tags: Option<&serde_json::Value>,
    ) -> crate::Result<()> {
        let id = id.to_string();
        let record = self.get_record(table, &id).await?;
        let mut payload = record.payload.clone();
        if let Some(fields) = payload.as_object_mut() {
            fields.insert(
                "at".to_string(),
                serde_json::json!(chrono::Utc::now().timestamp()),
            );
            if let Some(serde_json::Value::Array(new)) = tags {
                let mut all = match fields.get("tags") {
                    Some(serde_json::Value::Array(old)) => old.clone(),
                    _ => Vec::new(),
                };
                for tag in new {
                    if !all.contains(tag) {
                        all.push(tag.clone());
                    }
                }
                fields.insert("tags".to_string(), serde_json::Value::Array(all));
            }
        }
        self.update_record(
            table,
            &id,
            CreateRecordRequest {
                id: id.clone(),
                vector: record.vector.clone(),
                text: None,
                sparse_vector: None,
                named_vectors: None,
                payload: Some(payload),
            },
        )
        .await?;
        Ok(())
    }

    /// The memories that best match `query`, by meaning and by keyword, newest-first among
    /// equals; superseded ones left out unless asked for.
    pub async fn recall(&self, request: RecallRequest) -> crate::Result<Vec<serde_json::Value>> {
        let table = table(self)?;
        let mut must = Vec::new();
        if let Some(kind) = request.kind {
            must.push(condition(
                "kind",
                FilterOperation::Equal(serde_json::json!(kind)),
            ));
        }
        if let Some(tags) = request.tags.filter(|t| !t.is_empty()) {
            must.push(condition(
                "tags",
                FilterOperation::In(tags.into_iter().map(serde_json::Value::String).collect()),
            ));
        }
        if let Some(since) = request.since.as_ref() {
            must.push(condition(
                "at",
                FilterOperation::GreaterThanOrEqual(serde_json::json!(seconds(since)?)),
            ));
        }
        if !request.include_superseded {
            must.push(condition("superseded_by", FilterOperation::IsEmpty));
        }
        let search = crate::types::record::HybridSearchRequest {
            text: Some(request.query),
            limit: request.limit.unwrap_or(10).clamp(1, 100),
            filter: if must.is_empty() {
                None
            } else {
                Some(Filter {
                    must: Some(must),
                    must_not: None,
                    should: None,
                })
            },
            ..serde_json::from_value(serde_json::json!({}))
                .map_err(|e| crate::Error::Internal(e.to_string()))?
        };
        let hits = self.hybrid_search(&table, search).await?;
        Ok(hits
            .hits
            .iter()
            .map(|hit| {
                view(
                    &hit.id,
                    hit.payload.as_ref().unwrap_or(&serde_json::Value::Null),
                    Some(hit.score),
                )
            })
            .collect())
    }

    /// Delete memories for good. How many there were.
    pub async fn forget(&self, ids: Vec<String>) -> crate::Result<usize> {
        let table = table(self)?;
        self.ensure_kb_table(&table).await?;
        self.delete_records(&table, ids).await
    }

    /// Memories, newest first, of one kind or all; the page and the total.
    pub async fn list_memories(
        &self,
        kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> crate::Result<(Vec<serde_json::Value>, usize)> {
        let table = table(self)?;
        if self.list_tables().await?.iter().all(|t| t.name != table) {
            return Ok((Vec::new(), 0));
        }
        let filter = kind.map(|k| Filter {
            must: Some(vec![condition(
                "kind",
                FilterOperation::Equal(serde_json::json!(k)),
            )]),
            must_not: None,
            should: None,
        });
        let (records, total) = self
            .list_records(&table, filter.as_ref(), limit.clamp(1, 1000), offset, false)
            .await?;
        Ok((
            records
                .iter()
                .map(|r| view(&r.id, &r.payload, None))
                .collect(),
            total,
        ))
    }
}
