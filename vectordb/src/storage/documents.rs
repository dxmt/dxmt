//! Documents: a long text stored as overlapping chunks, one record each, written and removed
//! as one thing.
//!
//! A record has one vector, and an embedding model reads only so much: bge-m3 through ollama
//! refused a 60,975-character web page outright, and a text cut to fit is represented by its
//! beginning alone (`embeddings::service::embed_fitting`). A knowledge base is mostly long
//! texts, so they are split here, on paragraph and sentence boundaries, into chunks a model
//! reads whole.
//!
//! **A chunk is an ordinary record.** Its id is `{doc_id}#{n}` and its payload carries
//! `doc_id`, `chunk_index` and `chunk_count` beside the caller's metadata, so search, filters,
//! `records/list` and grouped search (`group_by: "doc_id"`) see documents with no new code
//! path, and nothing about a record changes meaning. Writing a document replaces it: the new
//! chunks are upserted, then the chunks of the previous version past the new count are
//! deleted. A write that fails between the two leaves the new chunks and some old ones, all
//! still tagged with the document; writing it again finishes the job.

use serde::{Deserialize, Serialize};

use crate::storage::IntelligentIndexedStorage;
use crate::types::record::{Condition, CreateRecordRequest, Filter, FilterOperation};

/// Characters, not tokens: a CJK character is about one token for the multilingual models
/// this is for, and 500 of them is a passage a reader takes in at once, well inside bge-m3's
/// context. English runs about four characters a token, so the same chunk is ~125 tokens.
pub const DEFAULT_CHUNK_SIZE: usize = 500;
pub const DEFAULT_CHUNK_OVERLAP: usize = 50;
const MIN_CHUNK_SIZE: usize = 50;
/// The payload keys a chunk carries; metadata may not use them.
pub const DOC_ID: &str = "doc_id";
pub const CHUNK_INDEX: &str = "chunk_index";
pub const CHUNK_COUNT: &str = "chunk_count";
/// More chunks than any document has, and fewer than would make listing them a problem.
const MAX_CHUNKS: usize = 100_000;

/// The body of `PUT /tables/{t}/documents/{doc_id}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "api-swagger", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct PutDocumentRequest {
    /// The whole document.
    pub text: String,
    /// Payload fields copied onto every chunk -- a title, a path, tags -- so a filter on them
    /// selects the document's chunks.
    #[serde(default)]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// Characters per chunk, at most. Default 500, at least 50.
    #[serde(default)]
    pub chunk_size: Option<usize>,
    /// Characters each chunk repeats from the end of the one before it, so a sentence cut by a
    /// boundary is whole in one of the two. Default 50; under half of `chunk_size`.
    #[serde(default)]
    pub chunk_overlap: Option<usize>,
}

/// What a document write did.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentWritten {
    pub doc_id: String,
    /// Chunks the document is now stored as.
    pub chunks: usize,
    /// Chunks of the previous version that the new one no longer has, now deleted.
    pub removed: usize,
}

/// One stored chunk, as `GET /tables/{t}/documents/{doc_id}` returns it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentChunk {
    pub id: String,
    pub chunk_index: usize,
    pub text: String,
}

/// A document's chunks, in order, and the metadata they carry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentChunks {
    pub doc_id: String,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub chunks: Vec<DocumentChunk>,
}

/// Split `text` into chunks of at most `size` characters, each repeating up to `overlap`
/// characters from the end of the previous one.
///
/// A chunk ends at the last paragraph break in the second half of its window, else the last
/// sentence end (`。！？；!?;` or a `.` before whitespace), else the last whitespace, else at
/// the window's edge -- unspaced Chinese with no punctuation for 500 characters is rare and
/// has no better place. The overlap starts at a word boundary where there is one, so an
/// English chunk does not open mid-word. Chunks are trimmed, and empty ones dropped.
pub fn split_into_chunks(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut start = 0;
    while start < n {
        let hard = (start + size).min(n);
        let end = if hard == n {
            n
        } else {
            best_break(&chars, start + size / 2, hard).unwrap_or(hard)
        };
        let piece: String = chars[start..end].iter().collect();
        let piece = piece.trim();
        if !piece.is_empty() {
            out.push(piece.to_string());
        }
        if end == n {
            break;
        }
        let from = end.saturating_sub(overlap).max(start + 1);
        start = (from..end)
            .find(|&i| is_space(chars[i - 1]) || is_sentence_end(&chars, i - 1))
            .unwrap_or(from);
    }
    out
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// Whether the character at `i` ends a sentence.
fn is_sentence_end(chars: &[char], i: usize) -> bool {
    match chars[i] {
        '。' | '！' | '？' | '；' | '!' | '?' | ';' | '…' => true,
        '.' => chars.get(i + 1).is_none_or(|c| c.is_whitespace()),
        _ => false,
    }
}

/// The best place in `lo..=hi` to end a chunk: a position just after a boundary.
fn best_break(chars: &[char], lo: usize, hi: usize) -> Option<usize> {
    let window = lo.max(1)..=hi;
    let last = |test: &dyn Fn(usize) -> bool| window.clone().rev().find(|&i| test(i));
    last(&|i| i >= 2 && chars[i - 1] == '\n' && chars[i - 2] == '\n')
        .or_else(|| last(&|i| is_sentence_end(chars, i - 1) || chars[i - 1] == '\n'))
        .or_else(|| last(&|i| is_space(chars[i - 1])))
}

fn by_document(doc_id: &str) -> Filter {
    Filter {
        must: Some(vec![Condition {
            key: DOC_ID.to_string(),
            operation: FilterOperation::Equal(serde_json::Value::String(doc_id.to_string())),
        }]),
        must_not: None,
        should: None,
    }
}

impl IntelligentIndexedStorage {
    /// Store `request.text` as the document `doc_id`, replacing any previous version.
    pub async fn put_document(
        &self,
        table: &str,
        doc_id: &str,
        request: PutDocumentRequest,
    ) -> crate::Result<DocumentWritten> {
        if doc_id.trim().is_empty() {
            return Err(crate::Error::InvalidInput(
                "a document needs a non-empty id".to_string(),
            ));
        }
        let size = request.chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
        let overlap = request.chunk_overlap.unwrap_or(DEFAULT_CHUNK_OVERLAP);
        if size < MIN_CHUNK_SIZE || overlap * 2 >= size {
            return Err(crate::Error::InvalidInput(format!(
                "chunk_size must be at least {MIN_CHUNK_SIZE} and chunk_overlap under half of it \
                 (got {size} and {overlap})"
            )));
        }
        let config = self.get_table(table).await?;
        let embedding = config.embedding.as_ref().ok_or_else(|| {
            crate::Error::InvalidInput(format!(
                "table '{table}' has no embedding model, and a document is stored as text the \
                 table's model embeds: create the table with `embedding`, or use the `kb` table"
            ))
        })?;
        let field = embedding.field().to_string();
        let metadata = request.metadata.unwrap_or_default();
        for reserved in [DOC_ID, CHUNK_INDEX, CHUNK_COUNT, field.as_str()] {
            if metadata.contains_key(reserved) {
                return Err(crate::Error::InvalidInput(format!(
                    "metadata may not set `{reserved}`: every chunk carries it already"
                )));
            }
        }
        let chunks = split_into_chunks(&request.text, size, overlap);
        if chunks.is_empty() {
            return Err(crate::Error::InvalidInput(
                "the document has no text".to_string(),
            ));
        }

        let previous = self.chunk_ids(table, doc_id).await?;
        let count = chunks.len();
        let records: Vec<CreateRecordRequest> = chunks
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let mut payload = metadata.clone();
                payload.insert(DOC_ID.to_string(), doc_id.into());
                payload.insert(CHUNK_INDEX.to_string(), i.into());
                payload.insert(CHUNK_COUNT.to_string(), count.into());
                CreateRecordRequest {
                    id: format!("{doc_id}#{i}"),
                    vector: Vec::new(),
                    sparse_vector: None,
                    named_vectors: None,
                    payload: Some(serde_json::Value::Object(payload)),
                    text: Some(text),
                }
            })
            .collect();
        let written: std::collections::HashSet<String> =
            records.iter().map(|r| r.id.clone()).collect();
        self.insert_records(table, records).await?;

        let stale: Vec<String> = previous
            .into_iter()
            .filter(|id| !written.contains(id))
            .collect();
        let removed = if stale.is_empty() {
            0
        } else {
            self.delete_records(table, stale).await?
        };
        Ok(DocumentWritten {
            doc_id: doc_id.to_string(),
            chunks: count,
            removed,
        })
    }

    /// The chunks of `doc_id`, in order. `RecordNotFound` when the table has none.
    pub async fn get_document(&self, table: &str, doc_id: &str) -> crate::Result<DocumentChunks> {
        let config = self.get_table(table).await?;
        let field = config
            .embedding
            .as_ref()
            .map(|e| e.field().to_string())
            .unwrap_or_else(|| "text".to_string());
        let (records, _) = self
            .list_records(table, Some(&by_document(doc_id)), MAX_CHUNKS, 0, false)
            .await?;
        if records.is_empty() {
            return Err(crate::Error::RecordNotFound(format!(
                "no document '{doc_id}' in table '{table}'"
            )));
        }
        let mut metadata = serde_json::Map::new();
        let mut chunks: Vec<DocumentChunk> = records
            .into_iter()
            .map(|record| {
                let mut payload = match record.payload {
                    serde_json::Value::Object(map) => map,
                    _ => serde_json::Map::new(),
                };
                let chunk_index = payload
                    .get(CHUNK_INDEX)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let text = payload
                    .remove(&field)
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_default();
                for key in [DOC_ID, CHUNK_INDEX, CHUNK_COUNT] {
                    payload.remove(key);
                }
                if metadata.is_empty() {
                    metadata = payload;
                }
                DocumentChunk {
                    id: record.id,
                    chunk_index,
                    text,
                }
            })
            .collect();
        chunks.sort_by_key(|c| c.chunk_index);
        Ok(DocumentChunks {
            doc_id: doc_id.to_string(),
            metadata,
            chunks,
        })
    }

    /// Delete every chunk of `doc_id`; how many there were.
    pub async fn delete_document(&self, table: &str, doc_id: &str) -> crate::Result<usize> {
        self.delete_records_matching(table, &by_document(doc_id))
            .await
    }

    async fn chunk_ids(&self, table: &str, doc_id: &str) -> crate::Result<Vec<String>> {
        let (records, _) = self
            .list_records(table, Some(&by_document(doc_id)), MAX_CHUNKS, 0, false)
            .await?;
        Ok(records.into_iter().map(|r| r.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every chunk is a piece of the text, found in order, and every non-space character of
    /// the text falls inside some chunk: nothing is invented and nothing is lost.
    fn covers(text: &str, chunks: &[String]) {
        let mut covered = vec![false; text.len()];
        let mut from = 0;
        for c in chunks {
            let at = from
                + text[from..]
                    .find(c.as_str())
                    .unwrap_or_else(|| panic!("chunk not in the text, in order: {c:?}"));
            covered[at..at + c.len()].iter_mut().for_each(|b| *b = true);
            from = at + 1;
            while !text.is_char_boundary(from) {
                from += 1;
            }
        }
        for (i, ch) in text.char_indices() {
            assert!(ch.is_whitespace() || covered[i], "lost {ch:?} at byte {i}");
        }
    }

    #[test]
    fn chinese_is_cut_at_sentence_ends_within_the_size() {
        // Numbered, so every sentence is distinct and a chunk has one place in the text.
        let text: String = (0..80)
            .map(|i| format!("第{i}句：向量数据库把文本变成向量，再按相似度检索。"))
            .collect();
        let chunks = split_into_chunks(&text, 200, 20);
        assert!(chunks.len() >= 8, "{}", chunks.len());
        for c in &chunks {
            assert!(c.chars().count() <= 200, "{} characters", c.chars().count());
            assert!(c.ends_with('。'), "cut mid-sentence: {c}");
        }
        covers(&text, &chunks);
    }

    #[test]
    fn english_chunks_do_not_open_or_close_mid_word() {
        let text: String = (0..40)
            .map(|i| format!("Sentence {i} says ownership keeps Rust memory safe. "))
            .collect();
        let chunks = split_into_chunks(&text, 300, 40);
        for c in &chunks {
            assert!(c.chars().count() <= 300);
            let first = c.split_whitespace().next().unwrap();
            assert!(
                text.contains(&format!(" {first} ")) || text.starts_with(first),
                "{c}"
            );
            assert!(c.ends_with('.'), "{c}");
        }
        covers(&text, &chunks);
    }

    #[test]
    fn a_paragraph_break_wins_over_a_sentence_end() {
        let text = format!("{}\n\n{}", "第一段。".repeat(30), "第二段。".repeat(30));
        let chunks = split_into_chunks(&text, 200, 0);
        assert!(
            chunks[0].chars().all(|c| "第一段。".contains(c)),
            "{}",
            chunks[0]
        );
    }

    #[test]
    fn neighbours_overlap_and_short_text_is_one_chunk() {
        let text = "甲乙丙丁戊己庚辛壬癸".repeat(30); // no punctuation at all
        let chunks = split_into_chunks(&text, 100, 10);
        for pair in chunks.windows(2) {
            let tail: String = pair[0]
                .chars()
                .rev()
                .take(10)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            assert!(pair[1].starts_with(&tail), "no overlap between {pair:?}");
        }
        assert_eq!(
            split_into_chunks("短笔记", 500, 50),
            vec!["短笔记".to_string()]
        );
        assert!(split_into_chunks("  \n ", 500, 50).is_empty());
    }
}
