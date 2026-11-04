//! The knowledge-base table: one table a first run can put notes in without configuring
//! anything. Named by `DXMT_KB_TABLE` (default `kb`, `off` disables), and made the first time
//! something writes to or searches it while the server has a default embedding model -- not at
//! startup, so a server nobody uses for notes never gains a table it did not ask for.

use crate::types::table::{
    EmbeddingConfig, RerankConfig, RerankFormat, TableConfig, TextAnalyzer, TextFieldConfig,
    TextSource,
};

/// The knowledge-base table's name: `DXMT_KB_TABLE`, default `kb`; `off` disables it.
pub fn kb_table_name() -> Option<String> {
    match crate::config::env::var("DXMT_KB_TABLE")
        .ok()
        .map(|v| v.trim().to_string())
    {
        Some(name) if name.is_empty() || name.eq_ignore_ascii_case("off") => None,
        Some(name) => Some(name),
        None => Some("kb".to_string()),
    }
}

/// The memory table's name: `DXMT_MEMORY_TABLE`, default `memory`; `off` disables it. Made on
/// first use like the knowledge-base table, with the same configuration; see `storage::agent_memory`.
pub fn memory_table_name() -> Option<String> {
    match crate::config::env::var("DXMT_MEMORY_TABLE")
        .ok()
        .map(|v| v.trim().to_string())
    {
        Some(name) if name.is_empty() || name.eq_ignore_ascii_case("off") => None,
        Some(name) => Some(name),
        None => Some("memory".to_string()),
    }
}

/// The server's reranker, given to the tables it makes on first use: `DXMT_RERANK_MODEL` and
/// `DXMT_RERANK_BASE_URL` (both, or none), `DXMT_RERANK_FORMAT` (`cohere`, the default, or
/// `tei` for text-embeddings-inference), `DXMT_RERANK_API_KEY_ENV`.
///
/// Off unless set, because a cross-encoder is worth having only where it is fast: on LoCoMo
/// it took the evidence in recall's top 10 from 0.71 to 0.82 on the first 400 questions
/// (`eval/memory/locomo_rerank.py`), and on CPU bge-reranker-v2-m3 spent ten seconds a query.
pub fn default_rerank() -> Option<RerankConfig> {
    let var = |k: &str| {
        crate::config::env::var(k)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let model = var("DXMT_RERANK_MODEL")?;
    let Some(base_url) = var("DXMT_RERANK_BASE_URL") else {
        tracing::warn!("DXMT_RERANK_MODEL is set without DXMT_RERANK_BASE_URL; no reranker");
        return None;
    };
    let format = match var("DXMT_RERANK_FORMAT").as_deref() {
        Some(f) if f.eq_ignore_ascii_case("tei") => RerankFormat::Tei,
        _ => RerankFormat::Cohere,
    };
    Some(RerankConfig {
        model,
        base_url,
        format,
        api_key_env: var("DXMT_RERANK_API_KEY_ENV"),
        field: None,
        top_n: None,
    })
}

/// The configuration of the knowledge-base table: the default model, and full-text search
/// over the same `text` field with the Chinese analyzer, so `/search/hybrid` has both halves.
/// jieba keeps runs of Latin letters together, so English notes are searchable too.
pub fn kb_table_config(name: &str, embedding: EmbeddingConfig) -> TableConfig {
    let text_fields = text_search_fields(embedding.field());
    TableConfig {
        name: name.to_string(),
        dimension: 0,
        distance_metric: crate::types::vector::DistanceMetric::Cosine,
        description: Some(
            "Notes and documents: insert records with `text`, search with `text`".to_string(),
        ),
        index_type: None,
        enable_full_text_search: text_fields.is_some(),
        text_fields,
        memory_optimization: None,
        payload_schema: None,
        schema_version: 1,
        named_vectors: None,
        hnsw: None,
        quantization: None,
        payload_index: None,
        embedding: Some(embedding),
        rerank: default_rerank(),
    }
}

/// Keyword search over the payload field a table's model embeds, for a table that stores
/// text: what makes `/search/hybrid` (and the MCP `hybrid_search` tool) fuse two rankings
/// rather than return the dense one alone. `None` in a build without `text-bm25`.
///
/// The Chinese analyzer, because jieba keeps runs of Latin letters together, so it serves
/// English notes as well as Chinese ones.
pub fn text_search_fields(payload_field: &str) -> Option<Vec<TextFieldConfig>> {
    cfg!(feature = "text-bm25").then(|| {
        // `name` is the text index's own field, which knows `content` (and title,
        // description, tags), not the payload key: named `text` here, nothing was indexed
        // and text search answered empty -- the trap `tests/api/test_full_text.py` warns of.
        vec![TextFieldConfig {
            name: "content".to_string(),
            source: TextSource::MetadataField(payload_field.to_string()),
            analyzer: TextAnalyzer::Chinese,
            boost: 1.0,
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_knowledge_base_table_searches_text_both_ways() {
        let config = kb_table_config(
            "kb",
            EmbeddingConfig {
                model: "bge-m3".into(),
                base_url: None,
                field: None,
                api_key_env: None,
                batch_size: None,
            },
        );
        assert_eq!(config.dimension, 0, "the width comes from the model");
        assert!(config.embedding.is_some());
        if cfg!(feature = "text-bm25") {
            assert!(config.enable_full_text_search);
            let fields = config.text_fields.unwrap();
            assert_eq!(fields[0].analyzer, TextAnalyzer::Chinese);
            assert_eq!(fields[0].name, "content");
            assert_eq!(fields[0].source, TextSource::MetadataField("text".into()));
        }
    }
}
