use crate::types::{vector::RecordId, Payload};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tantivy::{
    collector::TopDocs,
    query::{BooleanQuery, FuzzyTermQuery, Occur, Query, QueryParser},
    schema::*,
    Index, IndexReader, IndexWriter, TantivyDocument, Term,
};

#[cfg(feature = "text-bm25")]
use tantivy::schema::{IndexRecordOption, TextFieldIndexing, TextOptions};
use tokio::sync::RwLock;

/// Full-text search engine using Tantivy
///
/// This engine provides high-performance full-text search capabilities
/// integrated with DXMT's hybrid search architecture.
pub struct FullTextSearchEngine {
    /// Tantivy index
    index: Index,

    /// Index writer (protected by mutex for thread safety)
    writer: Arc<RwLock<IndexWriter>>,

    /// Index reader
    reader: IndexReader,

    /// Schema field references
    schema_fields: SchemaFields,

    /// Query parser cache
    query_parsers: Arc<RwLock<HashMap<String, QueryParser>>>,

    /// Statistics
    stats: Arc<RwLock<FullTextStats>>,
}

/// Schema field references for quick access
#[derive(Clone)]
struct SchemaFields {
    /// Document ID field
    id_field: Field,

    /// Main content field (searchable)
    content_field: Field,

    /// Title field (searchable, higher boost)
    title_field: Field,

    /// Description field (searchable)
    description_field: Field,

    /// Tags field (searchable, faceted)
    tags_field: Field,

    /// Metadata field (stored, not searchable)
    metadata_field: Field,

    /// Timestamp field
    timestamp_field: Field,
}

/// Full-text search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextConfig {
    /// Index directory path
    pub index_path: String,

    /// Memory budget for indexing (MB)
    pub index_memory_mb: usize,

    /// Number of indexing threads
    pub num_threads: usize,

    /// Enable phrase queries
    pub enable_phrase_queries: bool,

    /// Enable fuzzy search
    pub enable_fuzzy_search: bool,

    /// Default search fields
    pub default_search_fields: Vec<String>,

    /// Field boosts
    pub field_boosts: HashMap<String, f32>,
}

impl Default for FullTextConfig {
    fn default() -> Self {
        let mut field_boosts = HashMap::new();
        field_boosts.insert("title".to_string(), 2.0);
        field_boosts.insert("content".to_string(), 1.0);
        field_boosts.insert("description".to_string(), 0.8);
        field_boosts.insert("tags".to_string(), 1.5);

        Self {
            index_path: "/tmp/dxmt_full_text".to_string(),
            index_memory_mb: 256,
            num_threads: 4,
            enable_phrase_queries: true,
            enable_fuzzy_search: true,
            default_search_fields: vec![
                "title".to_string(),
                "content".to_string(),
                "description".to_string(),
            ],
            field_boosts,
        }
    }
}

/// Full-text document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextDocument {
    /// Document ID (must match RecordId)
    pub id: RecordId,

    /// Title of the document
    pub title: Option<String>,

    /// Main content
    pub content: String,

    /// Description or summary
    pub description: Option<String>,

    /// Tags or categories
    pub tags: Vec<String>,

    /// Additional metadata
    pub payload: Option<Payload>,

    /// Timestamp (for freshness ranking)
    pub timestamp: i64,
}

/// Full-text search request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextSearchRequest {
    /// Query string
    pub query: String,

    /// Fields to search (None = use defaults)
    pub search_fields: Option<Vec<String>>,

    /// Maximum results
    pub limit: usize,

    /// Offset for pagination
    pub offset: usize,

    /// Enable highlighting
    pub highlight: bool,

    /// Highlight pre-tag
    pub highlight_pre_tag: Option<String>,

    /// Highlight post-tag
    pub highlight_post_tag: Option<String>,

    /// Enable fuzzy matching
    pub fuzzy: Option<FuzzyConfig>,

    /// Filters to apply
    pub filters: Option<Vec<FieldFilter>>,

    /// Sort order
    pub sort_by: Option<SortConfig>,
}

/// Fuzzy search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyConfig {
    /// Maximum edit distance (0-2)
    pub max_distance: u8,

    /// Prefix length that must match exactly
    pub prefix_length: u8,

    /// Transposition cost
    pub transpositions: bool,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            max_distance: 1,
            prefix_length: 2,
            transpositions: true,
        }
    }
}

/// Field filter for structured filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldFilter {
    pub field: String,
    pub value: FilterValue,
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterValue {
    Text(String),
    Number(f64),
    Range { min: f64, max: f64 },
    Terms(Vec<String>),
}

/// Sort configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortConfig {
    pub field: String,
    pub order: SortOrder,
}

/// Sort order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Asc,
    Desc,
}

/// Full-text search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextSearchResult {
    /// Document ID
    pub id: RecordId,

    /// Relevance score
    pub score: f32,

    /// Title (if available)
    pub title: Option<String>,

    /// Content snippet
    pub snippet: Option<String>,

    /// Highlighted content
    pub highlights: Option<HashMap<String, Vec<String>>>,

    /// Metadata
    pub payload: Option<Payload>,
}

/// Full-text statistics
#[derive(Debug, Clone, Default)]
pub struct FullTextStats {
    pub total_documents: u64,
    pub total_searches: u64,
    pub avg_search_time_ms: f64,
    pub index_sizebytes: u64,
}

impl FullTextSearchEngine {
    /// Create a new full-text search engine
    pub async fn new(config: FullTextConfig) -> crate::Result<Self> {
        // Build schema
        let mut schema_builder = Schema::builder();

        // ID field (stored, indexed as string)
        let id_field =
            schema_builder.add_text_field("id", tantivy::schema::STRING | tantivy::schema::STORED);

        // Create text field options with default tokenizer for English text
        // Note: For Chinese text, change .set_tokenizer("default") to .set_tokenizer("jieba")
        #[cfg(feature = "text-bm25")]
        let text_options = {
            let text_indexing = TextFieldIndexing::default()
                .set_tokenizer("default") // Use default tokenizer for English (was: "jieba" for Chinese)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions);
            TextOptions::default()
                .set_indexing_options(text_indexing)
                .set_stored()
        };

        #[cfg(not(feature = "text-bm25"))]
        let text_options = tantivy::schema::TEXT | tantivy::schema::STORED;

        // Content fields with jieba tokenizer
        #[cfg(feature = "text-bm25")]
        let content_field = schema_builder.add_text_field("content", text_options.clone());
        #[cfg(feature = "text-bm25")]
        let title_field = schema_builder.add_text_field("title", text_options.clone());
        #[cfg(feature = "text-bm25")]
        let description_field = schema_builder.add_text_field("description", text_options.clone());
        #[cfg(feature = "text-bm25")]
        let tags_field = schema_builder.add_text_field("tags", text_options.clone());

        // Fallback for non-text-bm25 builds
        #[cfg(not(feature = "text-bm25"))]
        let content_field = schema_builder
            .add_text_field("content", tantivy::schema::TEXT | tantivy::schema::STORED);
        #[cfg(not(feature = "text-bm25"))]
        let title_field =
            schema_builder.add_text_field("title", tantivy::schema::TEXT | tantivy::schema::STORED);
        #[cfg(not(feature = "text-bm25"))]
        let description_field = schema_builder.add_text_field(
            "description",
            tantivy::schema::TEXT | tantivy::schema::STORED,
        );
        #[cfg(not(feature = "text-bm25"))]
        let tags_field = schema_builder.add_text_field(
            "tags",
            tantivy::schema::TEXT | tantivy::schema::STORED | tantivy::schema::FAST,
        );

        // Metadata as JSON string (stored only)
        let metadata_field = schema_builder.add_text_field("metadata", tantivy::schema::STORED);

        // Timestamp for sorting
        let timestamp_field = schema_builder.add_i64_field(
            "timestamp",
            tantivy::schema::INDEXED | tantivy::schema::STORED | tantivy::schema::FAST,
        );

        let schema = schema_builder.build();

        // Create or open index
        // Check if meta.json exists to determine if this is an existing index
        let meta_path = Path::new(&config.index_path).join("meta.json");
        // An empty path means no durable storage at all — embedded use, and most tests. The
        // index then lives in memory and dies with the process, which is what "no durable
        // storage" has to mean for it too.
        //
        // It used to fall back to a path under the temporary directory named after the table's
        // id, and once that id became a function of the table's *name* the path stopped being
        // unique to a process: two tests using the same table name shared an index, and a
        // second `cargo test` found the documents left by the first. Three integration tests
        // were failing with one more hit than they expected.
        let index = if config.index_path.trim().is_empty() {
            Index::create_in_ram(schema.clone())
        } else if meta_path.exists() {
            Index::open_in_dir(&config.index_path)
                .map_err(|e| crate::Error::Internal(format!("Failed to open index: {}", e)))?
        } else {
            std::fs::create_dir_all(&config.index_path).map_err(|e| {
                crate::Error::Internal(format!("Failed to create index directory: {}", e))
            })?;

            Index::create_in_dir(&config.index_path, schema.clone())
                .map_err(|e| crate::Error::Internal(format!("Failed to create index: {}", e)))?
        };

        // Register Chinese tokenizer (Jieba) if text-bm25 feature is enabled
        #[cfg(feature = "text-bm25")]
        {
            use crate::search::jieba_tokenizer::JiebaTokenizer;
            let tokenizer = JiebaTokenizer::new();
            index.tokenizers().register("jieba", tokenizer);
        }

        // Create index writer with configured memory
        let writer = index
            .writer(config.index_memory_mb * 1_000_000)
            .map_err(|e| crate::Error::Internal(format!("Failed to create index writer: {}", e)))?;

        // Create reader with OnCommit reload so searchers see writes after each commit
        let reader = index
            .reader_builder()
            .reload_policy(tantivy::ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| crate::Error::Internal(format!("Failed to create index reader: {}", e)))?;

        let schema_fields = SchemaFields {
            id_field,
            content_field,
            title_field,
            description_field,
            tags_field,
            metadata_field,
            timestamp_field,
        };

        Ok(Self {
            index,
            writer: Arc::new(RwLock::new(writer)),
            reader,
            schema_fields,
            query_parsers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(FullTextStats::default())),
        })
    }

    /// Add a document to the index
    pub async fn add_document(&self, doc: FullTextDocument) -> crate::Result<()> {
        let mut tantivy_doc = tantivy::doc!();

        // Add fields
        tantivy_doc.add_text(self.schema_fields.id_field, &doc.id);
        tantivy_doc.add_text(self.schema_fields.content_field, &doc.content);

        if let Some(title) = &doc.title {
            tantivy_doc.add_text(self.schema_fields.title_field, title);
        }
        if let Some(description) = &doc.description {
            tantivy_doc.add_text(self.schema_fields.description_field, description);
        }
        for tag in &doc.tags {
            tantivy_doc.add_text(self.schema_fields.tags_field, tag);
        }

        if let Some(metadata) = &doc.payload {
            let metadata_json = serde_json::to_string(metadata).map_err(|e| {
                crate::Error::Internal(format!("Failed to serialize payload: {}", e))
            })?;
            tantivy_doc.add_text(self.schema_fields.metadata_field, &metadata_json);
        }

        tantivy_doc.add_i64(self.schema_fields.timestamp_field, doc.timestamp);

        // Add to index
        let writer = self.writer.write().await;
        writer
            .add_document(tantivy_doc)
            .map_err(|e| crate::Error::Internal(format!("Failed to add document: {}", e)))?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_documents += 1;

        Ok(())
    }

    /// Update a document in the index
    pub async fn update_document(&self, doc: FullTextDocument) -> crate::Result<()> {
        // Delete existing document
        self.delete_document(&doc.id).await?;

        // Add new version
        self.add_document(doc).await
    }

    /// Delete a document from the index
    pub async fn delete_document(&self, id: &RecordId) -> crate::Result<()> {
        let writer = self.writer.write().await;

        let term = Term::from_field_text(self.schema_fields.id_field, id);
        writer.delete_term(term);

        // Update stats
        let mut stats = self.stats.write().await;
        if stats.total_documents > 0 {
            stats.total_documents -= 1
        }

        Ok(())
    }

    /// Commit changes to the index
    pub async fn commit(&self) -> crate::Result<()> {
        let mut writer = self.writer.write().await;
        writer
            .commit()
            .map_err(|e| crate::Error::Internal(format!("Failed to commit: {}", e)))?;

        // Reload reader to see committed changes
        self.reader
            .reload()
            .map_err(|e| crate::Error::Internal(format!("Failed to reload reader: {}", e)))?;

        Ok(())
    }

    /// Search for documents
    pub async fn search(
        &self,
        request: &FullTextSearchRequest,
    ) -> crate::Result<Vec<FullTextSearchResult>> {
        let start_time = std::time::Instant::now();

        // Get searcher
        let searcher = self.reader.searcher();

        // Build query
        let query = self.build_query(request).await?;

        // Create collector
        let limit = request.limit;
        let offset = request.offset;
        let collector = TopDocs::with_limit(limit + offset);

        // Execute search
        let top_docs = searcher
            .search(&query, &collector)
            .map_err(|e| crate::Error::Internal(format!("Search failed: {}", e)))?;

        // Process results
        let mut results = Vec::new();
        for (score, doc_address) in top_docs.into_iter().skip(offset) {
            let doc: TantivyDocument = searcher.doc(doc_address).map_err(|e| {
                crate::Error::Internal(format!("Failed to retrieve document: {}", e))
            })?;

            // Extract fields
            let id = doc
                .get_first(self.schema_fields.id_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::Internal("Missing document ID".to_string()))?
                .to_string();

            let title = doc
                .get_first(self.schema_fields.title_field)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let content = doc
                .get_first(self.schema_fields.content_field)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let metadata = doc
                .get_first(self.schema_fields.metadata_field)
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_str(s).ok());

            // Generate snippet
            let snippet = content
                .as_ref()
                .map(|c| self.generate_snippet(c, &request.query, 150));

            // Generate highlights if requested
            let highlights = if request.highlight {
                Some(
                    self.generate_highlights(&doc, &query, request)
                        .await
                        .unwrap_or_default(),
                )
            } else {
                None
            };

            results.push(FullTextSearchResult {
                id,
                score,
                title,
                snippet,
                highlights,
                payload: metadata,
            });
        }

        // Update stats
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.write().await;
        stats.total_searches += 1;
        stats.avg_search_time_ms = (stats.avg_search_time_ms * (stats.total_searches - 1) as f64
            + elapsed.as_millis() as f64)
            / stats.total_searches as f64;

        Ok(results)
    }

    /// Build Tantivy query from search request
    async fn build_query(&self, request: &FullTextSearchRequest) -> crate::Result<Box<dyn Query>> {
        let search_fields = request.search_fields.clone().unwrap_or_else(|| {
            vec![
                "title".to_string(),
                "content".to_string(),
                "description".to_string(),
            ]
        });

        // Get or create query parser
        let parser_key = search_fields.join(",");
        let mut parsers = self.query_parsers.write().await;

        let query_parser = parsers.entry(parser_key).or_insert_with(|| {
            let mut fields = Vec::new();
            for field_name in &search_fields {
                match field_name.as_str() {
                    "title" => fields.push(self.schema_fields.title_field),
                    "content" => fields.push(self.schema_fields.content_field),
                    "description" => fields.push(self.schema_fields.description_field),
                    "tags" => fields.push(self.schema_fields.tags_field),
                    _ => {} // Ignore unknown fields
                }
            }
            // Tantivy QueryParser 默认使用 OR 逻辑 (disjunction)
            // 这正是我们想要的：提高中文分词后的召回率
            // 例如："向量数据库" 分词后 → ["向量", "数据库"]
            // 默认OR: 匹配包含"向量"或"数据库"的文档（召回率高）
            // 如果调用 set_conjunction_by_default(): 仅匹配同时包含两者的文档（召回率低）
            QueryParser::for_index(&self.index, fields)
        });

        // Parse query, or build a fuzzy one.
        //
        // `fuzzy` was carried from the HTTP request all the way into this struct and then read
        // by nothing: a caller could ask for edit-distance matching, get a 200, and get the
        // exact-match answer. Tantivy's `QueryParser` has no fuzzy syntax to hand it, so the
        // query is built term by term instead -- one `FuzzyTermQuery` per (field, term), any of
        // which may match. `max_distance` is clamped to 2 because Levenshtein automata beyond
        // that are enormous and Tantivy refuses them.
        let mut query: Box<dyn Query> = if let Some(fuzzy) = &request.fuzzy {
            let distance = fuzzy.max_distance.min(2);
            let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
            for term_text in request.query.split_whitespace() {
                let lowered = term_text.to_lowercase();
                for field_name in &search_fields {
                    let field = match field_name.as_str() {
                        "title" => self.schema_fields.title_field,
                        "content" => self.schema_fields.content_field,
                        "description" => self.schema_fields.description_field,
                        "tags" => self.schema_fields.tags_field,
                        _ => continue,
                    };
                    let term = Term::from_field_text(field, &lowered);
                    let fuzzy_query = if fuzzy.prefix_length > 0 {
                        FuzzyTermQuery::new_prefix(term, distance, fuzzy.transpositions)
                    } else {
                        FuzzyTermQuery::new(term, distance, fuzzy.transpositions)
                    };
                    clauses.push((Occur::Should, Box::new(fuzzy_query)));
                }
            }
            if clauses.is_empty() {
                return Err(crate::Error::InvalidInput(
                    "a fuzzy query needs at least one term and one known field".to_string(),
                ));
            }
            Box::new(BooleanQuery::new(clauses))
        } else {
            query_parser
                .parse_query(&request.query)
                .map_err(|e| crate::Error::InvalidInput(format!("Invalid query: {}", e)))?
        };

        // Apply filters if any
        if let Some(filters) = &request.filters {
            let mut clauses = vec![(Occur::Must, query)];

            for filter in filters {
                let filter_query = match self.build_filter_query(filter) {
                    Ok(query) => query,
                    Err(e) => {
                        return Err(crate::Error::Internal(format!(
                            "Filter query build failed: {:?}",
                            e
                        )))
                    }
                };
                clauses.push((Occur::Must, filter_query))
            }

            query = Box::new(BooleanQuery::new(clauses));
        }

        Ok(query)
    }

    /// Build filter query
    fn build_filter_query(&self, filter: &FieldFilter) -> crate::Result<Box<dyn Query>> {
        match &filter.value {
            FilterValue::Text(text) => {
                let field = match filter.field.as_str() {
                    "tags" => self.schema_fields.tags_field,
                    "id" => self.schema_fields.id_field,
                    _ => {
                        return Err(crate::Error::InvalidInput(format!(
                            "Unknown filter field: {}",
                            filter.field
                        )));
                    }
                };

                let term = Term::from_field_text(field, text);
                Ok(Box::new(tantivy::query::TermQuery::new(
                    term,
                    tantivy::schema::IndexRecordOption::Basic,
                )))
            }
            FilterValue::Number(num) => {
                if filter.field == "timestamp" {
                    let term =
                        Term::from_field_i64(self.schema_fields.timestamp_field, *num as i64);
                    Ok(Box::new(tantivy::query::TermQuery::new(
                        term,
                        tantivy::schema::IndexRecordOption::Basic,
                    )))
                } else {
                    Err(crate::Error::InvalidInput(format!(
                        "Numeric filter not supported for field: {}",
                        filter.field
                    )))
                }
            }
            FilterValue::Range { min, max } => {
                if filter.field == "timestamp" {
                    // For tantivy 0.22, we need to use field name string
                    // Create range query using tantivy 0.25 API
                    use std::ops::Bound;
                    let min_term =
                        Term::from_field_i64(self.schema_fields.timestamp_field, *min as i64);
                    let max_term =
                        Term::from_field_i64(self.schema_fields.timestamp_field, *max as i64);
                    let range_query = tantivy::query::RangeQuery::new(
                        Bound::Included(min_term),
                        Bound::Included(max_term),
                    );
                    Ok(Box::new(range_query))
                } else {
                    Err(crate::Error::InvalidInput(format!(
                        "Range filter not supported for field: {}",
                        filter.field
                    )))
                }
            }
            FilterValue::Terms(terms) => {
                let mut clauses = Vec::new();

                for term_str in terms {
                    let field = match filter.field.as_str() {
                        "tags" => self.schema_fields.tags_field,
                        _ => {
                            return Err(crate::Error::InvalidInput(format!(
                                "Terms filter not supported for field: {}",
                                filter.field
                            )));
                        }
                    };

                    let term = Term::from_field_text(field, term_str);
                    let term_query = tantivy::query::TermQuery::new(
                        term,
                        tantivy::schema::IndexRecordOption::Basic,
                    );
                    clauses.push((
                        Occur::Should,
                        Box::new(term_query) as Box<dyn tantivy::query::Query>,
                    ));
                }

                Ok(Box::new(BooleanQuery::new(clauses)))
            }
        }
    }

    /// Generate snippet from content
    fn generate_snippet(&self, content: &str, query: &str, max_length: usize) -> String {
        // Simple snippet generation - find first occurrence of query terms
        let query_lower = query.to_lowercase();
        let content_lower = content.to_lowercase();

        if let Some(pos) = content_lower.find(&query_lower) {
            let start = pos.saturating_sub(50);
            let end = (pos + query_lower.len() + 50).min(content.len());

            let mut snippet = String::new();
            if start > 0 {
                snippet.push_str("...");
            }
            snippet.push_str(&content[start..end]);
            if end < content.len() {
                snippet.push_str("...");
            }

            snippet
        } else {
            // Fallback to beginning of content
            if content.len() > max_length {
                format!("{}...", &content[..max_length])
            } else {
                content.to_string()
            }
        }
    }

    /// Generate highlights for matched terms
    async fn generate_highlights(
        &self,
        doc: &TantivyDocument,
        _query: &dyn Query,
        request: &FullTextSearchRequest,
    ) -> crate::Result<HashMap<String, Vec<String>>> {
        let mut highlights = HashMap::new();

        // For now, return a simple implementation
        // In production, you'd use Tantivy's highlighting features

        let pre_tag = request.highlight_pre_tag.as_deref().unwrap_or("<em>");
        let post_tag = request.highlight_post_tag.as_deref().unwrap_or("</em>");

        // Highlight title
        if let Some(title) = doc
            .get_first(self.schema_fields.title_field)
            .and_then(|v| v.as_str())
        {
            let highlighted = self.highlight_text(title, &request.query, pre_tag, post_tag);
            highlights.insert("title".to_string(), vec![highlighted]);
        }
        // Highlight content
        if let Some(content) = doc
            .get_first(self.schema_fields.content_field)
            .and_then(|v| v.as_str())
        {
            let highlighted = self.highlight_text(content, &request.query, pre_tag, post_tag);
            highlights.insert("content".to_string(), vec![highlighted]);
        }
        Ok(highlights)
    }

    /// Simple text highlighting
    fn highlight_text(&self, text: &str, query: &str, pre_tag: &str, post_tag: &str) -> String {
        // Simple case-insensitive highlighting
        let query_lower = query.to_lowercase();
        let text_lower = text.to_lowercase();

        if let Some(pos) = text_lower.find(&query_lower) {
            let mut result = String::new();
            result.push_str(&text[..pos]);
            result.push_str(pre_tag);
            result.push_str(&text[pos..pos + query.len()]);
            result.push_str(post_tag);
            result.push_str(&text[pos + query.len()..]);
            result
        } else {
            text.to_string()
        }
    }

    /// How many documents the index on disk actually holds.
    ///
    /// Distinct from `get_stats().total_documents`, which counts what *this process* has added
    /// and is therefore zero for an index that was opened rather than filled. Telling those two
    /// states apart is the point: an engine reporting zero here over a table that has records is
    /// an index that lost its contents, and the only other symptom is a text search that answers
    /// 200 with nothing at all.
    pub fn document_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Get search statistics
    pub async fn get_stats(&self) -> FullTextStats {
        self.stats.read().await.clone()
    }

    /// Optimize the index (merge all segments into one)
    pub async fn optimize(&self) -> crate::Result<()> {
        let segment_ids: Vec<_> = self
            .index
            .searchable_segment_ids()
            .map_err(|e| crate::Error::Internal(format!("Failed to list segments: {}", e)))?;
        if segment_ids.len() <= 1 {
            return Ok(());
        }
        let mut writer = self.writer.write().await;
        writer
            .merge(&segment_ids)
            .await
            .map_err(|e| crate::Error::Internal(format!("Index merge failed: {}", e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_full_text_engine() -> crate::Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = FullTextConfig {
            index_path: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };

        let engine = FullTextSearchEngine::new(config).await?;

        // Add documents
        let doc1 = FullTextDocument {
            id: "doc1".to_string(),
            title: Some("Introduction to Machine Learning".to_string()),
            content: "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.".to_string(),
            description: Some("A comprehensive guide to ML basics".to_string()),
            tags: vec!["ml".to_string(), "ai".to_string(), "tutorial".to_string()],
            payload: Some(serde_json::json!({"category": "education"})),
            timestamp: 1000,
        };

        let doc2 = FullTextDocument {
            id: "doc2".to_string(),
            title: Some("Deep Learning Fundamentals".to_string()),
            content: "Deep learning uses neural networks with multiple layers to progressively extract features.".to_string(),
            description: Some("Understanding deep neural networks".to_string()),
            tags: vec!["deep-learning".to_string(), "neural-networks".to_string()],
            payload: Some(serde_json::json!({"category": "advanced"})),
            timestamp: 2000,
        };

        engine.add_document(doc1).await?;
        engine.add_document(doc2).await?;
        engine.commit().await?;

        // Search
        let search_request = FullTextSearchRequest {
            query: "machine learning".to_string(),
            search_fields: None,
            limit: 10,
            offset: 0,
            highlight: true,
            highlight_pre_tag: None,
            highlight_post_tag: None,
            fuzzy: None,
            filters: None,
            sort_by: None,
        };

        let results = engine.search(&search_request).await?;

        assert!(!results.is_empty(), "Search should return results");

        // Verify both documents are in results
        let doc1_found = results.iter().any(|r| r.id == "doc1");
        let doc2_found = results.iter().any(|r| r.id == "doc2");

        assert!(doc1_found, "doc1 should be in results");
        assert!(doc2_found, "doc2 should be in results");

        // Verify all results have positive scores and highlights
        for result in &results {
            assert!(
                result.score > 0.0,
                "Result {} should have positive score",
                result.id
            );
        }

        // Verify highlights are generated
        assert!(
            results.iter().any(|r| r.highlights.is_some()),
            "At least one result should have highlights"
        );

        // Test with filters
        let filtered_request = FullTextSearchRequest {
            query: "learning".to_string(),
            search_fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            highlight_pre_tag: None,
            highlight_post_tag: None,
            fuzzy: None,
            filters: Some(vec![FieldFilter {
                field: "tags".to_string(),
                value: FilterValue::Text("ml".to_string()),
            }]),
            sort_by: None,
        };

        let filtered_results = engine.search(&filtered_request).await?;
        assert_eq!(filtered_results.len(), 1);
        assert_eq!(filtered_results[0].id, "doc1");

        // Test update
        let updated_doc = FullTextDocument {
            id: "doc1".to_string(),
            title: Some("Advanced Machine Learning".to_string()),
            content: "Updated content about advanced ML techniques.".to_string(),
            description: None,
            tags: vec!["ml".to_string(), "advanced".to_string()],
            payload: None,
            timestamp: 3000,
        };

        engine.update_document(updated_doc).await?;
        engine.commit().await?;

        // Verify update - search for "Advanced Machine" which should match the updated title
        let advanced_search = FullTextSearchRequest {
            query: "Advanced Machine".to_string(),
            search_fields: None,
            limit: 10,
            offset: 0,
            highlight: false,
            highlight_pre_tag: None,
            highlight_post_tag: None,
            fuzzy: None,
            filters: None,
            sort_by: None,
        };
        let updated_results = engine.search(&advanced_search).await?;

        // The updated doc1 should be found with new title
        assert!(
            !updated_results.is_empty(),
            "Should find updated document with 'Advanced Machine'"
        );
        assert!(
            updated_results.iter().any(|r| {
                r.id == "doc1"
                    && r.title
                        .as_ref()
                        .map(|t| t.contains("Advanced"))
                        .unwrap_or(false)
            }),
            "Updated doc1 should have 'Advanced' in title"
        );

        // Test deletion
        engine.delete_document(&"doc2".to_string()).await?;
        engine.commit().await?;

        // Search for "deep AND learning" - must have both words (only doc2 has "deep")
        let after_delete = engine
            .search(&FullTextSearchRequest {
                query: "deep AND learning".to_string(),
                search_fields: None,
                limit: 10,
                offset: 0,
                highlight: false,
                highlight_pre_tag: None,
                highlight_post_tag: None,
                fuzzy: None,
                filters: None,
                sort_by: None,
            })
            .await?;

        assert!(
            after_delete.is_empty(),
            "Expected no results after deleting doc2"
        );

        // Check stats
        let stats = engine.get_stats().await;
        assert_eq!(stats.total_documents, 1);
        assert!(stats.total_searches > 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_complex_queries() -> crate::Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let config = FullTextConfig {
            index_path: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };

        let engine = FullTextSearchEngine::new(config).await?;

        // Add test documents
        let docs = vec![
            FullTextDocument {
                id: "rust1".to_string(),
                title: Some("Rust Programming Language".to_string()),
                content:
                    "Rust is a systems programming language focused on safety and performance."
                        .to_string(),
                description: Some("Introduction to Rust".to_string()),
                tags: vec![
                    "rust".to_string(),
                    "systems".to_string(),
                    "programming".to_string(),
                ],
                payload: None,
                timestamp: 1000,
            },
            FullTextDocument {
                id: "rust2".to_string(),
                title: Some("Async Programming in Rust".to_string()),
                content: "Asynchronous programming in Rust uses futures and async/await syntax."
                    .to_string(),
                description: Some("Guide to async Rust".to_string()),
                tags: vec![
                    "rust".to_string(),
                    "async".to_string(),
                    "programming".to_string(),
                ],
                payload: None,
                timestamp: 2000,
            },
            FullTextDocument {
                id: "python1".to_string(),
                title: Some("Python for Data Science".to_string()),
                content:
                    "Python is widely used in data science for its rich ecosystem of libraries."
                        .to_string(),
                description: Some("Python data science guide".to_string()),
                tags: vec![
                    "python".to_string(),
                    "data-science".to_string(),
                    "programming".to_string(),
                ],
                payload: None,
                timestamp: 3000,
            },
        ];

        for doc in docs {
            engine.add_document(doc).await?;
        }
        engine.commit().await?;

        // Test phrase query
        let phrase_results = engine
            .search(&FullTextSearchRequest {
                query: "\"systems programming\"".to_string(),
                search_fields: None,
                limit: 10,
                offset: 0,
                highlight: false,
                highlight_pre_tag: None,
                highlight_post_tag: None,
                fuzzy: None,
                filters: None,
                sort_by: None,
            })
            .await?;

        assert_eq!(phrase_results.len(), 1);
        assert_eq!(phrase_results[0].id, "rust1");

        // Test boolean query - Tantivy requires UPPERCASE boolean operators (AND, OR, NOT)
        let bool_results = engine
            .search(&FullTextSearchRequest {
                query: "rust AND async".to_string(),
                search_fields: None,
                limit: 10,
                offset: 0,
                highlight: false,
                highlight_pre_tag: None,
                highlight_post_tag: None,
                fuzzy: None,
                filters: None,
                sort_by: None,
            })
            .await?;

        assert_eq!(bool_results.len(), 1);
        assert_eq!(bool_results[0].id, "rust2");

        // Test field-specific search
        let field_results = engine
            .search(&FullTextSearchRequest {
                query: "programming".to_string(),
                search_fields: Some(vec!["tags".to_string()]),
                limit: 10,
                offset: 0,
                highlight: false,
                highlight_pre_tag: None,
                highlight_post_tag: None,
                fuzzy: None,
                filters: None,
                sort_by: None,
            })
            .await?;

        assert_eq!(field_results.len(), 3); // All docs have "programming" tag

        // Test range filter
        let range_results = engine
            .search(&FullTextSearchRequest {
                query: "programming".to_string(),
                search_fields: None,
                limit: 10,
                offset: 0,
                highlight: false,
                highlight_pre_tag: None,
                highlight_post_tag: None,
                fuzzy: None,
                filters: Some(vec![FieldFilter {
                    field: "timestamp".to_string(),
                    value: FilterValue::Range {
                        min: 1500.0,
                        max: 2500.0,
                    },
                }]),
                sort_by: None,
            })
            .await?;

        assert_eq!(range_results.len(), 1);
        assert_eq!(range_results[0].id, "rust2");

        Ok(())
    }
}
