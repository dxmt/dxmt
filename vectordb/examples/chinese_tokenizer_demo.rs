// Demonstration of Chinese word segmentation using Jieba tokenizer with Tantivy
//
// This example shows how to:
// 1. Create a custom Jieba tokenizer
// 2. Register it with Tantivy
// 3. Index Chinese text
// 4. Search Chinese text with proper word segmentation
//
// Run with: cargo run --example chinese_tokenizer_demo --features text-bm25

use tantivy::schema::{Schema, TextFieldIndexing, TextOptions, IndexRecordOption, Value};
use tantivy::{doc, Index, IndexWriter, TantivyDocument};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;

// Import from the vectordb library
use dxmt_vectordb::search::jieba_tokenizer::JiebaTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chinese Word Segmentation Demo with Jieba ===\n");

    // 1. Register Jieba tokenizer first (before creating schema)
    println!("📝 Creating index and registering Jieba tokenizer...");

    // Create a temporary index to register the tokenizer
    let temp_schema = Schema::builder().build();
    let index = Index::create_in_ram(temp_schema);

    let tokenizer = JiebaTokenizer::new();
    index.tokenizers().register("jieba", tokenizer);
    println!("✅ Jieba tokenizer registered\n");

    // 2. Create schema with jieba tokenizer
    println!("📋 Creating schema with Jieba tokenizer for Chinese text...");
    let mut schema_builder = Schema::builder();

    // Configure text fields to use jieba tokenizer
    let text_indexing = TextFieldIndexing::default()
        .set_tokenizer("jieba")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);

    let text_options = TextOptions::default()
        .set_indexing_options(text_indexing)
        .set_stored();

    let title = schema_builder.add_text_field("title", text_options.clone());
    let content = schema_builder.add_text_field("content", text_options);
    let schema = schema_builder.build();

    // 3. Create new index with proper schema
    let index = Index::create_in_ram(schema.clone());

    // Re-register tokenizer on the new index
    let tokenizer = JiebaTokenizer::new();
    index.tokenizers().register("jieba", tokenizer);

    println!("✅ Schema created with Jieba tokenizer\n");

    // 4. Index some Chinese documents
    println!("📚 Indexing Chinese documents...");
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    let documents = vec![
        ("机器学习入门", "机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习。"),
        ("深度学习基础", "深度学习使用神经网络模型处理复杂的数据，在图像识别和自然语言处理中表现出色。"),
        ("自然语言处理", "自然语言处理是让计算机理解和生成人类语言的技术，包括分词、词性标注等任务。"),
        ("向量数据库", "向量数据库专门用于存储和检索高维向量数据，在相似性搜索中非常高效。"),
        ("全文搜索引擎", "全文搜索引擎可以快速检索文本内容，支持中文分词、模糊匹配等功能。"),
    ];

    for (title_text, content_text) in documents {
        index_writer.add_document(doc!(
            title => title_text,
            content => content_text
        ))?;
        println!("  ✓ Indexed: {}", title_text);
    }

    index_writer.commit()?;
    println!("✅ All documents indexed\n");

    // 5. Search
    let reader = index.reader()?;
    let searcher = reader.searcher();

    let queries = vec![
        "机器学习",
        "深度学习",
        "自然语言",
        "向量",
        "搜索引擎",
        "人工智能",
    ];

    println!("🔍 Searching with Chinese queries:\n");

    for query_text in queries {
        println!("Query: \"{}\"", query_text);

        let query_parser = QueryParser::for_index(&index, vec![title, content]);
        let query = query_parser.parse_query(query_text)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(3))?;

        if top_docs.is_empty() {
            println!("  No results found\n");
        } else {
            println!("  Found {} result(s):", top_docs.len());
            for (_score, doc_address) in top_docs {
                let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;

                // Extract text values as strings
                if let Some(title_text) = retrieved_doc.get_first(title).and_then(|v| v.as_str()) {
                    println!("    - Title: {}", title_text);
                }
                if let Some(content_text) = retrieved_doc.get_first(content).and_then(|v| v.as_str()) {
                    println!("      Content: {}", content_text);
                }
            }
            println!();
        }
    }

    println!("=== Demo completed successfully! ===");
    Ok(())
}
