# Working with Embeddings in DXMT VectorDB

This tutorial covers everything you need to know about working with embeddings in DXMT VectorDB.

## Table of Contents
1. [Understanding Embeddings](#understanding-embeddings)
2. [Embedding Providers](#embedding-providers)
3. [Generating Embeddings](#generating-embeddings)
4. [Multimodal Embeddings](#multimodal-embeddings)
5. [Best Practices](#best-practices)

## Understanding Embeddings

### What are Embeddings?

Embeddings are dense numerical representations that capture semantic meaning:

```
Text: "The cat sat on the mat"
  ↓ (Embedding Model)
Vector: [0.12, -0.34, 0.56, ..., 0.78]  (384 dimensions)
```

### Why Embeddings Matter

1. **Semantic Search**: Find content by meaning, not keywords
2. **Similarity**: Measure how related two pieces of content are
3. **Clustering**: Group similar items together
4. **Classification**: Categorize content based on embeddings

### Embedding Dimensions

Common dimensions and their trade-offs:

| Model | Dimensions | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose |
| all-mpnet-base-v2 | 768 | Medium | Better | Balanced |
| text-embedding-ada-002 | 1536 | Slow | Best | High accuracy |
| e5-large-v2 | 1024 | Medium | Excellent | Multilingual |

## Embedding Providers

### OpenAI Embeddings

```rust
use dxmt_vectordb::embeddings::openai::OpenAIEmbedding;

// Initialize OpenAI embeddings
let openai = OpenAIEmbedding::new(
    std::env::var("OPENAI_API_KEY")?,
    "text-embedding-ada-002".to_string(),
);

// Generate embedding
let text = "DXMT VectorDB is amazing for semantic search";
let embedding = openai.embed(text).await?;
println!("Embedding dimension: {}", embedding.len()); // 1536

// Batch embeddings for efficiency
let texts = vec![
    "First document",
    "Second document",
    "Third document",
];
let embeddings = openai.batch_embed(&texts).await?;
```

**OpenAI Models:**
- `text-embedding-ada-002`: Best overall (1536d)
- `text-embedding-3-small`: Faster, smaller (512d)
- `text-embedding-3-large`: Higher quality (3072d)

### Cohere Embeddings

```rust
use dxmt_vectordb::embeddings::cohere::CohereEmbedding;

// Initialize Cohere
let cohere = CohereEmbedding::new(
    std::env::var("COHERE_API_KEY")?,
    "embed-english-v3.0".to_string(),
);

// Configure input type for better results
let embedding = cohere.embed_with_type(
    text,
    "search_document"  // or "search_query", "classification", "clustering"
).await?;

// Multilingual support
let cohere_multi = CohereEmbedding::new(
    api_key,
    "embed-multilingual-v3.0".to_string(),
);
```

**Cohere Models:**
- `embed-english-v3.0`: English only (1024d)
- `embed-multilingual-v3.0`: 100+ languages (1024d)
- `embed-english-light-v3.0`: Faster, smaller (384d)

### SentenceTransformers (Local)

```rust
use dxmt_vectordb::embeddings::sentence_transformers::SentenceTransformer;

// Load local model (no API key needed!)
let model = SentenceTransformer::new(
    "sentence-transformers/all-MiniLM-L6-v2"
)?;

// Fast local embedding
let embedding = model.embed(text)?;

// Custom models
let custom = SentenceTransformer::from_path(
    "/path/to/your/model"
)?;
```

**Popular Models:**
- `all-MiniLM-L6-v2`: Fast, general purpose (384d)
- `all-mpnet-base-v2`: Higher quality (768d)
- `multi-qa-mpnet-base-dot-v1`: Q&A optimized (768d)
- `all-distilroberta-v1`: RoBERTa based (768d)

### Auto-Embedding with Fallback

```rust
use dxmt_vectordb::embeddings::{
    AutoEmbedding, AutoEmbeddingConfig, EmbeddingProvider
};

// Configure with fallback chain
let config = AutoEmbeddingConfig {
    providers: vec![
        EmbeddingProvider::OpenAI {
            api_key: env::var("OPENAI_API_KEY").ok(),
            model: "text-embedding-ada-002".to_string(),
        },
        EmbeddingProvider::Cohere {
            api_key: env::var("COHERE_API_KEY").ok(),
            model: "embed-english-v3.0".to_string(),
        },
        EmbeddingProvider::Local {
            model: "all-MiniLM-L6-v2".to_string(),
        },
    ],
    retry_strategy: RetryStrategy::ExponentialBackoff {
        max_retries: 3,
        initial_delay_ms: 100,
    },
    cache_enabled: true,
    cache_ttl_seconds: 3600,
};

let auto_embedder = AutoEmbedding::new(config).await?;

// Automatically selects available provider
let embedding = auto_embedder.embed(text).await?;
```

## Generating Embeddings

### Text Preprocessing

```rust
// Clean and normalize text before embedding
fn preprocess_text(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

let clean_text = preprocess_text("  Hello, WORLD!!! 123  ");
let embedding = embedder.embed(&clean_text).await?;
```

### Chunking Long Documents

```rust
use dxmt_vectordb::utils::text_splitter::TextSplitter;

let splitter = TextSplitter::new()
    .chunk_size(512)
    .chunk_overlap(50)
    .separator("\n\n");

let document = "Very long document text...";
let chunks = splitter.split(document)?;

// Embed each chunk
let mut chunk_embeddings = Vec::new();
for (i, chunk) in chunks.iter().enumerate() {
    let embedding = embedder.embed(chunk).await?;
    
    let record = CreateRecordRequest {
        id: format!("doc_chunk_{}", i),
        vector: embedding,
        payload: Some(json!({
            "content": chunk,
            "chunk_index": i,
            "total_chunks": chunks.len(),
        })),
        ..Default::default()
    };
    
    storage.insert_record("documents", record).await?;
    chunk_embeddings.push(embedding);
}
```

### Caching Embeddings

```rust
use dxmt_vectordb::cache::EmbeddingCache;
use std::time::Duration;

// Create embedding cache
let cache = EmbeddingCache::new()
    .max_size(10000)
    .ttl(Duration::from_secs(3600));

// Check cache first
let cache_key = hash_text(text);
if let Some(cached) = cache.get(&cache_key).await {
    return Ok(cached);
}

// Generate and cache
let embedding = embedder.embed(text).await?;
cache.insert(cache_key, embedding.clone()).await;
```

## Multimodal Embeddings

### Text + Image Embeddings (CLIP)

```rust
use dxmt_vectordb::multimodal::clip::CLIPEmbedding;

let clip = CLIPEmbedding::new("openai/clip-vit-base-patch32")?;

// Embed text
let text_embedding = clip.embed_text("A photo of a cat").await?;

// Embed image
let image_path = "cat.jpg";
let image_embedding = clip.embed_image(image_path).await?;

// Cross-modal search: find images using text
let results = storage.search("images", SearchQuery {
    vector: text_embedding,
    top_k: 10,
    ..Default::default()
}).await?;
```

### Audio Embeddings (Whisper)

```rust
use dxmt_vectordb::multimodal::whisper::WhisperEmbedding;

let whisper = WhisperEmbedding::new("openai/whisper-base")?;

// Transcribe and embed audio
let audio_path = "speech.mp3";
let (transcription, embedding) = whisper.transcribe_and_embed(audio_path).await?;

println!("Transcription: {}", transcription);

// Store with metadata
let record = CreateRecordRequest {
    id: "audio_001".to_string(),
    vector: embedding,
    payload: Some(json!({
        "transcription": transcription,
        "duration_seconds": 120,
        "format": "mp3",
    })),
    ..Default::default()
};
```

### Unified Multimodal Search

```rust
use dxmt_vectordb::multimodal::unified::UnifiedEmbedding;

let unified = UnifiedEmbedding::new()
    .with_text_model("all-mpnet-base-v2")
    .with_image_model("clip-vit-base-patch32")
    .with_audio_model("whisper-base");

// Embed any content type
let content = MultimodalContent::from_path("content.jpg")?;
let embedding = unified.embed(content).await?;

// Search across all modalities
let query = unified.embed_text("sunset at beach").await?;
let results = storage.search("multimodal", SearchQuery {
    vector: query,
    filter: Some(json!({
        "modality": {"$in": ["image", "video"]}
    })),
    ..Default::default()
}).await?;
```

## Best Practices

### 1. Choose the Right Model

```rust
// For general text search
let general = SentenceTransformer::new("all-mpnet-base-v2")?;

// For question answering
let qa = SentenceTransformer::new("multi-qa-mpnet-base-dot-v1")?;

// For multilingual content
let multilingual = CohereEmbedding::new(
    api_key,
    "embed-multilingual-v3.0".to_string(),
);

// For maximum accuracy
let accurate = OpenAIEmbedding::new(
    api_key,
    "text-embedding-3-large".to_string(),
);
```

### 2. Normalize Embeddings

```rust
fn normalize_embedding(embedding: &mut Vec<f32>) {
    let norm: f32 = embedding.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    
    if norm > 0.0 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}

// Use normalized embeddings for dot product similarity
let mut embedding = embedder.embed(text).await?;
normalize_embedding(&mut embedding);
```

### 3. Handle Rate Limits

```rust
use backoff::{retry, ExponentialBackoff};
use std::time::Duration;

let backoff = ExponentialBackoff {
    max_elapsed_time: Some(Duration::from_secs(60)),
    ..Default::default()
};

let embedding = retry(backoff, || async {
    embedder.embed(text).await
        .map_err(|e| {
            if e.is_rate_limit() {
                backoff::Error::transient(e)
            } else {
                backoff::Error::permanent(e)
            }
        })
}).await?;
```

### 4. Batch Processing

```rust
use futures::stream::{self, StreamExt};

// Process in batches of 100
let texts: Vec<String> = load_documents();
let batch_size = 100;

let embeddings = stream::iter(texts.chunks(batch_size))
    .map(|batch| async move {
        embedder.batch_embed(batch).await
    })
    .buffer_unordered(4)  // Process 4 batches concurrently
    .collect::<Vec<_>>()
    .await;
```

### 5. Monitor Embedding Quality

```rust
// Track embedding statistics
struct EmbeddingStats {
    mean: f32,
    std_dev: f32,
    sparsity: f32,
    min: f32,
    max: f32,
}

fn analyze_embedding(embedding: &[f32]) -> EmbeddingStats {
    let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
    
    let variance = embedding.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / embedding.len() as f32;
    
    let std_dev = variance.sqrt();
    
    let sparsity = embedding.iter()
        .filter(|&&x| x.abs() < 0.01)
        .count() as f32 / embedding.len() as f32;
    
    EmbeddingStats {
        mean,
        std_dev,
        sparsity,
        min: *embedding.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        max: *embedding.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
    }
}
```

## Advanced Techniques

### Fine-tuning Embeddings

```python
# Fine-tune embeddings for your domain (Python example)
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare training data
train_examples = [
    InputExample(texts=['query', 'positive_doc'], label=1.0),
    InputExample(texts=['query', 'negative_doc'], label=0.0),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
)

# Save fine-tuned model
model.save('my-fine-tuned-model')
```

### Hybrid Embeddings

```rust
// Combine multiple embedding types
struct HybridEmbedding {
    text_weight: f32,
    metadata_weight: f32,
}

impl HybridEmbedding {
    async fn embed(&self, text: &str, metadata: &serde_json::Value) -> Vec<f32> {
        // Get text embedding
        let text_emb = text_embedder.embed(text).await?;
        
        // Get metadata embedding
        let metadata_str = serde_json::to_string(metadata)?;
        let meta_emb = metadata_embedder.embed(&metadata_str).await?;
        
        // Weighted combination
        let mut hybrid = Vec::with_capacity(text_emb.len());
        for (t, m) in text_emb.iter().zip(meta_emb.iter()) {
            hybrid.push(t * self.text_weight + m * self.metadata_weight);
        }
        
        normalize_embedding(&mut hybrid);
        hybrid
    }
}
```

### Cross-lingual Embeddings

```rust
// Search across languages
let multilingual = CohereEmbedding::new(
    api_key,
    "embed-multilingual-v3.0".to_string(),
);

// Index documents in multiple languages
let documents = vec![
    ("Hello world", "en"),
    ("Bonjour le monde", "fr"),
    ("Hola mundo", "es"),
    ("你好世界", "zh"),
];

for (text, lang) in documents {
    let embedding = multilingual.embed(text).await?;
    storage.insert_record("multilingual", CreateRecordRequest {
        id: format!("doc_{}", lang),
        vector: embedding,
        payload: Some(json!({
            "text": text,
            "language": lang,
        })),
        ..Default::default()
    }).await?;
}

// Search in any language
let query = "Welt";  // German for "world"
let query_emb = multilingual.embed(query).await?;
let results = storage.search("multilingual", SearchQuery {
    vector: query_emb,
    top_k: 5,
    ..Default::default()
}).await?;
```

## Troubleshooting

### Common Issues

**Issue**: Inconsistent embedding dimensions
```rust
// Always verify dimensions match
assert_eq!(embedding.len(), table_config.dimension,
    "Embedding dimension mismatch");
```

**Issue**: Poor search results
```rust
// Check embedding quality
let stats = analyze_embedding(&embedding);
if stats.std_dev < 0.01 {
    println!("Warning: Low variance in embedding");
}
```

**Issue**: Slow embedding generation
```rust
// Use local models for speed
let local = SentenceTransformer::new("all-MiniLM-L6-v2")?;
// 50-100x faster than API calls
```

## Next Steps

- [Tutorial 3: Advanced Search Techniques](./03_advanced_search.md)
- [Tutorial 4: Production Deployment](./04_production.md)
- [API Reference: Embeddings](/docs/api/embeddings)
- [Benchmarks: Embedding Performance](/benchmarks/embeddings)

## Summary

You've learned:
- ✅ Different embedding providers and their trade-offs
- ✅ How to generate and manage embeddings
- ✅ Multimodal embedding techniques
- ✅ Best practices for production use
- ✅ Advanced embedding strategies

Ready to master advanced search? Continue to the next tutorial!