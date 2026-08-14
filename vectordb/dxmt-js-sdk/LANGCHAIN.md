# DXMT LangChain Integration

Official LangChain integration for DXMT VectorDB JavaScript SDK.

## Installation

```bash
npm install dxmt-js-sdk @langchain/core
```

For embeddings, also install:

```bash
# OpenAI embeddings
npm install @langchain/openai

# Or other embedding providers
npm install @langchain/cohere
npm install @langchain/huggingface
```

## Quick Start

```typescript
import { DXMTVectorStore } from 'dxmt-js-sdk/langchain';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';

// Create embeddings
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Create DXMT vector store
const vectorStore = new DXMTVectorStore(embeddings, {
  collectionName: 'documents',
  host: 'localhost',
  port: 9999,
  distanceMetric: 'cosine',
});

// Add documents
await vectorStore.addDocuments([
  new Document({
    pageContent: 'DXMT is a high-performance vector database',
    metadata: { source: 'docs', category: 'tech' },
  }),
  new Document({
    pageContent: 'LangChain enables building LLM applications',
    metadata: { source: 'docs', category: 'tech' },
  }),
]);

// Search
const results = await vectorStore.similaritySearch(
  'What is a vector database?',
  5
);

results.forEach((doc) => {
  console.log('Content:', doc.pageContent);
  console.log('Metadata:', doc.metadata);
});
```

## Configuration

### DXMTVectorStoreConfig

```typescript
interface DXMTVectorStoreConfig {
  // Required
  collectionName: string;

  // Connection (optional)
  host?: string; // default: 'localhost'
  port?: number; // default: 9999
  ssl?: boolean; // default: false

  // Vector configuration (optional)
  dimension?: number; // auto-detected from embeddings
  distanceMetric?: 'cosine' | 'euclidean' | 'dot' | 'manhattan';
  indexType?: 'hnsw' | 'hnsw_pq' | 'ivf_pq' | 'diskann';

  // Client options (optional)
  timeout?: number; // default: 30000
  apiKey?: string;
  headers?: Record<string, string>;
}
```

## Usage Examples

### 1. Basic Usage

```typescript
import { DXMTVectorStore } from 'dxmt-js-sdk/langchain';
import { OpenAIEmbeddings } from '@langchain/openai';

const vectorStore = new DXMTVectorStore(new OpenAIEmbeddings(), {
  collectionName: 'my_docs',
  host: 'localhost',
  port: 9999,
});
```

### 2. Add Documents

```typescript
import { Document } from '@langchain/core/documents';

// Method 1: Add documents
await vectorStore.addDocuments([
  new Document({
    pageContent: 'Document content',
    metadata: { source: 'file.txt' },
  }),
]);

// Method 2: Add with custom IDs
await vectorStore.addDocuments(
  [
    new Document({
      pageContent: 'Document content',
      metadata: { source: 'file.txt' },
    }),
  ],
  { ids: ['custom-id-1'] }
);
```

### 3. Similarity Search

```typescript
// Basic search
const results = await vectorStore.similaritySearch('query text', 5);

// Search with scores
const scoredResults = await vectorStore.similaritySearchWithScore(
  'query text',
  5
);

scoredResults.forEach(([doc, score]) => {
  console.log(`Score: ${score}`);
  console.log(`Content: ${doc.pageContent}`);
});

// Search with metadata filter
const filtered = await vectorStore.similaritySearch(
  'query text',
  5,
  { category: 'tech', year: 2024 }
);
```

### 4. Create from Texts

```typescript
const vectorStore = await DXMTVectorStore.fromTexts(
  ['Text 1', 'Text 2', 'Text 3'],
  [
    { source: 'doc1' },
    { source: 'doc2' },
    { source: 'doc3' },
  ],
  new OpenAIEmbeddings(),
  {
    collectionName: 'texts_collection',
    host: 'localhost',
    port: 9999,
  }
);
```

### 5. Create from Documents

```typescript
const docs = [
  new Document({ pageContent: 'Doc 1', metadata: { id: 1 } }),
  new Document({ pageContent: 'Doc 2', metadata: { id: 2 } }),
];

const vectorStore = await DXMTVectorStore.fromDocuments(
  docs,
  new OpenAIEmbeddings(),
  {
    collectionName: 'docs_collection',
    host: 'localhost',
    port: 9999,
  }
);
```

### 6. Delete Documents

```typescript
// Delete by IDs
await vectorStore.delete({ ids: ['id1', 'id2', 'id3'] });
```

## Metadata Filtering

DXMT supports rich metadata filtering:

```typescript
// Simple equality filter
const results = await vectorStore.similaritySearch('query', 5, {
  category: 'tech',
});

// Multiple filters (all conditions must match)
const results = await vectorStore.similaritySearch('query', 5, {
  category: 'tech',
  year: 2024,
  language: 'en',
});
```

**Note**: Currently supports equality filters. Advanced filtering (range, in, etc.) coming soon.

## Integration with LangChain Chains

### Retrieval QA Chain

```typescript
import { RetrievalQAChain } from 'langchain/chains';
import { ChatOpenAI } from '@langchain/openai';

const model = new ChatOpenAI();
const retriever = vectorStore.asRetriever({ k: 5 });

const chain = RetrievalQAChain.fromLLM(model, retriever);

const response = await chain.call({
  query: 'What is DXMT?',
});

console.log(response.text);
```

### Conversational Retrieval Chain

```typescript
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { ChatOpenAI } from '@langchain/openai';
import { BufferMemory } from 'langchain/memory';

const model = new ChatOpenAI();
const retriever = vectorStore.asRetriever();

const memory = new BufferMemory({
  memoryKey: 'chat_history',
  returnMessages: true,
});

const chain = ConversationalRetrievalQAChain.fromLLM(
  model,
  retriever,
  {
    memory,
  }
);

// First question
const response1 = await chain.call({
  question: 'What is DXMT?',
});

// Follow-up question (uses chat history)
const response2 = await chain.call({
  question: 'What are its main features?',
});
```

## Performance Tips

1. **Batch Document Insertion**

   ```typescript
   // Good: Insert many documents at once
   await vectorStore.addDocuments(largeArrayOfDocs);

   // Bad: Insert one at a time
   for (const doc of docs) {
     await vectorStore.addDocuments([doc]); // Don't do this!
   }
   ```

2. **Reuse Vector Store Instance**

   ```typescript
   // Good: Create once, use multiple times
   const vectorStore = new DXMTVectorStore(embeddings, config);

   await vectorStore.similaritySearch('query1', 5);
   await vectorStore.similaritySearch('query2', 5);
   ```

3. **Use Metadata Filters**

   ```typescript
   // Narrows search space, improves performance
   const results = await vectorStore.similaritySearch('query', 5, {
     category: 'tech',
   });
   ```

4. **Adjust K Parameter**

   ```typescript
   // Only retrieve what you need
   const results = await vectorStore.similaritySearch('query', 3);
   ```

## Comparison with Other Vector Stores

| Feature | DXMT | Pinecone | Qdrant | Chroma |
| ------- | ---- | -------- | ------ | ------ |
| **LangChain Integration** | ✅ | ✅ | ✅ | ✅ |
| **Self-Hosted** | ✅ | ❌ | ✅ | ✅ |
| **TypeScript Support** | ✅ | ✅ | ✅ | ✅ |
| **Metadata Filtering** | ✅ | ✅ | ✅ | ⚠️ Limited |
| **Cost (98% savings)** | ✅ | ❌ | ❌ | ❌ |
| **Tiered Storage** | ✅ | ❌ | ❌ | ❌ |

## API Reference

### DXMTVectorStore

**Constructor**

```typescript
constructor(
  embeddings: EmbeddingsInterface,
  config: DXMTVectorStoreConfig
)
```

**Methods**

- `addDocuments(documents, options?)` - Add documents
- `addVectors(vectors, documents, options?)` - Add pre-computed vectors
- `similaritySearch(query, k, filter?)` - Search similar documents
- `similaritySearchWithScore(query, k, filter?)` - Search with scores
- `similaritySearchVectorWithScore(vector, k, filter?)` - Search by vector
- `delete(params)` - Delete documents by IDs

**Static Methods**

- `fromTexts(texts, metadatas, embeddings, config)` - Create from texts
- `fromDocuments(docs, embeddings, config)` - Create from documents

**Utility Methods**

- `getClient()` - Get underlying DXMT client
- `getCollectionName()` - Get collection name

## Troubleshooting

### Issue: "collectionName is required"

**Solution**: Always provide `collectionName` in config:

```typescript
const vectorStore = new DXMTVectorStore(embeddings, {
  collectionName: 'my_collection', // Required!
  host: 'localhost',
  port: 9999,
});
```

### Issue: "Failed to connect to DXMT server"

**Solution**: Ensure DXMT server is running:

```bash
cargo run -- serve --host 0.0.0.0 --port 9999
```

### Issue: "Dimension mismatch"

**Solution**: Ensure embedding dimension matches table dimension. Let DXMT auto-detect:

```typescript
const vectorStore = new DXMTVectorStore(embeddings, {
  collectionName: 'docs',
  // Don't specify dimension - let it auto-detect
});
```

## Examples

See the [examples](./examples) directory:

- [langchain-usage.ts](./examples/langchain-usage.ts) - Complete LangChain integration examples

## Related Documentation

- [DXMT JavaScript SDK](./README.md)
- [LangChain Documentation](https://js.langchain.com/)
- [DXMT VectorDB](https://github.com/your-org/dxmt)

## License

Apache License 2.0 - See [LICENSE](./LICENSE) for details.

---

Made with ❤️ by the DXMT Team
