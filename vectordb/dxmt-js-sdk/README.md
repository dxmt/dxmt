# DXMT JavaScript SDK

Official JavaScript/TypeScript SDK for [DXMT VectorDB](https://github.com/dxmt/dxmt) - A lightweight, horizontally scalable vector database with storage native architecture.

[![npm version](https://badge.fury.io/js/dxmt-js-sdk.svg)](https://www.npmjs.com/package/dxmt-js-sdk)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)

## Breaking changes in 2.0.0

The 1.x client did not match the server's wire format in several places, so these are fixes
that change the published types:

- `listTables()` returns `TableInfo[]`, not `string[]`. The server has always returned table
  objects.
- Successful responses are unwrapped from the `{ success, data }` envelope before they are
  returned. 1.x handed the envelope back for some calls and the payload for others.
- `searchVectors()` reads the server's `hits` (it read `results`, which is never sent) and
  sends `top_k` (it sent `k`, which the server ignores in favour of its default of 10).
- `deleteRecords()` sends `ids`, the field the server reads. Deleting by `filter` now throws
  instead of silently deleting nothing — the server has no such route yet.
- `insertRecords()` maps `metadata` to the server's `payload`; 1.x dropped it.
- New: `updateRecord()` and `healthMetrics()`.

## Features

- ✅ **Full TypeScript support** with complete type definitions
- ✅ **Modern async/await API** for all operations
- ✅ **Comprehensive error handling** with custom error classes
- ✅ **Works in Node.js and browsers** (with bundler)
- ✅ **Zero dependencies** (except cross-fetch for compatibility)
- ✅ **Full DXMT API coverage** - tables, records, search, filters
- ✅ **Production-ready** with timeout handling and retry logic

## Installation

```bash
npm install dxmt-js-sdk
```

Or with yarn:

```bash
yarn add dxmt-js-sdk
```

Or with pnpm:

```bash
pnpm add dxmt-js-sdk
```

## Quick Start

```typescript
import { DXMTClient } from 'dxmt-js-sdk';

// Create client
const client = new DXMTClient({
  host: 'localhost',
  port: 9999
});

// Create table
await client.createTable({
  name: 'documents',
  dimension: 1536,
  distance_metric: 'cosine'
});

// Insert records
await client.insertRecords({
  table_name: 'documents',
  records: [
    {
      id: 'doc1',
      vector: [0.1, 0.2, ...],  // 1536-dim vector
      metadata: { text: 'Sample document' }
    }
  ]
});

// Search
const results = await client.search({
  table_name: 'documents',
  vector: [0.1, 0.2, ...],  // Query vector
  k: 5
});

console.log('Results:', results);
```

## Documentation

### Client Configuration

```typescript
const client = new DXMTClient({
  host: 'localhost',      // DXMT server host
  port: 9999,             // DXMT server port
  ssl: false,             // Use HTTPS
  timeout: 30000,         // Request timeout (ms)
  apiKey: 'your-key',     // For a server started with DXMT_API_KEY; sent as a Bearer token
  headers: {              // Optional custom headers
    'X-Custom-Header': 'value'
  }
});
```

### Table Operations

#### Create Table

```typescript
const table = await client.createTable({
  name: 'my_vectors',
  dimension: 1536,
  distance_metric: 'cosine',  // 'cosine', 'euclidean', 'dot', 'manhattan'
  index_type: 'hnsw',         // Optional: 'hnsw', 'ivf_pq', 'diskann', etc.
  description: 'My vector collection'
});
```

#### List Tables

```typescript
const tables = await client.listTables();
console.log('Tables:', tables);
```

#### Get Table Info

```typescript
const info = await client.getTable('my_vectors');
console.log('Dimension:', info.dimension);
console.log('Record count:', info.record_count);
```

#### Delete Table

```typescript
await client.deleteTable('old_table');
```

### Record Operations

#### Insert Records

```typescript
await client.insertRecords({
  table_name: 'my_vectors',
  records: [
    {
      id: 'doc1',
      vector: [0.1, 0.2, 0.3, ...],
      metadata: { text: 'Document 1', category: 'tech' }
    },
    {
      id: 'doc2',
      vector: [0.4, 0.5, 0.6, ...],
      metadata: { text: 'Document 2', category: 'science' }
    }
  ],
  mode: 'replace'  // 'append', 'replace', 'error_on_duplicate'
});
```

#### Get Record

```typescript
const record = await client.getRecord('my_vectors', 'doc1');
console.log('Vector:', record.vector);
console.log('Metadata:', record.metadata);
```

#### Delete Records

```typescript
// Delete by IDs
await client.deleteRecords({
  table_name: 'my_vectors',
  record_ids: ['doc1', 'doc2']
});

// Delete by filter
await client.deleteRecords({
  table_name: 'my_vectors',
  filter: {
    must: [
      { key: 'category', operation: { type: 'Equal', value: 'old' } }
    ]
  }
});
```

### Search Operations

#### Vector Similarity Search

```typescript
const results = await client.search({
  table_name: 'my_vectors',
  vector: [0.1, 0.2, 0.3, ...],  // Query vector
  k: 10,                          // Number of results
  include_metadata: true,
  include_vectors: false
});

results.forEach(result => {
  console.log('ID:', result.id);
  console.log('Score:', result.score);
  console.log('Metadata:', result.metadata);
});
```

#### Search with Metadata Filter

```typescript
const results = await client.search({
  table_name: 'my_vectors',
  vector: queryVector,
  k: 10,
  filter: {
    must: [
      { key: 'category', operation: { type: 'Equal', value: 'tech' } },
      { key: 'year', operation: { type: 'GreaterThan', value: 2020 } }
    ],
    should: [
      { key: 'language', operation: { type: 'In', value: ['en', 'zh'] } }
    ]
  }
});
```

#### Full-Text Search

```typescript
const results = await client.fullTextSearch({
  table_name: 'my_vectors',
  query: 'machine learning',
  k: 10
});
```

#### Hybrid Search (Vector + Text)

```typescript
const results = await client.hybridSearch({
  table_name: 'my_vectors',
  vector: queryVector,
  text_query: 'machine learning',
  k: 10,
  vector_weight: 0.7,
  text_weight: 0.3
});
```

### Filter Operations

DXMT supports rich metadata filtering:

```typescript
// Supported filter operations
const filter = {
  must: [      // AND conditions
    { key: 'category', operation: { type: 'Equal', value: 'tech' } },
    { key: 'price', operation: { type: 'GreaterThan', value: 100 } }
  ],
  should: [    // OR conditions
    { key: 'tag', operation: { type: 'Contains', value: 'ai' } },
    { key: 'tag', operation: { type: 'Contains', value: 'ml' } }
  ],
  must_not: [  // NOT conditions
    { key: 'status', operation: { type: 'Equal', value: 'deleted' } }
  ]
};
```

**Supported Operations:**
- `Equal`, `NotEqual`
- `GreaterThan`, `GreaterThanOrEqual`, `LessThan`, `LessThanOrEqual`
- `In`, `NotIn`
- `Contains`, `StartsWith`, `EndsWith`

### Error Handling

The SDK provides custom error classes for different failure scenarios:

```typescript
import {
  DXMTError,
  DXMTConnectionError,
  DXMTTableError,
  DXMTRecordError,
  DXMTSearchError,
  DXMTValidationError,
  DXMTTimeoutError
} from 'dxmt-js-sdk';

try {
  await client.search({ ... });
} catch (error) {
  if (error instanceof DXMTConnectionError) {
    console.error('Failed to connect to DXMT server');
  } else if (error instanceof DXMTValidationError) {
    console.error('Invalid parameters:', error.message);
  } else if (error instanceof DXMTTimeoutError) {
    console.error('Request timed out');
  } else {
    console.error('Unknown error:', error);
  }
}
```

### Health Check

```typescript
const health = await client.healthCheck();
console.log('Status:', health.status);
console.log('Version:', health.version);
console.log('Uptime:', health.uptime_seconds);
```

## Examples

See the [examples](./examples) directory for complete examples:

- [basic-usage.ts](./examples/basic-usage.ts) - Complete workflow from table creation to search
- More examples coming soon!

## TypeScript Support

The SDK is written in TypeScript and provides full type definitions:

```typescript
import {
  DXMTClient,
  TableConfig,
  SearchParams,
  SearchResult,
  Filter
} from 'dxmt-js-sdk';

// All types are exported and available
const config: TableConfig = {
  name: 'vectors',
  dimension: 1536,
  distance_metric: 'cosine'
};

const searchParams: SearchParams = {
  table_name: 'vectors',
  vector: [0.1, 0.2, ...],
  k: 10
};
```

## Browser Usage

The SDK works in browsers when bundled with tools like Webpack, Rollup, or Vite:

```typescript
import { DXMTClient } from 'dxmt-js-sdk';

const client = new DXMTClient({
  host: 'api.example.com',
  port: 443,
  ssl: true
});

// Use normally
const results = await client.search({ ... });
```

**Note:** Make sure your DXMT server has CORS enabled for browser requests.

## Performance Tips

1. **Batch Operations**: Insert multiple records at once for better performance
   ```typescript
   await client.insertRecords({
     table_name: 'vectors',
     records: largeArrayOfRecords  // Insert in batches of 1000-10000
   });
   ```

2. **Reuse Client**: Create one client instance and reuse it
   ```typescript
   // Good: Single instance
   const client = new DXMTClient({ ... });

   // Bad: Creating new client for each request
   // Don't do this!
   ```

3. **Optimize Search**: Use filters to reduce search space
   ```typescript
   await client.search({
     table_name: 'vectors',
     vector: queryVector,
     k: 10,
     filter: { must: [...] }  // Pre-filter before vector search
   });
   ```

4. **Timeouts**: Adjust timeout for large operations
   ```typescript
   const client = new DXMTClient({
     timeout: 60000  // 60 seconds for large batch operations
   });
   ```

## API Reference

There is no generated reference yet; the public surface is the source:
- [Client](./src/client.ts)
- [Types](./src/types.ts)
- [Errors](./src/errors.ts)

## DXMT Server

This SDK requires a running DXMT VectorDB server. To start the server:

```bash
# Clone DXMT repository
git clone https://github.com/dxmt/dxmt
cd dxmt/vectordb

# Build and run
cargo run -- serve --host 0.0.0.0 --port 9999
```

Or use Docker:

```bash
docker run -p 9999:9999 dxmt/vectordb:latest
```

## Contributing

Contributions are welcome — please open an issue or a pull request.

## License

Apache License 2.0 - See [LICENSE](./LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/dxmt/dxmt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dxmt/dxmt/discussions)
- **Documentation**: [DXMT Docs](https://docs.dxmt.io)

## Related Projects

- [DXMT Python SDK](../dxmt-python-sdk) - Official Python SDK
- [DXMT VectorDB](https://github.com/dxmt/dxmt) - Core database
- [LangChain DXMT](../langchain-dxmt) - LangChain integration

---

Made with ❤️ by the DXMT Team
