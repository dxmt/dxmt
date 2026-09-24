/**
 * DXMT VectorDB JavaScript SDK
 *
 * Official JavaScript/TypeScript SDK for DXMT VectorDB - A lightweight,
 * horizontally scalable vector database with storage native architecture.
 *
 * @packageDocumentation
 *
 * @example
 * Basic usage:
 * ```typescript
 * import { DXMTClient } from 'dxmt-js-sdk';
 *
 * // Create client
 * const client = new DXMTClient({
 *   host: 'localhost',
 *   port: 9999
 * });
 *
 * // Create table
 * await client.createTable({
 *   name: 'documents',
 *   dimension: 1536,
 *   distance_metric: 'cosine'
 * });
 *
 * // Insert records
 * await client.insertRecords({
 *   table_name: 'documents',
 *   records: [{
 *     id: 'doc1',
 *     vector: [0.1, 0.2, ...],
 *     metadata: { text: 'Sample' }
 *   }]
 * });
 *
 * // Search
 * const results = await client.search({
 *   table_name: 'documents',
 *   vector: [0.1, 0.2, ...],
 *   k: 5
 * });
 * ```
 */

// Export main client
export { DXMTClient } from './client';

// Export all types
export type {
  // Configuration
  ClientConfig,
  TableConfig,
  TableInfo,
  TableStats,

  // Records
  Record,
  Vector,
  Metadata,

  // Search
  SearchParams,
  SearchResult,
  FullTextSearchParams,
  HybridSearchParams,

  // Operations
  InsertRecordsParams,
  UpdateRecordParams,
  DeleteParams,
  BatchOperationResult,
  ScrollOptions,
  UploadOptions,
  UploadResult,
  UploadFailure,

  // Filters
  Filter,
  FilterCondition,
  FilterOperation,
  FilterOperationType,

  // Enums
  DistanceMetric,
  IndexType,
  HnswParams,
  QuantizationParams,
  PayloadIndexPolicy,
  PayloadIndexField,
  PayloadIndexReport,

  // Utility
  HealthCheckResponse,
  ErrorResponse,
} from './types';

// Export error classes
export {
  DXMTError,
  DXMTConnectionError,
  DXMTTableError,
  DXMTRecordError,
  DXMTSearchError,
  DXMTValidationError,
  DXMTTimeoutError,
  DXMTAuthenticationError,
} from './errors';

// Default export
export { DXMTClient as default } from './client';
