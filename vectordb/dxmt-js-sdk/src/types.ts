/**
 * Type definitions for DXMT VectorDB JavaScript SDK.
 *
 * This module provides TypeScript type definitions for all DXMT API operations.
 */

/**
 * Distance metric types supported by DXMT.
 */
export type DistanceMetric = 'cosine' | 'euclidean' | 'dot' | 'manhattan';

/**
 * Index types supported by DXMT.
 */
export type IndexType =
  | 'hnsw'
  | 'hnsw_pq'
  | 'ivf_pq'
  | 'diskann'
  | 'diskann_enhanced'
  | 'brute_force'
  | 'lsh';

/**
 * Quantization types for compression.
 */
export type QuantizationType = 'none' | 'product' | 'scalar' | 'binary';

/**
 * Insert mode for handling duplicates.
 */
export type InsertMode = 'append' | 'replace' | 'error_on_duplicate';

/**
 * Vector representation - array of numbers.
 */
export type Vector = number[];

/**
 * Metadata - flexible key-value pairs.
 */
export type Metadata = Record<string, any>;

/**
 * Filter operation types.
 */
export type FilterOperationType =
  | 'Equal'
  | 'NotEqual'
  | 'GreaterThan'
  | 'GreaterThanOrEqual'
  | 'LessThan'
  | 'LessThanOrEqual'
  | 'In'
  | 'NotIn'
  | 'Contains'
  | 'StartsWith'
  | 'EndsWith';

/**
 * Filter operation for metadata filtering.
 */
export interface FilterOperation {
  type: FilterOperationType;
  value: any;
}

/**
 * Filter condition for metadata.
 */
export interface FilterCondition {
  key: string;
  operation: FilterOperation;
}

/**
 * Filter for search queries.
 */
export interface Filter {
  must?: FilterCondition[];
  should?: FilterCondition[];
  must_not?: FilterCondition[];
}

/**
 * Record in DXMT - combines vector with metadata.
 */
export interface Record {
  id: string;
  vector: Vector;
  metadata?: Metadata;
  score?: number;
}

/**
 * Table configuration for creating tables.
 */
export interface TableConfig {
  name: string;
  dimension: number;
  distance_metric?: DistanceMetric;
  index_type?: IndexType;
  quantization?: QuantizationType;
  description?: string;
  enable_full_text_search?: boolean;
  max_elements?: number;
  ef_construction?: number;
  m?: number;
}

/**
 * Table information returned by the server.
 */
export interface TableInfo {
  name: string;
  dimension: number;
  distance_metric: DistanceMetric;
  index_type: IndexType;
  quantization: QuantizationType;
  record_count: number;
  created_at: string;
  updated_at: string;
  description?: string;
}

/**
 * Parameters for inserting records.
 */
export interface InsertRecordsParams {
  table_name: string;
  records: Record[];
  mode?: InsertMode;
}

/**
 * Parameters for vector search.
 */
export interface SearchParams {
  table_name: string;
  vector: Vector;
  k?: number;
  filter?: Filter;
  ef_search?: number;
  include_metadata?: boolean;
  include_vectors?: boolean;
}

/**
 * Search result from vector search.
 */
export interface SearchResult {
  id: string;
  score: number;
  vector?: Vector;
  metadata?: Metadata;
}

/**
 * Parameters for full-text search.
 */
export interface FullTextSearchParams {
  table_name: string;
  query: string;
  k?: number;
  filter?: Filter;
}

/**
 * Parameters for hybrid search (vector + text).
 */
export interface HybridSearchParams {
  table_name: string;
  vector?: Vector;
  text_query?: string;
  k?: number;
  filter?: Filter;
  vector_weight?: number;
  text_weight?: number;
}

/**
 * Delete parameters.
 */
export interface DeleteParams {
  table_name: string;
  record_ids?: string[];
  filter?: Filter;
}

/**
 * Batch operation result.
 */
export interface BatchOperationResult {
  success: boolean;
  inserted?: number;
  updated?: number;
  deleted?: number;
  errors?: string[];
}

/**
 * Client configuration.
 */
export interface ClientConfig {
  host?: string;
  port?: number;
  ssl?: boolean;
  timeout?: number;
  headers?: Record<string, string>;
  apiKey?: string;
}

/**
 * Error response from server.
 */
export interface ErrorResponse {
  error: string;
  message: string;
  code?: string;
  details?: any;
}

/**
 * Statistics for a table.
 */
export interface TableStats {
  name: string;
  record_count: number;
  dimension: number;
  index_size_bytes: number;
  memory_usage_bytes: number;
  disk_usage_bytes: number;
}

/**
 * Health check response.
 */
export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  version: string;
  uptime_seconds: number;
}
