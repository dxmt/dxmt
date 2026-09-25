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
 * Vector representation - array of numbers.
 */
export type Vector = number[];

/**
 * Metadata - flexible key-value pairs.
 */
export type Metadata = globalThis.Record<string, any>;

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
  | 'EndsWith'
  /** The field is absent, null, or an empty array. Sent with no `value`. */
  | 'IsEmpty'
  /** The field is present and null, or holds a null element. Sent with no `value`. */
  | 'IsNull';

/**
 * Filter operation for metadata filtering.
 */
export interface FilterOperation {
  type: FilterOperationType;
  /** Omitted for `IsEmpty` and `IsNull`, which are about the field rather than a value in it.
   * The server refuses a member it does not know, so sending `value: null` with those is a
   * 400 rather than a no-op. */
  value?: unknown;
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
  /** Sent through to the server as written. Searchable with {@link DXMTClient.sparseSearch}. */
  sparse_vector?: SparseVector;
  metadata?: Metadata;
  /** Raw server payload (DXMT's wire name for metadata). */
  payload?: Metadata;
  /** Present on GET responses serialized by the Rust storage layer. */
  payload_json?: string;
  score?: number;
}

/** A sparse vector: the non-zero dimensions and their weights. */
export interface SparseVector {
  /** The non-zero dimensions, each below `dimension`. */
  indices: number[];
  /** One weight per index, in the same order. */
  values: number[];
  /** The width of the space the indices are drawn from. */
  dimension: number;
}

/** One group from {@link DXMTClient.searchGroups}. */
export interface SearchGroup {
  /** The value of `group_by` every hit here holds. */
  group_id: unknown;
  /** Normalized the way every other method here returns hits: `metadata`, not `payload`. */
  hits: SearchResult[];
  /** How many of the fetched candidates fell in this group, which can exceed `hits.length`. */
  matched: number;
}

/** What a grouped search returns. */
export interface SearchGroupsResult {
  groups: SearchGroup[];
  /** How wide the window of nearest neighbours got. A short `groups` with this at the
   * server's cap means the cap bounded the answer, not the data. */
  candidates_fetched: number;
  /** Candidates with no value under `group_by`; they belong to no group. */
  ungrouped: number;
  tookms: number;
}

/** Which payload fields a table indexes for filtering. Unset indexes every field. */
export interface PayloadIndexPolicy {
  /** When set, the only fields indexed. */
  only?: string[];
  /** Fields never indexed. A filter naming one is refused, not answered empty. */
  except?: string[];
}

/** One field's entry in {@link PayloadIndexReport}. */
export interface PayloadIndexField {
  field: string;
  distinct_values: number;
  keyword_values: number;
  keyword_rows: number;
  numeric_entries: number;
  unkeyed_rows: number;
  /** Computed from the postings, not measured. Excludes what every field shares. */
  estimated_bytes: number;
  numeric_min: number | null;
  numeric_max: number | null;
}

/** What `GET /tables/{t}/payload_index` reports. */
export interface PayloadIndexReport {
  table: string;
  /** False when nothing has filtered yet: the index is built on demand, and describing it is
   * not worth a pass over every record. `fields` is then empty. */
  built: boolean;
  indexes_every_field: boolean;
  only?: string[] | null;
  except?: string[] | null;
  record_count?: number;
  estimated_bytes?: number;
  shared_bytes?: number;
  fields: PayloadIndexField[];
}

/** Graph parameters for one table; unset fields keep the server's defaults. */
export interface HnswParams {
  /** Edges per node: denser graph, better recall, more memory, slower build. */
  m?: number;
  /** Candidates considered while building. Cannot be recovered by searching harder later. */
  ef_construction?: number;
  /** Starting traversal budget at query time; `search_mode` scales it per query. */
  ef_search?: number;
}

/** Product-quantization parameters for one table. */
export interface QuantizationParams {
  /** Subspaces per vector; one byte of code each at 256 centroids. */
  num_subvectors?: number;
  num_centroids?: number;
  /** Rows at which the index trains codebooks and switches to codes. Set it above the table's
   * eventual size to keep the table exact. */
  quantize_at_rows?: number;
}

/**
 * Table configuration for creating tables.
 */
export interface TableConfig {
  name: string;
  dimension: number;
  /** Which payload fields to index for filtering. Unset indexes every field. */
  payload_index?: PayloadIndexPolicy;
  distance_metric?: DistanceMetric;
  index_type?: IndexType;
  description?: string;
  enable_full_text_search?: boolean;
  /**
   * Graph parameters for this table.
   *
   * `m` and `ef_construction` used to sit at the top level here, next to a `quantization` that
   * took a quantization-type string and a `max_elements` — four fields the server has never
   * had. The whole object is sent as the request body, so since unknown fields started being
   * refused, naming any of them failed the create outright.
   */
  hnsw?: HnswParams;
  /** Quantization parameters for this table. */
  quantization?: QuantizationParams;
}

/**
 * Table information returned by the server.
 */
export interface TableInfo {
  name: string;
  dimension: number;
  distance_metric: DistanceMetric;
  index_type: IndexType;
  record_count: number;
  created_at: string;
  updated_at: string;
  description?: string;
  /** The graph parameters this table was created with, or null for the server's defaults. */
  hnsw?: HnswParams | null;
  /**
   * The quantization parameters this table was created with, or null for the defaults.
   *
   * This was typed as a `QuantizationType` string — 'none' | 'product' | 'scalar' | 'binary' —
   * which the server has never returned for a table.
   */
  quantization?: QuantizationParams | null;
}

/**
 * Parameters for inserting records.
 */
export interface InsertRecordsParams {
  table_name: string;
  records: Record[];
}

/**
 * Parameters for vector search.
 */
export interface SearchParams {
  table_name: string;
  vector: Vector;
  k?: number;
  filter?: Filter;
  include_metadata?: boolean;
  include_vectors?: boolean;
  /**
   * How hard to search, as a mode rather than a raw `ef`.
   *
   * `ef_search` used to be here and was sent on every search. The server has no such field:
   * it derives the traversal budget from this mode and from the table's size, so the number
   * was ignored — and since the server started refusing unknown fields, sending it fails the
   * request outright. `exact` scores every row, which is how recall is measured on real data.
   */
  search_mode?: 'fast' | 'normal' | 'accurate' | 'auto' | 'exact';
  /** Skip this many hits before the page starts. Not free; see {@link DXMTClient.search}. */
  offset?: number;
  /** Drop hits scoring below this. Scores are similarities, so it is a floor: higher is nearer. */
  score_threshold?: number;
  /** Return only these payload keys instead of the whole payload. */
  payload_fields?: string[];
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
  /**
   * The vector side's share of the score, 0 to 1; the text side gets the rest.
   *
   * Setting it selects `weighted_sum` fusion, which is the method that reads it. Left unset,
   * the server fuses by distribution (`dbsf`), which needs no weight — and which ignores one,
   * so a weight sent without a method changed nothing at all.
   */
  vector_weight?: number;
  /** Which fusion to use. Defaults to the server's `dbsf`, or to `weighted_sum` with a weight. */
  fusion_method?: 'dbsf' | 'weighted_sum' | 'reciprocal_rank_fusion';
}

/**
 * Delete parameters.
 */
export interface DeleteParams {
  table_name: string;
  record_ids?: string[];
  filter?: Filter;
}

/** Parameters for updating a single record. */
export interface UpdateRecordParams {
  table_name: string;
  record_id: string;
  vector?: Vector;
  metadata?: Metadata;
  /** Replace the payload (default) or merge keys into the existing payload. */
  merge_mode?: 'replace' | 'merge';
  /** Create the record when it does not already exist. */
  upsert?: boolean;
}

/**
 * Batch operation result.
 */
export interface BatchOperationResult {
  success: boolean;
  inserted?: number;
  updated?: number;
  deleted?: number;
  inserted_ids?: string[];
  deleted_ids?: string[];
  count?: number;
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
  headers?: globalThis.Record<string, string>;
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
  status: string;
  service?: string;
  ready?: boolean;
  storage_healthy?: boolean;
  timestamp?: string;
  description?: string;
  retry_after_seconds?: number;
  metrics?: globalThis.Record<string, any>;
  storage?: globalThis.Record<string, any>;
}

/** Options for {@link DXMTClient.scroll}. */
export interface ScrollOptions {
  /** Only records matching this filter. */
  filter?: Filter;
  /** Records per request, at most 1,000. Default 100. */
  batchSize?: number;
  /** Include each record's vector. Default false. */
  withVector?: boolean;
}

/** Options for {@link DXMTClient.upload}. */
export interface UploadOptions {
  /** Records per request. Default 256. */
  batchSize?: number;
  /** Requests in flight at once. Default 1. */
  parallel?: number;
  /** Further attempts per batch after the first. Default 3. */
  maxRetries?: number;
  /** Milliseconds before the first retry, doubling after each. Default 500. */
  backoffMs?: number;
}

/** One batch that did not go in, after every retry it was worth. */
export interface UploadFailure {
  ids: string[];
  error: string;
  /** HTTP status, or undefined when the request never got an answer. */
  status?: number;
}

/** What {@link DXMTClient.upload} did. `failed` is empty when everything went in. */
export interface UploadResult {
  uploaded: number;
  batches: number;
  retries: number;
  failed: UploadFailure[];
}
