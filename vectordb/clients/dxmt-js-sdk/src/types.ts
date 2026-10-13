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
  /** The vector. Omit it and send `text` on a table that declares an `embedding` model. */
  vector?: Vector;
  /** Text for the server to embed, filed into the payload under the config's `field`. */
  text?: string;
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
  /** Keep the 4-bit block-scan codes beside the 8-bit ones (default true): faster filtered
   * search over large match sets, for about the code's bytes again and CPU under load. */
  fast_scan?: boolean;
}

/**
 * How a table turns text into a vector, server-side.
 *
 * With it set, records may carry `text` instead of `vector` and {@link SearchParams.text}
 * embeds a query with this same model. `base_url` reaches anything speaking the OpenAI
 * `/embeddings` protocol — text-embeddings-inference, Infinity, ollama, vLLM, llama.cpp.
 *
 * There is deliberately no field for the key: `api_key_env` names an environment variable of
 * the *server*, so no key travels in a request body or lands in a snapshot.
 */
export interface EmbeddingConfig {
  /** Passed to the provider verbatim, e.g. `text-embedding-3-small` or `BAAI/bge-m3`. */
  model: string;
  /** API root, without the trailing `/embeddings`. Default `https://api.openai.com/v1`. */
  base_url?: string;
  /** Payload field holding the text. Default `text`; a record's `text` is written here. */
  field?: string;
  /** Environment variable on the server holding the key. Default `OPENAI_API_KEY`. */
  api_key_env?: string;
  /** Texts per request to the provider. Default 64. */
  batch_size?: number;
}

/**
 * A rerank provider the server calls when a search asks for `rerank`.
 *
 * The server has no model of its own: it sends the query text and each candidate's text to
 * `base_url` + `/rerank` and returns the page in the provider's order, each hit carrying
 * `rerank_score`. The API key lives in the server's environment, never in this object.
 */
export interface RerankConfig {
  /** Passed to the provider verbatim, e.g. `rerank-v3.5` or `BAAI/bge-reranker-v2-m3`. */
  model: string;
  /** API root, without the trailing `/rerank`. */
  base_url: string;
  /** Wire shape: `cohere` (Cohere v2, Jina, Voyage; the default) or `tei` (text-embeddings-inference). */
  format?: 'cohere' | 'tei';
  /** Environment variable on the server holding the key. Default `RERANK_API_KEY`. */
  api_key_env?: string;
  /** Payload field holding each record's text. Default: the embedding field, else `text`. */
  field?: string;
  /** Candidates fetched and scored when a search names no `top_n`. Default: five times the page. */
  top_n?: number;
}

/**
 * Ask the table's rerank provider to re-score a page. `{}` takes the defaults.
 */
export interface RerankRequest {
  /** Candidates to fetch and score. Default: the table's `rerank.top_n`, else five times the page. */
  top_n?: number;
  /** Payload field holding each candidate's text. Default: the table's `rerank.field`. */
  field?: string;
}

/**
 * Table configuration for creating tables.
 */
export interface TableConfig {
  name: string;
  /** Vector width, fixed for the life of the table. Optional only with {@link
   * TableConfig.embedding}, where the server takes it from the model. */
  dimension?: number;
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
  /** The model this table embeds text with. Without it, the caller sends vectors. */
  embedding?: EmbeddingConfig;
  /** A cross-encoder a search may rerank its page with; see {@link RerankConfig}. */
  rerank?: RerankConfig;
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
  /** The query vector. Omit it when passing {@link SearchParams.text}. */
  vector?: Vector;
  /**
   * Query text, embedded by the *table's* model — the only model it may be embedded with,
   * since a query vector from a different model than the corpus is not a worse answer but an
   * unrelated one. Needs the table to have been created with an `embedding` config.
   *
   * Exclusive with `vector`: the server refuses a body carrying both rather than preferring
   * one silently.
   */
  text?: string;
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
  /** Rerank the page for diversity: `{ lambda_mult, fetch_k }`. 1.0 is the plain ranking,
   * 0.0 ignores the query. Only fetched candidates can contribute diversity, so `fetch_k` is
   * the cost. Done server-side, over the vectors the index already holds. */
  mmr?: { lambda_mult?: number; fetch_k?: number };
  /** Re-score the page with the table's rerank provider. Needs `text` and a table created
   * with `rerank`; the request fails rather than returning the page unranked. */
  rerank?: RerankRequest;
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
  /** The rerank provider's relevance score, when the page was reranked; higher is better. */
  rerank_score?: number;
}

/**
 * Parameters for full-text search.
 */
export interface FullTextSearchParams {
  table_name: string;
  query: string;
  k?: number;
  filter?: Filter;
  /** Re-score the BM25 page with the table's rerank provider against `query`. */
  rerank?: RerankRequest;
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
   * the server uses its default: `weighted_sum` at 0.6 (`dbsf`, which ignores any weight, when
   * the request carries a sparse vector).
   */
  vector_weight?: number;
  /** Which fusion to use. Defaults to `weighted_sum` with a weight, else to the server's default. */
  fusion_method?: 'dbsf' | 'weighted_sum' | 'reciprocal_rank_fusion';
  /** Re-score the fused page with the table's rerank provider against `text_query`. */
  rerank?: RerankRequest;
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

/** One bucket of a facet: a value and how many records hold it. */
export interface Facet {
  value: string | number | boolean | null;
  count: number;
}

/** What `POST /tables/{t}/facets` answers. */
export interface FacetResult {
  /** The most common values, count descending then value ascending. */
  hits: Facet[];
  /** How many distinct values there were, of which `hits` is the top `limit`. */
  distinct_values: number;
  /**
   * How many of the `matched` records contributed a value here.
   *
   * `matched - covered` is what the facet cannot describe: records with no value for the
   * field, or with an object, which the index records as presence only.
   */
  covered: number;
  /** How many records the filter selected; the whole table when there is no filter. */
  matched: number;
}
