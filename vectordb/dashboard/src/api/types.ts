/**
 * The wire, with the server's own names.
 *
 * Nothing here is renamed for comfort. `tookms` has no underscore, `payload_json` really is a
 * string, and `GET /tables/{t}` really does nest the table one level down beside a sibling
 * count. This is a console for looking at what the server returned, so the moment these types
 * start describing something tidier than the server sends, it is describing the wrong thing.
 * `normalize.ts` is where the two record shapes are reconciled, on purpose and in one place.
 */

/** Every route except `/health/live` and `/health/ready` wraps its answer in this. */
export interface Envelope<T> {
  success: boolean;
  data: T | null;
  error: ApiError | null;
  timestamp: string;
}

export interface ApiError {
  code: string;
  message: string;
  details: unknown;
}

export interface Table {
  id: string;
  name: string;
  dimension: number;
  distance_metric: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  record_count: number;
  index_type: string;
  text_fields: unknown;
  enable_full_text_search: boolean;
  schema_version: number;
  payload_schema: unknown;
  memory_optimization: boolean | null;
  named_vectors: unknown;
}

export interface TableList {
  tables: Table[];
  total: number;
  offset: number;
  limit: number;
}

/** `GET /tables/{t}`. The nesting is the server's, not ours. */
export interface TableWithCount {
  table: Table;
  record_count: number;
}

export interface TableStats {
  name: string;
  resident: boolean;
  record_count: number;
  dimension: number;
  index_type: string;
  index_size_bytes: number;
  memory_usage_bytes: number;
  disk_usage_bytes: number;
}

/**
 * A record as `records/list` and `GET records/{r}` serialise it.
 *
 * The payload arrives as a JSON **string** under `payload_json`, where search hits send an
 * object under `payload`. `normalize.ts` reconciles the two; nothing else should.
 *
 * `vector` is empty unless the request asked for it with `with_vector`. It used to be empty or
 * full depending on whether the server kept its vectors in a row file -- a setting the client
 * cannot see -- which is why the flag exists.
 */
export interface WireRecord {
  id: string;
  vector: number[];
  sparse_vector: unknown;
  named_vectors: unknown;
  payload_json: string | null;
  created_at: string;
  updated_at: string;
  version: number;
  status: string;
}

export interface RecordPage {
  records: WireRecord[];
  /** The whole filtered table, not the page. */
  total: number;
  limit: number;
  offset: number;
}

/** A search hit. Note `payload` is an object here, unlike `WireRecord.payload_json`. */
export interface WireHit {
  id: string;
  /** A similarity: higher is better, 1.0 for an exact match. */
  score: number;
  payload: Record<string, unknown> | null;
  vector: number[] | null;
}

export interface SearchResult {
  hits: WireHit[];
  /** `hits.length`, not a corpus count. */
  total: number;
  /** No underscore. */
  tookms: number;
}

/**
 * A BM25 hit. Same three leading fields as `WireHit`, and then it diverges.
 *
 * There is no `vector` — the text index does not hold one — and there is a `highlights` array
 * the dense path has no equivalent for.
 */
export interface WireTextHit {
  id: string;
  /** A BM25 score. Unbounded above and **not comparable to a similarity**: 6.3 here says
   *  nothing about how it would rank under the dense metric, and averaging the two is
   *  meaningless, which is why fusion is done by rank on the server rather than by score here. */
  score: number;
  payload: Record<string, unknown> | null;
  highlights: string[] | null;
}

/**
 * `POST /search/text`, and note the timing field.
 *
 * It is `took_ms` here and `tookms` on `/search` and `/search/hybrid` — two Rust structs,
 * `FullTextSearchResult` and `SearchResult`, that were written at different times and never
 * reconciled. Renaming either one would break whichever SDKs read it, so the asymmetry is real,
 * it is on the wire, and `timingOf()` in the search page is where it is absorbed. Writing
 * `tookms` here would produce a silent `undefined` and a page reporting `NaN ms`.
 */
export interface FullTextResult {
  hits: WireTextHit[];
  total: number;
  /** With the underscore. See above. */
  took_ms: number;
}

export interface FullTextQuery {
  query: string;
  limit: number;
  offset?: number;
  highlight?: boolean;
  search_fields?: string[];
}

/**
 * How the server fuses the rankings. Omitted, it is `weighted_sum` at `vector_weight` 0.6, or
 * `dbsf` when the request carries a sparse vector (since 2026-10-13; `dbsf` always, before).
 *
 * `vector_weight` applies to two of the three, which is not what their names suggest:
 * `weighted_sum` normalises each side and sums them weighted, and `reciprocal_rank_fusion` is
 * a *weighted* RRF — `fuse_reciprocal_rank` takes `text_weight = 1 - vector_weight` and scales
 * each side's `1/(k + rank)` by it. Only `dbsf` ignores the weight, normalising each list by
 * its own mean and standard deviation instead. Measured rather than assumed: the same query
 * under RRF at 0.7 and at 0.1 returns two entirely different lists.
 */
export type FusionMethod = 'weighted_sum' | 'reciprocal_rank_fusion' | 'dbsf';

/**
 * `POST /search/hybrid`, which answers with a plain `SearchResult`.
 *
 * The field names matter more than usual here, because the server takes the body through serde
 * with no `deny_unknown_fields`: a misspelled knob is **silently ignored** rather than refused.
 * `vector_weight` is the dense side's share and is read by `weighted_sum` **and** by
 * `reciprocal_rank_fusion`; `limit` is the result count. There is no `alpha` and no `top_k`,
 * whatever else may suggest otherwise.
 */
export interface HybridQuery {
  vector?: number[] | null;
  text?: string | null;
  limit: number;
  vector_weight?: number;
  fusion_method?: FusionMethod;
  rrf_k?: number;
  filter?: Filter | null;
  with_payload?: boolean;
  with_vector?: boolean;
}

/**
 * Four of these scale HNSW's traversal budget; `exact` skips the index and scores every row.
 *
 * That last one is the only answer with no recall in it, which is what makes measuring recall
 * on your own data possible at all. It is also the only one a server may not have: it was added
 * after the other four, and an older build rejects it with a plain-text 422 rather than the
 * error envelope -- see `probeExactMode` in `capabilities.ts`.
 */
export type SearchMode = 'fast' | 'normal' | 'accurate' | 'auto' | 'exact';

export const APPROXIMATE_MODES: SearchMode[] = ['fast', 'normal', 'accurate', 'auto'];

export interface SearchQuery {
  vector: number[];
  top_k?: number;
  limit?: number;
  filter?: Filter | null;
  with_payload?: boolean;
  with_vector?: boolean;
  search_mode?: SearchMode;
  vector_name?: string | null;
}

/** Operation names are PascalCase on the wire. The SDK's Mongo-style sugar is not accepted. */
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

export interface Condition {
  key: string;
  operation: { type: FilterOperationType; value: unknown };
}

export interface Filter {
  must?: Condition[];
  must_not?: Condition[];
  should?: Condition[];
}

export interface ListRecordsRequest {
  filter?: Filter | null;
  /** Clamped to 1000 by the server. */
  limit?: number;
  offset?: number;
  /**
   * Include each record's full vector. Off by default, and worth leaving off.
   *
   * A page of 100 at 768 dimensions is over a megabyte of JSON floats and there is no
   * compression on this server, so the record browser never asks. The visualize page does,
   * because it is the only bulk source of vectors there is.
   */
  with_vector?: boolean;
}

/** Shared by `/status` and `/health/metrics`; the most useful object the server exposes. */
export interface StorageHealthSnapshot {
  ready: boolean;
  wal_enabled: boolean;
  wal_directory_accessible: boolean;
  wal_next_sequence: number;
  snapshot_present: boolean;
  snapshot_manifest_valid: boolean;
  snapshot_wal_next_sequence: number;
  durability_degraded: boolean;
  durability_error: string | null;
  resident_table_count: number;
  table_count: number;
  total_records: number;
  last_wal_sync_at: string | null;
  last_snapshot_at: string | null;
  snapshot_failures: number;
  last_snapshot_error: string | null;
}

export interface ServerStatus {
  status: string;
  version: string;
  storage: { table_count: number; total_records: number };
  persistence: StorageHealthSnapshot;
}

export interface HealthMetrics {
  ongoing_operations: {
    inserts: number;
    searches: number;
    deletes: number;
    table_operations: number;
    total: number;
  };
  completed_operations: number;
  errors: { recent_count: number };
  index_status: { building: boolean };
  storage: StorageHealthSnapshot;
}

/** A k-nearest-neighbour graph over a sample, from `POST /tables/{t}/graph`. */
export interface KnnGraph {
  /** The sampled records, in node order. */
  ids: string[];
  /** Per node, its nearest neighbours as indices into `ids`, nearest first. */
  neighbours: number[][];
  /** Per node, the similarity to each of those. Higher is nearer, as in search. */
  scores: number[][];
  sampled: number;
  /** How many records the sample was drawn from, after any filter. */
  total: number;
  /** Per node, its payload as the same JSON string `records/list` sends. */
  payloads?: (string | null)[];
}

export interface KnnGraphRequest {
  sample?: number;
  neighbours?: number;
  filter?: Filter | null;
  seed?: number;
  /** Default true: a node with no label cannot be drawn usefully. */
  with_payload?: boolean;
}
