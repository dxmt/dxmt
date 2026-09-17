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
