/**
 * DXMT VectorDB JavaScript Client.
 *
 * This module provides the main client class for interacting with DXMT VectorDB
 * via its REST API.
 */

import fetch from 'cross-fetch';
import {
  ClientConfig,
  TableConfig,
  TableInfo,
  Record as VectorRecord,
  SearchParams,
  SearchResult,
  InsertRecordsParams,
  DeleteParams,
  UpdateRecordParams,
  BatchOperationResult,
  TableStats,
  HealthCheckResponse,
  FullTextSearchParams,
  HybridSearchParams,
  Metadata,
  Filter,
  ScrollOptions,
  UploadOptions,
  UploadResult,
  PayloadIndexReport,
  SearchGroupsResult,
  SparseVector,
} from './types';
import {
  DXMTError,
  DXMTConnectionError,
  DXMTTableError,
  DXMTRecordError,
  DXMTSearchError,
  DXMTValidationError,
  DXMTTimeoutError,
  parseErrorResponse,
} from './errors';

/**
 * Default configuration values.
 */
const DEFAULT_CONFIG: Required<Omit<ClientConfig, 'apiKey'>> = {
  host: 'localhost',
  port: 9999,
  ssl: false,
  timeout: 30000,
  headers: {},
};

/**
 * DXMT VectorDB Client.
 *
 * Provides methods to interact with DXMT VectorDB server via REST API.
 *
 * @example
 * ```typescript
 * const client = new DXMTClient({
 *   host: 'localhost',
 *   port: 9999
 * });
 *
 * // Create a table
 * await client.createTable({
 *   name: 'my_vectors',
 *   dimension: 1536,
 *   distance_metric: 'cosine'
 * });
 *
 * // Insert records
 * await client.insertRecords({
 *   table_name: 'my_vectors',
 *   records: [{
 *     id: 'doc1',
 *     vector: [0.1, 0.2, ...],
 *     metadata: { text: 'Sample document' }
 *   }]
 * });
 *
 * // Search
 * const results = await client.search({
 *   table_name: 'my_vectors',
 *   vector: [0.1, 0.2, ...],
 *   k: 5
 * });
 * ```
 */
/** Statuses a retry can fix: busy (429, 503) or momentarily wrong (5xx). */
const RETRYABLE = new Set([429, 500, 502, 503, 504]);

export class DXMTClient {
  private baseUrl: string;
  private timeout: number;
  private headers: Record<string, string>;

  /**
   * Create a new DXMT client instance.
   *
   * @param config - Client configuration
   */
  constructor(config: ClientConfig = {}) {
    const {
      host = DEFAULT_CONFIG.host,
      port = DEFAULT_CONFIG.port,
      ssl = DEFAULT_CONFIG.ssl,
      timeout = DEFAULT_CONFIG.timeout,
      headers = {},
      apiKey,
    } = config;

    // Build base URL
    const protocol = ssl ? 'https' : 'http';
    this.baseUrl = `${protocol}://${host}:${port}/api/v1`;

    // Set timeout
    this.timeout = timeout;

    // Build headers
    this.headers = {
      'Content-Type': 'application/json',
      ...headers,
    };

    // Add API key if provided
    if (apiKey) {
      this.headers['Authorization'] = `Bearer ${apiKey}`;
    }
  }

  /**
   * Make HTTP request with error handling.
   */
  private async request<T>(
    method: string,
    path: string,
    body?: any
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;

    try {
      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        method,
        headers: this.headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Parse response
      const data = await response.json();

      // Check for errors
      if (!response.ok) {
        throw parseErrorResponse(response.status, data);
      }

      // The Rust API wraps successful responses in { success, data, timestamp }, while a few
      // compatibility endpoints return a plain JSON body. Accept both forms so the SDK reflects
      // the wire contract instead of leaking the server envelope into every call site.
      if (
        data &&
        data.success === true &&
        Object.prototype.hasOwnProperty.call(data, 'data')
      ) {
        return data.data as T;
      }
      return data as T;
    } catch (error: any) {
      // Handle timeout
      if (error.name === 'AbortError') {
        throw new DXMTTimeoutError(
          `Request timed out after ${this.timeout}ms`
        );
      }

      // Handle network errors
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new DXMTConnectionError(
          `Failed to connect to DXMT server at ${url}: ${error.message}`
        );
      }

      // Re-throw DXMT errors
      if (error instanceof DXMTError) {
        throw error;
      }

      // Wrap unknown errors
      throw new DXMTError(`Request failed: ${error.message}`);
    }
  }

  // ============================================================================
  // Table Operations
  // ============================================================================

  /**
   * Create a new table.
   *
   * @param config - Table configuration
   * @returns Table information
   *
   * @example
   * ```typescript
   * const table = await client.createTable({
   *   name: 'documents',
   *   dimension: 1536,
   *   distance_metric: 'cosine',
   *   index_type: 'hnsw'
   * });
   * ```
   */
  async createTable(config: TableConfig): Promise<TableInfo> {
    // Validate config
    if (!config.name) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!config.dimension || config.dimension <= 0) {
      throw new DXMTValidationError(
        'Valid dimension (> 0) is required'
      );
    }

    try {
      return await this.request<TableInfo>('POST', '/tables', config);
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTTableError(
        `Failed to create table: ${(error as Error).message}`
      );
    }
  }

  /**
   * List all tables.
   *
   * @returns Array of table names
   *
   * @example
   * ```typescript
   * const tables = await client.listTables();
   * console.log('Tables:', tables);
   * ```
   */
  async listTables(): Promise<TableInfo[]> {
    try {
      const response = await this.request<{ tables: TableInfo[] }>(
        'GET',
        '/tables'
      );
      return response.tables;
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTTableError(
        `Failed to list tables: ${(error as Error).message}`
      );
    }
  }

  /**
   * Get table information.
   *
   * @param tableName - Name of the table
   * @returns Table information
   *
   * @example
   * ```typescript
   * const info = await client.getTable('documents');
   * console.log('Dimension:', info.dimension);
   * console.log('Record count:', info.record_count);
   * ```
   */
  async getTable(tableName: string): Promise<TableInfo> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }

    try {
      const response = await this.request<
        TableInfo | { table: TableInfo; record_count: number }
      >(
        'GET',
        `/tables/${encodeURIComponent(tableName)}`
      );
      if ('table' in response) {
        return { ...response.table, record_count: response.record_count };
      }
      return response;
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTTableError(
        `Failed to get table: ${(error as Error).message}`
      );
    }
  }

  /**
   * Delete a table.
   *
   * @param tableName - Name of the table to delete
   *
   * @example
   * ```typescript
   * await client.deleteTable('old_documents');
   * ```
   */
  async deleteTable(tableName: string): Promise<void> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }

    try {
      await this.request<void>(
        'DELETE',
        `/tables/${encodeURIComponent(tableName)}`
      );
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTTableError(
        `Failed to delete table: ${(error as Error).message}`
      );
    }
  }

  /**
   * Get table statistics.
   *
   * @param tableName - Name of the table
   * @returns Table statistics
   *
   * @example
   * ```typescript
   * const stats = await client.getTableStats('documents');
   * console.log('Memory usage:', stats.memory_usage_bytes);
   * ```
   */
  async getTableStats(tableName: string): Promise<TableStats> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }

    try {
      return await this.request<TableStats>(
        'GET',
        `/tables/${encodeURIComponent(tableName)}/stats`
      );
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTTableError(
        `Failed to get table stats: ${(error as Error).message}`
      );
    }
  }

  // ============================================================================
  // Record Operations
  // ============================================================================

  /**
   * Insert records into a table.
   *
   * @param params - Insert parameters
   * @returns Batch operation result
   *
   * @example
   * ```typescript
   * await client.insertRecords({
   *   table_name: 'documents',
   *   records: [
   *     {
   *       id: 'doc1',
   *       vector: [0.1, 0.2, 0.3],
   *       metadata: { text: 'Document 1' }
   *     },
   *     {
   *       id: 'doc2',
   *       vector: [0.4, 0.5, 0.6],
   *       metadata: { text: 'Document 2' }
   *     }
   *   ]
   * });
   * ```
   */
  /**
   * Retrieve by sparse vector alone, ranked by dot product.
   *
   * Sparse vectors could be stored and indexed here long before they could be searched: the
   * only path into the sparse index was the hybrid endpoint, which refuses a body with no
   * dense vector and no text and needs a `text-bm25` build. This needs neither.
   */
  async sparseSearch(
    tableName: string,
    sparseVector: SparseVector,
    options: {
      limit?: number;
      filter?: Filter;
      withPayload?: boolean;
      withVector?: boolean;
      payloadFields?: string[];
      scoreThreshold?: number;
    } = {}
  ): Promise<SearchResult[]> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!sparseVector || !sparseVector.indices?.length) {
      throw new DXMTValidationError('A sparse vector with at least one index is required');
    }
    const body: Record<string, unknown> = {
      sparse_vector: sparseVector,
      limit: options.limit ?? 10,
      with_payload: options.withPayload ?? true,
      with_vector: options.withVector ?? false,
    };
    if (options.filter !== undefined) body.filter = options.filter;
    if (options.payloadFields !== undefined) body.payload_fields = options.payloadFields;
    if (options.scoreThreshold !== undefined) body.score_threshold = options.scoreThreshold;
    const answer = await this.request<{
      hits: Array<SearchResult & { payload?: Metadata }>;
    }>('POST', `/tables/${encodeURIComponent(tableName)}/search/sparse`, body);
    return (answer.hits || []).map((hit) => this.normalizeRecord(hit));
  }

  /**
   * Nearest neighbours grouped by a payload field.
   *
   * A document chunked for retrieval takes a whole top-10 by itself — the ten nearest vectors
   * are ten pieces of the same text. Grouping by the document's id with `groupSize: 1` returns
   * ten documents instead.
   *
   * @example
   * ```typescript
   * const { groups } = await client.searchGroups('chunks', queryVector, 'doc_id', {
   *   groups: 10,
   *   groupSize: 1,
   * });
   * ```
   */
  async searchGroups(
    tableName: string,
    vector: number[],
    groupBy: string,
    options: {
      groups?: number;
      groupSize?: number;
      filter?: Filter;
      withPayload?: boolean;
      payloadFields?: string[];
      scoreThreshold?: number;
      searchMode?: 'fast' | 'normal' | 'accurate' | 'auto' | 'exact';
    } = {}
  ): Promise<SearchGroupsResult> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!vector || vector.length === 0) {
      throw new DXMTValidationError('Query vector is required');
    }
    if (!groupBy) {
      throw new DXMTValidationError('group_by is required');
    }
    const body: Record<string, unknown> = {
      vector,
      group_by: groupBy,
      groups: options.groups ?? 10,
      group_size: options.groupSize ?? 3,
      with_payload: options.withPayload ?? true,
    };
    if (options.filter !== undefined) body.filter = options.filter;
    if (options.payloadFields !== undefined) body.payload_fields = options.payloadFields;
    if (options.scoreThreshold !== undefined) body.score_threshold = options.scoreThreshold;
    if (options.searchMode !== undefined) body.search_mode = options.searchMode;
    const answer = await this.request<
      Omit<SearchGroupsResult, 'groups'> & {
        groups: Array<{
          group_id: unknown;
          matched: number;
          hits: Array<SearchResult & { payload?: Metadata }>;
        }>;
      }
    >('POST', `/tables/${encodeURIComponent(tableName)}/search/groups`, body);
    // Hits come back through the same normalization as every other read here, so a caller
    // reads `metadata` whichever method they used rather than `payload` in this one alone.
    return {
      ...answer,
      groups: (answer.groups || []).map((group) => ({
        ...group,
        hits: (group.hits || []).map((hit) => this.normalizeRecord(hit)),
      })),
    };
  }

  /**
   * Write one table — configuration and every record, vectors included — to a file.
   *
   * The unit `POST /backup` is not: a backup is the whole instance and restoring it means
   * stopping the server and replacing its directory. The file is written **by the server, on
   * the server's filesystem**, inside its `DXMT_BACKUP_ROOT` — `path` is not a path on this
   * machine unless the two are the same machine.
   */
  async exportTable(
    tableName: string,
    path: string
  ): Promise<{
    table: string;
    path: string;
    records: number;
    bytes: number;
    records_without_vectors: number;
  }> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!path) {
      throw new DXMTValidationError('A destination path is required');
    }
    return this.request('POST', `/tables/${encodeURIComponent(tableName)}/export`, { path });
  }

  /**
   * Read a table back out of a file written by {@link exportTable}.
   *
   * Records keep their ids, payloads, versions and `created_at`. An import onto a name already
   * in use is refused unless `overwrite` is set: a merge and a replacement are different
   * operations and neither is the obvious reading of "import".
   */
  async importTable(
    path: string,
    options: { name?: string; overwrite?: boolean } = {}
  ): Promise<{
    table: string;
    path: string;
    records: number;
    records_in_file: number;
    replaced: boolean;
  }> {
    if (!path) {
      throw new DXMTValidationError('A source path is required');
    }
    const body: Record<string, unknown> = { path, overwrite: options.overwrite ?? false };
    if (options.name !== undefined) body.name = options.name;
    return this.request('POST', '/tables/import', body);
  }

  /**
   * What the payload (filter) index holds for each field, and what it costs.
   *
   * Every payload field is indexed by default, which is what makes a filter fast here without
   * declaring anything — and what a field nobody filters on pays for. This is the answer to
   * "what is that costing me": distinct values, postings and an estimate of the bytes, worst
   * field first. It does not build the index in order to describe it, so a table that has not
   * filtered since it was restored reports `built: false`.
   */
  async payloadIndex(tableName: string): Promise<PayloadIndexReport> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    return this.request<PayloadIndexReport>(
      'GET',
      `/tables/${encodeURIComponent(tableName)}/payload_index`
    );
  }

  /**
   * Index a payload field for filtering.
   *
   * Idempotent, and a no-op on a table that indexes every field — the default — so code
   * written against a database where indexes must be declared works unchanged. The index is
   * rebuilt from the records, so rows written before the field was indexed are covered.
   */
  async indexPayloadField(tableName: string, field: string): Promise<{ message: string }> {
    if (!tableName || !field) {
      throw new DXMTValidationError('Table name and field are required');
    }
    return this.request<{ message: string }>(
      'PUT',
      `/tables/${encodeURIComponent(tableName)}/payload_index/${encodeURIComponent(field)}`
    );
  }

  /**
   * Stop indexing a payload field, freeing what its postings cost.
   *
   * The price is filtering on it: a filter naming an unindexed field is refused with a 400,
   * because a filter is answered only from the index and the only other honest answer would
   * be an empty page — which looks exactly like "nothing matched".
   */
  async dropPayloadIndexField(tableName: string, field: string): Promise<{ message: string }> {
    if (!tableName || !field) {
      throw new DXMTValidationError('Table name and field are required');
    }
    return this.request<{ message: string }>(
      'DELETE',
      `/tables/${encodeURIComponent(tableName)}/payload_index/${encodeURIComponent(field)}`
    );
  }

  /**
   * How many records a filter matches, or how many the table holds.
   *
   * Counted server-side from the same index that answers the filter. Without it the only way
   * to this number was to page every match and count them here, which puts every matching
   * record on the wire to produce one integer.
   *
   * @example
   * ```typescript
   * const total = await client.count('documents');
   * const english = await client.count('documents', {
   *   must: [{ key: 'lang', operation: { type: 'Equal', value: 'en' } }],
   * });
   * ```
   */
  async count(tableName: string, filter?: Filter): Promise<number> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    const answer = await this.request<{ count: number }>(
      'POST',
      `/tables/${encodeURIComponent(tableName)}/records/count`,
      filter === undefined ? {} : { filter }
    );
    return answer.count;
  }

  /**
   * Every record of a table, newest first, fetched a page at a time.
   *
   * Pages by cursor rather than offset: every record present for the whole scroll comes back
   * exactly once however the table is written to meanwhile, where an offset shifts with each
   * insert and repeats or skips records.
   *
   * @example
   * ```typescript
   * for await (const record of client.scroll('documents', { batchSize: 500 })) {
   *   console.log(record.id, record.metadata);
   * }
   * ```
   */
  async *scroll(tableName: string, options: ScrollOptions = {}): AsyncGenerator<VectorRecord> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    const batchSize = options.batchSize ?? 100;
    let cursor: string | null = null;
    for (;;) {
      const page: {
        records?: Array<VectorRecord & { payload?: Metadata; payload_json?: string }>;
        next_cursor?: string | null;
      } = await this.request('POST', `/tables/${encodeURIComponent(tableName)}/records/list`, {
        limit: batchSize,
        with_vector: options.withVector ?? false,
        ...(options.filter === undefined ? {} : { filter: options.filter }),
        ...(cursor === null ? {} : { cursor }),
      });
      const records = page.records ?? [];
      // A server from before cursors answers without the field. Taking that as "no next page"
      // would end every scroll after one page, silently; a full page without it is an error.
      if (!Object.prototype.hasOwnProperty.call(page, 'next_cursor') && records.length >= batchSize) {
        throw new DXMTRecordError('this server does not page by cursor; upgrade it to scroll');
      }
      for (const record of records) {
        yield this.normalizeRecord(record);
      }
      cursor = page.next_cursor ?? null;
      if (cursor === null) {
        return;
      }
    }
  }

  /**
   * Insert many records, in batches, retrying what a retry can fix.
   *
   * `records` may be any iterable or async iterable; it is drawn a batch at a time. A batch
   * that fails for a reason a retry can fix — no answer, 429, a 5xx — is sent again after an
   * exponential backoff; one the server refuses as wrong (a dimension mismatch, a missing table)
   * is not. Inserts are upserts, so a retried batch that had in fact landed duplicates nothing.
   * It does not throw for a failed batch: `failed` says which ids did not go in and why.
   *
   * @example
   * ```typescript
   * const result = await client.upload('documents', records, { batchSize: 256, parallel: 4 });
   * if (result.failed.length) console.error(result.failed);
   * ```
   */
  async upload(
    tableName: string,
    records: Iterable<VectorRecord> | AsyncIterable<VectorRecord>,
    options: UploadOptions = {}
  ): Promise<UploadResult> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    const batchSize = options.batchSize ?? 256;
    const parallel = Math.max(1, options.parallel ?? 1);
    const maxRetries = options.maxRetries ?? 3;
    const backoffMs = options.backoffMs ?? 500;
    const path = `/tables/${encodeURIComponent(tableName)}/records`;
    const result: UploadResult = { uploaded: 0, batches: 0, retries: 0, failed: [] };

    const send = async (batch: VectorRecord[]): Promise<void> => {
      const body = {
        records: batch.map(({ metadata, ...record }) => ({
          ...record,
          ...(metadata === undefined ? {} : { payload: metadata }),
        })),
      };
      for (let attempt = 0; ; attempt++) {
        let status: number | undefined;
        let error: string;
        try {
          const answer = await this.rawRequest('POST', path, body);
          if (answer.status >= 200 && answer.status < 300) {
            result.uploaded += batch.length;
            break;
          }
          status = answer.status;
          error = answer.text;
        } catch (e) {
          error = `no answer: ${(e as Error).message}`;
        }
        const retryable = status === undefined || RETRYABLE.has(status);
        if (!retryable || attempt >= maxRetries) {
          result.failed.push({ ids: batch.map((r) => String(r.id)), error: error.slice(0, 500), status });
          break;
        }
        result.retries += 1;
        await new Promise((resolve) => setTimeout(resolve, backoffMs * 2 ** attempt));
      }
      result.batches += 1;
    };

    // At most `parallel` batches in flight, so the input is drawn only as fast as it is sent.
    const inFlight = new Set<Promise<void>>();
    let batch: VectorRecord[] = [];
    const flush = async (): Promise<void> => {
      const task = send(batch);
      batch = [];
      inFlight.add(task);
      task.finally(() => inFlight.delete(task));
      if (inFlight.size >= parallel) {
        await Promise.race(inFlight);
      }
    };
    for await (const record of records as AsyncIterable<VectorRecord>) {
      batch.push(record);
      if (batch.length === batchSize) {
        await flush();
      }
    }
    if (batch.length > 0) {
      await flush();
    }
    await Promise.all(inFlight);
    return result;
  }

  /** A request that reports the status instead of throwing on it; throws only on no answer. */
  private async rawRequest(method: string, path: string, body?: unknown): Promise<{ status: number; text: string }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      return { status: response.status, text: await response.text() };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async insertRecords(
    params: InsertRecordsParams
  ): Promise<BatchOperationResult> {
    // Validate params
    if (!params.table_name) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!params.records || params.records.length === 0) {
      throw new DXMTValidationError('At least one record is required');
    }

    // Validate records
    for (const record of params.records) {
      if (!record.id) {
        throw new DXMTValidationError('Record ID is required');
      }
      if (!record.vector || !Array.isArray(record.vector) || record.vector.length === 0) {
        throw new DXMTValidationError('Record vector must be an array');
      }
    }

    try {
      const response = await this.request<{
        inserted_ids?: string[];
        count?: number;
      }>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/records`,
        {
          // The public SDK calls this field `metadata`; the Rust wire contract calls it
          // `payload`. Send the server shape explicitly so metadata is not silently discarded.
          records: params.records.map(({ metadata, ...record }) => ({
            ...record,
            ...(metadata === undefined ? {} : { payload: metadata }),
          })),
        }
      );
      const insertedIds = response.inserted_ids || [];
      return {
        success: true,
        inserted: response.count ?? insertedIds.length,
        inserted_ids: insertedIds,
        count: response.count ?? insertedIds.length,
      };
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTRecordError(
        `Failed to insert records: ${(error as Error).message}`
      );
    }
  }

  /**
   * Get a record by ID.
   *
   * @param tableName - Name of the table
   * @param recordId - ID of the record
   * @returns Record
   *
   * @example
   * ```typescript
   * const record = await client.getRecord('documents', 'doc1');
   * console.log('Vector:', record.vector);
   * console.log('Metadata:', record.metadata);
   * ```
   */
  async getRecord(tableName: string, recordId: string): Promise<VectorRecord> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!recordId) {
      throw new DXMTValidationError('Record ID is required');
    }

    try {
      const response = await this.request<
        VectorRecord & { payload?: Metadata; payload_json?: string }
      >(
        'GET',
        `/tables/${encodeURIComponent(tableName)}/records/${encodeURIComponent(recordId)}`
      );
      return this.normalizeRecord(response);
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTRecordError(
        `Failed to get record: ${(error as Error).message}`
      );
    }
  }

  /** Update a record's vector and/or metadata without losing auxiliary vectors. */
  async updateRecord(params: UpdateRecordParams): Promise<BatchOperationResult> {
    if (!params.table_name) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!params.record_id) {
      throw new DXMTValidationError('Record ID is required');
    }
    if (params.vector !== undefined &&
        (!Array.isArray(params.vector) || params.vector.length === 0)) {
      throw new DXMTValidationError('Record vector must be an array');
    }
    if (params.metadata === undefined && params.vector === undefined && !params.upsert) {
      throw new DXMTValidationError('At least one field (vector or metadata) is required');
    }

    try {
      await this.request<{
        message?: string;
        record?: { id: string; version?: number };
      }>(
        'PUT',
        `/tables/${encodeURIComponent(params.table_name)}/records/${encodeURIComponent(params.record_id)}`,
        {
          ...(params.vector === undefined ? {} : { vector: params.vector }),
          ...(params.metadata === undefined ? {} : { payload: params.metadata }),
          merge_mode: params.merge_mode || 'replace',
          upsert: params.upsert || false,
        }
      );
      return { success: true, updated: 1, count: 1 };
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTRecordError(
        `Failed to update record: ${(error as Error).message}`
      );
    }
  }

  /**
   * Delete records from a table.
   *
   * @param params - Delete parameters
   *
   * @example
   * ```typescript
   * // Delete by IDs
   * await client.deleteRecords({
   *   table_name: 'documents',
   *   record_ids: ['doc1', 'doc2']
   * });
   *
   * // Delete by filter
   * await client.deleteRecords({
   *   table_name: 'documents',
   *   filter: {
   *     must: [{ key: 'category', operation: { type: 'Equal', value: 'old' } }]
   *   }
   * });
   * ```
   */
  async deleteRecords(params: DeleteParams): Promise<void> {
    if (!params.table_name) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!params.record_ids === !params.filter) {
      // Both would have to mean intersection or union, and neither reading is obvious;
      // neither means "delete the whole table", which is not something to do by omission.
      throw new DXMTValidationError(
        'Give either record_ids or filter, not both and not neither'
      );
    }

    try {
      await this.request<void>(
        'DELETE',
        `/tables/${encodeURIComponent(params.table_name)}/records`,
        params.record_ids
          ? { ids: params.record_ids }
          : { filter: params.filter }
      );
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTRecordError(
        `Failed to delete records: ${(error as Error).message}`
      );
    }
  }

  // ============================================================================
  // Search Operations
  // ============================================================================

  /**
   * Perform vector similarity search.
   *
   * @param params - Search parameters
   * @returns Array of search results
   *
   * @example
   * ```typescript
   * const results = await client.search({
   *   table_name: 'documents',
   *   vector: [0.1, 0.2, 0.3],
   *   k: 5,
   *   filter: {
   *     must: [{ key: 'category', operation: { type: 'Equal', value: 'tech' } }]
   *   }
   * });
   *
   * results.forEach(result => {
   *   console.log('ID:', result.id);
   *   console.log('Score:', result.score);
   *   console.log('Metadata:', result.metadata);
   * });
   * ```
   */
  async search(params: SearchParams): Promise<SearchResult[]> {
    // Validate params
    if (!params.table_name) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!params.vector || !Array.isArray(params.vector) || params.vector.length === 0) {
      throw new DXMTValidationError('Query vector must be an array');
    }

    try {
      const response = await this.request<{ hits: Array<SearchResult & { payload?: Metadata }> }>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/search`,
        {
          vector: params.vector,
          top_k: params.k || 10,
          filter: params.filter,
          with_payload: params.include_metadata !== false,
          with_vector: params.include_vectors || false,
          ...(params.search_mode === undefined ? {} : { search_mode: params.search_mode }),
          ...(params.score_threshold === undefined
            ? {}
            : { score_threshold: params.score_threshold }),
          ...(params.payload_fields === undefined
            ? {}
            : { payload_fields: params.payload_fields }),
        }
      );

      return (response.hits || []).map((hit) => this.normalizeRecord(hit));
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTSearchError(
        `Failed to search: ${(error as Error).message}`
      );
    }
  }

  /**
   * Perform full-text search.
   *
   * @param params - Full-text search parameters
   * @returns Array of search results
   *
   * @example
   * ```typescript
   * const results = await client.fullTextSearch({
   *   table_name: 'documents',
   *   query: 'machine learning',
   *   k: 10
   * });
   * ```
   */
  async fullTextSearch(
    params: FullTextSearchParams
  ): Promise<SearchResult[]> {
    if (!params.table_name) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!params.query) {
      throw new DXMTValidationError('Query text is required');
    }

    try {
      const response = await this.request<{ hits: Array<SearchResult & { payload?: Metadata }> }>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/search/text`,
        {
          query: params.query,
          limit: params.k || 10,
          filter: params.filter,
        }
      );

      return (response.hits || []).map((hit) => this.normalizeRecord(hit));
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTSearchError(
        `Failed to perform full-text search: ${(error as Error).message}`
      );
    }
  }

  /**
   * Perform hybrid search (vector + text).
   *
   * @param params - Hybrid search parameters
   * @returns Array of search results
   *
   * @example
   * ```typescript
   * const results = await client.hybridSearch({
   *   table_name: 'documents',
   *   vector: [0.1, 0.2, 0.3],
   *   text_query: 'machine learning',
   *   k: 10,
   *   vector_weight: 0.7  // 70% vector, 30% text; selects weighted_sum fusion
   * });
   * ```
   */
  async hybridSearch(params: HybridSearchParams): Promise<SearchResult[]> {
    if (!params.table_name) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!params.vector && !params.text_query) {
      throw new DXMTValidationError(
        'Either vector or text_query is required'
      );
    }

    try {
      const response = await this.request<{ hits: Array<SearchResult & { payload?: Metadata }> }>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/search/hybrid`,
        {
          vector: params.vector,
          text: params.text_query,
          limit: params.k || 10,
          filter: params.filter,
          // A weight only means something to a fusion that reads one. This sent 0.5 by
          // default and never named a method, so every hybrid search asked for `dbsf` --
          // which ignores the weight -- and the parameter did nothing whichever value it had.
          ...(params.vector_weight === undefined
            ? {}
            : { vector_weight: params.vector_weight }),
          ...(params.fusion_method === undefined
            ? params.vector_weight === undefined
              ? {}
              : { fusion_method: 'weighted_sum' as const }
            : { fusion_method: params.fusion_method }),
        }
      );

      return (response.hits || []).map((hit) => this.normalizeRecord(hit));
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTSearchError(
        `Failed to perform hybrid search: ${(error as Error).message}`
      );
    }
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Check server health.
   *
   * @returns Health check response
   *
   * @example
   * ```typescript
   * const health = await client.healthCheck();
   * console.log('Status:', health.status);
   * console.log('Version:', health.version);
   * ```
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    try {
      return await this.request<HealthCheckResponse>('GET', '/health');
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTConnectionError(
        `Health check failed: ${(error as Error).message}`
      );
    }
  }

  /** Return detailed health and storage durability metrics. */
  async healthMetrics(): Promise<HealthCheckResponse> {
    try {
      return await this.request<HealthCheckResponse>('GET', '/health/metrics');
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTConnectionError(
        `Health metrics request failed: ${(error as Error).message}`
      );
    }
  }

  /**
   * Get the base URL of the client.
   */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Get the timeout value.
   */
  getTimeout(): number {
    return this.timeout;
  }

  /**
   * Set custom headers for requests.
   */
  setHeaders(headers: Record<string, string>): void {
    this.headers = {
      ...this.headers,
      ...headers,
    };
  }

  /** Convert the server's `payload` field to the SDK's `metadata` field. */
  private normalizeRecord<
    T extends { payload?: Metadata; payload_json?: string }
  >(record: T): T & VectorRecord {
    const normalized = { ...record } as T & VectorRecord;
    let payload = record.payload;
    if (payload === undefined && record.payload_json) {
      try {
        payload = JSON.parse(record.payload_json) as Metadata;
      } catch {
        // Keep malformed payloads non-fatal; the raw payload_json remains
        // available for diagnostics and metadata is simply omitted.
      }
    }
    if (normalized.metadata === undefined && payload !== undefined) {
      normalized.metadata = payload;
    }
    return normalized;
  }
}
