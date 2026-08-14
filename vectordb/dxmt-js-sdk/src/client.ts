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
  Record,
  SearchParams,
  SearchResult,
  InsertRecordsParams,
  DeleteParams,
  BatchOperationResult,
  TableStats,
  HealthCheckResponse,
  FullTextSearchParams,
  HybridSearchParams,
  Filter,
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
  async listTables(): Promise<string[]> {
    try {
      const response = await this.request<{ tables: string[] }>(
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
      return await this.request<TableInfo>(
        'GET',
        `/tables/${encodeURIComponent(tableName)}`
      );
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
   *   ],
   *   mode: 'replace'
   * });
   * ```
   */
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
      if (!record.vector || !Array.isArray(record.vector)) {
        throw new DXMTValidationError('Record vector must be an array');
      }
    }

    try {
      return await this.request<BatchOperationResult>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/records`,
        {
          records: params.records,
          mode: params.mode || 'append',
        }
      );
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
  async getRecord(tableName: string, recordId: string): Promise<Record> {
    if (!tableName) {
      throw new DXMTValidationError('Table name is required');
    }
    if (!recordId) {
      throw new DXMTValidationError('Record ID is required');
    }

    try {
      return await this.request<Record>(
        'GET',
        `/tables/${encodeURIComponent(tableName)}/records/${encodeURIComponent(recordId)}`
      );
    } catch (error) {
      if (error instanceof DXMTError) throw error;
      throw new DXMTRecordError(
        `Failed to get record: ${(error as Error).message}`
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
    if (!params.record_ids && !params.filter) {
      throw new DXMTValidationError(
        'Either record_ids or filter is required'
      );
    }

    try {
      await this.request<void>(
        'DELETE',
        `/tables/${encodeURIComponent(params.table_name)}/records`,
        {
          record_ids: params.record_ids,
          filter: params.filter,
        }
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
    if (!params.vector || !Array.isArray(params.vector)) {
      throw new DXMTValidationError('Query vector must be an array');
    }

    try {
      const response = await this.request<{ results: SearchResult[] }>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/search`,
        {
          vector: params.vector,
          k: params.k || 10,
          filter: params.filter,
          ef_search: params.ef_search,
          include_metadata: params.include_metadata !== false,
          include_vectors: params.include_vectors || false,
        }
      );

      return response.results;
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
      const response = await this.request<{ results: SearchResult[] }>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/search/text`,
        {
          query: params.query,
          k: params.k || 10,
          filter: params.filter,
        }
      );

      return response.results;
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
   *   vector_weight: 0.7,
   *   text_weight: 0.3
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
      const response = await this.request<{ results: SearchResult[] }>(
        'POST',
        `/tables/${encodeURIComponent(params.table_name)}/search/hybrid`,
        {
          vector: params.vector,
          text_query: params.text_query,
          k: params.k || 10,
          filter: params.filter,
          vector_weight: params.vector_weight || 0.5,
          text_weight: params.text_weight || 0.5,
        }
      );

      return response.results;
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
}
