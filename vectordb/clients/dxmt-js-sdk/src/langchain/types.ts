/**
 * Type definitions for DXMT LangChain integration.
 */

/**
 * Configuration for DXMT VectorStore.
 */
export interface DXMTVectorStoreConfig {
  /**
   * Name of the collection/table.
   */
  collectionName: string;

  /**
   * DXMT server host.
   * @default 'localhost'
   */
  host?: string;

  /**
   * DXMT server port.
   * @default 9999
   */
  port?: number;

  /**
   * Use SSL/HTTPS.
   * @default false
   */
  ssl?: boolean;

  /**
   * Vector dimension (auto-detected if not provided).
   */
  dimension?: number;

  /**
   * Distance metric for similarity calculation.
   * @default 'cosine'
   */
  distanceMetric?: 'cosine' | 'euclidean' | 'dot' | 'manhattan';

  /**
   * Request timeout in milliseconds.
   * @default 30000
   */
  timeout?: number;

  /**
   * API key for authentication.
   */
  apiKey?: string;

  /**
   * Custom HTTP headers.
   */
  headers?: Record<string, string>;
}

/**
 * Options for adding documents.
 */
export interface AddDocumentOptions {
  /**
   * Document IDs (auto-generated if not provided).
   */
  ids?: string[];
}

/**
 * Options for similarity search.
 */
export interface SimilaritySearchOptions {
  /**
   * Metadata filter.
   */
  filter?: Record<string, any>;

  /**
   * Number of results to return.
   */
  k?: number;
}
