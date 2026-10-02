/**
 * DXMT VectorStore for LangChain.
 *
 * This module provides a LangChain VectorStore implementation backed by DXMT VectorDB.
 */

import type { EmbeddingsInterface } from '@langchain/core/embeddings';
import { VectorStore } from '@langchain/core/vectorstores';
import { Document } from '@langchain/core/documents';
import { DXMTClient } from '../client';
import { DXMTValidationError } from '../errors';
import type { Filter } from '../types';
import type {
  DXMTVectorStoreConfig,
  AddDocumentOptions,
} from './types';

/**
 * DXMT VectorStore for LangChain.
 *
 * Implements LangChain's VectorStore interface to enable DXMT VectorDB
 * integration with LangChain applications.
 *
 * @example
 * ```typescript
 * import { DXMTVectorStore } from 'dxmt-js-sdk/langchain';
 * import { OpenAIEmbeddings } from '@langchain/openai';
 *
 * const embeddings = new OpenAIEmbeddings();
 * const vectorStore = new DXMTVectorStore(embeddings, {
 *   collectionName: 'documents',
 *   host: 'localhost',
 *   port: 9999
 * });
 *
 * // Add documents
 * await vectorStore.addDocuments([
 *   new Document({ pageContent: 'Hello', metadata: { source: 'test' } })
 * ]);
 *
 * // Search
 * const results = await vectorStore.similaritySearch('Hi', 5);
 * ```
 */
export class DXMTVectorStore extends VectorStore {
  declare FilterType: Record<string, any>;

  /** The underlying client, for anything the store does not wrap. */
  readonly client: DXMTClient;
  private collectionName: string;
  private dimension?: number;
  private distanceMetric: string;

  /**
   * Create a new DXMT vector store.
   *
   * @param embeddings - Embeddings instance for generating vectors
   * @param config - DXMT configuration
   */
  constructor(
    embeddings: EmbeddingsInterface,
    config: DXMTVectorStoreConfig
  ) {
    super(embeddings, config);

    // Validate config
    if (!config.collectionName) {
      throw new DXMTValidationError('collectionName is required');
    }

    this.collectionName = config.collectionName;
    this.dimension = config.dimension;
    this.distanceMetric = config.distanceMetric || 'cosine';

    // Create DXMT client
    this.client = new DXMTClient({
      host: config.host || 'localhost',
      port: config.port || 9999,
      ssl: config.ssl || false,
      timeout: config.timeout || 30000,
      apiKey: config.apiKey,
      headers: config.headers,
    });
  }

  /**
   * Return type identifier.
   */
  _vectorstoreType(): string {
    return 'dxmt';
  }

  /**
   * Ensure the collection exists, creating it if necessary.
   */
  private async ensureCollection(): Promise<void> {
    try {
      // Try to get existing table
      await this.client.getTable(this.collectionName);
    } catch (error) {
      // Only a missing table means "create it". This caught *every* failure -- a 401, a
      // timeout, a 500 -- and went on to create, so the user saw a create error (or a
      // duplicate-table error) instead of the one that actually happened.
      if ((error as { code?: string }).code !== 'TABLE_NOT_FOUND') {
        throw error;
      }
      if (!this.dimension) {
        // Auto-detect dimension from embeddings
        const testVector = await this.embeddings.embedQuery('test');
        this.dimension = testVector.length;
      }

      await this.client.createTable({
        name: this.collectionName,
        dimension: this.dimension,
        distance_metric: this.distanceMetric as any,
        description: `LangChain collection (dimension=${this.dimension})`,
      });
    }
  }

  /**
   * Convert simple filter to DXMT filter format.
   */
  private convertFilter(filter?: Record<string, any>): Filter | undefined {
    if (!filter) return undefined;

    const mustConditions = [];
    for (const [key, value] of Object.entries(filter)) {
      // Equality on a scalar is the whole language here. An object -- LangChain's
      // `{ year: { $gte: 2020 } }` shape -- used to be wrapped in `Equal` and sent, which the
      // server refuses as an object target; saying so here names the limit instead.
      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        throw new DXMTValidationError(
          `filter on "${key}": this store supports equality on scalars only; ` +
            `for ranges and other operators use DXMTClient.search with a DXMT filter`
        );
      }
      mustConditions.push({
        key,
        operation: { type: 'Equal' as const, value },
      });
    }

    return mustConditions.length > 0
      ? { must: mustConditions }
      : undefined;
  }

  /**
   * Generate unique ID.
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Add documents to the vector store.
   *
   * @param documents - Documents to add
   * @param options - Options including optional IDs
   * @returns Array of document IDs
   */
  async addDocuments(
    documents: Document[],
    options?: AddDocumentOptions
  ): Promise<string[]> {
    // Ensure collection exists
    await this.ensureCollection();

    // Extract texts
    const texts = documents.map((doc) => doc.pageContent);

    // Generate embeddings
    const vectors = await this.embeddings.embedDocuments(texts);

    // Explicit ids, then the document's own `id`, then a generated one. `Document.id` was
    // ignored and every add minted `Date.now()-random`; since inserts are upserts, a caller
    // re-adding a document they had given an id got a second copy rather than an update.
    const ids =
      options?.ids || documents.map((doc) => doc.id ?? this.generateId());

    if (ids.length !== documents.length) {
      throw new DXMTValidationError(
        'Number of IDs must match number of documents'
      );
    }

    // Prepare records
    const records = documents.map((doc, i) => ({
      id: ids[i],
      vector: vectors[i],
      metadata: {
        ...doc.metadata,
        text: doc.pageContent,
      },
    }));

    // Insert into DXMT
    await this.client.insertRecords({
      table_name: this.collectionName,
      records,
    });

    return ids;
  }

  /**
   * Add vectors with documents.
   *
   * @param vectors - Vectors to add
   * @param documents - Corresponding documents
   * @param options - Options including optional IDs
   * @returns Array of document IDs
   */
  async addVectors(
    vectors: number[][],
    documents: Document[],
    options?: AddDocumentOptions
  ): Promise<string[]> {
    // Ensure collection exists
    await this.ensureCollection();

    if (vectors.length !== documents.length) {
      throw new DXMTValidationError(
        'Number of vectors must match number of documents'
      );
    }

    // Generate IDs if not provided
    const ids =
      options?.ids || documents.map(() => this.generateId());

    if (ids.length !== documents.length) {
      throw new DXMTValidationError(
        'Number of IDs must match number of documents'
      );
    }

    // Prepare records
    const records = documents.map((doc, i) => ({
      id: ids[i],
      vector: vectors[i],
      metadata: {
        ...doc.metadata,
        text: doc.pageContent,
      },
    }));

    // Insert into DXMT
    await this.client.insertRecords({
      table_name: this.collectionName,
      records,
    });

    return ids;
  }

  /**
   * Similarity search without scores.
   *
   * @param query - Query text
   * @param k - Number of results
   * @param filter - Optional metadata filter
   * @returns Array of documents
   */
  async similaritySearch(
    query: string,
    k = 4,
    filter?: Record<string, any>
  ): Promise<Document[]> {
    const results = await this.similaritySearchWithScore(query, k, filter);
    return results.map(([doc]) => doc);
  }

  /**
   * Similarity search with scores.
   *
   * @param query - Query text
   * @param k - Number of results
   * @param filter - Optional metadata filter
   * @returns Array of [document, score] tuples
   */
  async similaritySearchWithScore(
    query: string,
    k = 4,
    filter?: Record<string, any>
  ): Promise<[Document, number][]> {
    // Generate query embedding
    const queryVector = await this.embeddings.embedQuery(query);

    // Convert filter
    const dxmtFilter = this.convertFilter(filter);

    // Search DXMT
    const results = await this.client.search({
      table_name: this.collectionName,
      vector: queryVector,
      k,
      filter: dxmtFilter,
      include_metadata: true,
    });

    // Convert results
    return results.map((result) => {
      const metadata = result.metadata || {};
      const text = metadata.text || '';
      delete metadata.text;

      const doc = new Document({
        pageContent: text,
        metadata: {
          ...metadata,
          id: result.id,
        },
      });

      return [doc, result.score];
    });
  }

  /**
   * Similarity search by vector.
   *
   * @param query - Query vector
   * @param k - Number of results
   * @param filter - Optional metadata filter
   * @returns Array of documents
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: Record<string, any>
  ): Promise<[Document, number][]> {
    // Convert filter
    const dxmtFilter = this.convertFilter(filter);

    // Search DXMT
    const results = await this.client.search({
      table_name: this.collectionName,
      vector: query,
      k,
      filter: dxmtFilter,
      include_metadata: true,
    });

    // Convert results
    return results.map((result) => {
      const metadata = result.metadata || {};
      const text = metadata.text || '';
      delete metadata.text;

      const doc = new Document({
        pageContent: text,
        metadata: {
          ...metadata,
          id: result.id,
        },
      });

      return [doc, result.score];
    });
  }

  /**
   * Delete documents by IDs.
   *
   * @param params - Deletion parameters
   */
  async delete(params?: { ids?: string[]; filter?: Record<string, any> }): Promise<void> {
    // The base class signature is `delete(params?)`; this took a required `{ ids }` and threw
    // a TypeError on anything else, while the client underneath deletes by filter as well.
    if (params?.ids?.length) {
      await this.client.deleteRecords({
        table_name: this.collectionName,
        record_ids: params.ids,
      });
      return;
    }
    if (params?.filter) {
      await this.client.deleteRecords({
        table_name: this.collectionName,
        filter: this.convertFilter(params.filter),
      });
      return;
    }
    throw new DXMTValidationError('delete() needs { ids } or { filter }');
  }

  /**
   * Create DXMT vector store from texts.
   *
   * @param texts - Texts to add
   * @param metadatas - Metadata for each text
   * @param embeddings - Embeddings instance
   * @param config - DXMT configuration
   * @returns DXMT vector store instance
   */
  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    config: DXMTVectorStoreConfig
  ): Promise<DXMTVectorStore> {
    // Normalize metadatas
    const metadataArray = Array.isArray(metadatas)
      ? metadatas
      : texts.map(() => metadatas);

    // Create documents
    const documents = texts.map(
      (text, i) =>
        new Document({
          pageContent: text,
          metadata: metadataArray[i] || {},
        })
    );

    // Create vector store
    const vectorStore = new DXMTVectorStore(embeddings, config);

    // Add documents
    await vectorStore.addDocuments(documents);

    return vectorStore;
  }

  /**
   * Create DXMT vector store from documents.
   *
   * @param docs - Documents to add
   * @param embeddings - Embeddings instance
   * @param config - DXMT configuration
   * @returns DXMT vector store instance
   */
  static async fromDocuments(
    docs: Document[],
    embeddings: EmbeddingsInterface,
    config: DXMTVectorStoreConfig
  ): Promise<DXMTVectorStore> {
    const vectorStore = new DXMTVectorStore(embeddings, config);
    await vectorStore.addDocuments(docs);
    return vectorStore;
  }

  /**
   * Get the DXMT client instance.
   */
  getClient(): DXMTClient {
    return this.client;
  }

  /**
   * Get the collection name.
   */
  getCollectionName(): string {
    return this.collectionName;
  }
}
