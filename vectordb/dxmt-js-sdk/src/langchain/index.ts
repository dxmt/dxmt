/**
 * DXMT LangChain Integration.
 *
 * This module provides LangChain integration for DXMT VectorDB.
 *
 * @example
 * ```typescript
 * import { DXMTVectorStore } from 'dxmt-js-sdk/langchain';
 * import { OpenAIEmbeddings } from '@langchain/openai';
 *
 * const vectorStore = new DXMTVectorStore(
 *   new OpenAIEmbeddings(),
 *   {
 *     collectionName: 'documents',
 *     host: 'localhost',
 *     port: 9999
 *   }
 * );
 * ```
 */

export { DXMTVectorStore } from './vectorstore';
export type {
  DXMTVectorStoreConfig,
  AddDocumentOptions,
  SimilaritySearchOptions,
} from './types';
