/**
 * LangChain Integration Example for DXMT JavaScript SDK.
 *
 * This example demonstrates:
 * 1. Creating a DXMT vector store for LangChain
 * 2. Adding documents with embeddings
 * 3. Similarity search
 * 4. Search with scores
 * 5. Filtered search
 * 6. Using from_texts and from_documents
 */

import { Document } from '@langchain/core/documents';
import { FakeEmbeddings } from '@langchain/core/utils/testing';
import { DXMTVectorStore } from '../src/langchain';
import { DXMTClient } from '../src';

async function main() {
  console.log('='.repeat(70));
  console.log('DXMT JavaScript SDK - LangChain Integration Example');
  console.log('='.repeat(70));

  // Check server
  console.log('\n📊 Step 1: Checking DXMT server...');
  const testClient = new DXMTClient({ host: 'localhost', port: 9999 });

  try {
    const health = await testClient.healthCheck();
    console.log('✅ Connected to DXMT server');
    console.log(`   Version: ${health.version}`);
    console.log(`   Status: ${health.status}`);
  } catch (error) {
    console.error('❌ Failed to connect to DXMT server');
    console.error('   Make sure DXMT server is running on localhost:9999');
    console.error('   Start with: cargo run -- serve --host 0.0.0.0 --port 9999');
    process.exit(1);
  }

  // For this example, we use FakeEmbeddings for testing
  // In production, use real embeddings like OpenAIEmbeddings
  console.log('\n📊 Step 2: Creating embeddings instance...');
  const embeddings = new FakeEmbeddings();
  console.log('✅ Using FakeEmbeddings (for testing)');
  console.log('   In production, use: new OpenAIEmbeddings()');

  const collectionName = 'langchain_example';

  try {
    // Example 1: Create vector store and add documents
    console.log('\n📊 Step 3: Creating DXMT vector store...');
    const vectorStore = new DXMTVectorStore(embeddings, {
      collectionName,
      host: 'localhost',
      port: 9999,
      dimension: 1536, // FakeEmbeddings uses 1536 dimensions
      distanceMetric: 'cosine',
    });

    console.log('✅ Vector store created');
    console.log(`   Collection: ${vectorStore.getCollectionName()}`);

    // Add documents
    console.log('\n📊 Step 4: Adding documents...');
    const documents = [
      new Document({
        pageContent: 'DXMT is a high-performance vector database',
        metadata: { category: 'tech', source: 'docs' },
      }),
      new Document({
        pageContent: 'LangChain enables building LLM applications',
        metadata: { category: 'tech', source: 'docs' },
      }),
      new Document({
        pageContent: 'Vector search finds similar content efficiently',
        metadata: { category: 'concept', source: 'blog' },
      }),
      new Document({
        pageContent: 'JavaScript and TypeScript are popular languages',
        metadata: { category: 'programming', source: 'wiki' },
      }),
    ];

    const ids = await vectorStore.addDocuments(documents);
    console.log(`✅ Added ${documents.length} documents`);
    console.log(`   IDs: ${ids.slice(0, 2).join(', ')}...`);

    // Example 2: Similarity search
    console.log('\n📊 Step 5: Performing similarity search...');
    const query = 'What is a vector database?';
    console.log(`   Query: "${query}"`);

    const searchResults = await vectorStore.similaritySearch(query, 3);
    console.log(`✅ Found ${searchResults.length} results:`);
    searchResults.forEach((doc, i) => {
      console.log(`   ${i + 1}. ${doc.pageContent.substring(0, 50)}...`);
      console.log(`      Category: ${doc.metadata.category}`);
    });

    // Example 3: Search with scores
    console.log('\n📊 Step 6: Search with similarity scores...');
    const scoredResults = await vectorStore.similaritySearchWithScore(
      query,
      3
    );
    console.log(`✅ Results with scores:`);
    scoredResults.forEach(([doc, score], i) => {
      console.log(`   ${i + 1}. Score: ${score.toFixed(4)}`);
      console.log(`      Content: ${doc.pageContent.substring(0, 50)}...`);
    });

    // Example 4: Filtered search
    console.log('\n📊 Step 7: Search with metadata filter...');
    const filteredResults = await vectorStore.similaritySearch(query, 5, {
      category: 'tech',
    });
    console.log(
      `✅ Found ${filteredResults.length} results (filtered by category='tech'):`
    );
    filteredResults.forEach((doc, i) => {
      console.log(`   ${i + 1}. ${doc.pageContent.substring(0, 50)}...`);
      console.log(`      Source: ${doc.metadata.source}`);
    });

    // Example 5: Using from_texts
    console.log('\n📊 Step 8: Creating vector store from texts...');
    const vectorStore2 = await DXMTVectorStore.fromTexts(
      [
        'TypeScript provides type safety',
        'Node.js enables server-side JavaScript',
        'React is a UI library',
      ],
      [
        { type: 'language', topic: 'typescript' },
        { type: 'runtime', topic: 'nodejs' },
        { type: 'library', topic: 'react' },
      ],
      embeddings,
      {
        collectionName: 'langchain_from_texts',
        host: 'localhost',
        port: 9999,
      }
    );

    console.log('✅ Vector store created from texts');

    const textsResults = await vectorStore2.similaritySearch(
      'programming languages',
      2
    );
    console.log(`   Found ${textsResults.length} results:`);
    textsResults.forEach((doc, i) => {
      console.log(`   ${i + 1}. ${doc.pageContent}`);
    });

    // Example 6: Using from_documents
    console.log('\n📊 Step 9: Creating vector store from documents...');
    const newDocs = [
      new Document({
        pageContent: 'Rust provides memory safety',
        metadata: { lang: 'rust' },
      }),
      new Document({
        pageContent: 'Go is great for concurrency',
        metadata: { lang: 'go' },
      }),
    ];

    const vectorStore3 = await DXMTVectorStore.fromDocuments(
      newDocs,
      embeddings,
      {
        collectionName: 'langchain_from_docs',
        host: 'localhost',
        port: 9999,
      }
    );

    console.log('✅ Vector store created from documents');

    const docsResults = await vectorStore3.similaritySearch(
      'systems programming',
      2
    );
    console.log(`   Found ${docsResults.length} results:`);
    docsResults.forEach((doc, i) => {
      console.log(`   ${i + 1}. ${doc.pageContent}`);
      console.log(`      Language: ${doc.metadata.lang}`);
    });

    // Cleanup
    console.log('\n📊 Step 10: Cleaning up...');
    const client = vectorStore.getClient();
    await client.deleteTable(collectionName);
    await client.deleteTable('langchain_from_texts');
    await client.deleteTable('langchain_from_docs');
    console.log('✅ Test collections deleted');

    console.log('\n' + '='.repeat(70));
    console.log('✅ All LangChain examples completed successfully!');
    console.log('='.repeat(70));
    console.log('\n📚 Key Takeaways:');
    console.log('  ✓ DXMT integrates seamlessly with LangChain');
    console.log('  ✓ Supports all LangChain VectorStore operations');
    console.log('  ✓ Metadata filtering for refined searches');
    console.log('  ✓ Multiple creation patterns (from_texts, from_documents)');
    console.log('  ✓ Type-safe with full TypeScript support');
    console.log('\n🚀 Ready for production use with LangChain!');
  } catch (error) {
    console.error('\n❌ Error:', error);

    // Cleanup on error
    try {
      const client = new DXMTClient({ host: 'localhost', port: 9999 });
      await client.deleteTable(collectionName);
      await client.deleteTable('langchain_from_texts');
      await client.deleteTable('langchain_from_docs');
      console.log('🧹 Cleaned up test collections');
    } catch {
      // Ignore cleanup errors
    }

    process.exit(1);
  }
}

// Run the example
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
