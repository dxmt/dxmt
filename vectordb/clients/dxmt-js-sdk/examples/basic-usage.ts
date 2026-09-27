/**
 * Basic Usage Example for DXMT JavaScript SDK.
 *
 * This example demonstrates:
 * 1. Creating a client
 * 2. Creating a table
 * 3. Inserting records
 * 4. Searching for similar vectors
 * 5. Getting records
 * 6. Deleting records
 * 7. Deleting a table
 */

import { DXMTClient } from '../src';

async function main() {
  console.log('='.repeat(70));
  console.log('DXMT JavaScript SDK - Basic Usage Example');
  console.log('='.repeat(70));

  // Step 1: Create client
  console.log('\n📊 Step 1: Creating DXMT client...');
  const client = new DXMTClient({
    host: 'localhost',
    port: 9999,
    timeout: 30000,
  });

  try {
    // Check server health
    const health = await client.healthCheck();
    console.log(`✅ Connected to DXMT server`);
    console.log(`   Version: ${health.version}`);
    console.log(`   Status: ${health.status}`);
  } catch (error) {
    console.error('❌ Failed to connect to DXMT server');
    console.error('   Make sure DXMT server is running on localhost:9999');
    console.error('   Start with: cargo run -- serve --host 0.0.0.0 --port 9999');
    process.exit(1);
  }

  const tableName = 'js_sdk_example';

  try {
    // Step 2: Create table
    console.log('\n📊 Step 2: Creating table...');
    const table = await client.createTable({
      name: tableName,
      dimension: 3,
      distance_metric: 'cosine',
      description: 'JavaScript SDK example table',
    });

    console.log('✅ Table created successfully');
    console.log(`   Name: ${table.name}`);
    console.log(`   Dimension: ${table.dimension}`);
    console.log(`   Distance Metric: ${table.distance_metric}`);

    // Step 3: Insert records
    console.log('\n📊 Step 3: Inserting records...');
    await client.insertRecords({
      table_name: tableName,
      records: [
        {
          id: 'vec1',
          vector: [0.1, 0.2, 0.3],
          metadata: { text: 'First vector', category: 'example' },
        },
        {
          id: 'vec2',
          vector: [0.4, 0.5, 0.6],
          metadata: { text: 'Second vector', category: 'example' },
        },
        {
          id: 'vec3',
          vector: [0.7, 0.8, 0.9],
          metadata: { text: 'Third vector', category: 'test' },
        },
        {
          id: 'vec4',
          vector: [0.15, 0.25, 0.35],
          metadata: { text: 'Fourth vector', category: 'example' },
        },
      ],
      mode: 'replace',
    });

    console.log('✅ Inserted 4 records');

    // Step 4: Search for similar vectors
    console.log('\n📊 Step 4: Searching for similar vectors...');
    const queryVector = [0.1, 0.2, 0.3];
    console.log(`   Query vector: [${queryVector.join(', ')}]`);

    const results = await client.search({
      table_name: tableName,
      vector: queryVector,
      k: 3,
      include_metadata: true,
    });

    console.log(`✅ Found ${results.length} results:`);
    results.forEach((result, i) => {
      console.log(`   ${i + 1}. ID: ${result.id}`);
      console.log(`      Score: ${result.score.toFixed(4)}`);
      console.log(`      Metadata: ${JSON.stringify(result.metadata)}`);
    });

    // Step 5: Search with filter
    console.log('\n📊 Step 5: Searching with metadata filter...');
    const filteredResults = await client.search({
      table_name: tableName,
      vector: queryVector,
      k: 5,
      filter: {
        must: [
          {
            key: 'category',
            operation: { type: 'Equal', value: 'example' },
          },
        ],
      },
    });

    console.log(`✅ Found ${filteredResults.length} results (filtered by category=example):`);
    filteredResults.forEach((result, i) => {
      console.log(`   ${i + 1}. ID: ${result.id} | Category: ${result.metadata?.category}`);
    });

    // Step 6: Get specific record
    console.log('\n📊 Step 6: Getting specific record...');
    const record = await client.getRecord(tableName, 'vec1');
    console.log('✅ Retrieved record:');
    console.log(`   ID: ${record.id}`);
    console.log(`   Vector: [${record.vector.join(', ')}]`);
    console.log(`   Metadata: ${JSON.stringify(record.metadata)}`);

    // Step 7: Get table info
    console.log('\n📊 Step 7: Getting table information...');
    const tableInfo = await client.getTable(tableName);
    console.log('✅ Table info:');
    console.log(`   Name: ${tableInfo.name}`);
    console.log(`   Record count: ${tableInfo.record_count}`);
    console.log(`   Dimension: ${tableInfo.dimension}`);

    // Step 8: List all tables
    console.log('\n📊 Step 8: Listing all tables...');
    const tables = await client.listTables();
    console.log(`✅ Found ${tables.length} tables:`);
    tables.forEach((name) => {
      console.log(`   - ${name}`);
    });

    // Step 9: Delete specific records
    console.log('\n📊 Step 9: Deleting specific records...');
    await client.deleteRecords({
      table_name: tableName,
      record_ids: ['vec3', 'vec4'],
    });
    console.log('✅ Deleted records: vec3, vec4');

    // Verify deletion
    const afterDelete = await client.search({
      table_name: tableName,
      vector: queryVector,
      k: 10,
    });
    console.log(`   Remaining records: ${afterDelete.length}`);

    // Step 10: Cleanup - delete table
    console.log('\n📊 Step 10: Cleaning up...');
    await client.deleteTable(tableName);
    console.log('✅ Table deleted successfully');

    console.log('\n' + '='.repeat(70));
    console.log('✅ All examples completed successfully!');
    console.log('='.repeat(70));
  } catch (error) {
    console.error('\n❌ Error:', error);

    // Cleanup on error
    try {
      await client.deleteTable(tableName);
      console.log('🧹 Cleaned up test table');
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
