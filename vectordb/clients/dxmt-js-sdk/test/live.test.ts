import { afterAll, describe, expect, it } from 'vitest';
import { DXMTClient } from '../src/client';

const liveUrl = process.env.DXMT_LIVE_URL;
const suite = liveUrl ? describe : describe.skip;

function liveClient(): DXMTClient {
  const url = new URL(liveUrl || 'http://127.0.0.1:9999');
  return new DXMTClient({
    host: url.hostname,
    port: Number(url.port || (url.protocol === 'https:' ? 443 : 80)),
    ssl: url.protocol === 'https:',
    // A server started with DXMT_API_KEY needs it on every request.
    apiKey: process.env.DXMT_API_KEY,
  });
}

suite('DXMT live REST contract', () => {
  const tableName = `js_live_${Date.now()}`;
  const client = liveClient();

  afterAll(async () => {
    try {
      await client.deleteTable(tableName);
    } catch {
      // Keep cleanup best-effort when a server is interrupted.
    }
  });

  it('supports table, record, search, update, delete and health metrics', async () => {
    const health = await client.healthCheck();
    expect(health.ready).toBe(true);

    const table = await client.createTable({
      name: tableName,
      dimension: 2,
      distance_metric: 'cosine',
    });
    expect(table.name).toBe(tableName);

    const inserted = await client.insertRecords({
      table_name: tableName,
      records: [{ id: 'r1', vector: [1, 0], metadata: { kind: 'probe' } }],
    });
    expect(inserted.inserted).toBe(1);

    const record = await client.getRecord(tableName, 'r1');
    expect(record.metadata).toEqual({ kind: 'probe' });

    const hits = await client.search({ table_name: tableName, vector: [1, 0], k: 1 });
    expect(hits).toHaveLength(1);
    expect(hits[0].id).toBe('r1');

    const update = await client.updateRecord({
      table_name: tableName,
      record_id: 'r1',
      metadata: { kind: 'updated' },
    });
    expect(update.updated).toBe(1);
    expect((await client.getRecord(tableName, 'r1')).metadata).toEqual({ kind: 'updated' });

    const metrics = await client.healthMetrics();
    expect(metrics.storage?.ready).toBe(true);

    await client.deleteRecords({ table_name: tableName, record_ids: ['r1'] });
  }, 30_000);
});

suite('DXMT live bulk load and export', () => {
  const tableName = `js_live_scroll_${Date.now()}`;
  const client = liveClient();

  afterAll(async () => {
    try {
      await client.deleteTable(tableName);
    } catch {
      // Keep cleanup best-effort when a server is interrupted.
    }
  });

  it('uploads in parallel and scrolls every record back exactly once', async () => {
    await client.createTable({ name: tableName, dimension: 4, distance_metric: 'cosine' });
    const all = Array.from({ length: 1000 }, (_, i) => ({
      id: `r${i}`,
      vector: [1, i % 7, i % 11, 1],
      metadata: { parity: i % 2 === 0 ? 'even' : 'odd' },
    }));
    const result = await client.upload(tableName, all, { batchSize: 128, parallel: 4 });
    expect(result).toEqual({ uploaded: 1000, batches: 8, retries: 0, failed: [] });

    // Writing between pages must neither repeat nor drop what was there from the start.
    const seen: string[] = [];
    let extra = 0;
    for await (const record of client.scroll(tableName, { batchSize: 97 })) {
      seen.push(String(record.id));
      if (seen.length % 97 === 0) {
        await client.insertRecords({ table_name: tableName, records: [{ id: `new${extra++}`, vector: [1, 1, 1, 1] }] });
      }
    }
    const original = seen.filter((id) => id.startsWith('r'));
    expect(new Set(seen).size).toBe(seen.length);
    expect(original.length).toBe(1000);

    const even = [];
    for await (const record of client.scroll(tableName, {
      batchSize: 64,
      withVector: true,
      filter: { must: [{ key: 'parity', operation: { type: 'Equal', value: 'even' } }] },
    })) {
      even.push(record);
    }
    expect(even.length).toBe(500);
    expect(even.every((r) => r.metadata?.parity === 'even' && r.vector?.length === 4)).toBe(true);
  });
});
