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
