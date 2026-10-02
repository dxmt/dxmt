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

suite('a record that came out of the client goes back in', () => {
  // `scroll()`, `getRecord()` and every search hit hand back `created_at`, `updated_at`,
  // `version`, `status`, `payload_json` and `score` beside the fields a write takes. The
  // server's request type is `deny_unknown_fields`, and `upload()` used to spread the whole
  // record, so read-modify-write -- the most ordinary thing to do with a record -- was a 400
  // on every batch, reported as `uploaded: 0` with the refusal in `failed[]`, not thrown.
  const tableName = `js_roundtrip_${Date.now()}`;
  const client = liveClient();
  afterAll(async () => {
    try { await client.deleteTable(tableName); } catch { /* best effort */ }
  });

  it('scroll -> edit -> upload and insertRecords both succeed', async () => {
    await client.createTable({ name: tableName, dimension: 4, distance_metric: 'cosine' });
    await client.insertRecords({
      table_name: tableName,
      records: [
        { id: 'a', vector: [1, 0, 0, 0], metadata: { k: 1 } },
        { id: 'b', vector: [0, 1, 0, 0], metadata: { k: 2 } },
      ],
    });
    const rows = [];
    for await (const r of client.scroll(tableName, { withVector: true })) rows.push(r);
    expect(rows.length).toBe(2);
    for (const r of rows) r.metadata = { ...(r.metadata ?? {}), touched: true };

    const uploaded = await client.upload(tableName, rows);
    expect(uploaded.failed, JSON.stringify(uploaded.failed).slice(0, 300)).toEqual([]);
    expect(uploaded.uploaded).toBe(2);

    const back = await client.insertRecords({ table_name: tableName, records: rows });
    expect(back.count).toBe(2);

    const a = await client.getRecord(tableName, 'a');
    expect(a.metadata?.touched).toBe(true);
    expect(a.metadata?.k).toBe(1);
  });
});

suite('an error carries the server\'s code', () => {
  // The class is chosen by status, the code was the class's constant: `DIMENSION_MISMATCH`
  // arrived as `VALIDATION_ERROR` and `TABLE_NOT_FOUND` as `NOT_FOUND`, so a caller could
  // see that the server refused and not why.
  const tableName = `js_errors_${Date.now()}`;
  const client = liveClient();
  afterAll(async () => {
    try { await client.deleteTable(tableName); } catch { /* best effort */ }
  });

  it('keeps TABLE_NOT_FOUND, DIMENSION_MISMATCH and the HTTP status', async () => {
    await client.createTable({ name: tableName, dimension: 4 });
    const missing = await client.getTable('js_no_such_table_zz').catch((e) => e);
    expect(missing.code).toBe('TABLE_NOT_FOUND');
    expect(missing.status).toBe(404);

    const wrongDim = await client
      .insertRecords({ table_name: tableName, records: [{ id: 'x', vector: [1, 0] }] })
      .catch((e) => e);
    expect(wrongDim.status).toBe(400);
    expect(String(wrongDim.code)).toMatch(/DIMENSION|VALIDATION_FAILED/);
    expect(wrongDim.code).not.toBe('VALIDATION_ERROR');
  });
});

suite('every method the other suites do not reach', () => {
  // Twelve methods had no test of any kind, live or mocked. Each is called once and judged
  // by its effect on the server, not by returning without throwing.
  const tableName = `js_sweep_${Date.now()}`;
  const imported = `${tableName}_imp`;
  const client = liveClient();
  afterAll(async () => {
    for (const t of [tableName, imported]) {
      try { await client.deleteTable(t); } catch { /* best effort */ }
    }
  });

  it('listTables, getTableStats, searchGroups, aliases, payload index, export/import, facets, sparse', async () => {
    await client.createTable({ name: tableName, dimension: 4 });
    await client.insertRecords({
      table_name: tableName,
      records: [0, 1, 2, 3].map((i) => ({
        id: `r${i}`,
        vector: [i === 0 ? 1 : 0.1, i === 1 ? 1 : 0.1, i === 2 ? 1 : 0.1, i === 3 ? 1 : 0.1],
        metadata: { g: i % 2 === 0 ? 'even' : 'odd', n: i },
        sparse_vector: { indices: [i % 2], values: [1], dimension: 4 },
      })),
    });

    const tables = await client.listTables();
    expect(tables.map((t) => t.name)).toContain(tableName);

    const stats = await client.getTableStats(tableName);
    expect(stats.record_count).toBe(4);

    const groups = await client.searchGroups(tableName, [1, 0, 0, 0], 'g', { groups: 2, groupSize: 1 });
    expect(groups.groups.length).toBe(2);

    const alias = `${tableName}_alias`;
    await client.setAlias(alias, tableName);
    expect((await client.aliases())[alias]).toBe(tableName);
    expect(await client.count(alias)).toBe(4);
    await client.deleteAlias(alias);
    expect((await client.aliases())[alias]).toBeUndefined();

    const report = await client.payloadIndex(tableName);
    expect(report.built).toBe(true);
    await client.dropPayloadIndexField(tableName, 'n');
    const refused = await client
      .search({ table_name: tableName, vector: [1, 0, 0, 0], top_k: 2,
        filter: { must: [{ key: 'n', operation: { type: 'Equal', value: 1 } }] } })
      .catch((e) => e);
    expect(refused).toBeInstanceOf(Error);
    await client.indexPayloadField(tableName, 'n');
    const filtered = await client.search({ table_name: tableName, vector: [1, 0, 0, 0], top_k: 2,
      filter: { must: [{ key: 'n', operation: { type: 'Equal', value: 1 } }] } });
    expect(filtered.map((h) => h.id)).toEqual(['r1']);

    const facets = await client.facets(tableName, 'g');
    expect(facets.hits.map((h) => h.value).sort()).toEqual(['even', 'odd']);

    const sparse = await client.sparseSearch(tableName, { indices: [1], values: [1], dimension: 4 }, { limit: 4 });
    expect(sparse.map((h) => h.id).sort()).toEqual(['r1', 'r3']);

    const root = process.env.DXMT_BACKUP_ROOT;
    if (root) {
      await client.exportTable(tableName, `${root}/${tableName}.dxmt`);
      await client.importTable(`${root}/${tableName}.dxmt`, { name: imported });
      expect(await client.count(imported)).toBe(4);
    }
  });
});
