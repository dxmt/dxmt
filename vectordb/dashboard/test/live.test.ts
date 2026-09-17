/**
 * The client against a real server, skipped when there is not one.
 *
 * The unit tests above assert what the client does with a response; this asserts that the
 * responses are what the client believes. That distinction matters here more than usual,
 * because three of the shapes it handles — `payload_json` as a string, `tookms` with no
 * underscore, the plain-text extractor rejection — are ones nobody would invent, so a hand
 * written fixture that drifts from the server looks perfectly reasonable while being wrong.
 *
 *     cd vectordb && cargo run --release -- serve
 *     cd dashboard && DXMT_LIVE_URL=http://127.0.0.1:9999 npx vitest run test/live.test.ts
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { ApiFailure, unwrap } from '../src/api/envelope';
import { normalizeRecord, vectorFor } from '../src/api/normalize';
import type { RecordPage, SearchResult, ServerStatus, TableWithCount } from '../src/api/types';

const base = process.env.DXMT_LIVE_URL;
const suite = base ? describe : describe.skip;
const table = `dash_live_${Date.now()}`;

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    method,
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const { value } = await unwrap<T>(response);
  return value;
}

suite('the wire is the shape the client expects', () => {
  beforeAll(async () => {
    // Writing here is deliberate and is the one exception: a test fixture is not the dashboard.
    // The client module under test still cannot express these calls, which is the point.
    await call('POST', '/api/v1/tables', {
      name: table,
      dimension: 4,
      distance_metric: 'cosine',
    });
    await call('POST', `/api/v1/tables/${table}/records`, {
      records: [
        { id: 'a', vector: [1, 0, 0, 0], payload: { rank: 1, city: 'London' } },
        { id: 'b', vector: [0, 1, 0, 0], payload: { rank: 2, city: 'Berlin' } },
        { id: 'c', vector: [0.9, 0.1, 0, 0], payload: { rank: 3, city: 'London' } },
      ],
    });
  });

  afterAll(async () => {
    if (base) await fetch(`${base}/api/v1/tables/${table}`, { method: 'DELETE' });
  });

  it('sends a listed record\'s payload as a JSON string, not an object', async () => {
    const page = await call<RecordPage>('POST', `/api/v1/tables/${table}/records/list`, {
      limit: 10,
    });
    const record = page.records[0]!;
    expect(typeof record.payload_json).toBe('string');
    expect(record).not.toHaveProperty('payload');
    // And always the full vector, with no flag to suppress it — the constraint the whole
    // record browser is designed around.
    expect(Array.isArray(record.vector)).toBe(true);
    expect(record.vector).toHaveLength(4);
  });

  it('sends a search hit\'s payload as an object, unlike a listed record', async () => {
    const result = await call<SearchResult>('POST', `/api/v1/tables/${table}/search`, {
      vector: [1, 0, 0, 0],
      top_k: 2,
    });
    expect(typeof result.hits[0]!.payload).toBe('object');
    // No underscore. Renaming this field would break every timing the dashboard prints.
    expect(typeof result.tookms).toBe('number');
  });

  it('normalises both shapes to the same thing', async () => {
    const page = await call<RecordPage>('POST', `/api/v1/tables/${table}/records/list`, {
      limit: 10,
    });
    const listed = page.records.find((r) => r.id === 'a')!;
    const record = normalizeRecord(table, listed);

    expect(record.payload).toEqual({ rank: 1, city: 'London' });
    expect(Array.from(vectorFor(table, 'a') ?? [])).toEqual([1, 0, 0, 0]);
  });

  it('nests the table one level down in GET /tables/{t}', async () => {
    const detail = await call<TableWithCount>('GET', `/api/v1/tables/${table}`);
    expect(detail.table.name).toBe(table);
    expect(detail.record_count).toBe(3);
  });

  it('exposes the snapshot fields the overview page derives its numbers from', async () => {
    const status = await call<ServerStatus>('GET', '/status');
    for (const key of [
      'wal_next_sequence',
      'snapshot_wal_next_sequence',
      'durability_degraded',
      'resident_table_count',
      'table_count',
    ]) {
      expect(status.persistence, key).toHaveProperty(key);
    }
  });

  it('refuses an unknown search_mode as plain text, which is how the probe detects absence', async () => {
    const response = await fetch(`${base}/api/v1/tables/${table}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vector: [1, 0, 0, 0], top_k: 1, search_mode: 'nonesuch' }),
    });
    const failure = (await unwrap(response).catch((e: unknown) => e)) as ApiFailure;

    expect(failure).toBeInstanceOf(ApiFailure);
    expect(failure.failure.kind).toBe('text');
    expect(failure.isExtractorRejection).toBe(true);
  });

  it('answers search_mode "exact" with the true nearest neighbours', async () => {
    const exact = await call<SearchResult>('POST', `/api/v1/tables/${table}/search`, {
      vector: [1, 0, 0, 0],
      top_k: 3,
      search_mode: 'exact',
    });
    // By construction: a is the query, c is 0.9/0.1 away, b is perpendicular.
    expect(exact.hits.map((h) => h.id)).toEqual(['a', 'c', 'b']);
    expect(exact.hits[0]!.score).toBeCloseTo(1, 4);
  });
});
