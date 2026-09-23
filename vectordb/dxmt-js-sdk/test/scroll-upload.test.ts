/**
 * scroll() and upload() against a mocked transport: what goes on the wire, and what a failure
 * turns into.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';

const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock('cross-fetch', () => ({ default: fetchMock }));
import { DXMTClient } from '../src/client';
import type { Record as VectorRecord } from '../src/types';

const ok = (data: unknown) =>
  new Response(JSON.stringify({ success: true, data }), { status: 200 });
const bodyOf = (call: number) => JSON.parse(fetchMock.mock.calls[call][1].body);
const records = (n: number, from = 0): VectorRecord[] =>
  Array.from({ length: n }, (_, i) => ({ id: `r${from + i}`, vector: [1, 0], metadata: { i } }));

async function collect<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of it) out.push(x);
  return out;
}

describe('scroll', () => {
  afterEach(() => fetchMock.mockReset());

  it('follows next_cursor until it is null', async () => {
    fetchMock
      .mockResolvedValueOnce(ok({ records: [{ id: 'a', payload_json: '{"x":1}' }, { id: 'b' }], next_cursor: 'c1' }))
      .mockResolvedValueOnce(ok({ records: [{ id: 'c' }], next_cursor: null }));
    const client = new DXMTClient();
    const got = await collect(client.scroll('t', { batchSize: 2, filter: { must: [] } }));

    expect(got.map((r) => r.id)).toEqual(['a', 'b', 'c']);
    expect(got[0].metadata).toEqual({ x: 1 });
    expect(fetchMock.mock.calls[0][0]).toBe('http://localhost:9999/api/v1/tables/t/records/list');
    expect(bodyOf(0)).toEqual({ limit: 2, with_vector: false, filter: { must: [] } });
    expect(bodyOf(1)).toEqual({ limit: 2, with_vector: false, filter: { must: [] }, cursor: 'c1' });
  });

  it('refuses a server that answers a full page without a cursor', async () => {
    // Stopping there would look like a complete scroll of the first page.
    fetchMock.mockResolvedValueOnce(ok({ records: [{ id: 'a' }, { id: 'b' }] }));
    const client = new DXMTClient();
    await expect(collect(client.scroll('t', { batchSize: 2 }))).rejects.toThrow(/cursor/);
  });

  it('encodes the table name', async () => {
    fetchMock.mockResolvedValueOnce(ok({ records: [], next_cursor: null }));
    await collect(new DXMTClient().scroll('a/b c'));
    expect(fetchMock.mock.calls[0][0]).toBe('http://localhost:9999/api/v1/tables/a%2Fb%20c/records/list');
  });
});

describe('upload', () => {
  afterEach(() => fetchMock.mockReset());

  it('sends batches with metadata as payload', async () => {
    fetchMock.mockImplementation(async () => ok({ inserted: 1 }));
    const result = await new DXMTClient().upload('t', records(5), { batchSize: 2 });

    expect(result).toEqual({ uploaded: 5, batches: 3, retries: 0, failed: [] });
    const sent = [0, 1, 2].map(bodyOf);
    expect(sent.map((b) => b.records.length)).toEqual([2, 2, 1]);
    expect(sent[0].records[0]).toEqual({ id: 'r0', vector: [1, 0], payload: { i: 0 } });
  });

  it('draws from an async iterable', async () => {
    fetchMock.mockImplementation(async () => ok({}));
    async function* gen() {
      yield* records(3);
    }
    const result = await new DXMTClient().upload('t', gen(), { batchSize: 2 });
    expect(result.uploaded).toBe(3);
  });

  it('retries a 503 and a dropped connection, then succeeds', async () => {
    fetchMock
      .mockResolvedValueOnce(new Response('busy', { status: 503 }))
      .mockRejectedValueOnce(new Error('ECONNRESET'))
      .mockResolvedValueOnce(ok({}));
    const result = await new DXMTClient().upload('t', records(2), { backoffMs: 1 });
    expect(result).toEqual({ uploaded: 2, batches: 1, retries: 2, failed: [] });
  });

  it('does not retry what the server refused as wrong', async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ success: false, error: { code: 'DIMENSION_MISMATCH' } }), { status: 400 })
    );
    const result = await new DXMTClient().upload('t', records(2), { backoffMs: 1 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.uploaded).toBe(0);
    expect(result.failed).toEqual([
      { ids: ['r0', 'r1'], status: 400, error: expect.stringContaining('DIMENSION_MISMATCH') },
    ]);
  });

  it('gives up after maxRetries and says which ids', async () => {
    fetchMock.mockImplementation(async () => new Response('down', { status: 502 }));
    const result = await new DXMTClient().upload('t', records(3), { batchSize: 2, maxRetries: 2, backoffMs: 1 });
    expect(fetchMock).toHaveBeenCalledTimes(6);
    expect(result.retries).toBe(4);
    expect(result.failed.map((f) => f.ids)).toEqual([['r0', 'r1'], ['r2']]);
    expect(result.failed.every((f) => f.status === 502)).toBe(true);
  });

  it('keeps at most `parallel` requests in flight', async () => {
    let inFlight = 0;
    let most = 0;
    fetchMock.mockImplementation(async () => {
      inFlight += 1;
      most = Math.max(most, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight -= 1;
      return ok({});
    });
    const result = await new DXMTClient().upload('t', records(20), { batchSize: 2, parallel: 3 });
    expect(result.uploaded).toBe(20);
    expect(most).toBe(3);
  });
});

describe('hybridSearch fusion', () => {
  afterEach(() => fetchMock.mockReset());

  const call = async (params: Record<string, unknown>) => {
    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().hybridSearch({ table_name: 't', vector: [1, 0], text_query: 'x', ...params });
    return bodyOf(0);
  };

  it('sends no weight and no method when none was asked for', async () => {
    const body = await call({});
    expect(body.vector_weight).toBeUndefined();
    expect(body.fusion_method).toBeUndefined();
  });

  it('a weight selects the fusion that reads it', async () => {
    // It used to send `vector_weight: 0.5` by default and never a method, so the server fused
    // with `dbsf` -- which ignores the weight -- and the parameter did nothing.
    expect(await call({ vector_weight: 0.9 })).toMatchObject({
      vector_weight: 0.9,
      fusion_method: 'weighted_sum',
    });
  });

  it('an explicit method wins', async () => {
    expect(await call({ vector_weight: 0.9, fusion_method: 'reciprocal_rank_fusion' })).toMatchObject({
      vector_weight: 0.9,
      fusion_method: 'reciprocal_rank_fusion',
    });
  });
});
