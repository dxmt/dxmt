/**
 * The client passes `rerank` through on create and on each search route, and omits it
 * otherwise. The server does the work; what the client owes is the field in the body, exactly
 * where the server reads it. `live.test.ts` has the live half.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';

const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock('cross-fetch', () => ({ default: fetchMock }));
import { DXMTClient } from '../src/client';

const RERANK = { model: 'rerank-v3.5', base_url: 'http://127.0.0.1:1/v2', format: 'cohere' as const };

function ok(data: unknown) {
  return new Response(JSON.stringify({ success: true, data }), { status: 200 });
}

function sentBody() {
  return JSON.parse(fetchMock.mock.calls.at(-1)![1].body as string);
}

describe('rerank passthrough', () => {
  afterEach(() => fetchMock.mockReset());

  it('sends the rerank config on create, and nothing when unset', async () => {
    fetchMock.mockResolvedValueOnce(ok({ name: 'docs' }));
    await new DXMTClient().createTable({ name: 'docs', dimension: 4, rerank: RERANK });
    expect(sentBody().rerank).toEqual(RERANK);

    fetchMock.mockResolvedValueOnce(ok({ name: 'plain' }));
    await new DXMTClient().createTable({ name: 'plain', dimension: 4 });
    expect(sentBody().rerank).toBeUndefined();
  });

  it('sends rerank beside text on search, and omits it otherwise', async () => {
    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().search({
      table_name: 'docs',
      text: 'memory management',
      k: 5,
      rerank: { top_n: 20 },
    });
    expect(sentBody().rerank).toEqual({ top_n: 20 });
    expect(sentBody().text).toBe('memory management');

    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().search({ table_name: 'docs', vector: [0.1, 0.2], k: 5 });
    expect(sentBody().rerank).toBeUndefined();
  });

  it('sends an empty rerank, which asks for the table defaults', async () => {
    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().search({ table_name: 'docs', text: 'memory', k: 5, rerank: {} });
    expect(sentBody().rerank).toEqual({});
  });

  it('sends rerank on full-text and hybrid search', async () => {
    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().fullTextSearch({
      table_name: 'docs',
      query: 'memory',
      k: 5,
      rerank: { field: 'body' },
    });
    expect(sentBody().rerank).toEqual({ field: 'body' });

    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().hybridSearch({
      table_name: 'docs',
      text_query: 'memory',
      k: 5,
      rerank: {},
    });
    expect(sentBody().rerank).toEqual({});
  });

  it('keeps rerank_score on the hit', async () => {
    fetchMock.mockResolvedValueOnce(
      ok({ hits: [{ id: 'a', score: 0.4, payload: {}, rerank_score: 0.93 }] })
    );
    const hits = await new DXMTClient().search({
      table_name: 'docs',
      text: 'memory',
      k: 1,
      rerank: {},
    });
    expect(hits[0].rerank_score).toBe(0.93);
    expect(hits[0].score).toBe(0.4);
  });
});
