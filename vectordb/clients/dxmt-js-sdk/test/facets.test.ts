/**
 * Value counts from the client's side.
 *
 * The counting is the server's and is tested there. What is here is what the client decides:
 * the request shape, the default limit, and that the two honesty numbers survive the trip —
 * a client that dropped `covered` would leave every caller unable to tell that their chart
 * omits part of the corpus.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';

const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock('cross-fetch', () => ({ default: fetchMock }));
import { DXMTClient } from '../src/client';
import { DXMTValidationError } from '../src/errors';

function ok(data: unknown) {
  return new Response(JSON.stringify({ success: true, data }), { status: 200 });
}

function sentBody() {
  return JSON.parse(fetchMock.mock.calls.at(-1)![1].body as string);
}

const ANSWER = {
  hits: [{ value: 'rust', count: 2 }],
  distinct_values: 1,
  covered: 2,
  matched: 3,
};

describe('facets', () => {
  afterEach(() => fetchMock.mockReset());

  it('sends the key and a default limit', async () => {
    fetchMock.mockResolvedValueOnce(ok(ANSWER));
    await new DXMTClient().facets('docs', 'topic');
    expect(sentBody()).toEqual({ key: 'topic', limit: 20 });
    expect(fetchMock.mock.calls.at(-1)![0]).toContain('/tables/docs/facets');
  });

  it('passes a filter and a limit through', async () => {
    fetchMock.mockResolvedValueOnce(ok(ANSWER));
    const filter = { must: [{ key: 'lang', operation: { type: 'Equal' as const, value: 'en' } }] };
    await new DXMTClient().facets('docs', 'topic', { filter, limit: 5 });
    expect(sentBody()).toEqual({ key: 'topic', limit: 5, filter });
  });

  it('hands back covered and matched, not only the buckets', async () => {
    fetchMock.mockResolvedValueOnce(ok(ANSWER));
    const result = await new DXMTClient().facets('docs', 'topic');
    expect(result.covered).toBe(2);
    expect(result.matched).toBe(3);
    expect(result.hits[0]).toEqual({ value: 'rust', count: 2 });
  });

  it('refuses an empty key before the round trip', async () => {
    await expect(new DXMTClient().facets('docs', '')).rejects.toThrow(DXMTValidationError);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
