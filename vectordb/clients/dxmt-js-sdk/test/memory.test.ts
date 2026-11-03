/**
 * Memory from the client's side. What memory does -- the table made on first use, recall by
 * meaning and keyword, superseding -- is the server's (`tests/interfaces/mcp_stdio_test.rs`);
 * here: the paths, bodies carrying only what was set, an encoded id, and the unwrapped answers.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';

const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock('cross-fetch', () => ({ default: fetchMock }));
import { DXMTClient, DXMTValidationError } from '../src';

function reply(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status });
}

describe('memory', () => {
  afterEach(() => fetchMock.mockReset());

  it('remembers with only the fields that were set', async () => {
    fetchMock.mockResolvedValueOnce(
      reply(200, { success: true, data: { id: 'm2', superseded: ['m1'] } })
    );
    const kept = await new DXMTClient().remember({
      text: 'Prefers answers in Chinese', kind: 'preference', supersedes: ['m1'], source: undefined,
    });
    expect(kept).toEqual({ id: 'm2', superseded: ['m1'] });
    const [url, init] = fetchMock.mock.calls.at(-1)!;
    expect(url).toMatch(/\/api\/v1\/memories$/);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({
      text: 'Prefers answers in Chinese', kind: 'preference', supersedes: ['m1'],
    });
  });

  it('refuses an empty memory before sending it', async () => {
    await expect(new DXMTClient().remember({ text: '  ' })).rejects.toThrow(DXMTValidationError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('recalls and returns the memories themselves', async () => {
    const memory = { id: 'm1', text: 'tomatoes', kind: 'fact', at: 1, score: 0.9 };
    fetchMock.mockResolvedValueOnce(reply(200, { success: true, data: { memories: [memory] } }));
    const found = await new DXMTClient().recall({ query: 'garden', tags: ['home'], since: 0 });
    expect(found).toEqual([memory]);
    const [url, init] = fetchMock.mock.calls.at(-1)!;
    expect(url).toMatch(/\/memories\/recall$/);
    expect(JSON.parse(init.body)).toEqual({ query: 'garden', tags: ['home'], since: 0 });
  });

  it('forgets by an encoded id and lists with a query string', async () => {
    fetchMock.mockResolvedValueOnce(reply(200, { success: true, data: { forgotten: 1 } }));
    expect(await new DXMTClient().forget('a/b c')).toBe(1);
    expect(fetchMock.mock.calls.at(-1)![0]).toMatch(/\/memories\/a%2Fb%20c$/);
    expect(fetchMock.mock.calls.at(-1)![1].method).toBe('DELETE');

    fetchMock.mockResolvedValueOnce(
      reply(200, { success: true, data: { memories: [], total: 0 } })
    );
    expect(await new DXMTClient().listMemories({ kind: 'fact', limit: 5 })).toEqual({
      memories: [], total: 0,
    });
    expect(fetchMock.mock.calls.at(-1)![0]).toMatch(/\/memories\?limit=5&offset=0&kind=fact$/);
  });
});
