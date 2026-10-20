/**
 * Documents from the client's side. Chunking and replacement are the server's
 * (`tests/interfaces/documents_test.rs`); here: the encoded path, a body with only what was
 * set, and a missing document read as `null` while a missing table still throws.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';

const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock('cross-fetch', () => ({ default: fetchMock }));
import { DXMTClient } from '../src/client';

function reply(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status });
}

describe('documents', () => {
  afterEach(() => fetchMock.mockReset());

  it('puts to an encoded path with only the fields that were set', async () => {
    fetchMock.mockResolvedValueOnce(
      reply(200, { success: true, data: { doc_id: 'a/b.md', chunks: 3, removed: 0 } })
    );
    const written = await new DXMTClient().putDocument({
      table_name: 'kb', doc_id: 'notes/a b.md', text: '全文', metadata: { title: 'T' },
    });
    expect(written.chunks).toBe(3);
    const [url, init] = fetchMock.mock.calls.at(-1)!;
    expect(url).toContain('/tables/kb/documents/notes%2Fa%20b.md');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({ text: '全文', metadata: { title: 'T' } });
  });

  it('reads a missing document as null and a missing table as an error', async () => {
    fetchMock.mockResolvedValueOnce(reply(404, {
      success: false,
      error: { code: 'NOT_FOUND', message: "Record not found: no document 'd' in table 'kb'" },
    }));
    expect(await new DXMTClient().getDocument('kb', 'd')).toBeNull();
    fetchMock.mockResolvedValueOnce(reply(404, {
      success: false, error: { code: 'NOT_FOUND', message: 'Table not found: nope' },
    }));
    await expect(new DXMTClient().getDocument('nope', 'd')).rejects.toThrow();
  });

  it('returns the deleted chunk count', async () => {
    fetchMock.mockResolvedValueOnce(reply(200, { success: true, data: { deleted_chunks: 4 } }));
    expect(await new DXMTClient().deleteDocument('kb', 'd')).toBe(4);
  });
});
