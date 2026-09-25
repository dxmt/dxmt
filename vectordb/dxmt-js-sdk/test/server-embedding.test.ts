/**
 * The client half of server-side embedding.
 *
 * The server's behaviour is covered by `tests/server_embedding_test.rs`, which runs a real
 * OpenAI-protocol provider on loopback. What is left here is what the client decides: that
 * `embedding` reaches the create body, that a text query sends `text` and no `vector`, and
 * that asking for both is refused before the round trip rather than after it.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';

const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock('cross-fetch', () => ({ default: fetchMock }));
import { DXMTClient } from '../src/client';
import { DXMTValidationError } from '../src/errors';

const EMBEDDING = { model: 'text-embedding-3-small', base_url: 'http://127.0.0.1:1/v1' };

function ok(data: unknown) {
  return new Response(JSON.stringify({ success: true, data }), { status: 200 });
}

/** The JSON body of the last request the client made. */
function sentBody() {
  return JSON.parse(fetchMock.mock.calls.at(-1)![1].body as string);
}

describe('server-side embedding', () => {
  afterEach(() => fetchMock.mockReset());

  it('sends the embedding config on create', async () => {
    fetchMock.mockResolvedValueOnce(ok({ name: 'docs' }));
    await new DXMTClient().createTable({
      name: 'docs',
      dimension: 1536,
      embedding: EMBEDDING,
    });
    expect(sentBody().embedding).toEqual(EMBEDDING);
  });

  it('sends text and no vector for a text query', async () => {
    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().search({ table_name: 'docs', text: 'memory management', k: 5 });
    const body = sentBody();
    expect(body.text).toBe('memory management');
    expect(body.vector).toBeUndefined();
  });

  it('leaves a vector query alone', async () => {
    fetchMock.mockResolvedValueOnce(ok({ hits: [] }));
    await new DXMTClient().search({ table_name: 'docs', vector: [0.1, 0.2], k: 5 });
    const body = sentBody();
    expect(body.vector).toEqual([0.1, 0.2]);
    expect(body.text).toBeUndefined();
  });

  it('refuses both, and neither, without a request', async () => {
    const client = new DXMTClient();
    await expect(
      client.search({ table_name: 'docs', vector: [0.1], text: 'memory' })
    ).rejects.toThrow(DXMTValidationError);
    await expect(client.search({ table_name: 'docs' })).rejects.toThrow(DXMTValidationError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('inserts records carrying text instead of a vector', async () => {
    fetchMock.mockResolvedValueOnce(ok({ count: 1 }));
    await new DXMTClient().insertRecords({
      table_name: 'docs',
      records: [{ id: '1', text: 'rust borrows memory' }],
    });
    expect(sentBody().records[0]).toMatchObject({ id: '1', text: 'rust borrows memory' });
  });
});
