/**
 * The LangChain.js store against a real server.
 *
 * `src/langchain/` had no test of any kind -- not mocked, not live -- and `examples/
 * langchain-usage.ts` is run by nothing. Everything below needs a server (`DXMT_LIVE_URL`)
 * and nothing else: the embedder is a hashed bag of words, the same idea the Python
 * packages use, so the suite runs in CI with no key and retrieval assertions still mean
 * something (two texts sharing words share a direction).
 */
import { afterAll, describe, expect, it } from 'vitest';
import { Document } from '@langchain/core/documents';
import type { EmbeddingsInterface } from '@langchain/core/embeddings';
import { DXMTVectorStore } from '../src/langchain';

const liveUrl = process.env.DXMT_LIVE_URL;
const suite = liveUrl ? describe : describe.skip;

const DIM = 64;

function embed(text: string): number[] {
  const v = new Array(DIM).fill(0);
  for (const word of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    let h = 2166136261;
    for (const ch of word) h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
    v[h % DIM] += 1;
  }
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

const embeddings: EmbeddingsInterface = {
  embedDocuments: async (texts) => texts.map(embed),
  embedQuery: async (text) => embed(text),
};

function connection() {
  const url = new URL(liveUrl || 'http://127.0.0.1:9999');
  return {
    host: url.hostname,
    port: Number(url.port || 80),
    ssl: url.protocol === 'https:',
    apiKey: process.env.DXMT_API_KEY,
  };
}

suite('LangChain.js store', () => {
  const collectionName = `js_lc_${Date.now()}`;
  const store = new DXMTVectorStore(embeddings, { ...connection(), collectionName, dimension: DIM });

  afterAll(async () => {
    try { await store.client.deleteTable(collectionName); } catch { /* best effort */ }
  });

  it('adds documents, searches them, and keeps the ids a caller gave', async () => {
    const docs = [
      new Document({ id: 'cats', pageContent: 'cats purr and chase mice', metadata: { kind: 'pet' } }),
      new Document({ id: 'dogs', pageContent: 'dogs bark and fetch sticks', metadata: { kind: 'pet' } }),
      new Document({ id: 'rust', pageContent: 'rust compiles to native code', metadata: { kind: 'lang' } }),
    ];
    const ids = await store.addDocuments(docs);
    // `Document.id` used to be ignored in favour of a generated one, so re-adding a document
    // under its own id made a second copy instead of replacing the first.
    expect(ids).toEqual(['cats', 'dogs', 'rust']);
    expect(await store.client.count(collectionName)).toBe(3);
    await store.addDocuments([docs[0]]);
    expect(await store.client.count(collectionName)).toBe(3);

    const hits = await store.similaritySearch('mice and cats', 1);
    expect(hits[0].pageContent).toContain('cats');
    expect(hits[0].metadata.kind).toBe('pet');
    expect(hits[0].metadata.text).toBeUndefined();

    const scored = await store.similaritySearchWithScore('rust native', 1);
    expect(scored[0][0].pageContent).toContain('rust');
    expect(scored[0][1]).toBeGreaterThan(0);

    const pets = await store.similaritySearch('anything', 3, { kind: 'pet' });
    expect(pets.map((d) => d.metadata.kind)).toEqual(['pet', 'pet']);
  });

  it('refuses a filter it cannot express rather than sending something else', async () => {
    await expect(store.similaritySearch('x', 1, { year: { $gte: 2020 } })).rejects.toThrow(/equality on scalars/);
  });

  it('deletes by ids and by filter, and says so when given neither', async () => {
    await store.delete({ ids: ['rust'] });
    expect(await store.client.count(collectionName)).toBe(2);
    await store.delete({ filter: { kind: 'pet' } });
    expect(await store.client.count(collectionName)).toBe(0);
    await expect(store.delete()).rejects.toThrow(/ids.*filter/);
  });

  it('does not treat an unreachable server as a missing table', async () => {
    // `ensureCollection` caught every `getTable` failure as "absent" and went on to detect
    // the dimension and create the table. Against an unreachable server both fail the same
    // way, so the observable difference is the dimension probe: the old code embedded
    // 'test' before trying to create; the new one re-raises the connection error first.
    let probes = 0;
    const counting: EmbeddingsInterface = {
      embedDocuments: embeddings.embedDocuments,
      embedQuery: async (text) => { probes += 1; return embed(text); },
    };
    const broken = new DXMTVectorStore(counting, { host: '127.0.0.1', port: 1, collectionName: 'nope', timeout: 2000 });
    const err = await broken.addDocuments([new Document({ pageContent: 'x' })]).catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    expect(probes).toBe(0);
  });
});
