/**
 * The two record shapes, and the reasons the normaliser exists.
 *
 * A listed record and a search hit describe the same thing in different words — one sends the
 * payload as a JSON string, the other as an object — and the asymmetry is real enough that
 * `tests/api/test_crud.py` opens with a note about it. Everything downstream takes one type, so
 * this is where that is decided.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { forgetVectors, normalizeHit, normalizeRecord, vectorFor } from '../src/api/normalize';
import type { WireHit, WireRecord } from '../src/api/types';

const listed: WireRecord = {
  id: 'r1',
  vector: [1, 0, 0, 0],
  sparse_vector: null,
  named_vectors: null,
  payload_json: '{"rank":1,"city":"London"}',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  version: 1,
  status: 'Active',
};

const hit: WireHit = { id: 'r1', score: 1.0, payload: { rank: 1, city: 'London' }, vector: null };

describe('normalize', () => {
  beforeEach(() => forgetVectors('t'));

  it('turns payload_json into the same object a hit already carries', () => {
    expect(normalizeRecord('t', listed).payload).toEqual(normalizeHit('t', hit).payload);
  });

  it('keeps a malformed payload visible instead of throwing or blanking the cell', () => {
    const broken = normalizeRecord('t', { ...listed, payload_json: '{not json' });
    expect(broken.payload).toBeNull();
    expect(broken.payloadRaw).toBe('{not json');
    expect(broken.payloadError).toBeTruthy();
  });

  it('handles a record with no payload at all', () => {
    const none = normalizeRecord('t', { ...listed, payload_json: null });
    expect(none.payload).toBeNull();
    expect(none.payloadError).toBeNull();
  });

  it('keeps the vector out of the returned object and in the side map', () => {
    const record = normalizeRecord('t', { ...listed, vector: [3, 4, 0, 0] });

    // What React holds: an id, a dimension, a norm, a short preview. Not 768 boxed numbers.
    expect(record.dim).toBe(4);
    expect(record.norm).toBeCloseTo(5, 6);
    expect(record.preview).toEqual([3, 4, 0, 0]);
    expect(Object.values(record).some((v) => v instanceof Float32Array)).toBe(false);

    // And the full vector is retrievable without another request — which is what lets the
    // visualize page project a page of records it has already loaded.
    expect(Array.from(vectorFor('t', 'r1') ?? [])).toEqual([3, 4, 0, 0]);
  });

  it('scopes the vector map by table, so two tables\' r1 are not the same record', () => {
    normalizeRecord('a', { ...listed, vector: [1, 0, 0, 0] });
    normalizeRecord('b', { ...listed, vector: [0, 1, 0, 0] });
    expect(Array.from(vectorFor('a', 'r1') ?? [])).toEqual([1, 0, 0, 0]);
    expect(Array.from(vectorFor('b', 'r1') ?? [])).toEqual([0, 1, 0, 0]);
  });

  it('carries the score on a hit and not on a listed record', () => {
    expect(normalizeHit('t', hit).score).toBe(1);
    expect(normalizeRecord('t', listed).score).toBeNull();
  });
});
