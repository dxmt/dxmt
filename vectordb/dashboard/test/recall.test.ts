/**
 * The recall arithmetic, and the one rule that is easy to get wrong.
 *
 * Excluding the query point **by id, never by score** is the rule. A different record holding
 * an identical vector scores 1.0 too, and dropping it would turn a correct answer into a
 * fabricated miss — quietly, in a number the user is about to trust.
 */

import { describe, expect, it } from 'vitest';
import { compare, mulberry32, sampleIds, summarise } from '../src/quality/recall';

const hits = (...pairs: [string, number][]) => pairs.map(([id, score]) => ({ id, score }));

describe('compare', () => {
  it('drops the query point from both lists and scores the rest', () => {
    const exact = hits(['q', 1.0], ['a', 0.9], ['b', 0.8], ['c', 0.7]);
    const ann = hits(['q', 1.0], ['a', 0.9], ['b', 0.8], ['z', 0.5]);
    const probe = compare('q', exact, ann, 3);

    expect(probe.recall).toBeCloseTo(2 / 3);
    expect(probe.missing).toEqual(['c']);
  });

  it('keeps a duplicate-vector neighbour that also scores 1.0', () => {
    // `twin` holds the same vector as the query and is a genuine nearest neighbour. Excluding
    // by score would remove it from the exact list and invent a miss out of a perfect answer.
    const exact = hits(['q', 1.0], ['twin', 1.0], ['a', 0.9]);
    const ann = hits(['q', 1.0], ['twin', 1.0], ['a', 0.9]);
    const probe = compare('q', exact, ann, 2);

    expect(probe.recall).toBe(1);
    expect(probe.missing).toEqual([]);
  });

  it('is correct when the approximate run misses the query point itself', () => {
    // Not finding your own vector is the most severe recall failure there is. Removing by id
    // is a no-op on the ann list here, and the truncation must still leave k.
    const exact = hits(['q', 1.0], ['a', 0.9], ['b', 0.8]);
    const ann = hits(['a', 0.9], ['z', 0.4]);
    const probe = compare('q', exact, ann, 2);

    expect(probe.recall).toBeCloseTo(0.5);
    expect(probe.foundTop1).toBe(true); // both lists lead with `a` once `q` is gone
  });

  it('reports recall 1.0 for identical lists and 0 for disjoint ones', () => {
    expect(compare('q', hits(['q', 1], ['a', 0.9]), hits(['q', 1], ['a', 0.9]), 1).recall).toBe(1);
    expect(compare('q', hits(['q', 1], ['a', 0.9]), hits(['q', 1], ['z', 0.9]), 1).recall).toBe(0);
  });

  it('separates a near-tie substitution from a bad one, which recall alone cannot', () => {
    const exact = hits(['q', 1.0], ['a', 0.999], ['b', 0.998]);
    const nearTie = compare('q', exact, hits(['q', 1.0], ['a', 0.999], ['x', 0.997]), 2);
    const bad = compare('q', exact, hits(['q', 1.0], ['a', 0.999], ['x', 0.40]), 2);

    expect(nearTie.recall).toBeCloseTo(bad.recall); // same recall
    expect(nearTie.scoreRatio).toBeGreaterThan(0.99); // very different answers
    expect(bad.scoreRatio).toBeLessThan(0.75);
  });

  it('does not divide by zero on an empty exact list', () => {
    expect(compare('q', [], [], 5).recall).toBe(0);
  });
});

describe('summarise', () => {
  const probe = (id: string, recall: number) => ({
    id,
    recall,
    foundTop1: recall === 1,
    scoreRatio: 1,
    exactMs: 10,
    annMs: 1,
    missing: [],
  });

  it('separates "everyone misses one" from "a few are catastrophic"', () => {
    // Both average 0.9, and they are completely different indexes.
    const even = summarise([probe('a', 0.9), probe('b', 0.9), probe('c', 0.9)], 10, 'normal', true);
    const spiky = summarise([probe('a', 1), probe('b', 1), probe('c', 0.7)], 10, 'normal', true);

    expect(even.meanRecall).toBeCloseTo(spiky.meanRecall, 1);
    expect(even.stdevRecall).toBeLessThan(spiky.stdevRecall);
    expect(even.histogram[9]).toBe(3);
    expect(spiky.histogram[10]).toBe(2);
    expect(spiky.worst[0]?.id).toBe('c');
  });

  it('refuses to call it recall when there was no ground truth', () => {
    const measured = summarise([probe('a', 0.9)], 10, 'normal', true);
    const guessed = summarise([probe('a', 0.9)], 10, 'normal', false);

    expect(measured.label).toBe('recall@10');
    expect(guessed.label).not.toContain('recall@');
    expect(guessed.label).toContain('agreement');
  });

  it('reports what the approximation bought, not only what it cost', () => {
    const report = summarise([probe('a', 0.9), probe('b', 1)], 10, 'normal', true);
    expect(report.medianExactMs).toBe(10);
    expect(report.medianAnnMs).toBe(1);
  });
});

describe('sampling', () => {
  it('is reproducible from a seed, so a run can be repeated from a URL', () => {
    const ids = Array.from({ length: 100 }, (_, i) => `r${i}`);
    expect(sampleIds(ids, 10, 42)).toEqual(sampleIds(ids, 10, 42));
    expect(sampleIds(ids, 10, 42)).not.toEqual(sampleIds(ids, 10, 43));
  });

  it('never picks the same record twice', () => {
    const ids = Array.from({ length: 20 }, (_, i) => `r${i}`);
    const picked = sampleIds(ids, 20, 7);
    expect(new Set(picked).size).toBe(20);
  });

  it('asks for more than exists without looping forever', () => {
    expect(sampleIds(['a', 'b'], 10, 1)).toHaveLength(2);
  });

  it('produces values in [0, 1)', () => {
    const rand = mulberry32(1);
    for (let i = 0; i < 1000; i += 1) {
      const x = rand();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});
