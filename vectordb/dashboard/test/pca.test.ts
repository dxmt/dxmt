/**
 * The projection, and the number that decides whether the picture means anything.
 *
 * The explained-variance ratio is the addition to the borrowed method, and it is the point: two
 * components of a 768-dimension embedding typically account for five to fifteen per cent of the
 * variance, and a scatter presented without that figure gets read as a cluster map when it is
 * nothing of the kind. So it is tested against data whose answer is known by construction.
 */

import { describe, expect, it } from 'vitest';
import { pca2d } from '../src/viz/pca';
import { budgetFor, strata, MAX_FLOATS, MAX_REQUESTS } from '../src/lib/sample';
import { colorBy, ABSENT } from '../src/lib/colors';

function flat(rows: number[][]): { data: Float32Array; n: number; d: number } {
  const n = rows.length;
  const d = rows[0]!.length;
  const data = new Float32Array(n * d);
  rows.forEach((row, i) => data.set(row, i * d));
  return { data, n, d };
}

describe('pca2d', () => {
  it('finds the axis the variance is actually along', () => {
    // Everything on the first axis, jitter elsewhere: PC1 must be that axis and must carry
    // nearly all the variance.
    const rows = Array.from({ length: 200 }, (_, i) => {
      const row = new Array<number>(16).fill(0);
      row[0] = i - 100;
      row[5] = (i % 7) * 0.01;
      return row;
    });
    const { data, n, d } = flat(rows);
    const { positions, explained } = pca2d(data, n, d);

    expect(explained[0]).toBeGreaterThan(0.99);
    expect(explained[1]).toBeLessThan(0.01);

    // And the projection is ordered along that axis, up to a sign.
    const first = positions[0]!;
    const last = positions[(n - 1) * 2]!;
    expect(Math.abs(last - first)).toBeGreaterThan(100);
  });

  it('reports the small explained variance that makes a scatter misleading', () => {
    // Isotropic noise in 64 dimensions: no two directions carry much, and the page has to say
    // so rather than draw a confident picture.
    let seed = 7;
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff - 0.5;
    };
    const rows = Array.from({ length: 400 }, () =>
      Array.from({ length: 64 }, () => random()),
    );
    const { data, n, d } = flat(rows);
    const { explained } = pca2d(data, n, d);

    const together = explained[0] + explained[1];
    expect(together).toBeLessThan(0.2);
    expect(together).toBeGreaterThan(0.02);
  });

  it('explains no more than all of the variance', () => {
    const rows = Array.from({ length: 120 }, (_, i) => [i, i * 0.5, -i, (i % 5) * 3, 1]);
    const { data, n, d } = flat(rows);
    const { explained } = pca2d(data, n, d);
    expect(explained[0] + explained[1]).toBeLessThanOrEqual(1.0000001);
    expect(explained[0]).toBeGreaterThanOrEqual(explained[1]);
  });

  it('is reproducible, so the plot does not flip between runs', () => {
    const rows = Array.from({ length: 150 }, (_, i) => [Math.sin(i), Math.cos(i), i % 3, i * 0.01]);
    const a = pca2d(flat(rows).data, 150, 4);
    const b = pca2d(flat(rows).data, 150, 4);
    expect(Array.from(a.positions)).toEqual(Array.from(b.positions));
  });

  it('returns zeros rather than NaN when every point is identical', () => {
    const rows = Array.from({ length: 50 }, () => [1, 1, 1, 1]);
    const { data, n, d } = flat(rows);
    const { positions, explained } = pca2d(data, n, d);
    expect([...positions].every((v) => Number.isFinite(v))).toBe(true);
    expect(explained).toEqual([0, 0]);
  });

  it('refuses inputs it cannot project', () => {
    expect(() => pca2d(new Float32Array(4), 2, 2)).toThrow(/three points/);
    expect(() => pca2d(new Float32Array(9), 9, 1)).toThrow(/two dimensions/);
  });
});

describe('the sampling budget', () => {
  it('is bounded by the wire at high dimensions and by requests at low ones', () => {
    const wide = budgetFor(1_000_000, 768);
    expect(wide.points).toBe(Math.floor(MAX_FLOATS / 768));
    expect(wide.reason).toBe('floats');

    const narrow = budgetFor(1_000_000, 8);
    expect(narrow.points).toBe(MAX_REQUESTS * 1000);
    expect(narrow.reason).toBe('requests');
  });

  it('never asks for more than the table holds', () => {
    const small = budgetFor(120, 768);
    expect(small.points).toBe(120);
    expect(small.reason).toBe('table');
    expect(small.requests).toBe(1);
  });

  it('gets tighter as dimensions grow, which is the whole reason it is a pair', () => {
    const points = [128, 768, 1536, 3072].map((d) => budgetFor(1_000_000, d).points);
    for (let i = 1; i < points.length; i += 1) {
      expect(points[i]!).toBeLessThanOrEqual(points[i - 1]!);
    }
    expect(points[3]!).toBeLessThan(points[0]!);
  });

  it('spreads the offsets across the table rather than taking the newest rows', () => {
    // `records/list` is created_at descending, so offset 0 is the newest — a sample of last
    // Tuesday rather than of the table.
    expect(strata(100_000, 4)).toEqual([0, 25_000, 50_000, 75_000]);
    expect(strata(100_000, 1)).toEqual([0]);
  });
});

describe('colouring', () => {
  it('gives an absent key its own visible slot', () => {
    // Schemaless payloads mean a key missing on some records is normal, and it must not
    // silently share a colour with a real value.
    const { legend, colorAt } = colorBy(['a', 'b', undefined, 'a']);
    expect(colorAt(2)).toBe(ABSENT);
    expect(legend.entries.some((e) => e.label === '(no value)' && e.count === 1)).toBe(true);
  });

  it('folds a long tail into one entry rather than recycling colours', () => {
    const many = Array.from({ length: 30 }, (_, i) => `v${i}`);
    const { legend } = colorBy(many);
    expect(legend.entries.length).toBeLessThanOrEqual(11);
    expect(legend.entries.at(-1)!.label).toMatch(/^other \(\d+\)/);
  });

  it('treats a numeric field as a range, not as categories', () => {
    const { legend, colorAt } = colorBy([1, 5, 9]);
    expect(legend.kind).toBe('numeric');
    expect(legend.range).toEqual([1, 9]);
    expect(colorAt(0)).not.toBe(colorAt(2));
  });

  it('is stable: the same value is the same colour wherever it appears', () => {
    const { colorAt } = colorBy(['x', 'y', 'x', 'y', 'x']);
    expect(colorAt(0)).toBe(colorAt(2));
    expect(colorAt(1)).toBe(colorAt(3));
    expect(colorAt(0)).not.toBe(colorAt(1));
  });
});
