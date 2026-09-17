/**
 * The number that answers what the variance ratio cannot.
 *
 * Explained variance says how much of the *variance* two components carry. It says nothing
 * about whether the points near each other on screen are the points near each other in the
 * embedding — which is the only question a reader of a scatter plot is asking. The two come
 * apart in both directions, so the plot needs its own measurement.
 */

import { describe, expect, it } from 'vitest';
import { preservation } from '../src/viz/neighbourhood';
import { pca2d } from '../src/viz/pca';

function flat(rows: number[][]): Float32Array {
  const d = rows[0]!.length;
  const data = new Float32Array(rows.length * d);
  rows.forEach((row, i) => data.set(row, i * d));
  return data;
}

describe('neighbourhood preservation', () => {
  it('is near 1 when the data really is two-dimensional', () => {
    // A plane embedded in 32 dimensions: PCA recovers it exactly, so every neighbourhood
    // survives. This is the control — a low number here would mean the measure is broken.
    const rows: number[][] = [];
    for (let a = 0; a < 12; a += 1) {
      for (let b = 0; b < 12; b += 1) {
        const row = new Array<number>(32).fill(0);
        row[3] = a;
        row[17] = b;
        rows.push(row);
      }
    }
    const n = rows.length;
    const original = flat(rows);
    const { positions, explained } = pca2d(flat(rows), n, 32);

    expect(explained[0] + explained[1]).toBeGreaterThan(0.99);
    const { rate } = preservation(original, positions, n, 32, 'euclidean', 8);
    expect(rate).toBeGreaterThan(0.9);
  });

  it('is low when the data fills its dimensions, even though a plot still appears', () => {
    // Isotropic noise in 48 dimensions. The scatter looks like a cloud and means nothing;
    // this is the number that says so.
    let seed = 3;
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff - 0.5;
    };
    const rows = Array.from({ length: 200 }, () => Array.from({ length: 48 }, random));
    const original = flat(rows);
    const { positions } = pca2d(flat(rows), 200, 48);

    const { rate } = preservation(original, positions, 200, 48, 'euclidean', 10);
    expect(rate).toBeLessThan(0.35);
  });

  it('measures cosine in cosine, not in whatever centring left behind', () => {
    // PCA centres its input in place. That leaves Euclidean distances alone and changes every
    // cosine — so a measure taken after centring would silently be about a different metric
    // than the index uses. Same vectors at different magnitudes are identical under cosine and
    // far apart under Euclidean, which is what makes the two answers differ here.
    const directions = Array.from({ length: 40 }, (_, i) => {
      const angle = (i / 40) * Math.PI * 2;
      return [Math.cos(angle), Math.sin(angle), 0, 0];
    });
    const rows = directions.flatMap((direction, i) =>
      [1, 8].map((scale) => direction.map((v) => v * scale * (1 + i * 0.01))),
    );
    const n = rows.length;
    const original = flat(rows);
    const { positions } = pca2d(flat(rows), n, 4);

    const cosine = preservation(original, positions, n, 4, 'cosine', 4).rate;
    const euclidean = preservation(original, positions, n, 4, 'euclidean', 4).rate;
    expect(cosine).not.toBeCloseTo(euclidean, 2);
  });

  it('caps the probes rather than measuring every point', () => {
    // O(probes x n x d): uncapped, this would be quadratic in the sample size.
    const rows = Array.from({ length: 300 }, (_, i) => [i, i % 11, (i * 7) % 13, 1]);
    const original = flat(rows);
    const { positions } = pca2d(flat(rows), 300, 4);

    expect(preservation(original, positions, 300, 4, 'euclidean', 5).probes).toBe(50);
    expect(preservation(original, positions, 300, 4, 'euclidean', 5, 10).probes).toBe(10);
  });

  it('does not divide by zero on a sample too small to have neighbourhoods', () => {
    const rows = [[1, 2], [3, 4]];
    expect(preservation(flat(rows), flat(rows), 2, 2, 'euclidean', 5)).toEqual({
      rate: 0,
      k: 0,
      probes: 0,
    });
  });

  it('shrinks k rather than asking for more neighbours than exist', () => {
    const rows = Array.from({ length: 6 }, (_, i) => [i, -i, i * 2]);
    const original = flat(rows);
    const { positions } = pca2d(flat(rows), 6, 3);
    expect(preservation(original, positions, 6, 3, 'euclidean', 20).k).toBe(5);
  });
});

/**
 * Where the warning threshold sits, pinned against data whose answer is known.
 *
 * The page warns below 20% preservation, not 50%, and the reason is the first case here: a
 * projection can recover the structure perfectly and still scramble the ordering *inside* each
 * group. That is a good plot. Warning on it would be a false alarm, and a banner that cries
 * wolf is a banner people learn to close.
 */
describe('calibration', () => {
  function blobs(n: number, d: number) {
    let s = 7;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff - 0.5;
    };
    const flat = new Float32Array(n * d);
    for (let i = 0; i < n; i += 1) {
      const blob = i % 4;
      for (let j = 0; j < d; j += 1) flat[i * d + j] = rnd() * 0.15;
      flat[i * d + 5] = (blob % 2) * 10 + rnd();
      flat[i * d + 61] = Math.floor(blob / 2) * 10 + rnd();
    }
    return flat;
  }

  function uniform(n: number, d: number) {
    let s = 42;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
    const flat = new Float32Array(n * d);
    for (let i = 0; i < n * d; i += 1) flat[i] = rnd();
    return flat;
  }

  const WARNS_BELOW = 0.2;

  it('does not warn on four real clusters, which project well and reorder within themselves', () => {
    const n = 800;
    const d = 128;
    const original = blobs(n, d);
    const { positions, explained } = pca2d(original.slice(), n, d);

    expect(explained[0] + explained[1]).toBeGreaterThan(0.9);
    const { rate } = preservation(original, positions, n, d, 'euclidean');
    // Around 0.43 in practice: the groups are recovered, the order inside them is not.
    expect(rate).toBeGreaterThan(WARNS_BELOW);
    expect(rate).toBeLessThan(0.8);
  });

  it('warns on uniform noise, which is what a table of random vectors is', () => {
    const n = 600;
    const d = 128;
    const original = uniform(n, d);
    const { positions } = pca2d(original.slice(), n, d);

    const { rate } = preservation(original, positions, n, d, 'cosine');
    expect(rate).toBeLessThan(WARNS_BELOW);
  });

  it('reports 100% when the data really is two-dimensional', () => {
    // The anchor: if this is not 1.0 the measure itself is wrong, and every number above is
    // meaningless.
    const n = 400;
    const original = uniform(n, 2);
    const { positions } = pca2d(original.slice(), n, 2);
    expect(preservation(original, positions, n, 2, 'euclidean').rate).toBeCloseTo(1, 5);
  });
});
