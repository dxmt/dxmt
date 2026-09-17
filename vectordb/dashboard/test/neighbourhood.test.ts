/**
 * The number that answers what the variance ratio cannot.
 *
 * Explained variance says how much of the *variance* two components carry. It says nothing
 * about whether the points near each other on screen are the points near each other in the
 * embedding — which is the only question a reader of a scatter plot is asking. The two come
 * apart in both directions, so the plot needs its own measurement.
 */

import { describe, expect, it } from 'vitest';
import { preservation, preservationFromGraph } from '../src/viz/neighbourhood';
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

/**
 * Three situations the page must not describe with the same sentence.
 *
 * The preservation rate is only meaningful against what a random layout would score — `k` of
 * `n - 1`, about 1% for ten neighbours over a thousand points. Thirteen per cent and one per
 * cent are both "low", and they are as different as a photograph is from static: one has
 * structure the projection is coarsening, the other has none at all.
 */
describe('reading the preservation rate', () => {
  const chanceFor = (k: number, n: number) => k / (n - 1);

  it('calls random vectors noise, because they are', () => {
    // The shape of a table of arbitrary vectors: preservation sits at chance.
    const n = 600;
    const d = 128;
    let s = 42;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
    const flat = new Float32Array(n * d);
    for (let i = 0; i < n * d; i += 1) flat[i] = rnd();

    const { positions } = pca2d(flat.slice(), n, d);
    const { rate, k } = preservation(flat, positions, n, d, 'cosine');
    expect(rate).toBeLessThan(chanceFor(k, n) * 3);
  });

  it('does not call real embeddings noise, even when their fine ordering is lost', () => {
    // The shape of a sentence-embedding table: eight topics, tight within each. A point's ten
    // nearest in full dimensions are ten of its topic's hundred-odd members; on the plot they
    // are a *different* ten of the same hundred. Coarse structure intact, fine ordering gone —
    // which is a useful plot, and must not be described as noise.
    const n = 800;
    const d = 96;
    const topics = 8;
    let s = 3;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff - 0.5;
    };
    const flat = new Float32Array(n * d);
    const centres = Array.from({ length: topics }, () =>
      Array.from({ length: d }, () => rnd() * 4),
    );
    for (let i = 0; i < n; i += 1) {
      const centre = centres[i % topics]!;
      for (let j = 0; j < d; j += 1) flat[i * d + j] = centre[j]! + rnd() * 1.2;
    }

    const { positions } = pca2d(flat.slice(), n, d);
    const { rate, k } = preservation(flat, positions, n, d, 'cosine');

    expect(rate).toBeGreaterThan(chanceFor(k, n) * 3);
    // And still low in absolute terms, which is the whole reason the two tests differ.
    expect(rate).toBeLessThan(0.6);
  });

  it('puts the two on opposite sides of the rule, with room to spare', () => {
    // The rule the page uses is "three times chance", so that is what this measures — not some
    // arbitrary ratio between the two cases. If they landed a hair either side of the line it
    // would be a coin toss on real data; they do not.
    const n = 600;
    const d = 96;
    let s = 5;
    const rnd = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff - 0.5;
    };

    const noise = new Float32Array(n * d);
    for (let i = 0; i < n * d; i += 1) noise[i] = rnd();

    const structured = new Float32Array(n * d);
    const centres = Array.from({ length: 6 }, () => Array.from({ length: d }, () => rnd() * 4));
    for (let i = 0; i < n; i += 1) {
      const centre = centres[i % 6]!;
      for (let j = 0; j < d; j += 1) structured[i * d + j] = centre[j]! + rnd() * 1.2;
    }

    const noisy = preservation(noise, pca2d(noise.slice(), n, d).positions, n, d, 'cosine');
    const real = preservation(
      structured,
      pca2d(structured.slice(), n, d).positions,
      n,
      d,
      'cosine',
    );
    const chance = chanceFor(noisy.k, n);

    // Noise sits under the line; structure sits well over it. Measured: about 2x chance
    // against about 8x, either side of the 3x rule.
    expect(noisy.rate).toBeLessThan(chance * 3);
    expect(real.rate).toBeGreaterThan(chance * 5);
  });
});

/**
 * Measuring against a graph the server computed, which is the stricter of the two.
 *
 * With a k-nearest-neighbour graph in hand there is nothing to approximate: the neighbours are
 * exact, they came from the table's own metric, and every node is checked rather than a sample
 * of fifty. It is also the only way to measure a layout built *from* the graph, since that path
 * never downloads a vector.
 */
describe('preservationFromGraph', () => {
  it('scores a perfect layout at 1', () => {
    // Points on a line, and a graph that says each one's neighbours are its line neighbours.
    const n = 40;
    const positions = new Float32Array(n * 2);
    for (let i = 0; i < n; i += 1) positions[i * 2] = i;
    const truth = Array.from({ length: n }, (_, i) =>
      [i - 1, i + 1, i - 2, i + 2].filter((j) => j >= 0 && j < n).slice(0, 2),
    );

    expect(preservationFromGraph(positions, truth, 2).rate).toBeGreaterThan(0.95);
  });

  it('scores a layout that ignores the graph near zero', () => {
    const n = 40;
    const positions = new Float32Array(n * 2);
    // Deliberately the reverse of the graph's ordering.
    for (let i = 0; i < n; i += 1) positions[i * 2] = (i * 17) % n;
    const truth = Array.from({ length: n }, (_, i) =>
      [i - 1, i + 1].filter((j) => j >= 0 && j < n),
    );

    expect(preservationFromGraph(positions, truth, 2).rate).toBeLessThan(0.4);
  });

  it('checks every node rather than a sample, since there is nothing to save', () => {
    const n = 25;
    const positions = new Float32Array(n * 2);
    for (let i = 0; i < n; i += 1) positions[i * 2] = i;
    const truth = Array.from({ length: n }, (_, i) => [(i + 1) % n]);
    expect(preservationFromGraph(positions, truth, 1).probes).toBe(n);
  });

  it('shrinks k to what the graph actually carries', () => {
    const n = 10;
    const positions = new Float32Array(n * 2);
    const truth = Array.from({ length: n }, (_, i) => [(i + 1) % n, (i + 2) % n]);
    expect(preservationFromGraph(positions, truth, 20).k).toBe(2);
  });

  it('returns nothing rather than dividing by zero on an empty graph', () => {
    expect(preservationFromGraph(new Float32Array(0), [], 5)).toEqual({ rate: 0, k: 0, probes: 0 });
  });
});
