/**
 * Two principal components by power iteration, and the number that says whether to believe them.
 *
 * The method is the one in Qdrant's `pca.js` (Apache 2.0 — the method, not the code): iterate
 * `w = Xᶜᵀ(Xᶜv)` on the centered data, never forming the d×d covariance matrix, deflating the
 * second component against the first. At 768 dimensions the covariance would be 590,000 entries
 * to build and factor; this is O(n·d) a pass and converges in well under the iteration cap.
 *
 * Three differences from the reference, each measured in work saved or truth added:
 *
 * **Centre once.** The reference recomputes `row[j] - mean[j]` inside both inner loops of every
 * iteration — exactly twice the necessary arithmetic, sixty-four times over. The buffer here is
 * transferred and owned, so the mean comes off in a single pass up front.
 *
 * **A flat `Float32Array`.** An array of arrays is structure-cloned to a worker as boxed
 * numbers: 2,000 × 768 is about 24 MB copied. A transferred flat buffer is 6 MB and zero-copy,
 * and `normalize.ts` already holds the vectors in this shape.
 *
 * **Explained variance.** The reference does not compute it, and without it the scatter is a
 * picture people read as a map. Variance along a component is `Σ proj² / n`; the total is the
 * trace of the covariance, which is `Σⱼ var(Xⱼ)` and needs no matrix either. Two components of
 * a sentence embedding typically explain five to fifteen per cent, and the page has to say so.
 */

export interface Projection {
  /** Flat, `n × 2`. */
  positions: Float32Array;
  /** Fraction of total variance along each component. */
  explained: [number, number];
  iterations: number;
}

/** Seeded, so the same sample always lands the same way up. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MAX_ITERATIONS = 64;
const CONVERGED = 1e-9;

export function pca2d(data: Float32Array, n: number, d: number, seed = 42): Projection {
  if (n < 3) throw new Error(`PCA needs at least three points, got ${n}`);
  if (d < 2) throw new Error(`PCA needs at least two dimensions, got ${d}`);

  // Centre in place. `data` is ours: the caller transferred it.
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i += 1) {
    const row = i * d;
    for (let j = 0; j < d; j += 1) mean[j]! += data[row + j]!;
  }
  for (let j = 0; j < d; j += 1) mean[j]! /= n;
  for (let i = 0; i < n; i += 1) {
    const row = i * d;
    for (let j = 0; j < d; j += 1) data[row + j]! -= mean[j]!;
  }

  // Total variance is the trace of the covariance: no matrix needed for it either.
  let totalVariance = 0;
  for (let i = 0; i < n * d; i += 1) totalVariance += data[i]! * data[i]!;
  totalVariance /= n;

  const random = mulberry32(seed);
  const components: Float64Array[] = [];
  let iterations = 0;

  for (let c = 0; c < 2; c += 1) {
    let v = new Float64Array(d);
    for (let j = 0; j < d; j += 1) v[j] = random() - 0.5;
    orthonormalise(v, components);

    for (let iteration = 0; iteration < MAX_ITERATIONS; iteration += 1) {
      iterations += 1;
      const w = new Float64Array(d);
      for (let i = 0; i < n; i += 1) {
        const row = i * d;
        let dot = 0;
        for (let j = 0; j < d; j += 1) dot += data[row + j]! * v[j]!;
        if (dot === 0) continue;
        for (let j = 0; j < d; j += 1) w[j]! += data[row + j]! * dot;
      }
      orthonormalise(w, components);

      let agreement = 0;
      for (let j = 0; j < d; j += 1) agreement += w[j]! * v[j]!;
      v = w;
      if (Math.abs(1 - Math.abs(agreement)) < CONVERGED) break;
    }
    components.push(v);
  }

  const positions = new Float32Array(n * 2);
  const variance: [number, number] = [0, 0];
  for (let i = 0; i < n; i += 1) {
    const row = i * d;
    for (let c = 0; c < 2; c += 1) {
      const component = components[c]!;
      let dot = 0;
      for (let j = 0; j < d; j += 1) dot += data[row + j]! * component[j]!;
      positions[i * 2 + c] = dot;
      variance[c] = variance[c]! + dot * dot;
    }
  }

  return {
    positions,
    explained:
      totalVariance > 0
        ? [variance[0] / n / totalVariance, variance[1] / n / totalVariance]
        : [0, 0],
    iterations,
  };
}

/** Remove the projection onto every earlier component, then normalise. */
function orthonormalise(v: Float64Array, components: Float64Array[]): void {
  const d = v.length;
  for (const component of components) {
    let dot = 0;
    for (let j = 0; j < d; j += 1) dot += v[j]! * component[j]!;
    for (let j = 0; j < d; j += 1) v[j]! -= dot * component[j]!;
  }
  let norm = 0;
  for (let j = 0; j < d; j += 1) norm += v[j]! * v[j]!;
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < d; j += 1) v[j]! /= norm;
}
