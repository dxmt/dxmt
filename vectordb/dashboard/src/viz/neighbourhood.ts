/**
 * How much of the neighbourhood structure survived the projection, as a number.
 *
 * The explained-variance ratio says how much of the *variance* two components carry. It does
 * not say whether the points near each other on screen are the points near each other in the
 * embedding, and that is the question a reader of a scatter plot is actually asking. A layout
 * can carry a respectable share of the variance and still scramble neighbourhoods, and a layout
 * with very little variance can still place near things near.
 *
 * So: for a sample of the sample, take the k nearest in the plot and the k nearest in full
 * dimensions, and report the overlap. "31% of each point's ten nearest neighbours are still its
 * nearest neighbours on screen" is one sentence and it settles the question that paragraphs of
 * hedging cannot.
 *
 * Deliberately **not** called trustworthiness. That name denotes a specific rank-weighted
 * measure from the dimensionality-reduction literature, and this is the plain overlap; using
 * the word would claim a property that was not computed.
 */

export type Metric = 'cosine' | 'euclidean' | 'dot' | 'manhattan';

/** Distance in the table's own metric, so "near" here means what it means to the index. */
function distance(
  a: Float32Array,
  aOffset: number,
  b: Float32Array,
  bOffset: number,
  d: number,
  metric: Metric,
): number {
  if (metric === 'cosine' || metric === 'dot') {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let j = 0; j < d; j += 1) {
      const x = a[aOffset + j]!;
      const y = b[bOffset + j]!;
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    if (metric === 'dot') return -dot;
    const norm = Math.sqrt(na) * Math.sqrt(nb);
    return norm === 0 ? 1 : 1 - dot / norm;
  }
  if (metric === 'manhattan') {
    let sum = 0;
    for (let j = 0; j < d; j += 1) sum += Math.abs(a[aOffset + j]! - b[bOffset + j]!);
    return sum;
  }
  let sum = 0;
  for (let j = 0; j < d; j += 1) {
    const delta = a[aOffset + j]! - b[bOffset + j]!;
    sum += delta * delta;
  }
  return sum;
}

/** The `k` nearest to `from`, by index, excluding itself. */
function nearest(
  data: Float32Array,
  n: number,
  d: number,
  from: number,
  k: number,
  metric: Metric,
): number[] {
  const scored: { index: number; distance: number }[] = [];
  for (let i = 0; i < n; i += 1) {
    if (i === from) continue;
    scored.push({ index: i, distance: distance(data, from * d, data, i * d, d, metric) });
  }
  // Ties broken by index, so the answer does not depend on iteration order.
  scored.sort((a, b) => a.distance - b.distance || a.index - b.index);
  return scored.slice(0, k).map((entry) => entry.index);
}

export interface Preservation {
  /** Mean overlap of the k nearest, in [0, 1]. */
  rate: number;
  k: number;
  /** How many points it was measured on. */
  probes: number;
}

/**
 * Compare neighbourhoods in the original space against neighbourhoods in the plot.
 *
 * `highDim` must be the **uncentred** vectors: PCA centres its input in place, which leaves
 * Euclidean distances alone but changes every cosine. Measuring after that would quietly be
 * measuring a different metric than the index uses.
 *
 * O(probes × n × d). At fifty probes over 1,300 vectors of 768 dimensions that is about 50
 * million multiply-adds — a tenth of a second in the worker, which is why it is capped rather
 * than run over every point.
 */
export function preservation(
  highDim: Float32Array,
  positions: Float32Array,
  n: number,
  d: number,
  metric: Metric,
  k = 10,
  maxProbes = 50,
): Preservation {
  const effectiveK = Math.min(k, n - 1);
  if (n < 3 || effectiveK < 1) return { rate: 0, k: 0, probes: 0 };

  const probes = Math.min(maxProbes, n);
  const step = Math.max(1, Math.floor(n / probes));

  let total = 0;
  let counted = 0;
  for (let from = 0; from < n && counted < probes; from += step) {
    const inSpace = new Set(nearest(highDim, n, d, from, effectiveK, metric));
    const onScreen = nearest(positions, n, 2, from, effectiveK, 'euclidean');
    total += onScreen.filter((index) => inSpace.has(index)).length / effectiveK;
    counted += 1;
  }

  return { rate: counted === 0 ? 0 : total / counted, k: effectiveK, probes: counted };
}
