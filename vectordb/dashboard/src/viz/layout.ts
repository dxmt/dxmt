/**
 * A layout computed from the neighbour graph rather than from the vectors.
 *
 * PCA answers "which two directions carry the most variance", which is a question about the
 * whole cloud and often a poor proxy for the one a reader of a scatter plot is actually asking:
 * is this point near that one. On a 384-dimension embedding two components carry under a fifth
 * of the variance and keep under a tenth of the neighbourhoods.
 *
 * This optimises the second question directly. Every edge of the k-nearest-neighbour graph
 * pulls its endpoints together, every pair pushes apart, and the result settles into an
 * arrangement where near-in-the-embedding tends to mean near-on-screen. It is not a principled
 * embedding in the sense UMAP is — no manifold assumption, no probabilistic objective — and it
 * does not need to be: the page measures what came out and says so, rather than asking anyone
 * to trust the method.
 *
 * The repulsion is the expensive half, O(n²) a step, so it is approximated by sampling: each
 * node is pushed away from a fixed number of random others per step rather than from all of
 * them. Over a few hundred steps every pair is sampled many times and the aggregate is a
 * reasonable stand-in for the full force, at a fraction of the cost — the same trade every
 * force-directed layout makes, and the reason this is practical at four thousand nodes.
 */

export interface GraphLayoutOptions {
  n: number;
  /** Per node, the indices it is adjacent to. */
  neighbours: number[][];
  /** Per node, the similarity to each neighbour. Higher is nearer. */
  scores: number[][];
  steps?: number;
  seed?: number;
  onProgress?: (step: number, steps: number) => void;
}

/** Deterministic, so the same graph always settles the same way up. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const REPULSION_SAMPLES = 12;

export function graphLayout(options: GraphLayoutOptions): Float32Array {
  const { n, neighbours, scores, steps = 300, seed = 42, onProgress } = options;
  const positions = new Float32Array(n * 2);
  if (n === 0) return positions;

  const random = mulberry32(seed);
  // Seeded on a circle rather than uniformly in a square: a random square start leaves
  // clusters interleaved and the layout spends most of its steps untangling them.
  for (let i = 0; i < n; i += 1) {
    const angle = (i / n) * Math.PI * 2;
    positions[i * 2] = Math.cos(angle) * 100 + (random() - 0.5);
    positions[i * 2 + 1] = Math.sin(angle) * 100 + (random() - 0.5);
  }

  const dx = new Float32Array(n);
  const dy = new Float32Array(n);
  const repulsion = 140;

  for (let step = 0; step < steps; step += 1) {
    // Cools towards zero, so early steps rearrange and late ones settle. Without it the layout
    // keeps jittering and never looks the same twice at the pixel level.
    const alpha = 1 - step / steps;
    dx.fill(0);
    dy.fill(0);

    // Attraction along the edges, weighted by similarity: a neighbour at 0.99 pulls harder
    // than one at 0.60, so the graph's own confidence shapes the result.
    for (let i = 0; i < n; i += 1) {
      const edges = neighbours[i];
      const weights = scores[i];
      if (!edges) continue;
      for (let e = 0; e < edges.length; e += 1) {
        const j = edges[e]!;
        const w = Math.max(0.05, weights?.[e] ?? 0.5);
        const ax = positions[j * 2]! - positions[i * 2]!;
        const ay = positions[j * 2 + 1]! - positions[i * 2 + 1]!;
        dx[i]! += ax * w * 0.02;
        dy[i]! += ay * w * 0.02;
        dx[j]! -= ax * w * 0.02;
        dy[j]! -= ay * w * 0.02;
      }
    }

    // Repulsion, sampled. Every pair over a few hundred steps, not every pair every step.
    for (let i = 0; i < n; i += 1) {
      for (let s = 0; s < REPULSION_SAMPLES; s += 1) {
        const j = Math.floor(random() * n);
        if (j === i) continue;
        let rx = positions[i * 2]! - positions[j * 2]!;
        let ry = positions[i * 2 + 1]! - positions[j * 2 + 1]!;
        let d2 = rx * rx + ry * ry;
        if (d2 < 1e-6) {
          // Coincident points have no direction to separate along; give them one rather than
          // dividing by zero and scattering NaN through the whole layout.
          rx = random() - 0.5;
          ry = random() - 0.5;
          d2 = 1e-3;
        }
        const force = (repulsion / d2) * (n / REPULSION_SAMPLES) * 0.02;
        dx[i]! += rx * force;
        dy[i]! += ry * force;
      }
    }

    for (let i = 0; i < n; i += 1) {
      // Clamped: one large step early on can throw a node far enough that the repulsion never
      // brings it back, and the plot ends up with a halo of strays.
      positions[i * 2]! += Math.max(-20, Math.min(20, dx[i]!)) * alpha;
      positions[i * 2 + 1]! += Math.max(-20, Math.min(20, dy[i]!)) * alpha;
    }

    if (onProgress && step % 25 === 0) onProgress(step, steps);
  }

  return positions;
}
