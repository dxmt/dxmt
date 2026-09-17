/**
 * How much recall is the index actually losing, on this data.
 *
 * The method is Qdrant's (`qdrant-web-ui`, Apache 2.0 -- the method, not the code): take an
 * existing record as the query, run it twice, and call the overlap recall. Three things are
 * done differently here, and each difference is a correctness fix rather than a preference.
 *
 * # Excluding the query point: by id, at k+1, never by score
 *
 * The query is a stored record, so it comes back as its own nearest neighbour and would
 * otherwise occupy a slot in both lists and inflate recall. Qdrant drops it with a
 * `must_not: has_id` filter; DXMT has no id filter, so it is dropped after the fact -- which
 * turns out to be *better*, because adding a filter would send both runs down the filtered
 * HNSW path and measure the recall of filtered search rather than of the search production
 * traffic actually takes.
 *
 * Dropping it **by id and not by score** is the part that is easy to get wrong. A different
 * record holding an identical vector also scores 1.0; removing that one would invent a recall
 * miss out of a correct answer. And `k + 1` is requested precisely so that removing one still
 * leaves k.
 *
 * # What is measured, beyond the mean
 *
 * A mean recall of 0.95 made of "five per cent of queries are catastrophic" is a completely
 * different index from one made of "every query misses one neighbour", and the mean cannot tell
 * them apart -- hence the histogram and the worst offenders. Two runs both at 0.90 are also
 * very different if one substituted a neighbour at similarity 0.998 and the other at 0.61 --
 * hence the score ratio. And recall without the latency it bought is half a result, so both
 * timings travel with it.
 */

import { api } from '../api/client';
import type { Filter, SearchMode } from '../api/types';

export interface Probe {
  id: string;
  /** Overlap of the two top-k lists, after the query point is removed from both. */
  recall: number;
  /** Did the approximate run find the true nearest neighbour? Often the more actionable number. */
  foundTop1: boolean;
  /**
   * Sum of the approximate run's similarities over the sum of the exact run's. Near 1.0 means
   * the misses were near-ties; well below means it substituted genuinely worse neighbours.
   */
  scoreRatio: number;
  exactMs: number;
  annMs: number;
  missing: string[];
}

export interface RecallReport {
  k: number;
  mode: SearchMode;
  /** False when there is no exact mode: then this is agreement, not recall. See `label`. */
  groundTruth: boolean;
  label: string;
  probes: Probe[];
  meanRecall: number;
  stdevRecall: number;
  recallAt1: number;
  meanScoreRatio: number;
  medianExactMs: number;
  medianAnnMs: number;
  /** Per-recall-bucket counts, 0.0 to 1.0 in tenths. */
  histogram: number[];
  worst: Probe[];
}

/** Deterministic sampling, so a run is reproducible and a URL reproduces it. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleIds(ids: string[], count: number, seed: number): string[] {
  const rand = mulberry32(seed);
  const pool = [...ids];
  // Fixed before the loop. `pool.length` shrinks as items are drawn, so comparing against it
  // each round stops at roughly half the requested sample -- which a caller cannot see, because
  // a short sample is indistinguishable from a small table.
  const want = Math.min(count, pool.length);
  const picked: string[] = [];
  while (picked.length < want) {
    const i = Math.floor(rand() * pool.length);
    picked.push(pool.splice(i, 1)[0] as string);
  }
  return picked;
}

/** The overlap of two lists with the query point already removed. Exported for its test. */
export function compare(
  queryId: string,
  exact: { id: string; score: number }[],
  ann: { id: string; score: number }[],
  k: number,
): Omit<Probe, 'exactMs' | 'annMs'> {
  // By id. A different record with an identical vector also scores 1.0 and is a real neighbour.
  const exactTop = exact.filter((h) => h.id !== queryId).slice(0, k);
  const annTop = ann.filter((h) => h.id !== queryId).slice(0, k);

  const annIds = new Set(annTop.map((h) => h.id));
  const hit = exactTop.filter((h) => annIds.has(h.id));

  const exactSum = exactTop.reduce((sum, h) => sum + h.score, 0);
  const annSum = annTop.reduce((sum, h) => sum + h.score, 0);

  return {
    id: queryId,
    recall: exactTop.length === 0 ? 0 : hit.length / exactTop.length,
    foundTop1: exactTop.length > 0 && annTop.length > 0 && exactTop[0]!.id === annTop[0]!.id,
    scoreRatio: exactSum === 0 ? 1 : annSum / exactSum,
    missing: exactTop.filter((h) => !annIds.has(h.id)).map((h) => h.id),
  };
}

export interface RunOptions {
  table: string;
  ids: string[];
  k: number;
  mode: SearchMode;
  /** The mode that stands in for ground truth: `exact` when available, else `accurate`. */
  truthMode: SearchMode;
  filter?: Filter | null;
  vectorOf: (id: string) => Promise<number[]>;
  onProbe?: (probe: Probe, index: number, total: number) => void;
  signal?: AbortSignal;
}

export async function run(options: RunOptions): Promise<RecallReport> {
  const { table, ids, k, mode, truthMode, filter, vectorOf, onProbe, signal } = options;
  const probes: Probe[] = [];

  for (const [index, id] of ids.entries()) {
    if (signal?.aborted) break;
    const vector = await vectorOf(id);
    const body = {
      vector,
      // One more than wanted, so removing the query point still leaves k.
      top_k: k + 1,
      limit: k + 1,
      with_payload: false,
      filter: filter ?? null,
    };

    // Sequentially, never in parallel: two concurrent searches contend for the same index and
    // the same page cache, and the timings are half of what this page reports.
    const exact = await api.search(table, { ...body, search_mode: truthMode }, signal);
    const ann = await api.search(table, { ...body, search_mode: mode }, signal);

    const probe: Probe = {
      ...compare(id, exact.value.hits, ann.value.hits, k),
      exactMs: exact.value.tookms,
      annMs: ann.value.tookms,
    };
    probes.push(probe);
    onProbe?.(probe, index, ids.length);
  }

  return summarise(probes, k, mode, truthMode === 'exact');
}

export function summarise(
  probes: Probe[],
  k: number,
  mode: SearchMode,
  groundTruth: boolean,
): RecallReport {
  const recalls = probes.map((p) => p.recall);
  const mean = average(recalls);

  const histogram = new Array<number>(11).fill(0);
  for (const r of recalls) histogram[Math.round(r * 10)]! += 1;

  return {
    k,
    mode,
    groundTruth,
    // If there is no exact mode this is not recall and must not be called recall: `accurate`
    // is 4x ef and measures 0.97-0.99 itself, so the number is an upper bound on agreement and
    // is blind to every neighbour both modes miss.
    label: groundTruth ? `recall@${k}` : `agreement@${k} vs accurate`,
    probes,
    meanRecall: mean,
    stdevRecall: stdev(recalls, mean),
    recallAt1: probes.length === 0 ? 0 : probes.filter((p) => p.foundTop1).length / probes.length,
    meanScoreRatio: average(probes.map((p) => p.scoreRatio)),
    medianExactMs: median(probes.map((p) => p.exactMs)),
    medianAnnMs: median(probes.map((p) => p.annMs)),
    histogram,
    worst: [...probes].sort((a, b) => a.recall - b.recall).slice(0, 5),
  };
}

const average = (xs: number[]) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);

/** Bessel's correction: these are a sample of the table's queries, not the population. */
function stdev(xs: number[], mean: number): number {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (xs.length - 1));
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? ((sorted[mid - 1]! + sorted[mid]!) / 2) : sorted[mid]!;
}
