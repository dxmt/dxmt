/**
 * PCA off the main thread.
 *
 * At 1,300 points and 768 dimensions the iteration is a few hundred million multiply-adds —
 * under a second, and a frozen tab for all of it if it runs on the UI thread. The buffer is
 * transferred in and the result transferred out, so nothing is copied in either direction.
 *
 * `jobId` exists because the sample-size slider is a slider: a run started two adjustments ago
 * must not land on top of the current one.
 */

import { pca2d } from './pca';
import { preservation, type Metric, type Preservation } from './neighbourhood';

export interface PcaRequest {
  jobId: number;
  data: Float32Array;
  n: number;
  d: number;
  seed: number;
  /** The table's own metric, so "near" means what it means to the index. */
  metric: Metric;
}

export type PcaResponse =
  | {
      jobId: number;
      ok: true;
      positions: Float32Array;
      explained: [number, number];
      preservation: Preservation;
      ms: number;
    }
  | { jobId: number; ok: false; error: string };

self.onmessage = (event: MessageEvent<PcaRequest>) => {
  const { jobId, data, n, d, seed, metric } = event.data;
  const started = performance.now();
  try {
    // Neighbourhoods in the original space, **before** `pca2d` centres the buffer in place.
    // Centring leaves Euclidean distances alone and changes every cosine, so measuring after
    // it would compare against a metric the index does not use.
    const original = data.slice();

    const { positions, explained } = pca2d(data, n, d, seed);
    const preserved = preservation(original, positions, n, d, metric);

    const reply: PcaResponse = {
      jobId,
      ok: true,
      positions,
      explained,
      preservation: preserved,
      ms: performance.now() - started,
    };
    (self as unknown as Worker).postMessage(reply, [positions.buffer]);
  } catch (cause) {
    const reply: PcaResponse = {
      jobId,
      ok: false,
      error: cause instanceof Error ? cause.message : String(cause),
    };
    (self as unknown as Worker).postMessage(reply);
  }
};
