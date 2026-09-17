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

export interface PcaRequest {
  jobId: number;
  data: Float32Array;
  n: number;
  d: number;
  seed: number;
}

export type PcaResponse =
  | { jobId: number; ok: true; positions: Float32Array; explained: [number, number]; ms: number }
  | { jobId: number; ok: false; error: string };

self.onmessage = (event: MessageEvent<PcaRequest>) => {
  const { jobId, data, n, d, seed } = event.data;
  const started = performance.now();
  try {
    const { positions, explained } = pca2d(data, n, d, seed);
    const reply: PcaResponse = {
      jobId,
      ok: true,
      positions,
      explained,
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
