/**
 * Both layouts, off the main thread.
 *
 * PCA at 1,300 points and 768 dimensions is a few hundred million multiply-adds; the graph
 * layout is three hundred steps over four thousand nodes. Either is a frozen tab if it runs on
 * the UI thread, and both are easier to reason about with the buffers transferred in and out —
 * nothing is copied in either direction.
 *
 * `jobId` exists because the controls are a slider and a toggle: a run started two adjustments
 * ago must not land on the current one.
 */

import { pca2d } from './pca';
import { graphLayout } from './layout';
import { preservation, preservationFromGraph, type Metric, type Preservation } from './neighbourhood';

export interface PcaRequest {
  kind: 'pca';
  jobId: number;
  data: Float32Array;
  n: number;
  d: number;
  seed: number;
  /** The table's own metric, so "near" means what it means to the index. */
  metric: Metric;
}

export interface GraphRequest {
  kind: 'graph';
  jobId: number;
  n: number;
  neighbours: number[][];
  scores: number[][];
  seed: number;
  steps: number;
}

export type LayoutRequest = PcaRequest | GraphRequest;

export type PcaResponse =
  | {
      jobId: number;
      ok: true;
      positions: Float32Array;
      /** PCA only: the graph layout has no notion of variance along a component. */
      explained: [number, number] | null;
      preservation: Preservation;
      ms: number;
    }
  | { jobId: number; ok: false; error: string }
  | { jobId: number; progress: number };

const reply = (message: PcaResponse, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(message, transfer);

self.onmessage = (event: MessageEvent<LayoutRequest>) => {
  const request = event.data;
  const started = performance.now();

  try {
    if (request.kind === 'pca') {
      const { data, n, d, seed, metric } = request;
      // Neighbourhoods in the original space, **before** `pca2d` centres the buffer in place.
      // Centring leaves Euclidean distances alone and changes every cosine, so measuring after
      // it would compare against a metric the index does not use.
      const original = data.slice();
      const { positions, explained } = pca2d(data, n, d, seed);
      reply(
        {
          jobId: request.jobId,
          ok: true,
          positions,
          explained,
          preservation: preservation(original, positions, n, d, metric),
          ms: performance.now() - started,
        },
        [positions.buffer],
      );
      return;
    }

    const { n, neighbours, scores, seed, steps, jobId } = request;
    const positions = graphLayout({
      n,
      neighbours,
      scores,
      steps,
      seed,
      onProgress: (step, total) => reply({ jobId, progress: step / total }),
    });
    reply(
      {
        jobId,
        ok: true,
        positions,
        // Variance along a component is a PCA idea; this layout does not have one, and
        // inventing a number here would be worse than admitting the absence.
        explained: null,
        // Measured against the server's own graph: exact, every node, the table's metric.
        preservation: preservationFromGraph(positions, neighbours),
        ms: performance.now() - started,
      },
      [positions.buffer],
    );
  } catch (cause) {
    reply({
      jobId: request.jobId,
      ok: false,
      error: cause instanceof Error ? cause.message : String(cause),
    });
  }
};
