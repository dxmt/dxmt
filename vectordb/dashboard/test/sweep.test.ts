/**
 * The sweep's cost model, which is the thing that would be quietly wrong.
 *
 * Comparing four modes against ground truth looks like four recall runs. Done that way it is
 * four exhaustive passes per sample point — at twenty points, eighty full scans of the table,
 * seventy-five per cent of them recomputing an answer already in hand. The result would be
 * identical and the page would take four times as long, which is exactly the kind of waste
 * nobody notices because nothing is wrong with the output.
 *
 * So: one exhaustive search per point, reused. These tests count the requests.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { sweep } from '../src/quality/recall';
import { APPROXIMATE_MODES } from '../src/api/types';

interface Issued {
  mode: string;
  at: number;
  done?: number;
}

/** `exact` ranks a,b,c,d; `fast` misses two of them; the rest agree. */
const RANKINGS: Record<string, string[]> = {
  exact: ['q', 'a', 'b', 'c', 'd'],
  accurate: ['q', 'a', 'b', 'c', 'd'],
  auto: ['q', 'a', 'b', 'c', 'd'],
  normal: ['q', 'a', 'b', 'x', 'd'],
  fast: ['q', 'a', 'x', 'y', 'd'],
};

function stub() {
  const issued: Issued[] = [];
  vi.stubGlobal('fetch', async (_url: string, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? '{}')) as { search_mode?: string };
    const mode = body.search_mode ?? 'normal';
    const record: Issued = { mode, at: performance.now() };
    issued.push(record);
    await new Promise((resolve) => setTimeout(resolve, 3));
    record.done = performance.now();
    return new Response(
      JSON.stringify({
        success: true,
        data: {
          hits: (RANKINGS[mode] ?? []).map((id, i) => ({ id, score: 1 - i * 0.05, payload: null, vector: null })),
          total: 5,
          tookms: mode === 'exact' ? 40 : 4,
        },
        error: null,
        timestamp: 't',
      }),
      { headers: { 'content-type': 'application/json' } },
    );
  });
  return issued;
}

const options = (issuedIds: string[]) => ({
  table: 't',
  ids: issuedIds,
  k: 4,
  modes: [...APPROXIMATE_MODES],
  truthMode: 'exact' as const,
  vectorOf: () => Promise.resolve([1, 0, 0, 0]),
});

describe('sweep', () => {
  beforeEach(() => vi.unstubAllGlobals());

  it('runs one exhaustive search per point, not one per mode', async () => {
    const issued = stub();
    await sweep(options(['p1', 'p2', 'p3']));

    const exhaustive = issued.filter((i) => i.mode === 'exact');
    expect(exhaustive).toHaveLength(3);
    // Four modes over three points, minus nothing: `exact` is not one of the four being
    // measured, so each point costs one truth plus four comparisons.
    expect(issued).toHaveLength(3 * 5);
  });

  it('never overlaps two searches', async () => {
    const issued = stub();
    await sweep(options(['p1', 'p2']));

    for (let i = 1; i < issued.length; i += 1) {
      expect(
        issued[i]!.at,
        `${issued[i]!.mode} started before ${issued[i - 1]!.mode} finished`,
      ).toBeGreaterThanOrEqual(issued[i - 1]!.done!);
    }
  });

  it('scores each mode against the same truth', async () => {
    stub();
    const reports = await sweep(options(['p1', 'p2']));

    const byMode = Object.fromEntries(reports.map((r) => [r.mode, r]));
    // fast keeps a and d of a,b,c,d.
    expect(byMode.fast!.meanRecall).toBeCloseTo(0.5);
    // normal keeps a, b and d.
    expect(byMode.normal!.meanRecall).toBeCloseTo(0.75);
    expect(byMode.accurate!.meanRecall).toBe(1);
    expect(byMode.auto!.meanRecall).toBe(1);
  });

  it('carries both timings so the trade-off is visible, not just the recall', async () => {
    stub();
    const reports = await sweep(options(['p1']));
    for (const report of reports) {
      expect(report.medianExactMs).toBe(40);
      expect(report.medianAnnMs).toBe(4);
    }
  });

  it('stops when the run is cancelled and keeps what it measured', async () => {
    stub();
    const controller = new AbortController();
    const seen: number[] = [];

    const promise = sweep({
      ...options(['p1', 'p2', 'p3', 'p4', 'p5']),
      signal: controller.signal,
      onProgress: (_line, done) => {
        seen.push(done);
        if (done === 2) controller.abort();
      },
    });

    const reports = await promise;
    expect(seen).toEqual([1, 2]);
    // Two points measured, not five, and a report rather than an exception.
    expect(reports[0]!.probes).toHaveLength(2);
  });

  it('labels the result agreement when it had no ground truth', async () => {
    stub();
    const reports = await sweep({ ...options(['p1']), truthMode: 'accurate' });
    expect(reports[0]!.groundTruth).toBe(false);
    expect(reports[0]!.label).toContain('agreement');
  });
});
