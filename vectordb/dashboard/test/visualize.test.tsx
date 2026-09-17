import { describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TableVisualize } from '../src/pages/TableVisualize';
import { renderAt } from './render';
import type { LayoutRequest, PcaResponse } from '../src/viz/pca.worker';
import { Scatter } from '../src/viz/scatter';

const TABLE = { name: 't', dimension: 8, record_count: 500, distance_metric: 'cosine' };
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return { ...actual, useOutletContext: () => ({ table: TABLE }) };
});

/** jsdom has no Worker; this answers with a layout so the page can reach its rendered state. */
class FakeWorker {
  onmessage: ((event: MessageEvent<PcaResponse>) => void) | null = null;

  postMessage(request: LayoutRequest) {
    const positions = new Float32Array(request.n * 2);
    for (let i = 0; i < request.n; i += 1) {
      positions[i * 2] = i % 17;
      positions[i * 2 + 1] = i % 13;
    }
    const data: PcaResponse = {
      jobId: request.jobId,
      ok: true,
      positions,
      // Honouring `kind` matters: the graph layout has no variance, and a stub that invents
      // one would hide the page's own admission of that.
      explained: request.kind === 'pca' ? [0.4, 0.2] : null,
      preservation: { rate: 0.5, k: 10, probes: 50 },
      ms: 5,
    };
    setTimeout(() => this.onmessage?.({ data } as MessageEvent<PcaResponse>), 0);
  }

  terminate() {}
}
vi.stubGlobal('Worker', FakeWorker);
vi.stubGlobal('URL', globalThis.URL);

/**
 * The colour-by control is usable before anything is projected.
 *
 * It was not: the keys came only from the projected sample, so the control sat there enabled
 * and holding one meaningless option until Project had already been pressed — and nothing on
 * the page said why. The natural order is the other way round, because what you colour by is
 * part of deciding what you are looking for.
 */
describe('colour by', () => {
  it('offers the payload fields before Project is pressed', async () => {
    vi.stubGlobal('fetch', async () => {
      const records = Array.from({ length: 200 }, (_, i) => ({
        id: `d${i}`, vector: Array.from({ length: 8 }, (_, j) => (i + j) % 5),
        sparse_vector: null, named_vectors: null,
        payload_json: JSON.stringify({ topic: `t${i % 4}`, index: i, content: 'x' }),
        created_at: 'x', updated_at: 'x', version: 1, status: 'Active',
      }));
      return new Response(JSON.stringify({ success: true,
        data: { records, total: 500, limit: 200, offset: 0 }, error: null, timestamp: 't' }),
        { headers: { 'content-type': 'application/json' } });
    });

    const user = userEvent.setup();
    renderAt('/tables/t/visualize?layout=pca', '/tables/:name/visualize', <TableVisualize />);

    const options = () =>
      [...screen.getByLabelText(/colour by/i).querySelectorAll('option')].map((o) => o.textContent);

    // The assertion that was failing in the browser, before anything is clicked.
    await waitFor(() => expect(options()).toContain('topic'), { timeout: 5000 });
    expect(screen.getByLabelText(/colour by/i)).toBeEnabled();

    // And still there once a projection exists, rather than being replaced by it.
    await user.click(screen.getByRole('button', { name: /lay out/i }));
    await waitFor(() => expect(screen.getByText(/of variance/)).toBeInTheDocument(), { timeout: 5000 });
    expect(options()).toContain('topic');
  }, 20000);

  it('hands the projected points to the renderer, not just to the header', async () => {
    // The bug this exists for: the renderer was built in an effect with an empty dependency
    // list, while the canvas was still inside `{positions ? … : null}` and therefore absent.
    // The effect bailed, nothing was ever drawn, and every number on the page was still
    // correct — they come from the worker. A test that checked the text would have passed.
    const points = vi.spyOn(Scatter.prototype, 'setPoints');
    const user = userEvent.setup();
    renderAt('/tables/t/visualize?layout=pca', '/tables/:name/visualize', <TableVisualize />);

    await user.click(screen.getByRole('button', { name: /lay out/i }));
    await waitFor(() => expect(screen.getByText(/of variance/)).toBeInTheDocument(), {
      timeout: 5000,
    });

    await waitFor(() => expect(points).toHaveBeenCalled(), { timeout: 5000 });
    const drawn = points.mock.calls.at(-1)![0];
    expect(drawn.length).toBeGreaterThan(0);
    // Real coordinates, not a list of NaNs, which is the other way this goes quietly wrong.
    expect(drawn.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y))).toBe(true);
    points.mockRestore();
  }, 20000);
});
/**
 * The graph layout, which is the reason the server grew a `/graph` route.
 *
 * Its whole claim is that it never downloads a vector: the server answers with the neighbour
 * structure instead of the inputs, which is what lets the sample be four thousand nodes at any
 * dimension rather than thirteen hundred at 768. A layout that quietly fell back to fetching
 * records would keep working and lose exactly that.
 */
describe('the neighbour-graph layout', () => {
  it('asks for the graph and never for the vectors', async () => {
    const asked: string[] = [];
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      asked.push(url);
      if (url.endsWith('/graph')) {
        const n = 60;
        const body = JSON.parse(String(init?.body ?? '{}')) as { with_payload?: boolean };
        return new Response(
          JSON.stringify({
            success: true,
            data: {
              ids: Array.from({ length: n }, (_, i) => `d${i}`),
              // A ring: every node's neighbours are its two on either side.
              neighbours: Array.from({ length: n }, (_, i) =>
                [(i + 1) % n, (i - 1 + n) % n, (i + 2) % n, (i - 2 + n) % n],
              ),
              scores: Array.from({ length: n }, () => [0.99, 0.98, 0.9, 0.89]),
              sampled: n,
              total: 5000,
              payloads:
                body.with_payload === false
                  ? undefined
                  : Array.from({ length: n }, (_, i) => JSON.stringify({ topic: `t${i % 4}` })),
            },
            error: null,
            timestamp: 't',
          }),
          { headers: { 'content-type': 'application/json' } },
        );
      }
      // The payload-key probe on arrival is a records/list; anything else would be the
      // fallback this test exists to forbid.
      return new Response(
        JSON.stringify({
          success: true,
          data: { records: [], total: 0, limit: 20, offset: 0 },
          error: null,
          timestamp: 't',
        }),
        { headers: { 'content-type': 'application/json' } },
      );
    });

    const points = vi.spyOn(Scatter.prototype, 'setPoints');
    const user = userEvent.setup();
    renderAt('/tables/t/visualize', '/tables/:name/visualize', <TableVisualize />);

    await user.click(screen.getByRole('button', { name: /lay out/i }));
    await waitFor(() => expect(points).toHaveBeenCalled(), { timeout: 10000 });

    expect(asked.some((u) => u.endsWith('/graph'))).toBe(true);
    // The probe asks for twenty records without vectors; nothing else may.
    const vectorFetches = asked.filter((u) => u.includes('/records/list'));
    expect(vectorFetches.length).toBeLessThanOrEqual(1);

    // A ring laid out from its own graph should keep most neighbourhoods — that is the claim.
    expect(screen.getByText(/neighbourhoods kept/)).toBeInTheDocument();
    // And it says it has no variance rather than inventing one, since that is a PCA idea.
    expect(screen.getByText(/variance is a PCA idea/)).toBeInTheDocument();
    points.mockRestore();
  }, 30000);
});
