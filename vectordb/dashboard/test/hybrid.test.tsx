/**
 * The text and hybrid columns, and the three ways they could quietly lie.
 *
 * 1. **`took_ms` is not `tookms`.** `/search/text` answers with the underscore and `/search`
 *    and `/search/hybrid` answer without it — two structs on the Rust side, written at
 *    different times. Reading the wrong one yields `undefined`, and `undefined ms server` is
 *    the kind of thing that survives review because it looks like a loading state.
 * 2. **Overlap with `exact` is not recall.** `exact` is ground truth for the dense metric only.
 *    A percentage there would read as a quality score for BM25, when a low overlap is the
 *    entire reason to fuse the two.
 * 3. **The hybrid body's field names are not checked by the server.** serde takes it without
 *    `deny_unknown_fields`, so `alpha` or `top_k` would be accepted, ignored, and leave a
 *    weight control that visibly does nothing.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TableSearch } from '../src/pages/TableSearch';
import { forgetCapabilities } from '../src/api/capabilities';
import { renderAt } from './render';

const table = { name: 't', dimension: 4, record_count: 100, enable_full_text_search: true };

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return { ...actual, useOutletContext: () => ({ table }) };
});

interface Sent {
  url: string;
  body: Record<string, unknown>;
}

/** Dense returns a and b; BM25 returns b and c — one shared id out of two, on purpose. */
function stub({ textFeature = true }: { textFeature?: boolean } = {}) {
  const sent: Sent[] = [];
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
    sent.push({ url, body });

    const envelope = (data: unknown) =>
      Promise.resolve(
        new Response(JSON.stringify({ success: true, data, error: null, timestamp: 't' }), {
          headers: { 'content-type': 'application/json' },
        }),
      );

    if (url.endsWith('/search/text')) {
      if (!textFeature) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              success: false,
              data: null,
              error: {
                code: 'FEATURE_DISABLED',
                message:
                  'Full-text search feature is not enabled. Compile with --features text-bm25',
                details: null,
              },
              timestamp: 't',
            }),
            { headers: { 'content-type': 'application/json' } },
          ),
        );
      }
      const probe = body.query === 'dxmt_capability_probe';
      return envelope({
        hits: probe
          ? []
          : [
              { id: 'b', score: 6.25, payload: null, highlights: null },
              { id: 'c', score: 4.5, payload: null, highlights: null },
            ],
        total: probe ? 0 : 2,
        // With the underscore. This is the trap.
        took_ms: 7,
      });
    }

    if (url.endsWith('/search/hybrid')) {
      return envelope({
        hits: [
          { id: 'a', score: 0.9, payload: null, vector: null },
          { id: 'b', score: 0.8, payload: null, vector: null },
        ],
        total: 2,
        tookms: 11,
      });
    }

    return envelope({
      hits: [
        { id: 'a', score: 1, payload: null, vector: null },
        { id: 'b', score: 0.9, payload: null, vector: null },
      ],
      total: 2,
      tookms: 3,
    });
  });
  return sent;
}

async function run(text: string) {
  const user = userEvent.setup();
  renderAt('/tables/t/search', '/tables/:name/search', <TableSearch />);
  await user.click(screen.getByLabelText(/Query vector/i));
  await user.paste('[1, 0, 0, 0]');
  const box = await screen.findByLabelText(/Text query/i);
  await user.click(box);
  await user.paste(text);
  await user.click(await screen.findByRole('button', { name: /Compare/ }));
  return user;
}

describe('BM25 and hybrid beside the dense modes', () => {
  beforeEach(() => {
    forgetCapabilities();
    vi.unstubAllGlobals();
  });

  it('reads the BM25 timing from took_ms, not tookms', async () => {
    stub();
    await run('quasar');

    await waitFor(() => expect(screen.getByText('BM25')).toBeInTheDocument(), { timeout: 4000 });
    expect(screen.getByText(/7 ms server/)).toBeInTheDocument();
    // The failure this guards against renders as text, not as an exception.
    expect(screen.queryByText(/NaN|undefined ms/)).not.toBeInTheDocument();
  }, 10_000);

  it('counts the shared ids instead of calling the overlap recall', async () => {
    stub();
    await run('quasar');

    // Dense exact returned a and b; BM25 returned b and c.
    await waitFor(() => expect(screen.getByText('shares 1/2 with dense exact')).toBeInTheDocument(), {
      timeout: 4000,
    });
    // Hybrid returned both.
    expect(screen.getByText('shares 2/2 with dense exact')).toBeInTheDocument();
    // The disclaimer is split across elements, so match on the paragraph's text as a whole.
    expect(
      screen.getByText(
        (_, element) => element?.textContent?.includes('is not recall') ?? false,
        { selector: 'p.note' },
      ),
    ).toBeInTheDocument();
  }, 10_000);

  it('sends the field names the server actually reads', async () => {
    const sent = stub();
    await run('quasar');

    await waitFor(() => expect(sent.some((s) => s.url.endsWith('/search/hybrid'))).toBe(true), {
      timeout: 4000,
    });
    const hybrid = sent.find((s) => s.url.endsWith('/search/hybrid'))!;
    expect(hybrid.body).toMatchObject({
      text: 'quasar',
      limit: 10,
      fusion_method: 'reciprocal_rank_fusion',
      vector_weight: 0.7,
    });
    // serde ignores unknown fields here rather than refusing them, so a misspelling would be
    // invisible on both sides.
    expect(hybrid.body).not.toHaveProperty('alpha');
    expect(hybrid.body).not.toHaveProperty('top_k');
  }, 10_000);

  it('names which absence it is when the build has no text-bm25', async () => {
    stub({ textFeature: false });
    const user = userEvent.setup();
    renderAt('/tables/t/search', '/tables/:name/search', <TableSearch />);
    await user.click(screen.getByLabelText(/Query vector/i));
    await user.paste('[1, 0, 0, 0]');

    await waitFor(
      () => expect(screen.getByText(/built without the text-bm25 feature/)).toBeInTheDocument(),
      { timeout: 4000 },
    );
    // And no control that cannot work.
    expect(screen.queryByLabelText(/Text query/i)).not.toBeInTheDocument();
  }, 10_000);
});
