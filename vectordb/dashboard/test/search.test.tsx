/**
 * The search inspector's two promises: it compares the modes fairly, and it says so honestly
 * when it cannot.
 *
 * "Fairly" means sequentially. Two searches running at once contend for the index and the page
 * cache, and the `tookms` each reports is then partly a measurement of the other — which would
 * quietly invalidate the only number on the page anyone would act on.
 *
 * "Honestly" means that without `search_mode: "exact"` there is no ground truth, and a column
 * labelled as a percentage of exact must not appear.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TableSearch } from '../src/pages/TableSearch';
import { forgetCapabilities } from '../src/api/capabilities';
import { renderAt } from './render';

const TABLE = { name: 't', dimension: 4, record_count: 100 };

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return { ...actual, useOutletContext: () => ({ table: TABLE }) };
});

/** Ids per mode, so a column can be told from its neighbour. */
const BY_MODE: Record<string, string[]> = {
  fast: ['a', 'z'],
  normal: ['a', 'b'],
  accurate: ['a', 'b'],
  auto: ['a', 'b'],
  exact: ['a', 'b'],
};

interface Call {
  mode: string;
  at: number;
  done?: number;
}

function stubSearch({ exact = true }: { exact?: boolean } = {}) {
  const calls: Call[] = [];
  vi.stubGlobal('fetch', async (_url: string, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? '{}')) as { search_mode?: string };
    const mode = body.search_mode ?? 'normal';

    if (mode === 'exact' && !exact) {
      // Verbatim from a server that predates the variant: the extractor refuses the body
      // before any handler runs, so this is plain text and not the error envelope.
      return new Response(
        'Failed to deserialize the JSON body into the target type: search_mode: unknown variant `exact`',
        { status: 422, headers: { 'content-type': 'text/plain' } },
      );
    }

    const call: Call = { mode, at: performance.now() };
    calls.push(call);
    await new Promise((resolve) => setTimeout(resolve, 12));
    call.done = performance.now();

    return new Response(
      JSON.stringify({
        success: true,
        data: {
          hits: (BY_MODE[mode] ?? []).map((id, i) => ({ id, score: 1 - i * 0.1, payload: null, vector: null })),
          total: 2,
          tookms: 3,
        },
        error: null,
        timestamp: 't',
      }),
      { headers: { 'content-type': 'application/json' } },
    );
  });
  return calls;
}

async function runComparison() {
  const user = userEvent.setup();
  renderAt('/tables/t/search', '/tables/:name/search', <TableSearch />);
  // `paste`, not `type`: user-event reads `[` as the start of a key descriptor, and pasting
  // is what someone actually does with a vector.
  await user.click(screen.getByLabelText(/Query vector/i));
  await user.paste('[1, 0, 0, 0]');
  const button = await screen.findByRole('button', { name: /Compare \d+ modes/ });
  await user.click(button);
  return user;
}

describe('the search inspector', () => {
  beforeEach(() => {
    forgetCapabilities();
    vi.unstubAllGlobals();
  });

  it('runs the modes one at a time, never overlapping', async () => {
    const calls = stubSearch();
    await runComparison();

    await waitFor(() => expect(calls.filter((c) => c.done).length).toBeGreaterThanOrEqual(5), {
      timeout: 4000,
    });

    // Not "five requests were made" — five requests that never overlapped. Concurrency here
    // would make every `tookms` on the page partly a measurement of the other four.
    const searches = calls.filter((c) => c.done);
    for (let i = 1; i < searches.length; i += 1) {
      expect(
        searches[i]!.at,
        `${searches[i]!.mode} started before ${searches[i - 1]!.mode} finished`,
      ).toBeGreaterThanOrEqual(searches[i - 1]!.done!);
    }
  }, 10_000);

  it('marks what each approximate mode missed against exact', async () => {
    stubSearch();
    await runComparison();

    // `fast` returns z where exact returns b: one of two, so half.
    await waitFor(() => expect(screen.getByText('50% of exact')).toBeInTheDocument(), {
      timeout: 4000,
    });
    // The modes that agree with exact say so too.
    expect(screen.getAllByText('100% of exact').length).toBeGreaterThan(0);
  }, 10_000);

  it('says there is no ground truth rather than calling agreement recall', async () => {
    stubSearch({ exact: false });
    await runComparison();

    await waitFor(() => expect(screen.getByText(/No ground truth on this server/i)).toBeInTheDocument(), {
      timeout: 4000,
    });
    expect(screen.getByText(/upper bound/i)).toBeInTheDocument();
    // And no column claims a percentage of something that was never measured.
    expect(screen.queryByText(/of exact/)).not.toBeInTheDocument();
  }, 10_000);

  it('warns about a wrong-length vector but still lets it be sent', async () => {
    stubSearch();
    const user = userEvent.setup();
    renderAt('/tables/t/search', '/tables/:name/search', <TableSearch />);
    await user.click(screen.getByLabelText(/Query vector/i));
    await user.paste('[1, 0]');

    expect(await screen.findByText(/this table has 4 dimensions and that is 2/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Compare/ })).toBeDisabled();

    // Seeing the server's own DIMENSION_MISMATCH is sometimes the point of being here.
    await user.click(screen.getByLabelText(/send it anyway/i));
    expect(screen.getByRole('button', { name: /Compare/ })).toBeEnabled();
  }, 10_000);
});
