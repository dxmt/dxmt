/**
 * The record browser's two load-bearing behaviours, asserted in a DOM.
 *
 * Both are about not doing something. A vector cell must not render the vector, and a large
 * table must not be fetched because someone navigated to it — the endpoint answers a page by
 * sorting the entire table, so an automatic fetch is an automatic table scan.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { TableRecords } from '../src/pages/TableRecords';
import { forgetVectors } from '../src/api/normalize';
import { renderAt, stubFetch } from './render';

const DIM = 768;

function record(i: number) {
  return {
    id: `r${i}`,
    vector: Array.from({ length: DIM }, (_, d) => (d === i % DIM ? 1 : 0.0625)),
    sparse_vector: null,
    named_vectors: null,
    payload_json: JSON.stringify({ topic: 'graphs', rank: i }),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    version: 1,
    status: 'Active',
  };
}

// `TableRecords` reads the table from the router outlet; supply it directly instead.
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return { ...actual, useOutletContext: () => ({ table: tableStub }) };
});

let tableStub: { record_count: number; dimension: number } = { record_count: 300, dimension: DIM };

describe('the record browser', () => {
  beforeEach(() => {
    forgetVectors('t');
    vi.unstubAllGlobals();
  });

  it('never renders the vector itself', async () => {
    tableStub = { record_count: 3, dimension: DIM };
    vi.stubGlobal(
      'fetch',
      stubFetch(() => ({ records: [record(1), record(2)], total: 2, limit: 25, offset: 0 })),
    );

    renderAt('/tables/t/records?vectors=1', '/tables/:name/records', <TableRecords />);
    await screen.findByText('r1');

    // The summary is on screen...
    expect(screen.getAllByText(/768d/).length).toBeGreaterThan(0);
    // ...and the 0.0625 that fills 767 of the 768 components is not, anywhere.
    expect(document.body.textContent).not.toContain('0.0625');
  });

  it('does not fetch a large table until asked, and says why', async () => {
    tableStub = { record_count: 2_000_000, dimension: DIM };
    const calls: string[] = [];
    vi.stubGlobal(
      'fetch',
      stubFetch((url) => {
        calls.push(url);
        return { records: [], total: 0, limit: 25, offset: 0 };
      }),
    );

    renderAt('/tables/t/records', '/tables/:name/records', <TableRecords />);

    await screen.findByText(/not loaded automatically/i);
    expect(screen.getByText(/sorting the whole table/i)).toBeInTheDocument();
    // The important half: nothing was requested.
    await waitFor(() => expect(calls).toHaveLength(0));
  });

  it('fetches a small table without being asked', async () => {
    tableStub = { record_count: 8, dimension: DIM };
    const calls: string[] = [];
    vi.stubGlobal(
      'fetch',
      stubFetch((url) => {
        calls.push(url);
        return { records: [record(1)], total: 1, limit: 25, offset: 0 };
      }),
    );

    renderAt('/tables/t/records', '/tables/:name/records', <TableRecords />);
    await screen.findByText('r1');
    expect(calls).toHaveLength(1);
    expect(calls[0]).toContain('/records/list');
  });

  it('shows a payload that will not parse rather than blanking the row', async () => {
    tableStub = { record_count: 1, dimension: DIM };
    vi.stubGlobal(
      'fetch',
      stubFetch(() => ({
        records: [{ ...record(1), payload_json: '{not json' }],
        total: 1,
        limit: 25,
        offset: 0,
      })),
    );

    renderAt('/tables/t/records', '/tables/:name/records', <TableRecords />);
    await screen.findByText('r1');
    expect(screen.getByText('unparseable')).toBeInTheDocument();
  });

  it('says the ordering is not stable under concurrent writes', async () => {
    tableStub = { record_count: 2, dimension: DIM };
    vi.stubGlobal(
      'fetch',
      stubFetch(() => ({ records: [record(1)], total: 1, limit: 25, offset: 0 })),
    );

    renderAt('/tables/t/records', '/tables/:name/records', <TableRecords />);
    await screen.findByText('r1');
    expect(screen.getByText(/shifts every page/i)).toBeInTheDocument();
  });
});
