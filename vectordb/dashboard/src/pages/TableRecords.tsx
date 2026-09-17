import { useMemo, useState } from 'react';
import { Link, useOutletContext, useParams, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { forgetVectors, normalizeRecord, type NormalRecord } from '../api/normalize';
import type { Filter, Table } from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { FilterEditor } from '../components/FilterEditor';
import { JsonBlock } from '../components/JsonBlock';
import { Pager } from '../components/Pager';
import { RequestPreview } from '../components/RequestPreview';
import { VectorCell } from '../components/VectorCell';
import { count, since } from '../lib/format';

/**
 * The record browser, designed around one fact about the endpoint.
 *
 * `POST /records/list` does not page cheaply. The handler collects every matching record, sorts
 * the whole set by `created_at`, and then slices — so `limit` and `offset` shape the response,
 * not the work, and every page of a million-row table costs a million-row sort. Everything here
 * follows from that:
 *
 *   - **nothing is fetched automatically** above `AUTO_LOAD` rows. The page shows the count, the
 *     cost, and a button, and says why;
 *   - page sizes stop at 100. The server pays the same either way, but the browser does not, and
 *     nobody reads 1,000 rows;
 *   - no infinite scroll, no prefetching the next page, no polling. `staleTime: Infinity` and an
 *     explicit Refresh, so a tab switch cannot trigger a table scan;
 *   - vectors are off by default and behind a toggle that states what they cost.
 *
 * The ordering note is not pedantry either: `created_at` descending means a concurrent insert
 * shifts every page by one, and this is a read-only console attached to a writable server.
 */
const AUTO_LOAD = 50_000;
const PAGE_SIZES = [25, 50, 100];

export function TableRecords() {
  const { name = '' } = useParams();
  const { table } = useOutletContext<{ table: Table | undefined }>();
  const [params, setParams] = useSearchParams();

  const offset = Number(params.get('offset') ?? 0);
  const limit = Number(params.get('limit') ?? 25);
  const withVector = params.get('vectors') === '1';
  const [filter, setFilter] = useState<Filter | null>(null);
  const [asked, setAsked] = useState(false);

  const big = (table?.record_count ?? 0) > AUTO_LOAD;
  const enabled = asked || !big;

  const body = useMemo(
    () => ({ limit, offset, with_vector: withVector, ...(filter ? { filter } : {}) }),
    [limit, offset, withVector, filter],
  );

  const page = useQuery({
    queryKey: ['records', name, body],
    enabled,
    // The one setting that matters here: without it, every return to this tab is another full
    // table scan on the server.
    staleTime: Infinity,
    queryFn: async ({ signal }) => {
      forgetVectors(name);
      const response = await api.records(name, body, signal);
      return {
        records: response.value.records.map((r) => normalizeRecord(name, r)),
        total: response.value.total,
        ms: response.ms,
        wire: response.bytes,
      };
    },
  });

  const payloadKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const record of page.data?.records ?? []) {
      for (const key of Object.keys(record.payload ?? {})) keys.add(key);
    }
    return [...keys].sort();
  }, [page.data]);

  const move = (patch: Record<string, string>) => {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(patch)) next.set(key, value);
    setParams(next, { replace: true });
  };

  return (
    <>
      <FilterEditor
        keys={payloadKeys}
        busy={page.isFetching}
        onApply={(next) => {
          setFilter(next);
          move({ offset: '0' });
          setAsked(true);
        }}
      />

      <div className="toolbar">
        <label>
          rows
          <select
            value={limit}
            onChange={(event) => move({ limit: event.target.value, offset: '0' })}
          >
            {PAGE_SIZES.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </label>
        <label title="A page of 100 rows at 768 dimensions is over a megabyte of JSON floats, and this server does not compress its API responses.">
          <input
            type="checkbox"
            checked={withVector}
            onChange={(event) => move({ vectors: event.target.checked ? '1' : '0' })}
          />
          fetch vectors
        </label>
        <button type="button" disabled={page.isFetching} onClick={() => void page.refetch()}>
          {page.isFetching ? 'loading…' : 'Refresh'}
        </button>
      </div>

      {!enabled ? (
        <div className="banner warn">
          <strong>{count(table?.record_count ?? 0)} records — not loaded automatically</strong>
          <span className="detail">
            Listing records is answered by collecting every matching record and sorting the whole
            table, so each page costs the same as scanning the table once. Loading is a
            deliberate click here rather than something a tab switch does.
          </span>
          <p style={{ marginBottom: 0 }}>
            <button type="button" onClick={() => setAsked(true)}>
              Load {limit} records anyway
            </button>
          </p>
        </div>
      ) : null}

      {page.error ? <ErrorBanner error={page.error} /> : null}

      {page.data ? (
        <>
          <RequestPreview
            method="POST"
            path={`/api/v1/tables/${encodeURIComponent(name)}/records/list`}
            body={body}
            ms={page.data.ms}
            wire={page.data.wire}
          />

          <table>
            <thead>
              <tr>
                <th>id</th>
                <th>vector</th>
                <th>payload</th>
                <th>created</th>
                <th className="num">v</th>
              </tr>
            </thead>
            <tbody>
              {page.data.records.map((record) => (
                <RecordRow key={record.id} table={name} record={record} />
              ))}
            </tbody>
          </table>

          <Pager
            offset={offset}
            limit={limit}
            total={page.data.total}
            busy={page.isFetching}
            onOffset={(next) => move({ offset: String(next) })}
          />

          <p className="note">
            Ordered by <span className="mono">created_at</span> descending. This console only
            reads, but the server still accepts writes — an insert while you are paging shifts
            every page by one.
          </p>
        </>
      ) : null}
    </>
  );
}

function RecordRow({ table, record }: { table: string; record: NormalRecord }) {
  const [open, setOpen] = useState(false);
  const entries = Object.entries(record.payload ?? {});

  return (
    <>
      <tr onClick={() => setOpen((on) => !on)} className="clickable">
        <td>
          <Link to={`/tables/${encodeURIComponent(table)}/records/${encodeURIComponent(record.id)}`} onClick={(e) => e.stopPropagation()}>
            {record.id}
          </Link>
        </td>
        <td>
          <VectorCell table={table} id={record.id} dim={record.dim} norm={record.norm} />
        </td>
        <td>
          {record.payloadError ? (
            <span className="badge warn" title={record.payloadError}>
              unparseable
            </span>
          ) : entries.length === 0 ? (
            <span className="muted">—</span>
          ) : (
            entries.slice(0, 3).map(([key, value]) => (
              <span className="chip" key={key}>
                {key}: {String(JSON.stringify(value)).slice(0, 24)}
              </span>
            ))
          )}
          {entries.length > 3 ? <span className="muted"> +{entries.length - 3}</span> : null}
        </td>
        <td title={record.createdAt ?? undefined}>{since(record.createdAt)}</td>
        <td className="num">{record.version ?? '—'}</td>
      </tr>
      {open ? (
        <tr>
          <td colSpan={5} className="expanded">
            <JsonBlock
              value={record.payloadError ? undefined : record.payload}
              raw={record.payloadRaw ?? undefined}
            />
            {record.payloadError ? (
              <p className="err">
                The server sent a payload_json this could not parse: {record.payloadError}
              </p>
            ) : null}
          </td>
        </tr>
      ) : null}
    </>
  );
}
