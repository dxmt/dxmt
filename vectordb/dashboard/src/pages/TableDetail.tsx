import { NavLink, Outlet, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { ErrorBanner } from '../components/ErrorBanner';
import { bytes, count } from '../lib/format';

/**
 * One table, with the numbers that are not on the list page.
 *
 * Two are derived rather than served. **Bytes per record** is the figure the disk-index work
 * was accepted on — about 0.5 KB per 768-dimension row, against 8.3 KB when everything lived in
 * the process — and nothing prints it. And when the three sources of `record_count` disagree,
 * all three are shown: a console that quietly picks one is throwing away the bug report.
 */
export function TableDetail() {
  const { name = '' } = useParams();

  const detail = useQuery({
    queryKey: ['table', name],
    queryFn: ({ signal }) => api.table(name, signal).then((r) => r.value),
  });
  const stats = useQuery({
    queryKey: ['stats', name],
    queryFn: ({ signal }) => api.stats(name, signal).then((r) => r.value),
  });

  if (detail.error) return <ErrorBanner error={detail.error} />;

  const table = detail.data?.table;
  const counts = new Set(
    [table?.record_count, detail.data?.record_count, stats.data?.record_count].filter(
      (value): value is number => typeof value === 'number',
    ),
  );
  const perRecord =
    stats.data && stats.data.record_count > 0
      ? stats.data.memory_usage_bytes / stats.data.record_count
      : null;

  return (
    <>
      <h2>{name}</h2>
      <p className="sub">
        {table ? (
          <>
            {table.dimension} dimensions · {table.distance_metric} ·{' '}
            <span className="mono">{table.index_type}</span>
            {stats.data ? (
              <>
                {' · '}
                <span className={`badge ${stats.data.resident ? 'ok' : ''}`}>
                  {stats.data.resident ? 'resident' : 'evicted'}
                </span>
              </>
            ) : null}
          </>
        ) : (
          'loading…'
        )}
      </p>

      {counts.size > 1 ? (
        <div className="banner warn">
          <strong>Three record counts disagree</strong>
          <span className="detail">
            /tables says {table?.record_count}, /tables/{name} says {detail.data?.record_count},
            /stats says {stats.data?.record_count}. Showing all three rather than choosing one.
          </span>
        </div>
      ) : null}

      <div className="rowstats">
        <span>
          <b>{count([...counts][0] ?? 0)}</b> records
        </span>
        {stats.data ? (
          <>
            <span>
              <b>{bytes(stats.data.index_size_bytes)}</b> index
            </span>
            <span>
              <b>{bytes(stats.data.memory_usage_bytes)}</b> resident
            </span>
            <span>
              <b>{bytes(stats.data.disk_usage_bytes)}</b> on disk
            </span>
            {perRecord === null ? null : (
              <span title="Resident bytes divided by records — the figure the on-disk index was accepted on.">
                <b>{bytes(perRecord)}</b> per record
              </span>
            )}
          </>
        ) : null}
        {table?.enable_full_text_search ? <span className="badge">full text</span> : null}
      </div>

      <nav className="tabs">
        <NavLink to="records" className={({ isActive }) => (isActive ? 'on' : '')}>
          Records
        </NavLink>
      </nav>

      <Outlet context={{ table }} />
    </>
  );
}
