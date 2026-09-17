import { useRef, useState } from 'react';
import { useOutletContext, useParams, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { probeExactMode } from '../api/capabilities';
import { normalizeRecord, vectorFor } from '../api/normalize';
import { APPROXIMATE_MODES, type SearchMode, type Table } from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { useMetrics } from '../api/queries';
import { sampleIds, sweep, type RecallReport } from '../quality/recall';
import { Report } from '../quality/RecallReport';
import { count } from '../lib/format';

/**
 * How much recall is this index losing, on this data.
 *
 * The method is Qdrant's (Apache 2.0 — the method, not the code): take existing records as
 * queries, run each twice, and call the overlap recall. What it needs is a ground truth, which
 * is why `search_mode: "exact"` exists.
 *
 * The cost is the thing to be honest about. Every sample point is one exhaustive pass over the
 * table, so twenty points on a million rows is twenty full scans. The page therefore states
 * what a run will cost *before* it runs, defaults to twenty rather than Qdrant's hundred, and
 * can be stopped — with the partial result kept and labelled rather than thrown away.
 *
 * It also refuses to start while an index is rebuilding. Recall and latency measured during a
 * rebuild are numbers about the rebuild; publishing them as numbers about the index would be
 * worse than publishing nothing.
 */
const DEFAULT_N = 20;
const MAX_N = 200;

export function TableQuality() {
  const { name = '' } = useParams();
  const { table } = useOutletContext<{ table: Table | undefined }>();
  const [params, setParams] = useSearchParams();
  const metrics = useMetrics();

  const n = clamp(Number(params.get('n') ?? DEFAULT_N), 1, MAX_N);
  const k = clamp(Number(params.get('k') ?? 10), 1, 100);
  const seed = Number(params.get('seed') ?? 42);

  const [log, setLog] = useState<string[]>([]);
  const [reports, setReports] = useState<RecallReport[] | null>(null);
  const [partial, setPartial] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [running, setRunning] = useState(false);
  const abort = useRef<AbortController | null>(null);

  const dimension = table?.dimension;
  const capabilities = useQuery({
    queryKey: ['capabilities', name],
    enabled: dimension !== undefined,
    staleTime: Infinity,
    queryFn: ({ signal }) => probeExactMode(name, dimension ?? 0, signal),
  });

  const building = metrics.data?.index_status.building === true;
  const truthMode: SearchMode = capabilities.data?.exactMode ? 'exact' : 'accurate';
  const modes = capabilities.data?.exactMode ? APPROXIMATE_MODES : APPROXIMATE_MODES.filter((m) => m !== 'accurate');
  const tooSmall = (table?.record_count ?? 0) < k + 2;

  const move = (patch: Record<string, string>) => {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(patch)) next.set(key, value);
    setParams(next, { replace: true });
  };

  const start = async () => {
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    setLog([]);
    setReports(null);
    setPartial(false);
    setError(null);

    try {
      // One request for the sample, with vectors, rather than N requests for N records.
      const page = await api.records(
        name,
        { limit: Math.min(1000, Math.max(n * 4, 100)), offset: 0, with_vector: true },
        controller.signal,
      );
      for (const record of page.value.records) normalizeRecord(name, record);

      const available = page.value.records.map((r) => r.id);
      const ids = sampleIds(available, n, seed);
      if (ids.length < n) {
        setLog((lines) => [
          `sampling ${ids.length} of the ${n} asked for — the table has ${available.length} to draw from`,
          ...lines,
        ]);
      }

      const result = await sweep({
        table: name,
        ids,
        k,
        modes,
        truthMode,
        vectorOf: async (id) => Array.from(vectorFor(name, id) ?? []),
        onProgress: (line) => setLog((lines) => [line, ...lines].slice(0, 400)),
        signal: controller.signal,
      });

      setPartial(controller.signal.aborted);
      setReports(result);
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) setError(cause);
    } finally {
      setRunning(false);
      abort.current = null;
    }
  };

  const scans = n;

  return (
    <>
      <div className="searchbox">
        <div className="toolbar">
          <label>
            sample
            <input
              type="number"
              min={1}
              max={MAX_N}
              value={n}
              style={{ width: 70 }}
              onChange={(event) => move({ n: event.target.value })}
            />
          </label>
          <label>
            top_k
            <select value={k} onChange={(event) => move({ k: event.target.value })}>
              {[1, 5, 10, 20, 50].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label title="The same seed draws the same sample, so a run is reproducible from this URL.">
            seed
            <input
              type="number"
              value={seed}
              style={{ width: 80 }}
              onChange={(event) => move({ seed: event.target.value })}
            />
          </label>
          {running ? (
            <button type="button" onClick={() => abort.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="button" disabled={building || tooSmall} onClick={() => void start()}>
              Measure
            </button>
          )}
        </div>

        <p className="muted">
          {scans} exhaustive {scans === 1 ? 'pass' : 'passes'} over{' '}
          {count(table?.record_count ?? 0)} records, plus {scans * modes.length} indexed
          searches. Each point is one full scan — that is what makes the comparison worth
          anything and what makes it slow.
        </p>
      </div>

      {building ? (
        <div className="banner warn">
          <strong>An index is rebuilding</strong>
          <span className="detail">
            Every query on this server is waiting behind it right now. Recall measured here would
            be a number about the rebuild, so the run is held until it finishes.
          </span>
        </div>
      ) : null}

      {tooSmall ? (
        <div className="banner warn">
          <strong>Too few records to measure recall@{k}</strong>
          <span className="detail">
            The query point is excluded from its own results, so this needs at least {k + 2}{' '}
            records and the table has {count(table?.record_count ?? 0)}.
          </span>
        </div>
      ) : null}

      {capabilities.data && !capabilities.data.exactMode ? (
        <div className="banner warn">
          <strong>No ground truth — measuring agreement instead</strong>
          <span className="detail">{capabilities.data.exactModeReason}</span>
          <span className="mono">accurate</span> stands in for the truth below. It is 4× the
          traversal budget and about 0.98 itself, so what you get is an upper bound, blind to
          whatever every mode misses together.
        </div>
      ) : null}

      {error ? <ErrorBanner error={error} /> : null}

      {partial ? (
        <div className="banner warn">
          <strong>Stopped early</strong>
          <span className="detail">
            These figures cover the points that finished, not the {n} asked for.
          </span>
        </div>
      ) : null}

      {reports ? <Report table={name} reports={reports} /> : null}

      {log.length > 0 ? (
        <details className="request" open={running}>
          <summary>
            {running ? 'running…' : 'run log'} ({log.length}{' '}
            {log.length === 1 ? 'line' : 'lines'})
          </summary>
          <pre className="runlog">{log.join('\n')}</pre>
        </details>
      ) : null}
    </>
  );
}

const clamp = (value: number, low: number, high: number) =>
  Number.isFinite(value) ? Math.min(high, Math.max(low, Math.round(value))) : low;
