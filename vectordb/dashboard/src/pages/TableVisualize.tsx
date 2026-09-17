import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useOutletContext, useParams, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import { normalizeRecord, vectorFor } from '../api/normalize';
import type { Table } from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { Scatter } from '../viz/scatter';
import type { PcaRequest, PcaResponse } from '../viz/pca.worker';
import type { Metric, Preservation } from '../viz/neighbourhood';
import { budgetFor, strata, PAGE } from '../lib/sample';
import { colorBy, type Coloring } from '../lib/colors';
import { bytes, count } from '../lib/format';

/**
 * A 2-D projection of a sample, with the caveat attached rather than implied.
 *
 * Qdrant's equivalent lays out a server-computed k-nearest-neighbour graph and never moves raw
 * vectors to the browser; DXMT has no distance-matrix endpoint and this console adds none, so
 * PCA on sampled vectors is the only path — and PCA on a high-dimensional embedding is a weak
 * summary. Two components of a 768-dimension corpus typically explain five to fifteen per cent
 * of the variance. Drawn plainly, that is a picture people read as a cluster map.
 *
 * Five things make it honest instead, and none is optional:
 *
 *   1. the explained-variance ratio is in the header, not a tooltip;
 *   2. below a quarter, a caption states that distance here is not distance in the embedding;
 *   3. nothing is drawn that implies structure — no hulls, no density, no "clusters";
 *   4. **clicking a point runs a real search** and rings its true neighbours *in the plot*. If
 *      they scatter, the reader watches the projection fail, in the place they were about to be
 *      misled by it;
 *   5. the sample is stated — how many of how many, drawn how, and why the ceiling is there.
 */
export function TableVisualize() {
  const { name = '' } = useParams();
  const { table } = useOutletContext<{ table: Table | undefined }>();
  const [params, setParams] = useSearchParams();

  const canvas = useRef<HTMLCanvasElement>(null);
  const scatter = useRef<Scatter | null>(null);
  const worker = useRef<Worker | null>(null);
  const job = useRef(0);

  const [ids, setIds] = useState<string[]>([]);
  const [payloads, setPayloads] = useState<Record<string, unknown>[]>([]);
  const [positions, setPositions] = useState<Float32Array | null>(null);
  const [explained, setExplained] = useState<[number, number]>([0, 0]);
  const [preserved, setPreserved] = useState<Preservation | null>(null);
  const [shortfall, setShortfall] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [neighbours, setNeighbours] = useState<{ ids: string[]; outside: number } | null>(null);

  const dimension = table?.dimension ?? 0;
  const records = table?.record_count ?? 0;
  const budget = useMemo(() => budgetFor(records, dimension), [records, dimension]);
  const wanted = Math.min(Number(params.get('n') ?? budget.points), budget.points);
  const colorKey = params.get('by') ?? '';

  const keys = useMemo(() => {
    const seen = new Set<string>();
    for (const payload of payloads) for (const key of Object.keys(payload)) seen.add(key);
    return [...seen].sort();
  }, [payloads]);

  const coloring: Coloring = useMemo(
    () => colorBy(payloads.map((payload) => (colorKey ? payload[colorKey] : undefined))),
    [payloads, colorKey],
  );

  useEffect(() => {
    worker.current = new Worker(new URL('../viz/pca.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.current.onmessage = (event: MessageEvent<PcaResponse>) => {
      // A run started two slider adjustments ago must not land on the current one.
      if (event.data.jobId !== job.current) return;
      if (!event.data.ok) {
        setError(new Error(event.data.error));
        setStatus(null);
        return;
      }
      setPositions(event.data.positions);
      setExplained(event.data.explained);
      setPreserved(event.data.preservation);
      setStatus(`projected in ${Math.round(event.data.ms)} ms`);
    };
    return () => worker.current?.terminate();
  }, []);

  useEffect(() => {
    if (!canvas.current) return;
    scatter.current = new Scatter(canvas.current, {
      onHover: setHover,
      onClick: (index) => void verify(index),
    });
    return () => scatter.current?.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!positions || !scatter.current) return;
    scatter.current.setPoints(
      ids.map((_, index) => ({
        x: positions[index * 2]!,
        y: positions[index * 2 + 1]!,
        color: coloring.colorAt(index),
      })),
    );
  }, [positions, ids, coloring]);

  const load = async () => {
    setError(null);
    setNeighbours(null);
    setSelected(null);
    setPositions(null);
    setPreserved(null);
    setStatus('fetching…');

    try {
      const requests = Math.max(1, Math.ceil(wanted / PAGE));
      const offsets = strata(records, requests);
      const perRequest = Math.ceil(wanted / requests);

      const collectedIds: string[] = [];
      const collectedPayloads: Record<string, unknown>[] = [];
      for (const offset of offsets) {
        const response = await api.records(name, {
          limit: Math.min(perRequest, PAGE),
          offset,
          with_vector: true,
        });
        for (const wire of response.value.records) {
          const record = normalizeRecord(name, wire);
          if (record.dim === 0) continue;
          collectedIds.push(record.id);
          collectedPayloads.push(record.payload ?? {});
        }
      }

      if (collectedIds.length < 3) {
        setError(new Error(`only ${collectedIds.length} vectors came back; PCA needs three`));
        setStatus(null);
        return;
      }
      setShortfall(
        collectedIds.length < wanted
          ? `asked for ${count(wanted)}, received ${count(collectedIds.length)}`
          : null,
      );
      setIds(collectedIds);
      setPayloads(collectedPayloads);

      // One flat buffer, transferred. Never an array of arrays: structure-cloning that copies
      // every float as a boxed number.
      const flat = new Float32Array(collectedIds.length * dimension);
      collectedIds.forEach((id, index) => {
        const vector = vectorFor(name, id);
        if (vector) flat.set(vector, index * dimension);
      });

      job.current += 1;
      setStatus('projecting…');
      const request: PcaRequest = {
        jobId: job.current,
        data: flat,
        n: collectedIds.length,
        d: dimension,
        seed: 42,
        metric: (table?.distance_metric ?? 'cosine').toLowerCase() as Metric,
      };
      worker.current?.postMessage(request, [flat.buffer]);
    } catch (cause) {
      setError(cause);
      setStatus(null);
    }
  };

  /**
   * The honesty check, and the reason this page is worth having.
   *
   * A real search with the clicked point's own vector, its true neighbours ringed where they
   * actually landed. Neighbours that fell outside the sample are counted and said, because an
   * absence that is not stated reads as evidence.
   */
  const verify = async (index: number) => {
    const id = ids[index];
    const vector = id ? vectorFor(name, id) : undefined;
    if (!id || !vector) return;
    setSelected(index);
    try {
      const response = await api.search(name, {
        vector: Array.from(vector),
        top_k: 21,
        limit: 21,
        with_payload: false,
      });
      const returned = response.value.hits.map((hit) => hit.id).filter((hit) => hit !== id);
      const inSample = returned.filter((hit) => ids.includes(hit));
      setNeighbours({ ids: inSample, outside: returned.length - inSample.length });
      scatter.current?.setHighlighted(
        inSample.map((hit) => ids.indexOf(hit)).filter((i) => i >= 0),
        index,
      );
    } catch (cause) {
      setError(cause);
    }
  };

  const together = explained[0] + explained[1];
  // Either measure failing is enough, because they come apart. Measured on the demo tables:
  // one projection carried 31.4% of the variance — comfortably over any variance threshold —
  // while keeping only 17% of each point's ten nearest neighbours. A variance-only test would
  // have let that plot through without a word.
  const kept = preserved && preserved.probes > 0 ? preserved.rate : null;
  const weak = positions !== null && (together < 0.25 || (kept !== null && kept < 0.5));

  return (
    <>
      <div className="searchbox">
        <div className="toolbar">
          <label>
            points
            <input
              type="number"
              min={3}
              max={budget.points}
              value={wanted}
              style={{ width: 90 }}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                next.set('n', event.target.value);
                setParams(next, { replace: true });
              }}
            />
          </label>
          <label>
            colour by
            <select
              value={colorKey}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                next.set('by', event.target.value);
                setParams(next, { replace: true });
              }}
            >
              <option value="">nothing</option>
              {keys.map((key) => (
                <option key={key} value={key}>
                  {key}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => void load()} disabled={status !== null && status.endsWith('…')}>
            {status?.endsWith('…') ? status : positions ? 'Resample' : 'Project'}
          </button>
          {status && !status.endsWith('…') ? <span className="muted">{status}</span> : null}
        </div>

        <p className="muted">
          At most {count(budget.points)} points here:{' '}
          {budget.reason === 'floats'
            ? `${count(budget.points)} × ${dimension} dimensions is the 1M-float ceiling, about ${bytes(budget.estimatedBytes)} of JSON`
            : budget.reason === 'requests'
              ? `four requests of ${PAGE}, and each one makes the server sort the whole table`
              : 'the whole table fits'}
          . Vectors can only be fetched through <span className="mono">records/list</span>, which
          is ordered by <span className="mono">created_at</span> — so the sample is drawn from{' '}
          {Math.max(1, Math.ceil(wanted / PAGE))} evenly spaced windows by insertion time, not
          uniformly at random.
        </p>
      </div>

      {error ? <ErrorBanner error={error} /> : null}

      {positions ? (
        <>
          <div className="plothead">
            <span>
              PC1 <b>{(explained[0] * 100).toFixed(1)}%</b> · PC2{' '}
              <b>{(explained[1] * 100).toFixed(1)}%</b> · together{' '}
              <b>{(together * 100).toFixed(1)}%</b> of variance
            </span>
            {preserved && preserved.probes > 0 ? (
              <span
                title={`Measured on ${preserved.probes} sampled points: of each one's ${preserved.k} nearest neighbours in the embedding, this fraction are still among its ${preserved.k} nearest on screen.`}
              >
                neighbourhoods kept <b>{(preserved.rate * 100).toFixed(0)}%</b>
              </span>
            ) : null}
            <span className="muted">
              {count(ids.length)} of {count(records)} records
              {shortfall ? ` — ${shortfall}` : ''}
            </span>
          </div>

          {weak ? (
            <div className="banner warn">
              <strong>
                {kept !== null && kept < 0.5
                  ? `${(100 - kept * 100).toFixed(0)}% of each point's nearest neighbours are not its nearest neighbours on this plot`
                  : `These two components explain ${(together * 100).toFixed(1)}% of the variance`}
              </strong>
              <span className="detail">
                Distance on this plot is not distance in the embedding, and groups you see here
                may not be groups. The two components carry{' '}
                {(together * 100).toFixed(1)}% of the variance
                {preserved && preserved.probes > 0
                  ? `, and ${(preserved.rate * 100).toFixed(0)}% of each point's ${preserved.k} nearest neighbours survive the projection`
                  : ''}
                . Those are different questions and a plot can fail either one. Click a point to
                see it for yourself: its true nearest neighbours get ringed where they actually
                landed.
              </span>
            </div>
          ) : null}

          <div className="plot">
            {/* A canvas is opaque to a screen reader, so the figures it draws are stated in
                text as well — that is also the honest summary for anyone reading quickly. */}
            <canvas
              ref={canvas}
              role="img"
              aria-label={`Scatter plot of ${ids.length} sampled records. The two components carry ${(together * 100).toFixed(1)} per cent of the variance${preserved && preserved.probes > 0 ? `, and ${(preserved.rate * 100).toFixed(0)} per cent of each point's nearest neighbours survive the projection` : ''}.`}
            />
            {hover !== null && ids[hover] ? (
              <div className="plot-hover mono">
                {ids[hover]}
                {colorKey ? ` · ${colorKey}: ${String(payloads[hover]?.[colorKey] ?? '—')}` : ''}
              </div>
            ) : null}
          </div>

          {coloring.legend.entries.length > 0 || coloring.legend.range ? (
            <div className="legend">
              {coloring.legend.range ? (
                <span className="muted">
                  {colorKey}: {coloring.legend.range[0]} → {coloring.legend.range[1]}
                </span>
              ) : null}
              {coloring.legend.entries.map((entry) => (
                <span key={entry.label}>
                  <i style={{ background: entry.color }} /> {entry.label}{' '}
                  <span className="muted">({entry.count})</span>
                </span>
              ))}
            </div>
          ) : null}

          {selected !== null && neighbours ? (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>
                True neighbours of <span className="mono">{ids[selected]}</span>
              </h3>
              <p className="muted" style={{ marginTop: 0 }}>
                From a real search with that record&apos;s own vector — not from the plot.{' '}
                {neighbours.outside > 0 ? (
                  <b>
                    {neighbours.outside} of {neighbours.outside + neighbours.ids.length} are not in
                    this sample and cannot be shown.
                  </b>
                ) : null}{' '}
                If the ringed points are scattered, the projection is not preserving
                neighbourhoods — which is what the variance figure above was warning about.
              </p>
              <ol className="hits">
                {neighbours.ids.slice(0, 10).map((id) => (
                  <li key={id}>
                    <Link to={`/tables/${encodeURIComponent(name)}/records/${encodeURIComponent(id)}`}>
                      {id}
                    </Link>
                  </li>
                ))}
              </ol>
            </div>
          ) : null}

          <p className="note">
            Scroll to zoom, drag to pan, double-click to reset. Nothing is drawn beyond the points
            themselves: no hulls, no density, no groupings — at this explained variance, any of
            those would be an assertion the projection cannot support.
          </p>
        </>
      ) : null}
    </>
  );
}
