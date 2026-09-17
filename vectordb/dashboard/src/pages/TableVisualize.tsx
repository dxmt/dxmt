import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useOutletContext, useParams, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import { normalizeRecord, vectorFor } from '../api/normalize';
import type { Table } from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { Scatter } from '../viz/scatter';
import type { GraphRequest, LayoutRequest, PcaResponse } from '../viz/pca.worker';
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

  // A callback ref, not `useRef`: the canvas is inside `{positions ? … : null}`, so on mount
  // there is nothing to attach to. The first version constructed the renderer in an effect with
  // an empty dependency list, which ran while `canvas.current` was still null, bailed, and left
  // the plot permanently blank — the header rendered, because those numbers come from the
  // worker, and nothing else ever did. A callback ref fires when the element actually appears.
  const [canvasEl, setCanvasEl] = useState<HTMLCanvasElement | null>(null);
  const scatter = useRef<Scatter | null>(null);
  const worker = useRef<Worker | null>(null);
  const job = useRef(0);

  const mode = params.get('layout') === 'pca' ? 'pca' : 'graph';
  const [progress, setProgress] = useState<number | null>(null);
  const [ids, setIds] = useState<string[]>([]);
  const [payloads, setPayloads] = useState<Record<string, unknown>[]>([]);
  const [positions, setPositions] = useState<Float32Array | null>(null);
  const [explained, setExplained] = useState<[number, number] | null>(null);
  const [preserved, setPreserved] = useState<Preservation | null>(null);
  const [shortfall, setShortfall] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [neighbours, setNeighbours] = useState<{ ids: string[]; outside: number } | null>(null);

  const dimension = table?.dimension ?? 0;
  const records = table?.record_count ?? 0;
  /**
   * What bounds the sample, which is not the same question in the two modes.
   *
   * PCA needs the vectors here, so the ceiling is the wire: n x d floats as JSON, uncompressed.
   * The graph needs none of them — the server answers with the neighbours instead of the
   * inputs — so the ceiling is the server's own node cap, and it is four thousand whatever the
   * dimension. That difference is the entire reason the endpoint exists.
   */
  const GRAPH_MAX_NODES = 4_000;
  const budget = useMemo(
    () =>
      mode === 'graph'
        ? {
            points: Math.min(GRAPH_MAX_NODES, records),
            requests: 1,
            estimatedBytes: Math.min(GRAPH_MAX_NODES, records) * 15 * 24,
            reason: records <= GRAPH_MAX_NODES ? ('table' as const) : ('requests' as const),
          }
        : budgetFor(records, dimension),
    [mode, records, dimension],
  );
  const wanted = Math.min(Number(params.get('n') ?? budget.points), budget.points);
  const colorKey = params.get('by') ?? '';

  /**
   * The payload keys, learned before anything is projected.
   *
   * They used to come only from the sample, which meant the colour-by control sat there with
   * one meaningless option until you had already pressed Project — a visible, enabled, useless
   * control, and no hint as to why. The natural order is the other way round: choose what to
   * colour by, then look. So a handful of records are read on arrival (no vectors, so it is a
   * small response) purely to find out what the payloads contain.
   *
   * A handful rather than one, because DXMT payloads are schemaless and a key present on most
   * records can be absent from the first.
   */
  const probe = useQuery({
    queryKey: ['payload-keys', name],
    staleTime: Infinity,
    queryFn: ({ signal }) =>
      api
        .records(name, { limit: 20, offset: 0, with_vector: false }, signal)
        .then((r) => r.value.records.map((wire) => normalizeRecord(name, wire).payload ?? {})),
  });

  const keys = useMemo(() => {
    const seen = new Set<string>();
    for (const payload of [...(probe.data ?? []), ...payloads]) {
      for (const key of Object.keys(payload)) seen.add(key);
    }
    return [...seen].sort();
  }, [probe.data, payloads]);

  const coloring: Coloring = useMemo(
    () => colorBy(payloads.map((payload) => (colorKey ? payload[colorKey] : undefined))),
    [payloads, colorKey],
  );

  useEffect(() => {
    worker.current = new Worker(new URL('../viz/pca.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.current.onmessage = (event: MessageEvent<PcaResponse>) => {
      const message = event.data;
      // A run started two adjustments ago must not land on the current one.
      if (message.jobId !== job.current) return;
      if ('progress' in message) {
        setProgress(message.progress);
        return;
      }
      setProgress(null);
      if (!message.ok) {
        setError(new Error(message.error));
        setStatus(null);
        return;
      }
      setPositions(message.positions);
      setExplained(message.explained);
      setPreserved(message.preservation);
      setStatus(`laid out in ${Math.round(message.ms)} ms`);
    };
    return () => worker.current?.terminate();
  }, []);

  // The click handler closes over `ids`, which changes on every resample. Held in a ref and
  // read at call time so the renderer is built once per canvas rather than rebuilt whenever the
  // sample changes — and so a click never acts on the ids of a previous projection.
  const onPoint = useRef<(index: number) => void>(() => {});
  onPoint.current = (index: number) => void verify(index);

  useEffect(() => {
    if (!canvasEl) return;
    const instance = new Scatter(canvasEl, {
      onHover: setHover,
      onClick: (index) => onPoint.current(index),
    });
    scatter.current = instance;
    return () => {
      instance.destroy();
      scatter.current = null;
    };
  }, [canvasEl]);

  useEffect(() => {
    if (!positions || !scatter.current) return;
    scatter.current.setPoints(
      ids.map((_, index) => ({
        x: positions[index * 2]!,
        y: positions[index * 2 + 1]!,
        color: coloring.colorAt(index),
      })),
    );
    // `canvasEl` is a dependency because the renderer is built from it: without it here, the
    // first projection sets points on a renderer that does not exist yet and never retries.
  }, [positions, ids, coloring, canvasEl]);

  const load = async () => {
    setError(null);
    setNeighbours(null);
    setSelected(null);
    setPositions(null);
    setPreserved(null);
    setExplained(null);
    setShortfall(null);
    setStatus('fetching…');
    job.current += 1;
    const jobId = job.current;

    try {
      if (mode === 'graph') {
        // The whole point of the graph endpoint: the answer instead of the inputs. One request,
        // no vectors on the wire, and the node count stops being decided by how much JSON a tab
        // can parse.
        const response = await api.graph(name, { sample: wanted, neighbours: 15, seed: 42 });
        const graph = response.value;
        if (graph.sampled < 3) {
          setError(new Error(`only ${graph.sampled} nodes came back; a layout needs three`));
          setStatus(null);
          return;
        }
        setIds(graph.ids);
        // The labels come back with the nodes. They have to: the sample is scattered across
        // the table and no route fetches a list of ids, so a second request could not get them.
        setPayloads(
          (graph.payloads ?? []).map((raw) => {
            if (!raw) return {};
            try {
              const parsed: unknown = JSON.parse(raw);
              return typeof parsed === 'object' && parsed !== null
                ? (parsed as Record<string, unknown>)
                : {};
            } catch {
              return {};
            }
          }),
        );
        setShortfall(
          graph.sampled < wanted
            ? `asked for ${count(wanted)}, the table has ${count(graph.total)}`
            : null,
        );

        setStatus('laying out…');
        const request: GraphRequest = {
          kind: 'graph',
          jobId,
          n: graph.sampled,
          neighbours: graph.neighbours,
          scores: graph.scores,
          seed: 42,
          steps: 300,
        };
        worker.current?.postMessage(request);
        return;
      }

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

      setStatus('projecting…');
      const request: LayoutRequest = {
        kind: 'pca',
        jobId,
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

  const together = explained ? explained[0] + explained[1] : null;
  // Either measure failing is enough, because they come apart in both directions. One demo
  // table carries 31.4% of the variance — over any variance threshold — while keeping 17% of
  // each point's ten nearest neighbours; a variance-only test would pass that plot silently.
  //
  // The preservation line sits at 20% rather than 50% because of the opposite case, found by
  // calibrating against data with known structure: four well-separated clusters living on a
  // plane inside 128 dimensions project at 99.5% of the variance and still keep only 43% of
  // neighbourhoods — the groups are perfectly recovered, the ordering *within* each group is
  // not, and that is a good plot. Warning on it would be a false alarm, and false alarms teach
  // people to close the banner. Below 20% the overlap is near what chance gives (k of n: about
  // 1% for ten of a thousand), which is a picture of nothing.
  const kept = preserved && preserved.probes > 0 ? preserved.rate : null;
  // What a projection that preserved nothing would score: k neighbours drawn from n - 1. At
  // k=10 over a thousand points that is about one per cent, and the difference between one and
  // thirteen is the difference between two completely different situations that the first
  // version of this described with the same sentence.
  const chance = preserved && ids.length > 1 ? preserved.k / (ids.length - 1) : null;
  const noise = kept !== null && chance !== null && kept < chance * 3;
  const coarseOnly = kept !== null && !noise && kept < 0.2;
  const weak =
    positions !== null && ((together !== null && together < 0.25) || noise || coarseOnly);

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
              disabled={keys.length === 0}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                next.set('by', event.target.value);
                setParams(next, { replace: true });
              }}
            >
              <option value="">{keys.length === 0 ? 'no payload fields' : 'nothing'}</option>
              {keys.map((key) => (
                <option key={key} value={key}>
                  {key}
                </option>
              ))}
            </select>
          </label>
          <label
            title="Graph: the server computes the true nearest neighbours of the sample and this lays that graph out — no vectors on the wire, so the sample can be much larger. PCA: the two directions of most variance, computed here from the vectors themselves."
          >
            layout
            <select
              value={mode}
              onChange={(event) => {
                const next = new URLSearchParams(params);
                next.set('layout', event.target.value);
                next.delete('n');
                setParams(next, { replace: true });
              }}
            >
              <option value="graph">neighbour graph</option>
              <option value="pca">PCA</option>
            </select>
          </label>
          <button type="button" onClick={() => void load()} disabled={status !== null && status.endsWith('…')}>
            {status?.endsWith('…')
              ? progress !== null
                ? `${status} ${Math.round(progress * 100)}%`
                : status
              : positions
                ? 'Resample'
                : 'Lay out'}
          </button>
          {status && !status.endsWith('…') ? <span className="muted">{status}</span> : null}
        </div>

        {mode === 'graph' ? (
          <p className="muted">
            The server computes the true nearest neighbours of the sample and sends the graph —
            about {bytes(budget.estimatedBytes)} for {count(budget.points)} nodes, where the same
            nodes as {dimension}-dimension vectors would be{' '}
            {bytes(budget.points * dimension * 13)} of JSON floats. That is why the ceiling here
            is the server&apos;s node cap ({count(GRAPH_MAX_NODES)}, since the computation is
            quadratic) and not the wire. The sample is spread evenly across the table rather than
            taken from one end.
          </p>
        ) : (
          <p className="muted">
            At most {count(budget.points)} points here:{' '}
            {budget.reason === 'floats'
              ? `${count(budget.points)} × ${dimension} dimensions is the 1M-float ceiling, about ${bytes(budget.estimatedBytes)} of JSON`
              : budget.reason === 'requests'
                ? `four requests of ${PAGE}, and each one makes the server sort the whole table`
                : 'the whole table fits'}
            . PCA needs the vectors themselves, and they can only be fetched through{' '}
            <span className="mono">records/list</span>, which is ordered by{' '}
            <span className="mono">created_at</span> — so the sample is drawn from{' '}
            {Math.max(1, Math.ceil(wanted / PAGE))} evenly spaced windows by insertion time, not
            uniformly at random.
          </p>
        )}
      </div>

      {error ? <ErrorBanner error={error} /> : null}

      {positions ? (
        <>
          <div className="plothead">
            {explained && together !== null ? (
              <span>
                PC1 <b>{(explained[0] * 100).toFixed(1)}%</b> · PC2{' '}
                <b>{(explained[1] * 100).toFixed(1)}%</b> · together{' '}
                <b>{(together * 100).toFixed(1)}%</b> of variance
              </span>
            ) : (
              <span className="muted">
                laid out from the neighbour graph — variance is a PCA idea and this layout has
                none
              </span>
            )}
            {preserved && preserved.probes > 0 ? (
              <span
                title={
                  mode === 'graph'
                    ? `Every one of ${preserved.probes} nodes checked against the graph the server computed: of its ${preserved.k} true nearest neighbours, this fraction are still among its ${preserved.k} nearest on screen.`
                    : `Measured on ${preserved.probes} sampled points: of each one's ${preserved.k} nearest neighbours in the embedding, this fraction are still among its ${preserved.k} nearest on screen.`
                }
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
                {noise
                  ? 'This is a picture of noise'
                  : coarseOnly
                    ? 'Broad shape only — the fine ordering did not survive'
                    : `These two components explain ${((together ?? 0) * 100).toFixed(1)}% of the variance`}
              </strong>
              <span className="detail">
                {together !== null ? `${(together * 100).toFixed(1)}% of the variance, and ` : ''}
                {kept !== null
                  ? `${(kept * 100).toFixed(0)}% of each point's ${preserved?.k} nearest neighbours survive the layout — chance alone would give about ${((chance ?? 0) * 100).toFixed(0)}%`
                  : ''}
                .
              </span>
              {noise ? (
                <>
                  Neighbourhoods survive at about the rate a random layout would produce, so
                  there is no structure here to see. That is a fact about the data, not about
                  the projection: a table of arbitrary vectors has nothing a plot can show.
                </>
              ) : (
                <>
                  Distance on this plot is not distance in the embedding, so read the broad
                  arrangement and not the fine one — two points side by side are not
                  necessarily neighbours. Groups that look real may well be; colour by a payload
                  field to check one you already know about, and click any point to have its
                  true nearest neighbours ringed where they actually landed.
                </>
              )}
            </div>
          ) : null}

          <div className="plot">
            {/* A canvas is opaque to a screen reader, so the figures it draws are stated in
                text as well — that is also the honest summary for anyone reading quickly. */}
            <canvas
              ref={setCanvasEl}
              role="img"
              aria-label={`Scatter plot of ${ids.length} sampled records, laid out ${mode === 'graph' ? 'from the neighbour graph' : 'by PCA'}.${together !== null ? ` The two components carry ${(together * 100).toFixed(1)} per cent of the variance.` : ''}${preserved && preserved.probes > 0 ? ` ${(preserved.rate * 100).toFixed(0)} per cent of each point's nearest neighbours survive the layout.` : ''}`}
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
