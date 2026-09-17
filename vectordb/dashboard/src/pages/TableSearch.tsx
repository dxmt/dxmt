import { useState } from 'react';
import { Link, useOutletContext, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { ApiFailure } from '../api/envelope';
import { probeExactMode, probeTextSearch } from '../api/capabilities';
import {
  APPROXIMATE_MODES,
  type Filter,
  type FusionMethod,
  type SearchMode,
  type Table,
  type WireHit,
} from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { FilterEditor } from '../components/FilterEditor';
import { RecordPicker, type Picked } from '../components/RecordPicker';
import { RequestPreview } from '../components/RequestPreview';

/**
 * The same query under every mode, side by side.
 *
 * This is the page the exact mode was added for. Four of the five modes scale HNSW's traversal
 * budget and answer approximately; `exact` skips the index and scores every row. Run all of
 * them on one query and the recall-versus-latency trade-off stops being a README claim about
 * somebody else's corpus and becomes a column of ids you can read.
 *
 * Two things it deliberately does not do:
 *
 * **It does not run the modes in parallel.** Concurrent searches contend for the index and the
 * page cache, and `tookms` would then be measuring the contention. Sequential is slower to
 * watch and is the only way the numbers mean anything.
 *
 * **It does not pre-empt the server's validation.** A pasted vector of the wrong length gets a
 * warning and a button that sends it anyway, because seeing the real `DIMENSION_MISMATCH` is
 * sometimes exactly what someone came here to do.
 *
 * # The text and hybrid columns
 *
 * Given a text query as well, the same run adds BM25 and the server's fusion of the two. These
 * are not more search modes and they are not drawn as though they were: the five modes differ
 * only in how much of one index they traverse, while these ask a different index a different
 * question. Three things follow, and each is visible on the page rather than assumed:
 *
 * **Their overlap with `exact` is not recall.** `exact` is ground truth for *the dense metric*.
 * A BM25 result that shares two ids with it has not "missed eight" — it answered a lexical
 * question and lexical relevance is not what the dense metric ranks. The badge on those columns
 * says `shares N with dense exact`, which is an observation. Calling it recall would be a lie
 * with a percent sign on it, and the point of hybrid is precisely that the two disagree.
 *
 * **Scores across columns are not comparable.** BM25 is unbounded and depends on the corpus;
 * cosine similarity is in [-1, 1]. The numbers are shown per column and never averaged, and the
 * fusion that does combine them is done by the server, by rank.
 *
 * **The query has to be the same question asked two ways**, which on this server means starting
 * from a record: nothing here turns a typed sentence into a vector, because no route reaches an
 * embedding model. Picking a record fills in both halves, and both stay editable.
 */
const TOP_K = 10;

interface ModeResult {
  mode: SearchMode;
  hits: WireHit[];
  tookms: number;
  wallMs: number;
  error: unknown;
}

/** A retrieval strategy that is not a traversal budget: BM25, or the server's fusion. */
interface StrategyResult {
  strategy: 'text' | 'hybrid';
  hits: { id: string; score: number }[];
  tookms: number;
  wallMs: number;
  error: unknown;
}

export function TableSearch() {
  const { name = '' } = useParams();
  const { table } = useOutletContext<{ table: Table | undefined }>();

  const [text, setText] = useState('');
  const [query, setQuery] = useState('');
  const [fusion, setFusion] = useState<FusionMethod>('reciprocal_rank_fusion');
  const [weight, setWeight] = useState(0.7);
  const [source, setSource] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter | null>(null);
  const [topK, setTopK] = useState(TOP_K);
  const [run, setRun] = useState(0);
  const [force, setForce] = useState(false);

  const parsed = parseVector(text);
  const dimension = table?.dimension;
  const mismatch =
    parsed.vector && dimension !== undefined && parsed.vector.length !== dimension
      ? `this table has ${dimension} dimensions and that is ${parsed.vector.length}`
      : null;

  const capabilities = useQuery({
    queryKey: ['capabilities', name],
    enabled: dimension !== undefined,
    staleTime: Infinity,
    queryFn: ({ signal }) => probeExactMode(name, dimension ?? 0, signal),
  });

  const textCapability = useQuery({
    queryKey: ['capabilities', 'text', name],
    enabled: table !== undefined,
    staleTime: Infinity,
    queryFn: ({ signal }) => probeTextSearch(name, table?.enable_full_text_search ?? false, signal),
  });
  const canText = textCapability.data?.textSearch === true;
  const asked = canText && query.trim() !== '';

  const modes: SearchMode[] = capabilities.data?.exactMode
    ? [...APPROXIMATE_MODES, 'exact']
    : APPROXIMATE_MODES;

  const results = useQuery({
    queryKey: ['search', name, run],
    enabled: run > 0 && parsed.vector !== null,
    staleTime: Infinity,
    queryFn: async ({ signal }) => {
      const vector = parsed.vector ?? [];
      const modeResults: ModeResult[] = [];
      // One at a time, on purpose. See the note above.
      for (const mode of modes) {
        const started = performance.now();
        try {
          const response = await api.search(
            name,
            {
              vector,
              top_k: topK,
              limit: topK,
              with_payload: true,
              search_mode: mode,
              ...(filter ? { filter } : {}),
            },
            signal,
          );
          modeResults.push({
            mode,
            hits: response.value.hits,
            tookms: response.value.tookms,
            wallMs: response.ms,
            error: null,
          });
        } catch (cause) {
          if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
          modeResults.push({
            mode,
            hits: [],
            tookms: 0,
            wallMs: performance.now() - started,
            error: cause,
          });
        }
      }

      const strategies: StrategyResult[] = [];
      const wanted = query.trim();
      if (canText && wanted !== '') {
        // Still sequential, and for the same reason: these contend for the same page cache the
        // dense searches just warmed, and a parallel run would be timing the contention.
        const bm25 = performance.now();
        try {
          const response = await api.searchText(
            name,
            { query: wanted, limit: topK, highlight: false },
            signal,
          );
          strategies.push({
            strategy: 'text',
            // `took_ms` with the underscore here; `tookms` without it on the other two. Two
            // structs on the Rust side, and this is the wire, not a tidied version of it.
            hits: response.value.hits.map((hit) => ({ id: hit.id, score: hit.score })),
            tookms: response.value.took_ms,
            wallMs: response.ms,
            error: null,
          });
        } catch (cause) {
          if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
          strategies.push({
            strategy: 'text',
            hits: [],
            tookms: 0,
            wallMs: performance.now() - bm25,
            error: cause,
          });
        }

        const fused = performance.now();
        try {
          const response = await api.searchHybrid(
            name,
            {
              vector,
              text: wanted,
              limit: topK,
              fusion_method: fusion,
              // Read by `weighted_sum` and by `reciprocal_rank_fusion` (which is a weighted
              // RRF here, whatever the name suggests); ignored by `dbsf`. Sent in every case so
              // the request preview matches what actually went out.
              vector_weight: weight,
              with_payload: true,
              ...(filter ? { filter } : {}),
            },
            signal,
          );
          strategies.push({
            strategy: 'hybrid',
            hits: response.value.hits.map((hit) => ({ id: hit.id, score: hit.score })),
            tookms: response.value.tookms,
            wallMs: response.ms,
            error: null,
          });
        } catch (cause) {
          if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
          strategies.push({
            strategy: 'hybrid',
            hits: [],
            tookms: 0,
            wallMs: performance.now() - fused,
            error: cause,
          });
        }
      }

      return { modeResults, strategies };
    },
  });

  const truth = results.data?.modeResults.find((r) => r.mode === 'exact' && r.error === null);
  const truthIds = new Set(truth?.hits.map((h) => h.id) ?? []);

  return (
    <>
      <div className="searchbox">
        <label>
          Query vector
          <textarea
            className="mono"
            rows={3}
            spellCheck={false}
            placeholder="[0.1, -0.4, …] — or paste a record's id below"
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setSource(null);
            }}
          />
        </label>
        <RecordPicker
          table={name}
          busy={results.isFetching}
          onPick={(picked: Picked) => {
            setText(JSON.stringify(picked.vector.map((v) => Number(v.toFixed(6)))));
            setSource(picked.id);
            // Only when there is a text side to fill. Overwriting a typed query with null
            // would silently drop the text column from the next run.
            if (canText && picked.text !== null) setQuery(picked.text);
          }}
        />
        {source ? (
          <p className="muted">
            using <span className="mono">{source}</span>&apos;s vector — it should come back first,
            at similarity 1.0
          </p>
        ) : null}

        {canText ? (
          <>
            <label>
              Text query
              <input
                value={query}
                placeholder="the same question in words — BM25 and the fusion both use this"
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
            {asked ? (
              <div className="toolbar">
                <label>
                  fusion
                  <select
                    value={fusion}
                    onChange={(event) => setFusion(event.target.value as FusionMethod)}
                  >
                    <option value="reciprocal_rank_fusion">reciprocal_rank_fusion</option>
                    <option value="dbsf">dbsf</option>
                    <option value="weighted_sum">weighted_sum</option>
                  </select>
                </label>
                <label className={fusion === 'dbsf' ? 'muted' : ''}>
                  vector_weight
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={weight}
                    disabled={fusion === 'dbsf'}
                    onChange={(event) => setWeight(Number(event.target.value))}
                  />
                  <span className="mono">{weight.toFixed(2)}</span>
                </label>
                {fusion === 'dbsf' ? (
                  <span className="muted">
                    dbsf normalises each side by its own mean and deviation, so there is no
                    weight to set — it is what the server uses when the method is omitted
                  </span>
                ) : (
                  <span className="muted">
                    the text side gets {(1 - weight).toFixed(2)} —{' '}
                    {fusion === 'reciprocal_rank_fusion'
                      ? 'this RRF is weighted, so it scales each side\u2019s 1/(k+rank) rather than leaving the ranks alone'
                      : 'each side is min-max normalised, then summed at these weights'}
                  </span>
                )}
              </div>
            ) : null}
          </>
        ) : textCapability.data ? (
          <p className="muted">
            No text search here: {textCapability.data.textSearchReason}
          </p>
        ) : null}
        {parsed.error ? <p className="err">{parsed.error}</p> : null}
        {mismatch ? (
          <p className="warn-text">
            {mismatch}. The server will refuse it — send it anyway if that is what you want to
            see.
          </p>
        ) : null}

        <div className="toolbar">
          <label>
            top_k
            <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {[5, 10, 20, 50].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={parsed.vector === null || results.isFetching || (mismatch !== null && !force)}
            onClick={() => setRun((n) => n + 1)}
          >
            {results.isFetching
              ? `running ${modes.length + (asked ? 2 : 0)} searches…`
              : asked
                ? `Compare ${modes.length} modes + BM25 + fusion`
                : `Compare ${modes.length} modes`}
          </button>
          {mismatch ? (
            <label>
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
              send it anyway
            </label>
          ) : null}
        </div>
      </div>

      <FilterEditor keys={[]} busy={results.isFetching} onApply={setFilter} />

      {capabilities.data && !capabilities.data.exactMode ? (
        <div className="banner warn">
          <strong>No ground truth on this server</strong>
          <span className="detail">{capabilities.data.exactModeReason}</span>
          Without <span className="mono">search_mode: &quot;exact&quot;</span> the columns below
          can only be compared with each other, and <span className="mono">accurate</span> is
          itself about 0.98 — so agreement with it is an upper bound, blind to whatever every
          mode misses.
        </div>
      ) : null}

      {results.error ? <ErrorBanner error={results.error} /> : null}

      {results.data ? (
        <>
          <RequestPreview
            method="POST"
            path={`/api/v1/tables/${encodeURIComponent(name)}/search`}
            body={{
              vector: '…',
              top_k: topK,
              search_mode: modes[modes.length - 1],
              ...(filter ? { filter } : {}),
            }}
          />
          {results.data.strategies.length > 0 ? (
            <RequestPreview
              method="POST"
              path={`/api/v1/tables/${encodeURIComponent(name)}/search/hybrid`}
              body={{
                vector: '…',
                text: query.trim(),
                limit: topK,
                fusion_method: fusion,
                vector_weight: weight,
                ...(filter ? { filter } : {}),
              }}
            />
          ) : null}

          <div className="modes">
            {results.data.modeResults.map((result) => (
              <ModeColumn
                key={result.mode}
                table={name}
                result={result}
                truthIds={truthIds}
                haveTruth={truth !== undefined}
              />
            ))}
          </div>

          {truth ? (
            <p className="note">
              Highlighted ids are the ones <span className="mono">exact</span> returned. A mode
              that misses one is not broken — it spent a smaller traversal budget, which is what
              the millisecond figure bought.
            </p>
          ) : null}

          {results.data.strategies.length > 0 ? (
            <>
              <h3>A different index, a different question</h3>
              <div className="modes">
                {results.data.strategies.map((result) => (
                  <StrategyColumn
                    key={result.strategy}
                    table={name}
                    result={result}
                    truthIds={truthIds}
                    haveTruth={truth !== undefined}
                  />
                ))}
              </div>
              <p className="note">
                The badge counts ids these share with <span className="mono">exact</span>, and it
                is <strong>not</strong> recall: <span className="mono">exact</span> is ground
                truth for the dense metric only, and BM25 is answering a lexical question. A low
                overlap is the reason to fuse them, not evidence that either is wrong. Scores are
                not comparable across these columns either — BM25 is unbounded, a similarity is
                not — which is why the default fusion combines them by rank.
              </p>
            </>
          ) : null}
        </>
      ) : null}
    </>
  );
}

function ModeColumn({
  table,
  result,
  truthIds,
  haveTruth,
}: {
  table: string;
  result: ModeResult;
  truthIds: Set<string>;
  haveTruth: boolean;
}) {
  const isTruth = result.mode === 'exact';
  const overlap =
    haveTruth && !isTruth && result.hits.length > 0
      ? result.hits.filter((hit) => truthIds.has(hit.id)).length / truthIds.size
      : null;

  return (
    <section className={`mode ${isTruth ? 'truth' : ''}`}>
      <header>
        <strong>{result.mode}</strong>
        {result.error ? null : (
          <span className="muted">
            {result.tookms} ms server · {Math.round(result.wallMs)} ms wall
          </span>
        )}
        {overlap === null ? null : (
          <span className={`badge ${overlap === 1 ? 'ok' : 'warn'}`}>
            {(overlap * 100).toFixed(0)}% of exact
          </span>
        )}
      </header>

      {result.error ? (
        <p className="err">
          {result.error instanceof ApiFailure ? result.error.message : String(result.error)}
        </p>
      ) : (
        <ol className="hits">
          {result.hits.map((hit) => (
            <li key={hit.id} className={haveTruth && !truthIds.has(hit.id) ? 'extra' : ''}>
              <Link to={`/tables/${encodeURIComponent(table)}/records/${encodeURIComponent(hit.id)}`}>
                {hit.id}
              </Link>
              <span className="mono muted">{hit.score.toFixed(4)}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

/**
 * BM25 or the fusion, drawn deliberately unlike a mode column.
 *
 * The one number that could mislead is the overlap, so it is labelled by what it counts rather
 * than by what it would be convenient to call it. See the note under the columns.
 */
function StrategyColumn({
  table,
  result,
  truthIds,
  haveTruth,
}: {
  table: string;
  result: StrategyResult;
  truthIds: Set<string>;
  haveTruth: boolean;
}) {
  const shared =
    haveTruth && result.hits.length > 0
      ? result.hits.filter((hit) => truthIds.has(hit.id)).length
      : null;

  return (
    <section className="mode strategy">
      <header>
        <strong>{result.strategy === 'text' ? 'BM25' : 'hybrid'}</strong>
        {result.error ? null : (
          <span className="muted">
            {result.tookms} ms server · {Math.round(result.wallMs)} ms wall
          </span>
        )}
        {shared === null ? null : (
          <span className="badge">
            shares {shared}/{truthIds.size} with dense exact
          </span>
        )}
      </header>

      {result.error ? (
        <p className="err">
          {result.error instanceof ApiFailure ? result.error.message : String(result.error)}
        </p>
      ) : result.hits.length === 0 ? (
        <p className="muted">
          no hits — a BM25 query matches whole indexed terms, so a phrase that never appears in
          the payload returns nothing rather than something near it
        </p>
      ) : (
        <ol className="hits">
          {result.hits.map((hit) => (
            // The *shared* ids are marked here, where a mode column marks the divergent ones.
            // An id BM25 found that the dense metric did not is the ordinary case and often
            // the useful one; colouring it as a miss would contradict the note below.
            <li key={hit.id} className={haveTruth && truthIds.has(hit.id) ? 'shared' : ''}>
              <Link to={`/tables/${encodeURIComponent(table)}/records/${encodeURIComponent(hit.id)}`}>
                {hit.id}
              </Link>
              <span className="mono muted">{hit.score.toFixed(4)}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

/** Accept a JSON array, or bare numbers separated by commas or whitespace. */
function parseVector(text: string): { vector: number[] | null; error: string | null } {
  const trimmed = text.trim();
  if (trimmed === '') return { vector: null, error: null };

  const parts = trimmed.replace(/^\[|\]$/g, '').split(/[,\s]+/).filter((p) => p !== '');
  const numbers = parts.map(Number);
  const bad = numbers.findIndex((n) => !Number.isFinite(n));
  if (bad !== -1) {
    return { vector: null, error: `"${parts[bad]}" at position ${bad} is not a number` };
  }
  return { vector: numbers, error: null };
}
