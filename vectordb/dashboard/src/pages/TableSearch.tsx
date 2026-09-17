import { useState } from 'react';
import { Link, useOutletContext, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { ApiFailure } from '../api/envelope';
import { probeExactMode } from '../api/capabilities';
import { APPROXIMATE_MODES, type Filter, type SearchMode, type Table, type WireHit } from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { FilterEditor } from '../components/FilterEditor';
import { RecordPicker } from '../components/RecordPicker';
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
 */
const TOP_K = 10;

interface ModeResult {
  mode: SearchMode;
  hits: WireHit[];
  tookms: number;
  wallMs: number;
  error: unknown;
}

export function TableSearch() {
  const { name = '' } = useParams();
  const { table } = useOutletContext<{ table: Table | undefined }>();

  const [text, setText] = useState('');
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

  const modes: SearchMode[] = capabilities.data?.exactMode
    ? [...APPROXIMATE_MODES, 'exact']
    : APPROXIMATE_MODES;

  const results = useQuery({
    queryKey: ['search', name, run],
    enabled: run > 0 && parsed.vector !== null,
    staleTime: Infinity,
    queryFn: async ({ signal }) => {
      const vector = parsed.vector ?? [];
      const out: ModeResult[] = [];
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
          out.push({
            mode,
            hits: response.value.hits,
            tookms: response.value.tookms,
            wallMs: response.ms,
            error: null,
          });
        } catch (cause) {
          if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
          out.push({ mode, hits: [], tookms: 0, wallMs: performance.now() - started, error: cause });
        }
      }
      return out;
    },
  });

  const truth = results.data?.find((r) => r.mode === 'exact' && r.error === null);
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
          onVector={(vector, fromId) => {
            setText(JSON.stringify(vector.map((v) => Number(v.toFixed(6)))));
            setSource(fromId);
          }}
        />
        {source ? (
          <p className="muted">
            using <span className="mono">{source}</span>&apos;s vector — it should come back first,
            at similarity 1.0
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
            {results.isFetching ? `running ${modes.length} searches…` : `Compare ${modes.length} modes`}
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

          <div className="modes">
            {results.data.map((result) => (
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
