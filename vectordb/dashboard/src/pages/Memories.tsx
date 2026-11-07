import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import type { Memory, RecallRequest } from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { Pager } from '../components/Pager';
import { RequestPreview } from '../components/RequestPreview';
import { count, epochIso, since } from '../lib/format';

/**
 * What the agents remember about you, and the one page in this console that can change data.
 *
 * Two views of one table. With no query it is `GET /memories`, newest first and paged; with a
 * query it is `POST /memories/recall`, which is the same search an agent runs, so what comes
 * back here is what an agent asking the same question would be handed. Both state their
 * request, as every other page does.
 *
 * Reading is all this page does; correcting and forgetting happen on a memory's own page,
 * where the whole memory is in front of the reader when they decide.
 */
const KINDS = ['fact', 'preference', 'event', 'instruction', 'note'];
const PAGE_SIZES = [25, 50, 100];
/** The server's own default, and on LoCoMo the point where more stopped helping. */
const RECALL_LIMIT = 25;

export function Memories() {
  const [params, setParams] = useSearchParams();
  const kind = params.get('kind') || null;
  const offset = Number(params.get('offset') ?? 0);
  const limit = Number(params.get('limit') ?? 25);
  const query = params.get('q') ?? '';
  const history = params.get('history') === '1';
  const [draft, setDraft] = useState(query);

  // Following the URL, so Back to a search shows that search in the box as well as its hits.
  useEffect(() => setDraft(query), [query]);

  const move = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(patch)) {
      if (value === null || value === '') next.delete(key);
      else next.set(key, value);
    }
    setParams(next, { replace: true });
  };

  const list = useQuery({
    queryKey: ['memories', 'list', kind, limit, offset],
    enabled: query === '',
    queryFn: async ({ signal }) => {
      const response = await api.memories(kind, limit, offset, signal);
      return { ...response.value, ms: response.ms, wire: response.bytes };
    },
  });

  const recallBody: RecallRequest = {
    query,
    limit: RECALL_LIMIT,
    ...(kind ? { kind } : {}),
    ...(history ? { include_superseded: true } : {}),
  };
  const recall = useQuery({
    queryKey: ['memories', 'recall', recallBody],
    enabled: query !== '',
    queryFn: async ({ signal }) => {
      const response = await api.recall(recallBody, signal);
      return { ...response.value, ms: response.ms, wire: response.bytes };
    },
  });

  const searching = query !== '';
  const active = searching ? recall : list;
  const rows = active.data?.memories ?? [];
  const kinds = [...new Set([...KINDS, ...rows.map((m) => m.kind).filter(Boolean), ...(kind ? [kind] : [])])];

  return (
    <>
      <h2>Memories</h2>
      <p className="sub">
        {searching
          ? recall.data
            ? `${rows.length} recalled for “${query}”`
            : 'recalling…'
          : list.data
            ? `${count(list.data.total)} memor${list.data.total === 1 ? 'y' : 'ies'}${kind ? ` of kind ${kind}` : ''}`
            : 'loading…'}
      </p>

      <form
        className="toolbar"
        role="search"
        onSubmit={(event) => {
          event.preventDefault();
          move({ q: draft.trim(), offset: null });
        }}
      >
        <input
          type="search"
          aria-label="Search memories"
          placeholder="Recall: a question, a topic, a few words"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          style={{ flex: '1 1 320px' }}
        />
        <button type="submit" disabled={draft.trim() === ''}>
          Recall
        </button>
        {searching ? (
          <button type="button" onClick={() => move({ q: null })}>
            Clear
          </button>
        ) : null}
        <label title="Recall leaves out memories another one replaced, as an agent's recall does. Tick to see them as history.">
          <input
            type="checkbox"
            checked={history}
            onChange={(event) => move({ history: event.target.checked ? '1' : null })}
          />
          include superseded
        </label>
      </form>

      <div className="toolbar">
        <label>
          kind
          <select
            value={kind ?? ''}
            onChange={(event) => move({ kind: event.target.value || null, offset: null })}
          >
            <option value="">all</option>
            {kinds.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        {searching ? null : (
          <label>
            rows
            <select
              value={limit}
              onChange={(event) => move({ limit: event.target.value, offset: null })}
            >
              {PAGE_SIZES.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
        )}
        <button type="button" disabled={active.isFetching} onClick={() => void active.refetch()}>
          {active.isFetching ? 'loading…' : 'Refresh'}
        </button>
      </div>

      {active.error ? <ErrorBanner error={active.error} /> : null}

      {active.data ? (
        <RequestPreview
          method={searching ? 'POST' : 'GET'}
          path={
            searching
              ? '/api/v1/memories/recall'
              : `/api/v1/memories?${kind ? `kind=${encodeURIComponent(kind)}&` : ''}limit=${limit}&offset=${offset}`
          }
          body={searching ? recallBody : undefined}
          ms={active.data.ms}
          wire={active.data.wire}
        />
      ) : null}

      {!searching && list.data && list.data.total === 0 && !kind ? <NoMemories /> : null}

      {rows.length > 0 ? (
        <MemoryTable memories={rows} recalled={searching} />
      ) : active.data && (searching || kind) ? (
        <p className="note">
          {searching ? 'Nothing recalled for that query.' : `No memories of kind ${kind}.`}
        </p>
      ) : null}

      {!searching && list.data && list.data.total > 0 ? (
        <Pager
          offset={offset}
          limit={limit}
          total={list.data.total}
          busy={list.isFetching}
          onOffset={(next) => move({ offset: String(next) })}
        />
      ) : null}

      {searching ? (
        <p className="note">
          Recall ranks by meaning and by keyword together, the same search an agent runs.{' '}
          <span className="mono">score</span> is a rank, the best hit near 1.0, and not a
          similarity; a server new enough to send <span className="mono">similarity</span> has it
          shown instead.
        </p>
      ) : null}
    </>
  );
}

function MemoryTable({ memories, recalled }: { memories: Memory[]; recalled: boolean }) {
  return (
    <table>
      <thead>
        <tr>
          <th>memory</th>
          <th>kind</th>
          <th>tags</th>
          <th>source</th>
          <th>remembered</th>
          {recalled ? <th className="num">match</th> : null}
        </tr>
      </thead>
      <tbody>
        {memories.map((memory) => (
          <MemoryRow key={memory.id} memory={memory} recalled={recalled} />
        ))}
      </tbody>
    </table>
  );
}

function MemoryRow({ memory, recalled }: { memory: Memory; recalled: boolean }) {
  const iso = epochIso(memory.at);
  const superseded = typeof memory.superseded_by === 'string' && memory.superseded_by !== '';
  return (
    <tr data-superseded={superseded || undefined}>
      <td>
        <Link
          to={`/memories/${encodeURIComponent(memory.id)}`}
          state={{ memory }}
          style={superseded ? { opacity: 0.6, textDecoration: 'line-through' } : undefined}
        >
          {memory.text}
        </Link>
        {superseded ? (
          <span
            className="badge warn"
            style={{ marginLeft: 8 }}
            title={`Replaced by ${memory.superseded_by}. Kept as history; recall leaves it out.`}
          >
            superseded
          </span>
        ) : null}
      </td>
      <td>
        <span className="badge">{memory.kind}</span>
      </td>
      <td>
        {memory.tags && memory.tags.length > 0 ? (
          memory.tags.map((tag) => (
            <span className="chip" key={tag}>
              {tag}
            </span>
          ))
        ) : (
          <span className="muted">—</span>
        )}
      </td>
      <td>{memory.source ? memory.source : <span className="muted">—</span>}</td>
      <td title={iso ?? undefined}>{since(iso)}</td>
      {recalled ? <td className="num">{match(memory)}</td> : null}
    </tr>
  );
}

/** Similarity when the server sends it; the rank otherwise, labelled as what it is. */
export function match(memory: Memory): string {
  if (typeof memory.similarity === 'number') return `similarity ${memory.similarity.toFixed(3)}`;
  if (typeof memory.score === 'number') return `score ${memory.score.toFixed(3)}`;
  return '—';
}

/**
 * The empty state, which has to say two different things: nothing has been remembered yet, or
 * this server cannot remember at all. The list cannot tell them apart -- it answers an empty
 * page for both -- so it says both, and how memories get here.
 */
function NoMemories() {
  return (
    <section className="card">
      <h3>No memories yet</h3>
      <p>
        Agents write memories; this page reads them, and lets you correct or forget them.
      </p>
      <p>
        <strong>Memory needs an embedding model on the server.</strong> The memory table is made
        on first use with the server&apos;s model, so a server without one cannot keep any. Run{' '}
        <span className="mono">ollama pull bge-m3</span> (a local ollama is found on its own) and
        restart, or set <span className="mono">DXMT_EMBEDDING_MODEL</span> (with{' '}
        <span className="mono">OPENAI_API_KEY</span> for OpenAI, or{' '}
        <span className="mono">DXMT_EMBEDDING_BASE_URL</span> for another provider). With{' '}
        <span className="mono">DXMT_MEMORY_TABLE=off</span> memory is turned off and this list
        stays empty.
      </p>
      <p>How agents write one:</p>
      <ul>
        <li>
          over MCP, the <span className="mono">remember</span> tool at{' '}
          <span className="mono">/mcp</span> — for example{' '}
          <span className="mono">claude mcp add --transport http dxmt http://127.0.0.1:9999/mcp</span>
        </li>
        <li>
          over REST,{' '}
          <span className="mono">
            POST /api/v1/memories {'{"text": "Prefers answers in Chinese", "kind": "preference"}'}
          </span>
        </li>
      </ul>
    </section>
  );
}
