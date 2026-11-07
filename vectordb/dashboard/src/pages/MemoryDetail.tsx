import { useEffect, useRef, useState } from 'react';
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ownMemory } from '../api/client';
import { MEMORY_FIELDS, type Memory, type RememberRequest } from '../api/types';
import { ErrorBanner } from '../components/ErrorBanner';
import { JsonBlock } from '../components/JsonBlock';
import { Card, Stat } from '../components/Stat';
import { count, epochIso, since } from '../lib/format';

/**
 * One memory in full, and the only place this console writes.
 *
 * There is no `GET /memories/{id}`, so the memory is found by paging `GET /memories` -- which
 * also finds the memories this one replaced, since their `superseded_by` names it. A memory
 * store is a person's worth of sentences, not a table of vectors, and a thousand a page keeps
 * that to a request or two; the scan stops at `SCAN_PAGES` and says so rather than paging a
 * runaway table forever. Arriving from the list carries the row along, so the page is drawn
 * before the scan answers.
 *
 * Correcting makes a new memory that supersedes this one by default, because that is how a
 * fact changes without losing what was believed before; fixing in place is offered for the
 * typo that was never a different fact. Forgetting is permanent and asks first, naming the
 * memory, because a dialog that says only "are you sure?" is answered without reading.
 */
const SCAN_PAGE = 1000;
const SCAN_PAGES = 20;

export function MemoryDetail() {
  const { id = '' } = useParams();
  const location = useLocation();
  const carried = (location.state as { memory?: Memory } | null)?.memory;
  const handed = carried && carried.id === id ? carried : undefined;

  const scan = useQuery({
    queryKey: ['memories', 'scan', id],
    queryFn: async ({ signal }) => {
      let found: Memory | undefined;
      const replaced: Memory[] = [];
      let total = 0;
      let seen = 0;
      for (let page = 0; page < SCAN_PAGES; page += 1) {
        const response = await api.memories(null, SCAN_PAGE, page * SCAN_PAGE, signal);
        total = response.value.total;
        for (const memory of response.value.memories) {
          if (memory.id === id) found = memory;
          if (memory.superseded_by === id) replaced.push(memory);
        }
        seen += response.value.memories.length;
        if (response.value.memories.length < SCAN_PAGE || seen >= total) break;
      }
      return { found, replaced, total, complete: seen >= total };
    },
  });

  const memory = scan.data ? scan.data.found : handed;

  if (scan.error && !memory) return <ErrorBanner error={scan.error} />;

  if (!memory) {
    return (
      <>
        <h2 className="mono">{id}</h2>
        <p className="sub">
          <Link to="/memories">Memories</Link>
        </p>
        {scan.data ? (
          <div className="banner warn">
            <strong>No memory with this id</strong>
            {scan.data.complete
              ? 'It may have been forgotten. A memory another one superseded keeps pointing at its replacement even after that replacement is forgotten.'
              : `Only the newest ${count(SCAN_PAGE * SCAN_PAGES)} of ${count(scan.data.total)} memories were searched.`}
          </div>
        ) : (
          <p className="muted">looking…</p>
        )}
      </>
    );
  }

  const superseded = typeof memory.superseded_by === 'string' && memory.superseded_by !== '';
  const iso = epochIso(memory.at);
  const metadata = metadataOf(memory);

  return (
    <>
      <h2>{memory.text}</h2>
      <p className="sub">
        <Link to="/memories">Memories</Link> · <span className="mono">{memory.id}</span>
      </p>

      {scan.data && !scan.data.found ? (
        <div className="banner warn">
          <strong>This memory is no longer on the server</strong>
          What is shown is the copy the list had. It has been forgotten or corrected since.
        </div>
      ) : null}

      {superseded ? (
        <div className="banner warn">
          <strong>Superseded</strong>
          Replaced by{' '}
          <Link to={`/memories/${encodeURIComponent(memory.superseded_by as string)}`}>
            {memory.superseded_by as string}
          </Link>
          . Kept as history; recall leaves it out unless asked to include superseded memories.
        </div>
      ) : null}

      <div className="cards">
        <Card title="Memory">
          <Stat label="kind" value={memory.kind} />
          <Stat
            label="tags"
            value={memory.tags && memory.tags.length > 0 ? memory.tags.join(', ') : '—'}
          />
          <Stat label="source" value={memory.source || '—'} />
          <Stat label="remembered" value={since(iso)} title={iso ?? undefined} />
          <Stat
            label="superseded by"
            value={
              superseded ? (
                <Link to={`/memories/${encodeURIComponent(memory.superseded_by as string)}`}>
                  {memory.superseded_by as string}
                </Link>
              ) : (
                '—'
              )
            }
            mono={superseded}
          />
          <Stat label="id" value={memory.id} mono />
        </Card>

        <section className="card">
          <h3>Metadata</h3>
          {Object.keys(metadata).length === 0 ? (
            <p className="muted">None. Anything an agent passes as metadata is shown here.</p>
          ) : (
            <JsonBlock value={metadata} />
          )}
        </section>

        {scan.data && scan.data.replaced.length > 0 ? (
          <section className="card">
            <h3>Replaces</h3>
            <ul>
              {scan.data.replaced.map((old) => (
                <li key={old.id}>
                  <Link to={`/memories/${encodeURIComponent(old.id)}`} state={{ memory: old }}>
                    {old.text}
                  </Link>{' '}
                  <span className="muted">{since(epochIso(old.at))}</span>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>

      {superseded ? (
        <p className="note">
          A superseded memory is history and is not corrected here: correct{' '}
          <Link to={`/memories/${encodeURIComponent(memory.superseded_by as string)}`}>
            the one that replaced it
          </Link>
          .
        </p>
      ) : (
        <Correct key={memory.id} memory={memory} />
      )}

      <Forget memory={memory} replaced={scan.data?.replaced.length ?? 0} />
    </>
  );
}

/** Everything on the memory that is not one of its own fields: what was passed as metadata. */
export function metadataOf(memory: Memory): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(memory).filter(([key]) => !MEMORY_FIELDS.includes(key)),
  );
}

type Mode = 'supersede' | 'in-place';

/** The body a correction sends. Source and metadata are carried over; they are not edited. */
export function correction(
  memory: Memory,
  edit: { text: string; kind: string; tags: string[] },
  mode: Mode,
): RememberRequest {
  const metadata = metadataOf(memory);
  return {
    text: edit.text,
    ...(edit.kind ? { kind: edit.kind } : {}),
    ...(edit.tags.length > 0 ? { tags: edit.tags } : {}),
    ...(memory.source ? { source: memory.source } : {}),
    ...(Object.keys(metadata).length > 0 ? { metadata } : {}),
    ...(mode === 'supersede' ? { supersedes: [memory.id] } : { id: memory.id }),
  };
}

function Correct({ memory }: { memory: Memory }) {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [text, setText] = useState(memory.text);
  const [kind, setKind] = useState(memory.kind ?? '');
  const [tags, setTags] = useState((memory.tags ?? []).join(', '));
  const [mode, setMode] = useState<Mode>('supersede');

  const edit = {
    text: text.trim(),
    kind: kind.trim(),
    tags: tags
      .split(',')
      .map((tag) => tag.trim())
      .filter(Boolean),
  };
  const unchanged =
    edit.text === memory.text &&
    edit.kind === (memory.kind ?? '') &&
    edit.tags.join('\u0000') === (memory.tags ?? []).join('\u0000');

  const save = useMutation({
    mutationFn: () => ownMemory.remember(correction(memory, edit, mode)).then((r) => r.value),
    onSuccess: async (saved) => {
      await client.invalidateQueries({ queryKey: ['memories'] });
      if (saved.id !== memory.id) navigate(`/memories/${encodeURIComponent(saved.id)}`);
    },
  });

  return (
    <section className="card" style={{ marginTop: 20 }}>
      <h3>Correct</h3>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          save.mutate();
        }}
      >
        <label style={{ display: 'block', marginBottom: 10 }}>
          <span className="muted">text</span>
          <textarea
            aria-label="Memory text"
            rows={3}
            value={text}
            onChange={(event) => setText(event.target.value)}
          />
        </label>
        <div className="toolbar">
          <label>
            kind
            <input aria-label="Kind" value={kind} onChange={(event) => setKind(event.target.value)} />
          </label>
          <label style={{ flex: '1 1 260px' }}>
            tags
            <input
              aria-label="Tags"
              placeholder="comma, separated"
              value={tags}
              onChange={(event) => setTags(event.target.value)}
              style={{ flex: 1 }}
            />
          </label>
        </div>
        <fieldset style={{ border: 0, padding: 0, margin: '6px 0 12px' }}>
          <legend className="muted">how</legend>
          <label style={{ display: 'block' }}>
            <input
              type="radio"
              name="mode"
              checked={mode === 'supersede'}
              onChange={() => setMode('supersede')}
            />{' '}
            Save as a new memory that supersedes this one — this one is kept as history
          </label>
          <label style={{ display: 'block' }}>
            <input
              type="radio"
              name="mode"
              checked={mode === 'in-place'}
              onChange={() => setMode('in-place')}
            />{' '}
            Fix a typo in place — same id, the old wording is overwritten and not kept
          </label>
        </fieldset>
        <button type="submit" disabled={save.isPending || edit.text === '' || unchanged}>
          {save.isPending ? 'saving…' : mode === 'supersede' ? 'Save correction' : 'Fix in place'}
        </button>
        {save.isSuccess && save.data.id === memory.id ? (
          <span className="muted" style={{ marginLeft: 12 }}>
            saved
          </span>
        ) : null}
      </form>
      {save.error ? (
        <div style={{ marginTop: 12 }}>
          <ErrorBanner error={save.error} />
        </div>
      ) : null}
      <p className="note">
        Source and metadata are carried over unchanged. Either way the memory is embedded again
        from the new text, and its time moves to now.
      </p>
    </section>
  );
}

function Forget({ memory, replaced }: { memory: Memory; replaced: number }) {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [asking, setAsking] = useState(false);
  const cancel = useRef<HTMLButtonElement>(null);

  const forget = useMutation({
    mutationFn: () => ownMemory.forget(memory.id).then((r) => r.value),
    onSuccess: async (result) => {
      await client.invalidateQueries({ queryKey: ['memories'] });
      if (result.forgotten > 0) navigate('/memories', { replace: true });
    },
  });

  // The safe choice has the focus, so an Enter pressed out of habit does not delete anything.
  useEffect(() => {
    if (asking) cancel.current?.focus();
  }, [asking]);

  return (
    <section className="card" style={{ marginTop: 20 }}>
      <h3>Forget</h3>
      <p>Delete this memory permanently. No agent will recall it again, and it cannot be undone.</p>
      <button type="button" className="danger" onClick={() => setAsking(true)}>
        Forget…
      </button>

      {asking ? (
        <div
          className="dialog-backdrop"
          onKeyDown={(event) => {
            if (event.key === 'Escape' && !forget.isPending) setAsking(false);
          }}
        >
          <div
            className="dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="forget-title"
            aria-describedby="forget-what"
          >
            <h3 id="forget-title">Forget this memory permanently?</h3>
            <blockquote id="forget-what">“{memory.text}”</blockquote>
            <p>
              It is deleted, not superseded: no history is kept and there is no undo.
              {replaced > 0
                ? ` The ${replaced} memor${replaced === 1 ? 'y' : 'ies'} it replaced stay marked superseded and out of recall.`
                : ''}
            </p>
            {forget.data && forget.data.forgotten === 0 ? (
              <p className="err">
                Nothing was forgotten: the server holds no memory with this id any more.
              </p>
            ) : null}
            {forget.error ? <ErrorBanner error={forget.error} /> : null}
            <div className="toolbar" style={{ justifyContent: 'flex-end', marginBottom: 0 }}>
              <button
                type="button"
                ref={cancel}
                className="secondary"
                disabled={forget.isPending}
                onClick={() => setAsking(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="danger"
                disabled={forget.isPending}
                onClick={() => forget.mutate()}
              >
                {forget.isPending ? 'forgetting…' : 'Forget permanently'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
