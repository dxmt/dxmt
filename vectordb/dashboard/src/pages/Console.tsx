import { useEffect, useRef, useState } from 'react';
import { apiBase, sendRead } from '../api/client';
import { ApiFailure } from '../api/envelope';
import { JsonBlock } from '../components/JsonBlock';
import { CopyButton } from '../components/CopyButton';
import { isError, parse, readableRoutes } from '../lib/console';
import { asCurl } from '../lib/curl';
import { bytes } from '../lib/format';

/**
 * A scratchpad for requests, next to the Swagger page rather than instead of it.
 *
 * Swagger is per-operation: a form per route, a modal at a time, and the body reset whenever you
 * navigate. That is the right shape for reading what an API does and the wrong one for the loop
 * this console is for — vary one field, send it again, compare against what came back a minute
 * ago. So: free text, a history that persists, and the timing and size of every response.
 *
 * **It sends nothing that changes data.** `lib/console.ts` holds the gate and explains why it is
 * a second one: `client.ts` makes a write a compile error, which is worth everything for code we
 * wrote and nothing for a string someone types into a box. A refused request names the Swagger
 * page, because the answer to "I need to create a table" is a real one and should be given.
 *
 * No Monaco. It is two megabytes for syntax highlighting and completion on an editor holding two
 * lines of JSON, and it would be compiled into the server binary — where the entire console is
 * currently 112 KB.
 */
const KEY = 'dxmt.console.history';
const KEPT = 30;

const STARTER = `# Anything here is read-only. Ctrl+Enter sends.
GET /api/v1/tables`;

interface Entry {
  text: string;
  at: number;
}

function remembered(): Entry[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as Entry[]) : [];
  } catch {
    return [];
  }
}

interface Outcome {
  status: number;
  ms: number;
  wire: number;
  body: unknown;
  text: string;
  request: string;
}

export function Console() {
  const [text, setText] = useState(STARTER);
  const [history, setHistory] = useState<Entry[]>(remembered);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);

  const parsed = parse(text);
  const problem = isError(parsed) ? parsed : null;

  useEffect(() => {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(history.slice(0, KEPT)));
    } catch {
      // A history that cannot be saved is still a history for this session.
    }
  }, [history]);

  const send = async () => {
    if (problem || busy) return;
    const request = parse(text);
    if (isError(request)) return;

    setBusy(true);
    setFailure(null);
    try {
      // `sendRead` only accepts a `Parsed`, which only `parse()` can produce — so the read-only
      // gate is in the type of the argument rather than in anyone remembering to check.
      const reply = await sendRead(request);
      setOutcome({
        status: reply.status,
        ms: reply.ms,
        wire: reply.bytes,
        body: reply.body,
        text: reply.text,
        request: asCurl(request.method, request.path, request.body),
      });
      setHistory((entries) => [
        { text, at: Date.now() },
        ...entries.filter((entry) => entry.text !== text),
      ]);
    } catch (cause) {
      setFailure(
        cause instanceof ApiFailure
          ? cause.message
          : `cannot reach the server: ${cause instanceof Error ? cause.message : String(cause)}`,
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h2>Console</h2>
      <p className="sub">
        Read-only requests, with their timing and size. For anything that changes data, the{' '}
        <a href={`${apiBase()}/api/docs`} target="_blank" rel="noreferrer">
          Swagger page
        </a>
        .
      </p>

      <div className="consolegrid">
        <div>
          <div className="searchbox" style={{ marginBottom: 12 }}>
            <textarea
              ref={box}
              className="mono"
              rows={10}
              spellCheck={false}
              value={text}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                  event.preventDefault();
                  void send();
                }
              }}
              aria-label="Request"
            />
            {problem ? (
              <p className="err">
                {problem.line === undefined ? null : (
                  <span className="muted">line {problem.line + 1}: </span>
                )}
                {problem.error}
              </p>
            ) : null}
            <div className="toolbar" style={{ marginBottom: 0 }}>
              <button type="button" disabled={busy || problem !== null} onClick={() => void send()}>
                {busy ? 'sending…' : 'Send'}
              </button>
              <span className="muted">Ctrl+Enter</span>
            </div>
          </div>

          {failure ? <div className="banner bad">{failure}</div> : null}

          {outcome ? (
            <>
              <div className="toolbar" style={{ marginTop: 0 }}>
                <span className={`badge ${outcome.status < 400 ? 'ok' : 'warn'}`}>
                  {outcome.status}
                </span>
                <span className="muted">
                  {Math.round(outcome.ms)} ms · {bytes(outcome.wire)}
                </span>
                <CopyButton text={outcome.request} label="copy as curl" />
                <CopyButton text={outcome.text} label="copy response" />
              </div>
              {typeof outcome.body === 'string' ? (
                <JsonBlock raw={outcome.body} />
              ) : (
                <JsonBlock value={outcome.body} />
              )}
            </>
          ) : null}
        </div>

        <aside>
          <section className="card">
            <h3>What this can reach</h3>
            <ul className="routes">
              {readableRoutes().map((route) => (
                <li key={`${route.method} ${route.path}`}>
                  <button
                    type="button"
                    className="ghost mono"
                    onClick={() =>
                      setText(
                        route.method === 'GET'
                          ? `${route.method} ${route.path}`
                          : `${route.method} ${route.path}\n{}`,
                      )
                    }
                  >
                    {route.method} {route.path}
                  </button>
                  <span className="muted"> {route.what}</span>
                </li>
              ))}
            </ul>
          </section>

          {history.length > 0 ? (
            <section className="card" style={{ marginTop: 16 }}>
              <h3>Recent</h3>
              <ul className="routes">
                {history.slice(0, 12).map((entry) => (
                  <li key={entry.at}>
                    <button type="button" className="ghost mono" onClick={() => setText(entry.text)}>
                      {entry.text.split('\n').find((line) => line.trim() && !line.startsWith('#')) ??
                        entry.text}
                    </button>
                  </li>
                ))}
              </ul>
              <button type="button" className="ghost" onClick={() => setHistory([])}>
                clear
              </button>
            </section>
          ) : null}
        </aside>
      </div>
    </>
  );
}
