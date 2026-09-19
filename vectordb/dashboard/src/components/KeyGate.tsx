import { useEffect, useState } from 'react';
import { keyIsAccepted } from '../api/client';
import { getKey, onKeyState, setKey } from '../api/key';

/**
 * Ask for the API key, when the server wants one.
 *
 * The server's key is off by default and this renders nothing in that case. It appears when a
 * request comes back 401 — which is how it learns, rather than by asking the server about its
 * own authentication, because an endpoint that reports whether a key is required is an endpoint
 * that answers questions to people who have not presented one.
 *
 * It replaces the page rather than sitting beside it. Every panel behind it would be an error
 * with the same cause, and a console showing eight copies of one problem is worse at conveying
 * it than a console showing one.
 */
export function KeyGate({ children }: { children: React.ReactNode }) {
  const [rejected, setRejected] = useState(false);
  const [stale, setStale] = useState(false);
  const [draft, setDraft] = useState('');

  useEffect(
    () =>
      onKeyState((event) => {
        if (event === 'unauthorized') {
          // A key was stored and the server refused it, so say that rather than presenting an
          // empty form the reader will fill in with the same value.
          setStale(getKey() !== null);
          setRejected(true);
        } else {
          setRejected(false);
          setStale(false);
        }
      }),
    [],
  );

  useEffect(() => {
    let cancelled = false;
    // One probe at startup, so the gate appears before the pages behind it each render their
    // own copy of the same failure.
    const stored = getKey();
    keyIsAccepted().then((accepted) => {
      if (cancelled || accepted) return;
      setStale(stored !== null);
      setRejected(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!rejected) return <>{children}</>;

  return (
    <div className="gate">
      <h1>{stale ? 'That key was refused' : 'This server wants an API key'}</h1>
      <p>
        It was started with <code>DXMT_API_KEY</code> set. The console sends the key as the{' '}
        <code>api-key</code> header on every request.
      </p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          setKey(draft);
          setRejected(false);
          setDraft('');
        }}
      >
        <label htmlFor="api-key">API key</label>
        <input
          id="api-key"
          type="password"
          autoComplete="off"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button type="submit" disabled={!draft.trim()}>
          Use this key
        </button>
      </form>
      <p className="muted">
        Kept in <code>sessionStorage</code> and forgotten when this tab closes. It is never put
        in the URL, where it would reach history, referrers and every log in between.
      </p>
    </div>
  );
}
