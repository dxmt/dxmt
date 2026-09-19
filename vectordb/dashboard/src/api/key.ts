/**
 * The API key this console sends, when the server asks for one.
 *
 * The server's key is off by default; when `DXMT_API_KEY` is set every route but the health
 * probes answers 401, and without this the console would render an error on every page with no
 * way to act on it.
 *
 * **`sessionStorage`, not `localStorage`.** The key is a credential, and a credential that
 * outlives the tab is one more copy of it lying around on a shared machine than anybody asked
 * for. Closing the tab forgets it; that is the intended trade.
 *
 * It is never put in the URL. A query parameter lands in history, in referrers, and in every
 * access log between here and the server.
 */

const SLOT = 'dxmt.api-key';

/** What happened, because the gate reacts to the two differently. */
export type KeyEvent = 'unauthorized' | 'changed';

const listeners = new Set<(event: KeyEvent) => void>();

export function getKey(): string | null {
  try {
    const key = sessionStorage.getItem(SLOT);
    return key && key.trim() ? key : null;
  } catch {
    // Private mode, or storage disabled. The console still works against an open server.
    return null;
  }
}

export function setKey(key: string | null): void {
  try {
    if (key && key.trim()) sessionStorage.setItem(SLOT, key.trim());
    else sessionStorage.removeItem(SLOT);
  } catch {
    /* nothing to do; the next request simply goes without it */
  }
  for (const listener of listeners) listener('changed');
}

/// Called by the client on a 401, so the gate can appear without polling.
///
/// A 401 *with* a key already stored means the stored key is wrong, which is a different
/// situation from having none and has to be distinguishable: clearing it silently would look
/// like the console forgot, and keeping it would loop rejected requests behind a page that
/// never explains itself.
export function reportUnauthorized(): void {
  for (const listener of listeners) listener('unauthorized');
}

export function onKeyState(listener: (event: KeyEvent) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The header to attach, or nothing. Matches the server's `api-key`. */
export function keyHeader(): Record<string, string> {
  const key = getKey();
  return key ? { 'api-key': key } : {};
}
