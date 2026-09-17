/**
 * What this particular build can do, discovered by asking rather than assumed.
 *
 * Two capabilities change what the UI may claim, and neither is announced anywhere:
 *
 *   - **`search_mode: "exact"`** exists only on builds that have it. Without it there is no
 *     ground truth over HTTP, and the recall page must stop calling its output "recall".
 *   - **`text-bm25`** is off in the default build, so `/search/text` answers 501
 *     `FEATURE_DISABLED` and `/search/hybrid` quietly degrades to the dense component.
 *
 * The exact-mode probe is the one that needs care. An older server rejects the unknown enum
 * variant in axum's `Json` extractor, *before any handler runs*, so the answer is **422 with a
 * plain-text body** and not the error envelope. A probe written as
 * `(await res.json()).error.code` throws there and reports the feature as broken rather than
 * absent -- which is why `envelope.ts` classifies that shape explicitly and why this reads
 * `isExtractorRejection` instead of a status code.
 */

import { api } from './client';
import { ApiFailure } from './envelope';

export interface Capabilities {
  /** Ground truth is available, so recall is measurable rather than approximable. */
  exactMode: boolean;
  /** Why not, when not -- shown in the UI rather than swallowed. */
  exactModeReason: string | null;
}

const cache = new Map<string, Capabilities>();

/**
 * Ask a table whether this server has an exact mode, with the cheapest possible query.
 *
 * A zero vector of the right dimension and `top_k: 1`: the answer is meaningless and the
 * question is not. On a table too large for the exact scan's cap the server refuses with an
 * envelope error naming the cap -- which is a "yes, but not on this table", and is reported as
 * such rather than as an absent feature.
 */
export async function probeExactMode(
  table: string,
  dimension: number,
  signal?: AbortSignal,
): Promise<Capabilities> {
  const cached = cache.get(table);
  if (cached) return cached;

  const query = {
    vector: new Array<number>(dimension).fill(0),
    top_k: 1,
    limit: 1,
    with_payload: false,
    search_mode: 'exact' as const,
  };

  let result: Capabilities;
  try {
    await api.search(table, query, signal);
    result = { exactMode: true, exactModeReason: null };
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;

    if (cause instanceof ApiFailure && cause.isExtractorRejection) {
      // The server does not know the value. This is the absence, and it is the arm that a
      // naive probe gets wrong.
      result = {
        exactMode: false,
        exactModeReason: 'this server was built before search_mode "exact" existed',
      };
    } else if (cause instanceof ApiFailure && cause.code === 'INVALID_INPUT') {
      // The mode exists; this table is too big for it, or it is a named-vector query. Either
      // way the server's own sentence is more useful than anything invented here.
      result = { exactMode: false, exactModeReason: cause.failure.kind === 'envelope'
        ? cause.failure.error.message
        : 'the server refused the exact scan on this table' };
    } else {
      // Anything else is a real failure and must not be cached as a capability answer.
      throw cause;
    }
  }

  cache.set(table, result);
  return result;
}

export function forgetCapabilities(table?: string): void {
  if (table) cache.delete(table);
  else cache.clear();
}
