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

export interface TextCapability {
  /** BM25 and hybrid fusion can be asked for on this table. */
  textSearch: boolean;
  /** Which of the two absences this is, in the server's own words where it has them. */
  textSearchReason: string | null;
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

const textCache = new Map<string, TextCapability>();

/**
 * Whether this table can answer a text query, and if not, which kind of "not".
 *
 * There are two, they are not interchangeable, and a page that renders one sentence for both
 * sends people to fix the wrong thing:
 *
 *   - **the build has no `text-bm25`** -- `/search/text` answers `FEATURE_DISABLED` for every
 *     table, and `/search/hybrid` silently drops the text half rather than failing, which is
 *     the more dangerous of the two because the answer still looks like a hybrid answer;
 *   - **the table was created without `enable_full_text_search`** -- the feature is compiled in
 *     and this one table has no text index. That flag cannot be changed after creation, so the
 *     honest advice is to recreate the table, not to restart the server.
 *
 * The table's own flag settles the second without a request, so the probe only runs when the
 * flag says yes. A table flagged for text search that still answers `FEATURE_DISABLED` is the
 * first case, and is reported as such.
 */
export async function probeTextSearch(
  table: string,
  enabledOnTable: boolean,
  signal?: AbortSignal,
): Promise<TextCapability> {
  const cached = textCache.get(table);
  if (cached) return cached;

  let result: TextCapability;
  if (!enabledOnTable) {
    result = {
      textSearch: false,
      textSearchReason:
        'this table was created without enable_full_text_search, which cannot be turned on afterwards',
    };
  } else {
    try {
      // A token nothing will match, asking for one hit: the answer is uninteresting and the
      // question is whether the endpoint is there at all.
      await api.searchText(table, { query: 'dxmt_capability_probe', limit: 1 }, signal);
      result = { textSearch: true, textSearchReason: null };
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;

      if (cause instanceof ApiFailure && cause.code === 'FEATURE_DISABLED') {
        result = {
          textSearch: false,
          textSearchReason: 'this server was built without the text-bm25 feature',
        };
      } else if (cause instanceof ApiFailure && cause.code === 'FULL_TEXT_NOT_ENABLED') {
        // The table's flag said yes and the server says no. Trust the server and say so plainly
        // rather than quietly showing the control.
        result = {
          textSearch: false,
          textSearchReason: cause.message,
        };
      } else {
        throw cause;
      }
    }
  }

  textCache.set(table, result);
  return result;
}

export function forgetCapabilities(table?: string): void {
  if (table) {
    cache.delete(table);
    textCache.delete(table);
  } else {
    cache.clear();
    textCache.clear();
  }
}
