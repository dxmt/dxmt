/**
 * How many points the visualize page may fetch, and why that is not one number.
 *
 * There are two independent costs and they scale differently.
 *
 * **The wire** is `n × d`: JSON floats run about thirteen bytes a dimension, and nothing on
 * this server is compressed. Two thousand 768-dimension vectors is roughly twenty megabytes of
 * text the browser then has to parse.
 *
 * **The server** is the number of `records/list` calls, because each one collects every record
 * in the table, sorts them all by `created_at` and slices — the page cost is the table, not the
 * page. Four calls on a million rows is four full sorts.
 *
 * So the budget is a pair, and the page shows the arithmetic rather than a mystery ceiling.
 */

/** Roughly 13 MB of JSON, which is about as much as is reasonable to parse in a tab. */
export const MAX_FLOATS = 1_000_000;
/** Each one is a full table scan on the server. */
export const MAX_REQUESTS = 4;
/** The server's own clamp on `limit`. */
export const PAGE = 1000;

export interface Budget {
  points: number;
  requests: number;
  estimatedBytes: number;
  /** Why the ceiling is where it is, in words the page can print. */
  reason: 'requests' | 'floats' | 'table';
}

export function budgetFor(recordCount: number, dimension: number): Budget {
  const byFloats = Math.floor(MAX_FLOATS / Math.max(dimension, 1));
  const byRequests = MAX_REQUESTS * PAGE;

  let points = Math.min(byFloats, byRequests, recordCount);
  points = Math.max(Math.min(points, recordCount), 0);

  const reason: Budget['reason'] =
    points === recordCount ? 'table' : byFloats <= byRequests ? 'floats' : 'requests';

  return {
    points,
    requests: Math.max(1, Math.ceil(points / PAGE)),
    estimatedBytes: points * dimension * 13,
    reason,
  };
}

/**
 * Where to start each request.
 *
 * `records/list` is ordered `created_at` descending, so offset 0 is the newest rows — a badly
 * biased sample of a table that was loaded over time, and one that would make the scatter a
 * picture of last Tuesday. Striding across evenly spaced windows is a cluster sample over
 * insertion-time strata, not a uniform random sample, and the page says which it is rather than
 * letting the reader assume.
 */
export function strata(total: number, requests: number): number[] {
  if (requests <= 1) return [0];
  return Array.from({ length: requests }, (_, i) => Math.floor((i * total) / requests));
}
