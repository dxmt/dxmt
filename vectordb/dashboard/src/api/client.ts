/**
 * The only place in this app that talks to the network.
 *
 * # Read-only is a type, not a promise
 *
 * The server has no authentication: anyone who can reach the port can already rewrite every
 * table. That is exactly why this console must not be able to. So "read-only" is enforced where
 * it cannot be forgotten rather than in a code-review checklist:
 *
 *   - there is no `put`, `patch` or `del` function in this module, or anywhere else;
 *   - `post`'s path parameter is a closed union of the five read-only POST routes, so a write
 *     call is a compile error and not a review catch;
 *   - an eslint rule forbids bare `fetch(` outside this file.
 *
 * `POST /api/v1/query` is deliberately absent from that union: DXQL executes INSERT, UPDATE,
 * DELETE and DROP TABLE, so a query console would make the phrase "read-only dashboard" false.
 *
 * # Why not `dxmt-js-sdk`
 *
 * The same reason `tests/api/helpers/client.py` gives for not using it: the SDK renames
 * `payload` to `metadata` and unwraps the envelope, and a console whose whole job is showing
 * what the server returned must not show the SDK's opinion of what the server meant. It is also
 * released on its own cadence while this ships inside the binary, and it has no `search_mode` --
 * the one field two of these pages exist to vary.
 */

import { ApiFailure, unwrap } from './envelope';
import type {
  ListRecordsRequest,
  RecordPage,
  SearchQuery,
  SearchResult,
  Table,
  TableList,
  TableStats,
  TableWithCount,
  ServerStatus,
  HealthMetrics,
  WireRecord,
} from './types';

/**
 * Where the API is, derived from where this page was served.
 *
 * The bundle is mounted at `/dashboard/`, and the API lives at the root, so the base is this
 * page's URL with that segment removed. Every spelling has to work -- `/dashboard`,
 * `/dashboard/`, `/dashboard/index.html`, and any client-side route below it -- because a
 * reload on a deep link resolves relative URLs against wherever the user happened to be.
 *
 * In dev, Vite proxies `/api` to the server, so the same relative paths work with no special
 * case. That is the point of proxying rather than pointing at an absolute origin: the dev and
 * production request paths are identical, so they cannot drift.
 */
export function apiBase(): string {
  const path = window.location.pathname;
  const index = path.indexOf('/dashboard');
  return index === -1 ? '' : path.slice(0, index);
}

/** Every read-only POST this app is permitted to make. Adding a write here is the whole review. */
type ReadOnlyPost =
  | `/api/v1/tables/${string}/records/list`
  | `/api/v1/tables/${string}/search`
  | `/api/v1/tables/${string}/search/text`
  | `/api/v1/tables/${string}/search/hybrid`;

export interface Timed<T> {
  value: T;
  /** Wall clock at the browser, which includes everything `tookms` does not. */
  ms: number;
  /** Uncompressed JSON length. The only honest measure of what a page of records cost. */
  bytes: number;
}

async function request<T>(
  method: 'GET' | 'POST',
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<Timed<T>> {
  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(apiBase() + path, {
      method,
      signal,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') {
      throw cause;
    }
    throw new ApiFailure({
      kind: 'network',
      message: cause instanceof Error ? cause.message : String(cause),
    });
  }
  const { value, bytes } = await unwrap<T>(response);
  return { value, ms: performance.now() - started, bytes };
}

export function get<T>(path: string, signal?: AbortSignal): Promise<Timed<T>> {
  return request<T>('GET', path, undefined, signal);
}

export function post<T>(
  path: ReadOnlyPost,
  body: unknown,
  signal?: AbortSignal,
): Promise<Timed<T>> {
  return request<T>('POST', path, body, signal);
}

/**
 * Percent-encode a path segment.
 *
 * Not decoration: commit `9030ca3` in this repository fixed exactly this bug in the Python
 * client, where a table name or record id containing a slash, space or `#` silently addressed
 * the wrong resource. A console is more likely to meet such an id than a program is, because a
 * person types it.
 */
const seg = (value: string) => encodeURIComponent(value);

export const api = {
  status: (signal?: AbortSignal) => get<ServerStatus>('/status', signal),

  metrics: (signal?: AbortSignal) => get<HealthMetrics>('/health/metrics', signal),

  tables: (limit: number, offset: number, signal?: AbortSignal) =>
    get<TableList>(`/api/v1/tables?limit=${limit}&offset=${offset}`, signal),

  table: (name: string, signal?: AbortSignal) =>
    get<TableWithCount>(`/api/v1/tables/${seg(name)}`, signal),

  stats: (name: string, signal?: AbortSignal) =>
    get<TableStats>(`/api/v1/tables/${seg(name)}/stats`, signal),

  records: (name: string, body: ListRecordsRequest, signal?: AbortSignal) =>
    post<RecordPage>(`/api/v1/tables/${seg(name)}/records/list`, body, signal),

  record: (name: string, id: string, signal?: AbortSignal) =>
    get<WireRecord>(`/api/v1/tables/${seg(name)}/records/${seg(id)}`, signal),

  search: (name: string, query: SearchQuery, signal?: AbortSignal) =>
    post<SearchResult>(`/api/v1/tables/${seg(name)}/search`, query, signal),

  searchText: (name: string, body: unknown, signal?: AbortSignal) =>
    post<SearchResult>(`/api/v1/tables/${seg(name)}/search/text`, body, signal),

  searchHybrid: (name: string, body: unknown, signal?: AbortSignal) =>
    post<SearchResult>(`/api/v1/tables/${seg(name)}/search/hybrid`, body, signal),
};

export type { Table };
