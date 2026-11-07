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
 *   - `post`'s path parameter is a closed union of the six read-only POST routes, so a write
 *     call is a compile error and not a review catch;
 *   - an eslint rule forbids bare `fetch(` outside this file.
 *
 * # The one exception: your own memories
 *
 * `ownMemory` below makes exactly two writes, `POST /api/v1/memories` (correct a memory) and
 * `DELETE /api/v1/memories/{id}` (forget one), with both paths fixed in this file rather than
 * passed in. The memories are the user's own record of what their agents know about them, and
 * owning that record means being able to correct it and delete it from the place they read it.
 * Tables and records stay read-only; `tests/interfaces/api_contract_test.rs` allows these two
 * routes by exact method and path and fails on any other write in `dashboard-routes.json`.
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
import { keyHeader, reportUnauthorized } from './key';
import type { Parsed } from '../lib/console';
import type {
  Forgotten,
  FullTextQuery,
  FullTextResult,
  HybridQuery,
  KnnGraph,
  KnnGraphRequest,
  ListRecordsRequest,
  MemoryPage,
  RecallRequest,
  RecallResult,
  RecordPage,
  Remembered,
  RememberRequest,
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
  | `/api/v1/tables/${string}/search/hybrid`
  | `/api/v1/tables/${string}/graph`
  | '/api/v1/memories/recall';

export interface Timed<T> {
  value: T;
  /** Wall clock at the browser, which includes everything `tookms` does not. */
  ms: number;
  /** Uncompressed JSON length. The only honest measure of what a page of records cost. */
  bytes: number;
}

async function request<T>(
  // `DELETE` is here for `ownMemory.forget` alone; `request` is not exported, and nothing else
  // in this module passes it.
  method: 'GET' | 'POST' | 'DELETE',
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
      headers: {
        ...keyHeader(),
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
      },
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
  // Before `unwrap` throws: the gate needs to know, and every caller of this would otherwise
  // have to remember to tell it.
  if (response.status === 401) reportUnauthorized();
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

  /**
   * The nearest-neighbour structure of a sample, without its vectors.
   *
   * A thousand 768-dimension vectors is 8.5 MB of JSON; the same thousand nodes as a graph is
   * 0.36 MB, and four thousand nodes is 1.5 MB — still less than a quarter of the vectors it
   * replaces. That is what lifts the ceiling on how much of a table can be laid out.
   */
  graph: (name: string, body: KnnGraphRequest, signal?: AbortSignal) =>
    post<KnnGraph>(`/api/v1/tables/${seg(name)}/graph`, body, signal),

  record: (name: string, id: string, signal?: AbortSignal) =>
    get<WireRecord>(`/api/v1/tables/${seg(name)}/records/${seg(id)}`, signal),

  search: (name: string, query: SearchQuery, signal?: AbortSignal) =>
    post<SearchResult>(`/api/v1/tables/${seg(name)}/search`, query, signal),

  /** Answers with `FullTextResult`, whose timing field is `took_ms` and not `tookms`. */
  searchText: (name: string, query: FullTextQuery, signal?: AbortSignal) =>
    post<FullTextResult>(`/api/v1/tables/${seg(name)}/search/text`, query, signal),

  /** Answers with the same `SearchResult` the dense path does, fused server-side. */
  searchHybrid: (name: string, query: HybridQuery, signal?: AbortSignal) =>
    post<SearchResult>(`/api/v1/tables/${seg(name)}/search/hybrid`, query, signal),

  /** Newest first. `{memories: [], total: 0}` when memory is off or nothing is remembered. */
  memories: (kind: string | null, limit: number, offset: number, signal?: AbortSignal) =>
    get<MemoryPage>(
      `/api/v1/memories?${kind ? `kind=${encodeURIComponent(kind)}&` : ''}limit=${limit}&offset=${offset}`,
      signal,
    ),

  /** A POST because the query is a structured body; it changes nothing. */
  recall: (query: RecallRequest, signal?: AbortSignal) =>
    post<RecallResult>('/api/v1/memories/recall', query, signal),
};

/**
 * The only two writes this console can make, both to the user's own memories.
 *
 * Kept apart from `api` so the exception is one object to read rather than two entries among
 * eleven reads, and so the paths are literals here: no caller supplies a route, only the body
 * of a memory and the id of one. See the header of this file for why the exception exists.
 */
export const ownMemory = {
  /** Correct: a new memory that `supersedes` the old one, or the same `id` to fix in place. */
  remember: (body: RememberRequest) => request<Remembered>('POST', '/api/v1/memories', body),

  /** Forget, permanently. The server answers `forgotten: 0` for an id it did not hold. */
  forget: (id: string) => request<Forgotten>('DELETE', `/api/v1/memories/${seg(id)}`),
};

export type { Table };

export interface RawReply {
  status: number;
  /** Parsed when the body is JSON, the raw text when it is not — a plain-text rejection is a
   *  real answer the console should show, not a `SyntaxError` to swallow. */
  body: unknown;
  text: string;
  ms: number;
  bytes: number;
}

/**
 * Send a request the console composed, and hand back what came off the wire.
 *
 * The console needs a path decided at runtime, which the closed union above cannot express —
 * that is a compile-time construct and a typed string is not. So the guarantee moves into the
 * argument: this takes a `Parsed`, and a `Parsed` can only be produced by `parse()` in
 * `lib/console.ts`, which refuses everything that writes. There is no way to call this with a
 * path that has not been through the gate.
 *
 * It also stays in this module so the eslint rule holds — `fetch` appears in exactly one file,
 * which is what makes "nothing else can reach the network" checkable rather than a convention.
 */
export async function sendRead(request: Parsed, signal?: AbortSignal): Promise<RawReply> {
  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(apiBase() + request.path, {
      method: request.method,
      signal,
      headers: {
        ...keyHeader(),
        ...(request.body === undefined ? {} : { 'Content-Type': 'application/json' }),
      },
      body: request.body === undefined ? undefined : JSON.stringify(request.body),
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiFailure({
      kind: 'network',
      message: cause instanceof Error ? cause.message : String(cause),
    });
  }

  const text = await response.text();
  let body: unknown = text;
  try {
    body = JSON.parse(text);
  } catch {
    // Left as text on purpose.
  }
  return {
    status: response.status,
    body,
    text,
    ms: performance.now() - started,
    bytes: text.length,
  };
}

/**
 * Does this server want a key, and is the one we hold accepted?
 *
 * `/status` is the probe because it needs the key — an endpoint that reports whether
 * authentication is on would be answering a question to someone who has not authenticated.
 *
 * It lives here rather than in the gate that calls it for the same reason everything else
 * does: `fetch` appears in exactly one file, which is what makes "nothing else can reach the
 * network" a checkable property rather than a convention. The eslint rule caught the first
 * version of this sitting in the component.
 */
export async function keyIsAccepted(): Promise<boolean> {
  try {
    const response = await fetch(apiBase() + '/status', {
      headers: keyHeader(),
    });
    return response.status !== 401;
  } catch {
    // Unreachable is not unauthorized. The network banner covers that case, and returning
    // false here would put a key form in front of a server that is simply down.
    return true;
  }
}
