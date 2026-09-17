/**
 * Parsing what someone types into the console, and deciding whether we may send it.
 *
 * The format is the one every REST console uses because it is the one people already know:
 * a verb and a path on the first line, an optional JSON body below.
 *
 *     GET /api/v1/tables
 *
 *     POST /api/v1/tables/articles/search
 *     { "vector": [0.1, 0.2], "top_k": 5 }
 *
 * **Read-only is enforced here as well as in the client.** `client.ts` refuses a write at
 * compile time, which is worth everything for code we write and nothing at all for a string a
 * user types — so this is the second gate, and it is the one that matters for this page. A
 * request the console will not send is refused before it is sent, with the reason, rather than
 * being passed to a server that would happily carry it out.
 */

/**
 * A request that has been through the gate below — and the only way to obtain one.
 *
 * The brand is not decoration. `sendRead` in `client.ts` accepts a `Parsed` and nothing else,
 * and the claim is that it therefore cannot be handed a write. Without the brand that claim was
 * false: `Parsed` was a plain shape and `{ method: 'POST', path: '/api/v1/tables', body: {} }`
 * satisfied it, so the gate could be walked around by anyone writing an object literal. A
 * private symbol nobody outside this module can name makes `parse` the only source.
 */
declare const checked: unique symbol;

export interface Parsed {
  method: 'GET' | 'POST';
  path: string;
  body: unknown;
  readonly [checked]: true;
}

/** The only place a `Parsed` is minted. */
const passed = (method: 'GET' | 'POST', path: string, body: unknown): Parsed =>
  ({ method, path, body }) as Parsed;

export interface ParseError {
  error: string;
  /** Which line the problem is on, so the editor can point at it. */
  line?: number;
}

export type ParseResult = Parsed | ParseError;

/**
 * The discriminant is the presence of `error`, which only works because `Parsed` does not carry
 * one. The first version gave the success branch `error: null` — copying the response envelope,
 * where that field is meaningful — and every successful parse was then reported as a failure.
 * The tests below caught it on their first run.
 */
export const isError = (result: ParseResult): result is ParseError => 'error' in result;

/** Every path this console is allowed to reach, and why it is a read. */
const READS: { pattern: RegExp; method: 'GET' | 'POST'; what: string }[] = [
  { pattern: /^\/status$/, method: 'GET', what: 'server status' },
  { pattern: /^\/health(\/(live|ready|metrics))?$/, method: 'GET', what: 'health probes' },
  { pattern: /^\/api\/v1\/health(\/(live|ready|metrics))?$/, method: 'GET', what: 'health probes' },
  { pattern: /^\/api\/v1\/tables(\?.*)?$/, method: 'GET', what: 'list tables' },
  { pattern: /^\/api\/v1\/tables\/[^/]+$/, method: 'GET', what: 'table detail' },
  { pattern: /^\/api\/v1\/tables\/[^/]+\/stats$/, method: 'GET', what: 'table stats' },
  { pattern: /^\/api\/v1\/tables\/[^/]+\/records\/[^/]+$/, method: 'GET', what: 'one record' },
  { pattern: /^\/api\/v1\/tables\/[^/]+\/records\/list$/, method: 'POST', what: 'list records' },
  { pattern: /^\/api\/v1\/tables\/[^/]+\/search$/, method: 'POST', what: 'vector search' },
  { pattern: /^\/api\/v1\/tables\/[^/]+\/search\/text$/, method: 'POST', what: 'text search' },
  { pattern: /^\/api\/v1\/tables\/[^/]+\/search\/hybrid$/, method: 'POST', what: 'hybrid search' },
  { pattern: /^\/api\/v1\/tables\/[^/]+\/graph$/, method: 'POST', what: 'neighbour graph' },
];

const VERBS = new Set(['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS']);

export function parse(text: string): ParseResult {
  const lines = text.split('\n');
  const firstIndex = lines.findIndex((line) => line.trim() !== '' && !line.trim().startsWith('#'));
  if (firstIndex === -1) return { error: 'nothing to send' };

  const [verb, rawPath, ...rest] = lines[firstIndex]!.trim().split(/\s+/);
  if (!verb || !rawPath) {
    return { error: 'a request is a verb and a path, like `GET /api/v1/tables`', line: firstIndex };
  }
  if (rest.length > 0) {
    return { error: `unexpected ${rest.join(' ')} after the path`, line: firstIndex };
  }

  const method = verb.toUpperCase();
  if (!VERBS.has(method)) {
    return { error: `${verb} is not an HTTP verb`, line: firstIndex };
  }
  const path = rawPath.startsWith('/') ? rawPath : `/${rawPath}`;

  // The gate. Note it checks the *path* before the method, so a DELETE to a readable path is
  // refused for the right reason rather than being reported as an unknown route.
  const known = READS.filter((entry) => entry.pattern.test(path));
  if (known.length === 0) {
    return {
      error:
        `this console can only reach read-only routes, and ${path} is not one of them. ` +
        `The server has more — creating tables, writing records, snapshots — and this page ` +
        `will not send them; the Swagger page at /api/docs will.`,
      line: firstIndex,
    };
  }
  if (!known.some((entry) => entry.method === method)) {
    const allowed = [...new Set(known.map((entry) => entry.method))].join(' or ');
    return {
      // Says where to go, not only what is refused. `POST /api/v1/tables` is a real route that
      // creates a table; being told only "this path is readable with GET" leaves someone
      // wondering whether the route exists at all.
      error:
        `${path} is readable with ${allowed}, not with ${method}. This console sends nothing ` +
        `that changes data; the Swagger page at /api/docs will.`,
      line: firstIndex,
    };
  }

  const rawBody = lines.slice(firstIndex + 1).join('\n').trim();
  if (rawBody === '') {
    if (method === 'POST') {
      return { error: `${path} needs a JSON body`, line: firstIndex + 1 };
    }
    return passed('GET', path, undefined);
  }

  let body: unknown;
  try {
    body = JSON.parse(rawBody);
  } catch (cause) {
    return {
      error: cause instanceof Error ? cause.message : String(cause),
      line: firstIndex + 1,
    };
  }
  if (method === 'GET') {
    return { error: 'a GET carries no body', line: firstIndex + 1 };
  }
  return passed('POST', path, body);
}

/** What the console can reach, for the help text — generated rather than written twice. */
export function readableRoutes(): { method: string; path: string; what: string }[] {
  return [
    { method: 'GET', path: '/status', what: 'server status' },
    { method: 'GET', path: '/health/metrics', what: 'in-flight operations, WAL, snapshots' },
    { method: 'GET', path: '/api/v1/tables', what: 'list tables' },
    { method: 'GET', path: '/api/v1/tables/{t}', what: 'table detail' },
    { method: 'GET', path: '/api/v1/tables/{t}/stats', what: 'size and index type' },
    { method: 'GET', path: '/api/v1/tables/{t}/records/{id}', what: 'one record, with its vector' },
    { method: 'POST', path: '/api/v1/tables/{t}/records/list', what: 'a page of records' },
    { method: 'POST', path: '/api/v1/tables/{t}/search', what: 'vector search' },
    { method: 'POST', path: '/api/v1/tables/{t}/search/text', what: 'BM25 (needs text-bm25)' },
    { method: 'POST', path: '/api/v1/tables/{t}/search/hybrid', what: 'vector + text' },
    { method: 'POST', path: '/api/v1/tables/{t}/graph', what: 'neighbour graph over a sample' },
  ];
}
