/**
 * Classifying a response, which is harder here than it looks.
 *
 * There are five distinct failure shapes on this server, not one, and a client that assumes the
 * envelope breaks on three of them:
 *
 *   1. `fetch` itself rejects -- the server is down, DNS failed, or the browser blocked the
 *      request. JavaScript cannot tell those apart, so neither can the message.
 *   2. The envelope: `{success: false, error: {code, message, details}}`. The documented one.
 *   3. Raw JSON with no envelope: `POST /query` answers `{"error": "<string>"}` -- a plain
 *      string where every other route sends an object -- and `/health/live` and
 *      `/health/ready` are unwrapped by design.
 *   4. **Plain text.** axum's `Json` extractor rejects a malformed body or an unknown enum
 *      variant *before any handler runs*, so there is no `ApiResponse` to wrap it: the answer
 *      is 422 with `content-type: text/plain`. A current server no longer does this -- every
 *      body goes through `ApiJson`, which answers in the envelope -- but an older one does, and
 *      it is exactly what probing for `search_mode: "exact"` returns there. The arm stays
 *      load-bearing for those servers: `(await res.json()).error.code` throws on it, and a
 *      capability probe that throws reports a feature as broken instead of absent.
 *   5. A 200 that is not the envelope at all, which means the shape changed underneath us.
 *
 * Nothing downstream is allowed to reach into `error.code` without narrowing through this.
 */

import type { ApiError, Envelope } from './types';

export type Failure =
  | { kind: 'network'; message: string }
  | { kind: 'envelope'; status: number; error: ApiError }
  | { kind: 'rawJson'; status: number; body: unknown }
  | { kind: 'text'; status: number; text: string }
  | { kind: 'parse'; status: number; text: string };

export class ApiFailure extends Error {
  readonly failure: Failure;

  constructor(failure: Failure) {
    super(describe(failure));
    this.name = 'ApiFailure';
    this.failure = failure;
  }

  /** The server's own error code, when there was one. */
  get code(): string | null {
    return this.failure.kind === 'envelope' ? this.failure.error.code : null;
  }

  get status(): number | null {
    return this.failure.kind === 'network' ? null : this.failure.status;
  }

  /** A feature that was not compiled into this build, rather than anything being wrong. */
  get isFeatureDisabled(): boolean {
    return this.code === 'FEATURE_DISABLED' || this.code === 'NOT_IMPLEMENTED';
  }

  get isNotFound(): boolean {
    return this.code === 'TABLE_NOT_FOUND' || this.code === 'RECORD_NOT_FOUND';
  }

  /**
   * The body was refused before a handler saw it. Almost always means we sent a field or an
   * enum value this build does not know about -- which is a capability answer, not a bug.
   */
  get isExtractorRejection(): boolean {
    return this.failure.kind === 'text' && (this.status === 422 || this.status === 400);
  }
}

function describe(failure: Failure): string {
  switch (failure.kind) {
    case 'network':
      return `cannot reach the server (${failure.message})`;
    case 'envelope':
      return `${failure.error.code}: ${failure.error.message}`;
    case 'rawJson':
      return `HTTP ${failure.status}: ${JSON.stringify(failure.body).slice(0, 300)}`;
    case 'text':
      return `HTTP ${failure.status} (the request body was refused before any handler ran): ${failure.text.slice(0, 300)}`;
    case 'parse':
      return `HTTP ${failure.status}: the response was not in the expected shape`;
  }
}

/**
 * Turn a `Response` into `T`, or throw an `ApiFailure` carrying which of the five it was.
 *
 * Reads the body as text first and parses it here rather than calling `response.json()`, for
 * two reasons: `json()` throws an opaque `SyntaxError` on a plain-text error, discarding the
 * message that says what was actually wrong; and the text length is the only honest measure of
 * what a request cost, which the record browser displays.
 */
export async function unwrap<T>(response: Response): Promise<{ value: T; bytes: number }> {
  const text = await response.text();
  const bytes = text.length;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new ApiFailure({ kind: 'text', status: response.status, text });
  }

  if (isEnvelope(parsed)) {
    if (parsed.success) {
      return { value: parsed.data as T, bytes };
    }
    throw new ApiFailure({
      kind: 'envelope',
      status: response.status,
      error: parsed.error ?? { code: 'UNKNOWN', message: text.slice(0, 300), details: null },
    });
  }

  // Not the envelope. Either a route that never wraps (the two health probes), or `/query`.
  if (response.ok) {
    return { value: parsed as T, bytes };
  }
  throw new ApiFailure({ kind: 'rawJson', status: response.status, body: parsed });
}

function isEnvelope(value: unknown): value is Envelope<unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    'success' in value &&
    typeof (value as { success: unknown }).success === 'boolean'
  );
}
