import { asCurl } from '../lib/curl';
import { CopyButton } from './CopyButton';
import { bytes } from '../lib/format';

/**
 * What this page just asked for, what it cost, and how to ask again from a terminal.
 *
 * The byte count is measured rather than estimated — `client.ts` reads every response as text
 * before parsing it, so this is the real uncompressed JSON length. It is the number that
 * explains why a page felt slow, and it is not available anywhere else.
 */
export function RequestPreview({
  method,
  path,
  body,
  ms,
  wire,
  serverMs,
}: {
  method: string;
  path: string;
  body?: unknown;
  ms?: number;
  wire?: number;
  serverMs?: number;
}) {
  const command = asCurl(method, path, body);
  return (
    <details className="request">
      <summary>
        <span className="mono">
          {method} {path}
        </span>
        {ms === undefined ? null : (
          <span className="muted">
            {' · '}
            {Math.round(ms)} ms
            {serverMs === undefined ? '' : ` (server ${serverMs} ms)`}
            {wire === undefined ? '' : ` · ${bytes(wire)} of JSON`}
          </span>
        )}
      </summary>
      <div className="request-body">
        <div className="jsonblock-tools">
          <CopyButton text={command} label="copy as curl" />
        </div>
        <pre>{command}</pre>
      </div>
    </details>
  );
}
