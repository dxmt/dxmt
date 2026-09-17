/**
 * The five failure shapes, each asserted from a real response body.
 *
 * This is the highest-value file in the suite, because four of the five arms are shapes a
 * reasonable person would not expect a JSON API to produce, and a client that assumes the
 * documented envelope fails on three of them — silently, as a white screen, in exactly the
 * situation where the user needed the error message most.
 */

import { describe, expect, it } from 'vitest';
import { ApiFailure, unwrap } from '../src/api/envelope';

function response(body: string, status = 200, contentType = 'application/json'): Response {
  return new Response(body, { status, headers: { 'content-type': contentType } });
}

describe('unwrap', () => {
  it('returns the data out of a successful envelope', async () => {
    const { value } = await unwrap<{ total: number }>(
      response('{"success":true,"data":{"total":3},"error":null,"timestamp":"t"}'),
    );
    expect(value).toEqual({ total: 3 });
  });

  it('reports the byte count, which is the only honest measure of what a page cost', async () => {
    const body = '{"success":true,"data":{"total":3},"error":null,"timestamp":"t"}';
    const { bytes } = await unwrap(response(body));
    expect(bytes).toBe(body.length);
  });

  it('classifies the documented error envelope', async () => {
    const error = await unwrap(
      response(
        '{"success":false,"data":null,"error":{"code":"TABLE_NOT_FOUND","message":"Table not found: x","details":null},"timestamp":"t"}',
        404,
      ),
    ).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiFailure);
    const failure = error as ApiFailure;
    expect(failure.code).toBe('TABLE_NOT_FOUND');
    expect(failure.isNotFound).toBe(true);
    expect(failure.isFeatureDisabled).toBe(false);
  });

  it('recognises a disabled cargo feature as a capability answer, not a fault', async () => {
    const error = (await unwrap(
      response(
        '{"success":false,"data":null,"error":{"code":"FEATURE_DISABLED","message":"Full-text search feature is not enabled. Compile with --features text-bm25","details":null},"timestamp":"t"}',
        501,
      ),
    ).catch((e: unknown) => e)) as ApiFailure;

    expect(error.isFeatureDisabled).toBe(true);
    expect(error.message).toContain('text-bm25');
  });

  it('does not throw a SyntaxError on the plain-text extractor rejection', async () => {
    // The real body, verbatim, from a server asked for a search_mode it does not have. This is
    // what a capability probe reads, and `response.json()` throws an opaque SyntaxError on it.
    const body =
      'Failed to deserialize the JSON body into the target type: search_mode: unknown variant `exact`, expected one of `fast`, `normal`, `accurate`, `auto` at line 1 column 51';
    const error = (await unwrap(response(body, 422, 'text/plain; charset=utf-8')).catch(
      (e: unknown) => e,
    )) as ApiFailure;

    expect(error).toBeInstanceOf(ApiFailure);
    expect(error.failure.kind).toBe('text');
    expect(error.isExtractorRejection).toBe(true);
    expect(error.code).toBeNull();
    expect(error.message).toContain('unknown variant');
  });

  it('classifies /query\'s bare-string error, which is not the envelope shape', async () => {
    const error = (await unwrap(response('{"error":"parse error at line 1"}', 400)).catch(
      (e: unknown) => e,
    )) as ApiFailure;

    expect(error.failure.kind).toBe('rawJson');
    expect(error.code).toBeNull();
  });

  it('passes an unenveloped 200 through, because the health probes are unwrapped by design', async () => {
    const { value } = await unwrap<{ status: string }>(
      response('{"status":"alive","service":"vectordb"}'),
    );
    expect(value.status).toBe('alive');
  });
});
