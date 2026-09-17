/**
 * What the console will and will not send.
 *
 * `client.ts` makes a write a compile error, which is worth everything for code we write and
 * nothing at all for a string someone types into a box. So the console has its own gate, and
 * this is the file that says it holds — including for the cases a careless one would let
 * through: a write verb on a readable path, a path that only looks like one of ours, and the
 * routes the server does have and this page must still refuse.
 */

import { describe, expect, it } from 'vitest';
import { isError, parse, readableRoutes } from '../src/lib/console';

const ok = (text: string) => {
  const result = parse(text);
  if (isError(result)) throw new Error(`expected this to parse: ${result.error}`);
  return result;
};
const err = (text: string) => {
  const result = parse(text);
  if (!isError(result)) throw new Error('expected this to be refused');
  return result.error;
};

describe('parsing', () => {
  it('reads a verb and a path', () => {
    expect(ok('GET /api/v1/tables')).toMatchObject({
      method: 'GET',
      path: '/api/v1/tables',
      body: undefined,
    });
  });

  it('reads a body under the request line', () => {
    expect(ok('POST /api/v1/tables/t/search\n{"vector":[1,0],"top_k":3}')).toMatchObject({
      method: 'POST',
      path: '/api/v1/tables/t/search',
      body: { vector: [1, 0], top_k: 3 },
    });
  });

  it('ignores blank lines and comments above the request', () => {
    expect(ok('# what is in here\n\nGET /api/v1/tables').path).toBe('/api/v1/tables');
  });

  it('accepts a lowercase verb and a path without its slash', () => {
    expect(ok('get api/v1/tables')).toMatchObject({ method: 'GET', path: '/api/v1/tables' });
  });

  it('keeps a query string', () => {
    expect(ok('GET /api/v1/tables?limit=5&offset=10').path).toBe('/api/v1/tables?limit=5&offset=10');
  });

  it('points at the line a bad body is on', () => {
    const result = parse('POST /api/v1/tables/t/search\n{not json');
    expect(isError(result) && result.line).toBe(1);
  });
});

describe('the read-only gate', () => {
  it('refuses every write verb, even on a path it can read', () => {
    for (const verb of ['DELETE', 'PUT', 'PATCH']) {
      expect(err(`${verb} /api/v1/tables/t`)).toMatch(/readable with GET, not with/);
    }
  });

  it('refuses the routes that write, and says where they can be sent instead', () => {
    for (const request of [
      'POST /api/v1/tables\n{"name":"x","dimension":4}',
      'POST /api/v1/tables/t/records\n{"records":[]}',
      'POST /api/v1/tables/t/index/rebuild\n{}',
      'POST /api/v1/snapshot\n{}',
      'POST /api/v1/backup\n{"path":"/tmp/x"}',
    ]) {
      const message = err(request);
      // Two shapes of refusal — "not a route this console reaches" and "that path is readable
      // with a different verb" — and both have to point somewhere, or the reader is left
      // wondering whether the route exists at all.
      expect(message).toMatch(/read-only|not with POST/);
      expect(message).toMatch(/Swagger/);
    }
  });

  it('refuses POST /query, which executes DDL and DML', () => {
    // The one that looks like a read and is not: DXQL runs INSERT, UPDATE, DELETE and DROP.
    expect(err('POST /api/v1/query\n{"query":"SELECT * FROM t"}')).toMatch(/read-only/);
  });

  it('is not fooled by a path that merely contains a readable one', () => {
    expect(err('GET /api/v1/tables/t/records/list/../../../etc')).toMatch(/read-only/);
    expect(err('POST /api/v1/tables/t/searchx\n{}')).toMatch(/read-only/);
  });

  it('allows exactly the routes the help text advertises', () => {
    for (const route of readableRoutes()) {
      const path = route.path.replace('{t}', 'demo').replace('{id}', 'r1');
      const text = route.method === 'GET' ? `GET ${path}` : `POST ${path}\n{}`;
      const result = parse(text);
      expect(isError(result), `${route.method} ${path}: ${isError(result) ? result.error : ''}`).toBe(
        false,
      );
    }
  });
});

describe('shape errors', () => {
  it('says what a request looks like when given nonsense', () => {
    expect(err('hello')).toMatch(/verb and a path/);
    expect(err('   ')).toMatch(/nothing to send/);
    expect(err('FETCH /api/v1/tables')).toMatch(/not an HTTP verb/);
  });

  it('will not send a POST with no body, which the server answers 415', () => {
    expect(err('POST /api/v1/tables/t/search')).toMatch(/needs a JSON body/);
  });

  it('will not send a GET with one', () => {
    expect(err('GET /api/v1/tables\n{"a":1}')).toMatch(/carries no body/);
  });
});
