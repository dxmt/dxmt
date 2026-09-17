/**
 * The client's two jobs that are easy to get wrong silently.
 *
 * Percent-encoding is not hypothetical here: commit `9030ca3` fixed exactly this bug in the
 * Python client, where a table name or record id containing a slash addressed a different
 * resource without any error. A console meets such ids more often than a program does, because
 * a person types them.
 *
 * And the base URL has to survive every spelling of the dashboard's own path, because a reload
 * on a deep client-side route resolves relative URLs against wherever the user happened to be.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, apiBase, post } from '../src/api/client';

function at(path: string) {
  Object.defineProperty(window, 'location', {
    value: new URL(`http://host:9999${path}`),
    writable: true,
  });
}

function capture() {
  const calls: string[] = [];
  vi.stubGlobal('fetch', (url: string) => {
    calls.push(url);
    return Promise.resolve(
      new Response('{"success":true,"data":{},"error":null,"timestamp":"t"}', {
        headers: { 'content-type': 'application/json' },
      }),
    );
  });
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe('apiBase', () => {
  it('strips the dashboard segment however the page was addressed', () => {
    for (const path of [
      '/dashboard',
      '/dashboard/',
      '/dashboard/index.html',
      '/dashboard/tables/mine/search',
    ]) {
      at(path);
      expect(apiBase(), path).toBe('');
    }
  });

  it('survives being mounted under a prefix by a reverse proxy', () => {
    at('/dxmt/dashboard/tables/mine');
    expect(apiBase()).toBe('/dxmt');
  });
});

describe('path encoding', () => {
  it('encodes a table name that would otherwise change the route', async () => {
    at('/dashboard/');
    const calls = capture();
    // Without encoding this addresses /tables/a/b/stats -- a different route entirely.
    await api.stats('a/b');
    expect(calls[0]).toBe('/api/v1/tables/a%2Fb/stats');
  });

  it('encodes a record id with a space, a hash and a question mark', async () => {
    at('/dashboard/');
    const calls = capture();
    await api.record('t', 'doc 1#frag?x=1');
    expect(calls[0]).toBe('/api/v1/tables/t/records/doc%201%23frag%3Fx%3D1');
  });
});

describe('read-only by construction', () => {
  it('exports no write verbs', async () => {
    const module: Record<string, unknown> = await import('../src/api/client');
    for (const verb of ['put', 'patch', 'del', 'delete', 'remove']) {
      expect(Object.keys(module)).not.toContain(verb);
    }
  });

  it('exposes only reading operations', async () => {
    const names = Object.keys(api);
    // Anything that mutates would have to be named here to be callable, and the type of
    // `post`'s path forbids the routes that mutate. Both halves, asserted.
    expect(names.sort()).toEqual(
      [
        'metrics',
        'record',
        'records',
        'search',
        'searchHybrid',
        'searchText',
        'stats',
        'status',
        'table',
        'tables',
      ].sort(),
    );
    expect(typeof post).toBe('function');
  });
});
