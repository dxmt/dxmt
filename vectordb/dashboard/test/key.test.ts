import { beforeEach, describe, expect, it, vi } from 'vitest';
import { getKey, keyHeader, onKeyState, reportUnauthorized, setKey } from '../src/api/key';

/**
 * The key store, which is small and has three ways to be quietly wrong: keeping a credential
 * longer than the tab, sending a blank one as if it were real, and failing to tell the gate
 * that the server refused what we sent.
 */
describe('api key', () => {
  beforeEach(() => setKey(null));

  it('sends nothing when no key is set', () => {
    expect(getKey()).toBeNull();
    expect(keyHeader()).toEqual({});
  });

  it('sends the key under the header the server reads', () => {
    setKey('s3cret');
    expect(keyHeader()).toEqual({ 'api-key': 's3cret' });
  });

  /** A blank key is a mistake, and storing one gives a console that looks configured. */
  it('treats blank as absent', () => {
    for (const blank of ['', '   ', '\t']) {
      setKey(blank);
      expect(getKey(), `${JSON.stringify(blank)} was stored`).toBeNull();
      expect(keyHeader()).toEqual({});
    }
  });

  it('trims what the operator pasted', () => {
    setKey('  s3cret  ');
    expect(getKey()).toBe('s3cret');
  });

  /** sessionStorage, not localStorage: the credential must not outlive the tab. */
  it('keeps the key out of localStorage', () => {
    setKey('s3cret');
    expect(sessionStorage.getItem('dxmt.api-key')).toBe('s3cret');
    expect(localStorage.getItem('dxmt.api-key')).toBeNull();
  });

  it('forgets the key on clear', () => {
    setKey('s3cret');
    setKey(null);
    expect(getKey()).toBeNull();
    expect(sessionStorage.getItem('dxmt.api-key')).toBeNull();
  });

  /** The two events are distinguishable, which is what lets the gate say "that key was
   * refused" instead of presenting an empty form for the reader to refill identically. */
  it('reports a rejection separately from a change', () => {
    const seen: string[] = [];
    const stop = onKeyState((event) => seen.push(event));
    setKey('s3cret');
    reportUnauthorized();
    stop();
    setKey('ignored-after-unsubscribe');
    expect(seen).toEqual(['changed', 'unauthorized']);
  });

  /** Private mode throws on access. The console still has to work against an open server. */
  it('survives storage being unavailable', () => {
    const broken = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('denied');
    });
    expect(() => getKey()).not.toThrow();
    expect(getKey()).toBeNull();
    broken.mockRestore();
  });
});
