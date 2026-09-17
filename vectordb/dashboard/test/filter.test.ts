/**
 * The filter model, and the one mistake worth catching by name.
 *
 * `tests/api/helpers/filters.py` in the server repo notes that the Python SDK accepts
 * Mongo-style `{"rank": {"$gte": 3}}` and translates it client-side — the server does not take
 * it. Anyone who has used that SDK will paste it here, so the editor recognises it and says
 * what the right shape is instead of forwarding a request that fails with a parser error.
 */

import { describe, expect, it } from 'vitest';
import { coerce, newRow, parseFilter, toFilter, type Row } from '../src/lib/filter';

const row = (over: Partial<Row>): Row => ({ ...newRow(), ...over });

describe('toFilter', () => {
  it('emits exactly the shape the server takes', () => {
    expect(toFilter([row({ group: 'must', key: 'rank', operator: 'GreaterThanOrEqual', value: '3' })])).toEqual({
      must: [{ key: 'rank', operation: { type: 'GreaterThanOrEqual', value: 3 } }],
    });
  });

  it('omits an empty group rather than sending it as an empty list', () => {
    const filter = toFilter([row({ group: 'must', key: 'city', value: 'London' })]);
    expect(filter).toEqual({ must: [{ key: 'city', operation: { type: 'Equal', value: 'London' } }] });
    expect(filter).not.toHaveProperty('should');
    expect(filter).not.toHaveProperty('must_not');
  });

  it('is null when nothing is filled in, so no filter key is sent at all', () => {
    expect(toFilter([])).toBeNull();
    expect(toFilter([row({ key: '  ' })])).toBeNull();
  });

  it('keeps all three groups apart', () => {
    expect(
      toFilter([
        row({ group: 'must', key: 'a', value: '1' }),
        row({ group: 'must_not', key: 'b', value: '2' }),
        row({ group: 'should', key: 'c', value: '3' }),
      ]),
    ).toEqual({
      must: [{ key: 'a', operation: { type: 'Equal', value: 1 } }],
      must_not: [{ key: 'b', operation: { type: 'Equal', value: 2 } }],
      should: [{ key: 'c', operation: { type: 'Equal', value: 3 } }],
    });
  });
});

describe('coerce', () => {
  it('reads numbers, booleans and null as themselves', () => {
    expect(coerce('Equal', '3')).toBe(3);
    expect(coerce('Equal', '3.5')).toBe(3.5);
    expect(coerce('Equal', 'true')).toBe(true);
    expect(coerce('Equal', 'null')).toBeNull();
  });

  it('leaves a bare word alone', () => {
    expect(coerce('Equal', 'London')).toBe('London');
  });

  it('keeps a numeric-looking value as a string for the string operators', () => {
    // `Contains` on a number matches nothing; coercing here would produce a filter that is
    // silently always empty.
    expect(coerce('Contains', '42')).toBe('42');
    expect(coerce('StartsWith', '2026')).toBe('2026');
  });

  it('splits In and NotIn on commas and coerces each part', () => {
    expect(coerce('In', 'alpha, beta')).toEqual(['alpha', 'beta']);
    expect(coerce('In', '1, 2, 3')).toEqual([1, 2, 3]);
    expect(coerce('In', 'a, , b')).toEqual(['a', 'b']);
  });

  it('does not turn an overflowing number into null', () => {
    // JSON.parse('1e999') is Infinity, which JSON.stringify writes as `null` — the request
    // would silently become a filter for null.
    expect(coerce('Equal', '1e999')).toBe('1e999');
  });
});

describe('parseFilter', () => {
  it('accepts a filter written by hand', () => {
    const { filter, error } = parseFilter('{"must":[{"key":"rank","operation":{"type":"In","value":[1,2]}}]}');
    expect(error).toBeNull();
    expect(filter).toEqual({ must: [{ key: 'rank', operation: { type: 'In', value: [1, 2] } }] });
  });

  it('names the SDK shorthand instead of forwarding it', () => {
    const { error } = parseFilter('{"rank": {"$gte": 3}}');
    expect(error).toContain('Mongo-style');
    expect(error).toContain('GreaterThanOrEqual');
  });

  it('catches a lowercase operation name, which the server would reject as a variant', () => {
    const { error } = parseFilter('{"must":[{"key":"a","operation":{"type":"equal","value":1}}]}');
    expect(error).toContain('PascalCase');
  });

  it('catches a misspelled group', () => {
    expect(parseFilter('{"mustnot":[]}').error).toContain('mustnot');
  });

  it('reports malformed JSON rather than throwing', () => {
    expect(parseFilter('{not json').error).toBeTruthy();
    expect(parseFilter('[1,2]').error).toContain('must, must_not');
  });

  it('treats an empty box as no filter', () => {
    expect(parseFilter('   ')).toEqual({ filter: null, error: null });
  });
});
