/**
 * The filter editor's model, and the translation to what the server actually accepts.
 *
 * The server's shape is `{must, must_not, should}`, each a list of
 * `{key, operation: {type, value}}` with `type` in PascalCase. That is not what anyone types
 * from memory, and it is not what the Python SDK accepts either — the SDK offers Mongo-style
 * `{"rank": {"$gte": 3}}` as client-side sugar and translates it. Someone who has used the SDK
 * will paste that here and get a confusing rejection from the server, so this detects it and
 * says what the right shape is.
 *
 * Empty groups are omitted rather than sent as `[]`: the two are not the same request, and
 * sending an empty list where the server expects absence is the sort of thing that works until
 * it does not.
 */

import type { Condition, Filter, FilterOperationType } from '../api/types';

export const OPERATORS: FilterOperationType[] = [
  'Equal',
  'NotEqual',
  'GreaterThan',
  'GreaterThanOrEqual',
  'LessThan',
  'LessThanOrEqual',
  'In',
  'NotIn',
  'Contains',
  'StartsWith',
  'EndsWith',
];

/** Operators whose value is a list rather than a scalar. */
const LIST_OPERATORS = new Set<FilterOperationType>(['In', 'NotIn']);

/** Operators that only make sense on a string. */
const STRING_OPERATORS = new Set<FilterOperationType>(['Contains', 'StartsWith', 'EndsWith']);

export type Group = 'must' | 'must_not' | 'should';
export const GROUPS: Group[] = ['must', 'must_not', 'should'];

export interface Row {
  id: number;
  group: Group;
  key: string;
  operator: FilterOperationType;
  /** Always the raw text the user typed; coercion happens on the way out. */
  value: string;
}

let nextId = 1;
export const newRow = (group: Group = 'must'): Row => ({
  id: nextId++,
  group,
  key: '',
  operator: 'Equal',
  value: '',
});

/**
 * Coerce a typed string to what the operator wants.
 *
 * `Contains`/`StartsWith`/`EndsWith` stay strings even when they look numeric — they are string
 * operations, and turning "42" into a number there produces a filter the server will not match
 * anything with. Everything else parses as JSON when it can, so `3`, `true` and `null` mean
 * what they look like, and falls back to the literal text when it cannot.
 */
export function coerce(operator: FilterOperationType, text: string): unknown {
  if (LIST_OPERATORS.has(operator)) {
    return text
      .split(',')
      .map((part) => part.trim())
      .filter((part) => part.length > 0)
      .map((part) => scalar(part));
  }
  if (STRING_OPERATORS.has(operator)) return text;
  return scalar(text);
}

function scalar(text: string): unknown {
  const trimmed = text.trim();
  if (trimmed === '') return '';
  try {
    const parsed: unknown = JSON.parse(trimmed);
    // A bare word is not JSON, but `NaN`-ish input like `1e999` parses to Infinity, which
    // serialises back as `null` and would silently change the request.
    if (typeof parsed === 'number' && !Number.isFinite(parsed)) return trimmed;
    return parsed;
  } catch {
    return trimmed;
  }
}

/** The rows as the server wants them, or null when nothing is filled in. */
export function toFilter(rows: Row[]): Filter | null {
  const filter: Filter = {};
  for (const group of GROUPS) {
    const conditions: Condition[] = rows
      .filter((row) => row.group === group && row.key.trim() !== '')
      .map((row) => ({
        key: row.key.trim(),
        operation: { type: row.operator, value: coerce(row.operator, row.value) },
      }));
    // Omitted, not `[]`.
    if (conditions.length > 0) filter[group] = conditions;
  }
  return Object.keys(filter).length === 0 ? null : filter;
}

export interface ParseResult {
  filter: Filter | null;
  error: string | null;
}

/** Validate a hand-written filter, with a specific message for the mistake people make. */
export function parseFilter(text: string): ParseResult {
  const trimmed = text.trim();
  if (trimmed === '') return { filter: null, error: null };

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (cause) {
    return { filter: null, error: cause instanceof Error ? cause.message : String(cause) };
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return { filter: null, error: 'a filter is an object with must, must_not and/or should' };
  }

  const value = parsed as Record<string, unknown>;
  const mongo = Object.values(value).find(
    (v) =>
      typeof v === 'object' &&
      v !== null &&
      Object.keys(v as object).some((k) => k.startsWith('$')),
  );
  if (mongo) {
    return {
      filter: null,
      error:
        'this is the SDK\'s Mongo-style shorthand, which the client translates and the server ' +
        'does not accept. Use {"must": [{"key": "rank", "operation": {"type": ' +
        '"GreaterThanOrEqual", "value": 3}}]}',
    };
  }

  const unknown = Object.keys(value).filter((key) => !GROUPS.includes(key as Group));
  if (unknown.length > 0) {
    return {
      filter: null,
      error: `unknown group${unknown.length > 1 ? 's' : ''} ${unknown.join(', ')} — a filter has must, must_not and should`,
    };
  }

  for (const group of GROUPS) {
    const conditions = value[group];
    if (conditions === undefined) continue;
    if (!Array.isArray(conditions)) return { filter: null, error: `${group} must be a list` };
    for (const condition of conditions) {
      const c = condition as Partial<Condition>;
      if (typeof c?.key !== 'string' || typeof c?.operation?.type !== 'string') {
        return { filter: null, error: `each condition is {key, operation: {type, value}}` };
      }
      if (!OPERATORS.includes(c.operation.type)) {
        return {
          filter: null,
          error: `unknown operation ${c.operation.type} — the names are PascalCase: ${OPERATORS.join(', ')}`,
        };
      }
    }
  }
  return { filter: value as Filter, error: null };
}
