/**
 * Reconciling the two record shapes, in one place, on purpose.
 *
 * `records/list` and `GET records/{r}` send the payload as a JSON **string** under
 * `payload_json`; search hits send an **object** under `payload`. Both describe the same thing.
 * Every component downstream takes the normalised form, so the asymmetry is handled once rather
 * than in each table cell -- but `payloadRaw` keeps the original string, because when the parse
 * fails, the raw text is the bug report and silently dropping it would hide exactly the kind of
 * problem a console exists to surface.
 *
 * # The vectors do not go into React state
 *
 * `records/list` always sends the full vector and offers no way to suppress it, so a page of
 * 100 rows at 768 dimensions is over a megabyte of JSON holding 76,800 numbers. As boxed
 * JavaScript numbers inside component state, React diffs all of them on every keystroke
 * elsewhere on the page. So the floats are converted to `Float32Array` here and parked in a
 * module-level map keyed by `table:id`; what reaches the components is an id, a dimension, a
 * norm and a six-value preview. The visualize page then reads the same map with no refetch,
 * which is the other half of why the map exists.
 */

import type { WireHit, WireRecord } from './types';

export interface NormalRecord {
  id: string;
  dim: number;
  /** Euclidean norm, for the vector cell's one-line summary. */
  norm: number;
  /** The first few components, so a table can show something without holding the whole vector. */
  preview: number[];
  payload: Record<string, unknown> | null;
  /** The unparsed payload, kept only when parsing it failed. */
  payloadRaw: string | null;
  payloadError: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  version: number | null;
  status: string | null;
  /** Present on search hits, absent on listed records. */
  score: number | null;
}

const PREVIEW = 6;

/** `table:id` → the full vector, outside React's reach. */
const vectors = new Map<string, Float32Array>();

function remember(table: string, id: string, vector: number[] | null): Float32Array | null {
  if (!vector || vector.length === 0) return null;
  const typed = Float32Array.from(vector);
  vectors.set(`${table}:${id}`, typed);
  return typed;
}

/** The full vector for a record the app has already loaded, or undefined if it has not. */
export function vectorFor(table: string, id: string): Float32Array | undefined {
  return vectors.get(`${table}:${id}`);
}

/** Every vector currently held for a table, in the order the ids are given. */
export function vectorsFor(table: string, ids: string[]): (Float32Array | undefined)[] {
  return ids.map((id) => vectors.get(`${table}:${id}`));
}

/** Drop a table's vectors. Called when a table's records are refetched from scratch. */
export function forgetVectors(table: string): void {
  for (const key of vectors.keys()) {
    if (key.startsWith(`${table}:`)) vectors.delete(key);
  }
}

function norm(v: Float32Array | null): number {
  if (!v) return 0;
  let sum = 0;
  for (const x of v) sum += x * x;
  return Math.sqrt(sum);
}

export function normalizeRecord(table: string, wire: WireRecord): NormalRecord {
  const typed = remember(table, wire.id, wire.vector);

  let payload: Record<string, unknown> | null = null;
  let payloadRaw: string | null = null;
  let payloadError: string | null = null;
  if (wire.payload_json != null) {
    try {
      const parsed: unknown = JSON.parse(wire.payload_json);
      payload = typeof parsed === 'object' && parsed !== null
        ? (parsed as Record<string, unknown>)
        : { value: parsed };
    } catch (cause) {
      // serde wrote this string, so it should always parse. If it ever does not, showing the
      // raw text beats a blank cell and beats a white screen.
      payloadRaw = wire.payload_json;
      payloadError = cause instanceof Error ? cause.message : String(cause);
    }
  }

  return {
    id: wire.id,
    dim: typed?.length ?? 0,
    norm: norm(typed),
    preview: typed ? Array.from(typed.slice(0, PREVIEW)) : [],
    payload,
    payloadRaw,
    payloadError,
    createdAt: wire.created_at ?? null,
    updatedAt: wire.updated_at ?? null,
    version: wire.version ?? null,
    status: wire.status ?? null,
    score: null,
  };
}

export function normalizeHit(table: string, wire: WireHit): NormalRecord {
  // `with_vector` defaults to false, so a hit usually has no vector -- but when it has one it
  // is worth keeping for the same reasons a listed record's is.
  const typed = remember(table, wire.id, wire.vector);
  return {
    id: wire.id,
    dim: typed?.length ?? 0,
    norm: norm(typed),
    preview: typed ? Array.from(typed.slice(0, PREVIEW)) : [],
    payload: wire.payload,
    payloadRaw: null,
    payloadError: null,
    createdAt: null,
    updatedAt: null,
    version: null,
    status: null,
    score: wire.score,
  };
}
