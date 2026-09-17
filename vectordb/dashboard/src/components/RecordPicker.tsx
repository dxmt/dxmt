import { useState } from 'react';
import { api } from '../api/client';
import { normalizeRecord, vectorFor } from '../api/normalize';
import { ApiFailure } from '../api/envelope';

/**
 * Get a query vector without typing 768 numbers.
 *
 * Almost every real question starts from a record that is already in the table — "what does the
 * index think is near *this*". `GET /records/{id}` reads through to the row file, so it answers
 * with the vector whatever the storage mode, which makes it the one dependable source.
 *
 * It hands back the record's **text** as well, and that is not a convenience. Comparing dense
 * retrieval against BM25 needs the same question asked both ways, and there is no embedding
 * model on this server to turn a typed sentence into a vector — `src/embeddings/` reaches no
 * route. A record already has both halves, so it is the only place a matched pair comes from.
 */
export interface Picked {
  vector: number[];
  id: string;
  /** The record's indexable text, or null when nothing in the payload looks like text. */
  text: string | null;
}

/**
 * The text this record would have been indexed under.
 *
 * Mirrors what `extract_text_from_record` does on the server for a table that declares no
 * `text_fields`: `content` is the field it indexes, with `title` and `description` beside it.
 * The fallback to "the first string in the payload" is a guess, and it is only reached when the
 * conventional names are all absent — at which point a guess beats an empty box, because the
 * text is editable and the person can see what was filled in.
 */
function textOf(payload: Record<string, unknown> | null): string | null {
  if (!payload) return null;
  for (const key of ['content', 'text', 'title', 'description', 'body']) {
    const value = payload[key];
    if (typeof value === 'string' && value.trim() !== '') return value;
  }
  for (const value of Object.values(payload)) {
    if (typeof value === 'string' && value.trim() !== '') return value;
  }
  return null;
}

export function RecordPicker({
  table,
  onPick,
  busy,
}: {
  table: string;
  onPick: (picked: Picked) => void;
  busy: boolean;
}) {
  const [id, setId] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    const wanted = id.trim();
    if (wanted === '') return;
    setLoading(true);
    setError(null);
    try {
      const response = await api.record(table, wanted);
      const record = normalizeRecord(table, response.value);
      const vector = vectorFor(table, wanted);
      if (!vector || vector.length === 0) {
        setError(`${wanted} has no vector`);
        return;
      }
      onPick({ vector: Array.from(vector), id: wanted, text: textOf(record.payload) });
    } catch (cause) {
      setError(
        cause instanceof ApiFailure && cause.isNotFound
          ? `no record ${wanted} in ${table}`
          : String(cause),
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="picker">
      <input
        placeholder="record id"
        value={id}
        onChange={(event) => setId(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') void load();
        }}
      />
      <button type="button" disabled={busy || loading || id.trim() === ''} onClick={() => void load()}>
        {loading ? 'loading…' : 'use this record'}
      </button>
      {error ? <span className="err">{error}</span> : null}
    </div>
  );
}
