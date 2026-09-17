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
 */
export function RecordPicker({
  table,
  onVector,
  busy,
}: {
  table: string;
  onVector: (vector: number[], fromId: string) => void;
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
      normalizeRecord(table, response.value);
      const vector = vectorFor(table, wanted);
      if (!vector || vector.length === 0) {
        setError(`${wanted} has no vector`);
        return;
      }
      onVector(Array.from(vector), wanted);
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
        {loading ? 'loading…' : 'use its vector'}
      </button>
      {error ? <span className="err">{error}</span> : null}
    </div>
  );
}
