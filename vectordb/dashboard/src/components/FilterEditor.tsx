import { useState } from 'react';
import { GROUPS, OPERATORS, newRow, parseFilter, toFilter, type Group, type Row } from '../lib/filter';
import type { Filter } from '../api/types';
import { JsonBlock } from './JsonBlock';

/**
 * Build a filter without having to remember the shape, and see the shape while you build it.
 *
 * The emitted JSON is always on screen beside the editor. That is not decoration: this is the
 * only place in the product where someone learns what a DXMT filter looks like, and the API
 * documentation shows one example. A raw-JSON mode toggles in the other direction for anyone
 * who already knows, or has one to paste.
 */
export function FilterEditor({
  keys,
  onApply,
  busy,
}: {
  /** Payload keys seen in the current page, offered as completions. */
  keys: string[];
  onApply: (filter: Filter | null) => void;
  busy: boolean;
}) {
  const [rows, setRows] = useState<Row[]>([]);
  const [rawMode, setRawMode] = useState(false);
  const [raw, setRaw] = useState('');

  const built = rawMode ? parseFilter(raw) : { filter: toFilter(rows), error: null };

  const update = (id: number, patch: Partial<Row>) =>
    setRows((current) => current.map((row) => (row.id === id ? { ...row, ...patch } : row)));

  return (
    <div className="filter">
      <div className="filter-head">
        <strong>Filter</strong>
        <button type="button" className="ghost" onClick={() => setRawMode((on) => !on)}>
          {rawMode ? 'use the editor' : 'write JSON'}
        </button>
      </div>

      {rawMode ? (
        <textarea
          className="mono"
          rows={6}
          spellCheck={false}
          value={raw}
          placeholder='{"must": [{"key": "rank", "operation": {"type": "GreaterThanOrEqual", "value": 3}}]}'
          onChange={(event) => setRaw(event.target.value)}
        />
      ) : (
        <>
          {rows.map((row) => (
            <div className="filter-row" key={row.id}>
              <select
                value={row.group}
                onChange={(event) => update(row.id, { group: event.target.value as Group })}
              >
                {GROUPS.map((group) => (
                  <option key={group} value={group}>
                    {group}
                  </option>
                ))}
              </select>
              <input
                list="payload-keys"
                placeholder="payload key"
                value={row.key}
                onChange={(event) => update(row.id, { key: event.target.value })}
              />
              <select
                value={row.operator}
                onChange={(event) =>
                  update(row.id, { operator: event.target.value as Row['operator'] })
                }
              >
                {OPERATORS.map((operator) => (
                  <option key={operator} value={operator}>
                    {operator}
                  </option>
                ))}
              </select>
              <input
                placeholder={row.operator === 'In' || row.operator === 'NotIn' ? 'a, b, c' : 'value'}
                value={row.value}
                onChange={(event) => update(row.id, { value: event.target.value })}
              />
              <button
                type="button"
                className="ghost"
                onClick={() => setRows((current) => current.filter((r) => r.id !== row.id))}
              >
                ✕
              </button>
            </div>
          ))}
          <datalist id="payload-keys">
            {keys.map((key) => (
              <option key={key} value={key} />
            ))}
          </datalist>
          <button type="button" className="ghost" onClick={() => setRows((c) => [...c, newRow()])}>
            + condition
          </button>
        </>
      )}

      {built.error ? <p className="err">{built.error}</p> : null}

      {built.filter ? (
        <>
          <p className="muted" style={{ margin: '10px 0 4px' }}>sent as</p>
          <JsonBlock value={built.filter} />
        </>
      ) : null}

      <div className="filter-actions">
        <button
          type="button"
          disabled={busy || built.error !== null}
          onClick={() => onApply(built.filter)}
        >
          Apply
        </button>
        <button
          type="button"
          className="ghost"
          disabled={busy}
          onClick={() => {
            setRows([]);
            setRaw('');
            onApply(null);
          }}
        >
          Clear
        </button>
      </div>
    </div>
  );
}
