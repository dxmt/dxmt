/**
 * Colours for a categorical field, and a ramp for a numeric one.
 *
 * The palette is Okabe-Ito, which is designed to stay distinguishable under the common forms of
 * colour blindness — worth more here than a prettier ramp, because on this page the colour *is*
 * the data.
 *
 * The important case is the one that is easy to skip: DXMT payloads are schemaless, so a key
 * present on some records and absent on others is normal. Those must get their own visible slot
 * rather than silently sharing a colour with one of the real values.
 */
const PALETTE = [
  '#0072b2',
  '#e69f00',
  '#009e73',
  '#cc79a7',
  '#56b4e9',
  '#d55e00',
  '#8a6db1',
  '#6b8e23',
  '#b03a2e',
  '#4d4d4d',
];

export const ABSENT = '#9a9a93';
export const OTHER = '#b8b8b0';
const MAX_CATEGORIES = 11;

export interface Legend {
  kind: 'categorical' | 'numeric' | 'none';
  entries: { label: string; color: string; count: number }[];
  /** For a numeric field. */
  range?: [number, number];
}

export interface Coloring {
  legend: Legend;
  colorAt: (index: number) => string;
}

export function colorBy(values: unknown[]): Coloring {
  const present = values.filter((v) => v !== undefined && v !== null);
  const absent = values.length - present.length;

  if (present.length === 0) {
    return {
      legend: { kind: 'none', entries: [] },
      colorAt: () => PALETTE[0]!,
    };
  }

  if (present.every((v) => typeof v === 'number')) {
    const numbers = present as number[];
    const low = Math.min(...numbers);
    const high = Math.max(...numbers);
    return {
      legend: {
        kind: 'numeric',
        range: [low, high],
        entries: absent > 0 ? [{ label: '(no value)', color: ABSENT, count: absent }] : [],
      },
      colorAt: (index) => {
        const value = values[index];
        if (typeof value !== 'number') return ABSENT;
        return ramp(high === low ? 0.5 : (value - low) / (high - low));
      },
    };
  }

  const counts = new Map<string, number>();
  for (const value of present) {
    const label = String(value);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const named = ordered.slice(0, MAX_CATEGORIES - 1);
  const rest = ordered.slice(MAX_CATEGORIES - 1);

  const assigned = new Map(named.map(([label], i) => [label, PALETTE[i % PALETTE.length]!]));
  const entries = named.map(([label, count]) => ({ label, color: assigned.get(label)!, count }));
  if (rest.length > 0) {
    entries.push({
      label: `other (${rest.length})`,
      color: OTHER,
      count: rest.reduce((sum, [, n]) => sum + n, 0),
    });
  }
  if (absent > 0) entries.push({ label: '(no value)', color: ABSENT, count: absent });

  return {
    legend: { kind: 'categorical', entries },
    colorAt: (index) => {
      const value = values[index];
      if (value === undefined || value === null) return ABSENT;
      return assigned.get(String(value)) ?? OTHER;
    },
  };
}

/** A perceptually even-ish blue-to-orange ramp, in HSL. */
function ramp(t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const hue = 205 - clamped * 175;
  const light = 42 + clamped * 16;
  return `hsl(${hue.toFixed(0)}, 72%, ${light.toFixed(0)}%)`;
}
