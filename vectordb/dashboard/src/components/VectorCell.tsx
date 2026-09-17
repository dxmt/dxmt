import { useEffect, useRef } from 'react';
import { vectorFor } from '../api/normalize';

/**
 * A vector in a table cell, without putting a vector in a table cell.
 *
 * 768 numbers is not a cell, and rendering them is not reading them. What is actually useful at
 * a glance is the shape and the magnitude: a sparkline of the first components, the dimension,
 * and the norm — a norm of 1.0 says the vectors are normalised, a norm of 0 says the record has
 * none, and either is the thing you were looking for.
 *
 * The floats come from the module-level map in `normalize.ts` rather than from props, so React
 * never holds or diffs them.
 */
const WIDTH = 72;
const HEIGHT = 18;
const SHOWN = 96;

export function VectorCell({ table, id, dim, norm }: { table: string; id: string; dim: number; norm: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const element = canvas.current;
    const vector = vectorFor(table, id);
    if (!element || !vector || vector.length === 0) return;

    const scale = window.devicePixelRatio || 1;
    element.width = WIDTH * scale;
    element.height = HEIGHT * scale;
    const context = element.getContext('2d');
    if (!context) return;
    context.scale(scale, scale);
    context.clearRect(0, 0, WIDTH, HEIGHT);

    const shown = Math.min(SHOWN, vector.length);
    let low = Infinity;
    let high = -Infinity;
    for (let i = 0; i < shown; i += 1) {
      const v = vector[i]!;
      if (v < low) low = v;
      if (v > high) high = v;
    }
    const span = high - low || 1;

    context.strokeStyle = getComputedStyle(element).color;
    context.lineWidth = 1;
    context.beginPath();
    for (let i = 0; i < shown; i += 1) {
      const x = (i / Math.max(shown - 1, 1)) * (WIDTH - 1);
      const y = HEIGHT - 1 - ((vector[i]! - low) / span) * (HEIGHT - 2);
      if (i === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
  }, [table, id, dim]);

  if (dim === 0) {
    return (
      <span className="muted" title="Ask for vectors with the toggle above; they are omitted by default.">
        —
      </span>
    );
  }

  return (
    <span className="veccell">
      <canvas ref={canvas} style={{ width: WIDTH, height: HEIGHT }} />
      <span className="muted mono">
        {dim}d · ‖v‖&nbsp;{norm.toFixed(3)}
      </span>
    </span>
  );
}
