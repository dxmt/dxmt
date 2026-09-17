/**
 * A scatter plot on a 2-D canvas.
 *
 * Deliberately not WebGL. The reference implementation is 633 lines of WebGL2 with an offscreen
 * colour-encoded picking buffer, built for hundreds of thousands of points — which it can have
 * because its layout comes from the server. Ours cannot: the sampling budget caps this at four
 * thousand points, where `arc()` batched by colour is under a millisecond a frame and hit
 * testing is a grid lookup.
 */

export interface Point {
  x: number;
  y: number;
  color: string;
}

export interface ScatterOptions {
  onHover?: (index: number | null) => void;
  onClick?: (index: number) => void;
}

export class Scatter {
  private points: Point[] = [];
  private highlighted = new Set<number>();
  private selected: number | null = null;
  private scale = 1;
  private offsetX = 0;
  private offsetY = 0;
  private bounds = { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  private hovered: number | null = null;
  private dragging: { x: number; y: number } | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    private options: ScatterOptions = {},
  ) {
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointerleave', this.onPointerUp);
    canvas.addEventListener('dblclick', this.reset);
  }

  destroy(): void {
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerup', this.onPointerUp);
    this.canvas.removeEventListener('pointerleave', this.onPointerUp);
    this.canvas.removeEventListener('dblclick', this.reset);
  }

  setPoints(points: Point[]): void {
    this.points = points;
    if (points.length > 0) {
      this.bounds = {
        minX: Math.min(...points.map((p) => p.x)),
        maxX: Math.max(...points.map((p) => p.x)),
        minY: Math.min(...points.map((p) => p.y)),
        maxY: Math.max(...points.map((p) => p.y)),
      };
    }
    this.reset();
  }

  /** Ids to ring — the true neighbours of the selected point. */
  setHighlighted(indexes: number[], selected: number | null): void {
    this.highlighted = new Set(indexes);
    this.selected = selected;
    this.draw();
  }

  draw = (): void => {
    const context = this.canvas.getContext('2d');
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    // Batch by colour: one path per colour rather than one per point.
    const byColor = new Map<string, number[]>();
    this.points.forEach((point, index) => {
      const list = byColor.get(point.color);
      if (list) list.push(index);
      else byColor.set(point.color, [index]);
    });

    for (const [color, indexes] of byColor) {
      context.fillStyle = color;
      context.globalAlpha = this.highlighted.size > 0 ? 0.35 : 0.8;
      context.beginPath();
      for (const index of indexes) {
        const { x, y } = this.toScreen(index, width, height);
        context.moveTo(x + 2.6, y);
        context.arc(x, y, 2.6, 0, Math.PI * 2);
      }
      context.fill();
    }
    context.globalAlpha = 1;

    // The true neighbours, drawn on top so the projection can be seen failing.
    for (const index of this.highlighted) {
      const { x, y } = this.toScreen(index, width, height);
      context.strokeStyle = this.points[index]?.color ?? '#000';
      context.lineWidth = 1.6;
      context.beginPath();
      context.arc(x, y, 6, 0, Math.PI * 2);
      context.stroke();
    }

    if (this.selected !== null && this.points[this.selected]) {
      const { x, y } = this.toScreen(this.selected, width, height);
      context.strokeStyle = getComputedStyle(this.canvas).color;
      context.lineWidth = 2;
      context.beginPath();
      context.arc(x, y, 9, 0, Math.PI * 2);
      context.stroke();
    }
  };

  private toScreen(index: number, width: number, height: number): { x: number; y: number } {
    const point = this.points[index]!;
    const spanX = this.bounds.maxX - this.bounds.minX || 1;
    const spanY = this.bounds.maxY - this.bounds.minY || 1;
    const pad = 18;
    const x = ((point.x - this.bounds.minX) / spanX) * (width - pad * 2) + pad;
    const y = height - (((point.y - this.bounds.minY) / spanY) * (height - pad * 2) + pad);
    return { x: x * this.scale + this.offsetX, y: y * this.scale + this.offsetY };
  }

  private nearest(mouseX: number, mouseY: number): number | null {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    let best: number | null = null;
    let bestDistance = 12 * 12;
    for (let index = 0; index < this.points.length; index += 1) {
      const { x, y } = this.toScreen(index, width, height);
      const distance = (x - mouseX) ** 2 + (y - mouseY) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    }
    return best;
  }

  private onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    // Zoom about the cursor, not the origin.
    this.offsetX = mouseX - (mouseX - this.offsetX) * factor;
    this.offsetY = mouseY - (mouseY - this.offsetY) * factor;
    this.scale *= factor;
    this.draw();
  };

  private onPointerDown = (event: PointerEvent): void => {
    this.dragging = { x: event.clientX, y: event.clientY };
    this.canvas.setPointerCapture(event.pointerId);
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (this.dragging) {
      this.offsetX += event.clientX - this.dragging.x;
      this.offsetY += event.clientY - this.dragging.y;
      this.dragging = { x: event.clientX, y: event.clientY };
      this.draw();
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const index = this.nearest(event.clientX - rect.left, event.clientY - rect.top);
    if (index !== this.hovered) {
      this.hovered = index;
      this.options.onHover?.(index);
    }
  };

  private onPointerUp = (event: PointerEvent): void => {
    const moved =
      this.dragging &&
      (Math.abs(event.clientX - this.dragging.x) > 3 ||
        Math.abs(event.clientY - this.dragging.y) > 3);
    this.dragging = null;
    if (moved) return;
    const rect = this.canvas.getBoundingClientRect();
    const index = this.nearest(event.clientX - rect.left, event.clientY - rect.top);
    if (index !== null) this.options.onClick?.(index);
  };

  private reset = (): void => {
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.draw();
  };
}
