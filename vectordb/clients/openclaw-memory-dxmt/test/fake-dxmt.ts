import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

/**
 * An in-process stand-in for the DXMT server's memory API on a loopback port, so the real
 * `dxmt-js-sdk` client runs end to end without a server. It keeps the server's
 * shapes: the `{success, data}` envelope, dedup within a kind, tag filters that mean "any of",
 * supersede, newest-first listing, flattened metadata on every hit. Scores are word overlap.
 */
export type FakeMemory = Record<string, unknown> & {
  id: string;
  text: string;
  kind: string;
  tags: string[];
  at: number;
  superseded_by?: string;
};

export class FakeDxmt {
  memories = new Map<string, FakeMemory>();
  /** Off to stand in for a server before `GET /memories/{id}` and `similarity` (1.6). */
  getById = true;
  reportsSimilarity = true;
  requests: Array<{ method: string; path: string; body?: unknown }> = [];
  failNext: string | undefined;
  private counter = 0;
  private server: Server | undefined;
  url = "";

  async start(): Promise<string> {
    this.server = createServer((req, res) => void this.serve(req, res));
    await new Promise<void>((resolve) => this.server!.listen(0, "127.0.0.1", resolve));
    const address = this.server.address();
    const port = typeof address === "object" && address ? address.port : 0;
    this.url = `http://127.0.0.1:${port}`;
    return this.url;
  }

  async stop(): Promise<void> {
    await new Promise<void>((resolve) => (this.server ? this.server.close(() => resolve()) : resolve()));
  }

  private envelope(data: unknown, status = 200): { status: number; body: string } {
    return {
      status,
      body: JSON.stringify({ success: status < 400, data: status < 400 ? data : null, error: status < 400 ? null : data }),
    };
  }

  private async serve(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const raw = Buffer.concat(chunks).toString("utf8");
    const answer = this.handle(req.method ?? "GET", new URL(req.url ?? "/", this.url), raw ? JSON.parse(raw) : undefined);
    res.writeHead(answer.status, { "content-type": "application/json" });
    res.end(answer.body);
  }

  private handle(method: string, url: URL, body: Record<string, unknown> | undefined): { status: number; body: string } {
    method = method.toUpperCase();
    const path = url.pathname;
    this.requests.push({ method, path, body });
    if (this.failNext) {
      const message = this.failNext;
      this.failNext = undefined;
      return this.envelope({ code: "INTERNAL", message }, 500);
    }
    if (path === "/api/v1/health") {
      return this.envelope({ status: "Healthy", ready: true });
    }
    if (path === "/api/v1/memories" && method === "POST") {
      return this.envelope(this.remember(body ?? {}));
    }
    if (path === "/api/v1/memories/recall" && method === "POST") {
      return this.envelope({ memories: this.recall(body ?? {}) });
    }
    if (path === "/api/v1/memories" && method === "GET") {
      const limit = Number(url.searchParams.get("limit") ?? 50);
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const kind = url.searchParams.get("kind");
      const rows = [...this.memories.values()].filter((m) => !kind || m.kind === kind).sort((a, b) => b.at - a.at);
      return this.envelope({ memories: rows.slice(offset, offset + limit), total: rows.length });
    }
    const forget = path.match(/^\/api\/v1\/memories\/([^/]+)$/);
    if (forget && method === "GET" && this.getById) {
      const row = this.memories.get(decodeURIComponent(forget[1]!));
      return row ? this.envelope(row) : this.envelope({ code: "NOT_FOUND", message: "no such memory" }, 404);
    }
    if (forget && method === "GET") {
      // A server older than `GET /memories/{id}`: the path exists for DELETE only.
      return this.envelope({ code: "METHOD_NOT_ALLOWED", message: "GET" }, 405);
    }
    if (forget && method === "DELETE") {
      return this.envelope({ forgotten: this.memories.delete(decodeURIComponent(forget[1]!)) ? 1 : 0 });
    }
    return this.envelope({ code: "NOT_FOUND", message: `no route ${method} ${path}` }, 404);
  }

  private remember(body: Record<string, unknown>) {
    const text = String(body.text);
    const kind = String(body.kind ?? "note");
    for (const m of this.memories.values()) {
      if (m.text === text && m.kind === kind && !m.superseded_by && !body.id) {
        return { id: m.id, superseded: [], duplicate: true };
      }
    }
    const id = String(body.id ?? `mem-${String(++this.counter).padStart(4, "0")}`);
    const row: FakeMemory = {
      ...((body.metadata as Record<string, unknown>) ?? {}),
      id,
      text,
      kind,
      tags: Array.isArray(body.tags) ? (body.tags as string[]) : [],
      source: body.source,
      at: 1_700_000_000 + this.counter,
    };
    this.memories.set(id, row);
    const superseded: string[] = [];
    for (const old of (body.supersedes as string[] | undefined) ?? []) {
      const target = this.memories.get(old);
      if (target) {
        target.superseded_by = id;
        superseded.push(old);
      }
    }
    return { id, superseded };
  }

  private recall(body: Record<string, unknown>) {
    const words = new Set(String(body.query).toLowerCase().split(/\W+/).filter(Boolean));
    const tags = body.tags as string[] | undefined;
    const kind = body.kind as string | undefined;
    const hits: Array<FakeMemory & { score: number }> = [];
    for (const m of this.memories.values()) {
      if (m.superseded_by && !body.include_superseded) continue;
      if (kind && m.kind !== kind) continue;
      if (tags && !tags.some((t) => m.tags.includes(t))) continue;
      const own = new Set(m.text.toLowerCase().split(/\W+/).filter(Boolean));
      let overlap = 0;
      for (const w of words) if (own.has(w)) overlap++;
      hits.push({ ...m, score: words.size ? overlap / words.size : 0 });
    }
    hits.sort((a, b) => b.score - a.score || b.at - a.at);
    // Like the server: `score` is a rank normalised over the page (the best hit is 1.0 however
    // poor), `similarity` the closeness itself -- unless this stands in for an older server.
    const best = hits[0]?.score || 1;
    return hits.slice(0, Number(body.limit ?? 25)).map((h) => ({
      ...h,
      score: h.score / best,
      ...(this.reportsSimilarity ? { similarity: h.score } : {}),
    }));
  }
}
