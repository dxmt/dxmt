/**
 * The DXMT side: one SDK client, agent scope as tags, memories as the server's memory API.
 *
 * Every memory the plugin writes carries the tags `openclaw` and `openclaw:agent:<agentId>`;
 * category is the memory's `kind`, importance and origin ride in metadata (returned flattened
 * on every hit). With `scope: "agent"` recall filters on the agent tag; with `scope: "shared"`
 * it does not, and the agent sees what every other DXMT client on the machine wrote.
 *
 * The server embeds the text and deduplicates ("saying what is already remembered refreshes
 * it"), so no vector and no fingerprint lives on this side.
 */
import { DXMTClient } from "dxmt-js-sdk";
import { MEMORY_CATEGORIES, type MemoryCategory, type MemoryConfig, parseServerUrl } from "./config.js";
import type { RecalledMemory } from "./policy.js";

export const WRITER_TAG = "openclaw";
export const agentTag = (agentId: string) => `openclaw:agent:${agentId}`;

/** What the store needs from a client: the SDK's memory and health methods. Tests hand in a fake. */
export type MemoryClient = Pick<DXMTClient, "remember" | "recall" | "forget" | "listMemories" | "healthCheck">;

export function createDxmtClient(cfg: Pick<MemoryConfig, "url" | "apiKey">): DXMTClient {
  const { host, port, ssl } = parseServerUrl(cfg.url);
  return new DXMTClient({ host, port, ssl, apiKey: cfg.apiKey, timeout: 15_000 });
}

export type StoreParams = {
  text: string;
  category: MemoryCategory;
  importance: number;
  origin: "tool" | "auto_capture";
  supersedes?: string[];
  sessionKey?: string;
};

export type StoreResult = { id: string; duplicate: boolean; superseded: string[] };

const LIST_PAGE = 50;
const MAX_LIST_PAGES = 40;

function asCategory(kind: unknown): MemoryCategory {
  return (MEMORY_CATEGORIES as readonly string[]).includes(String(kind)) ? (kind as MemoryCategory) : "other";
}

function asImportance(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0.7;
}

export function toRecalled(row: Record<string, unknown>): RecalledMemory {
  return {
    id: String(row.id),
    text: String(row.text ?? ""),
    category: asCategory(row.kind),
    importance: asImportance(row.importance),
    score: typeof row.score === "number" ? row.score : 0,
    at: typeof row.at === "number" ? row.at : undefined,
  };
}

export class MemoryStore {
  constructor(
    readonly client: MemoryClient,
    private readonly scope: MemoryConfig["scope"],
  ) {}

  private recallTags(agentId: string): string[] | undefined {
    return this.scope === "agent" ? [agentTag(agentId)] : undefined;
  }

  async store(agentId: string, params: StoreParams): Promise<StoreResult> {
    const metadata: Record<string, unknown> = {
      importance: params.importance,
      origin: params.origin,
      agent_id: agentId,
    };
    if (params.sessionKey) {
      metadata.session_key = params.sessionKey;
    }
    const answer = await this.client.remember({
      text: params.text,
      kind: params.category,
      tags: [WRITER_TAG, agentTag(agentId)],
      source: `openclaw:${agentId}`,
      supersedes: params.supersedes?.length ? params.supersedes : undefined,
      metadata,
    });
    return { id: answer.id, duplicate: answer.duplicate === true, superseded: answer.superseded ?? [] };
  }

  /** Best matches first, scored by the server (cosine similarity, 1.0 identical), floored at `minScore`. */
  async search(agentId: string, query: string, limit: number, minScore: number): Promise<RecalledMemory[]> {
    const rows = await this.client.recall({
      query,
      limit: Math.min(100, Math.max(1, limit)),
      tags: this.recallTags(agentId),
    });
    return rows
      .map((row) => toRecalled(row as Record<string, unknown>))
      .filter((m) => m.text.trim().length > 0 && m.score >= minScore)
      .slice(0, limit);
  }

  async delete(agentId: string, id: string): Promise<boolean> {
    const existing = await this.get(agentId, id);
    if (!existing) {
      return false;
    }
    return (await this.client.forget(id)) > 0;
  }

  /** One memory by id, within this agent's scope; pages the list endpoint (no GET by id on the server). */
  async get(agentId: string, id: string): Promise<RecalledMemory | undefined> {
    let offset = 0;
    for (let page = 0; page < MAX_LIST_PAGES; page++) {
      const result = await this.client.listMemories({ limit: LIST_PAGE, offset });
      for (const row of result.memories) {
        if (row.id === id) {
          const tags = Array.isArray(row.tags) ? (row.tags as string[]) : [];
          if (this.scope === "agent" && !tags.includes(agentTag(agentId))) {
            return undefined;
          }
          return toRecalled(row as Record<string, unknown>);
        }
      }
      offset += result.memories.length;
      if (result.memories.length === 0 || offset >= result.total) {
        return undefined;
      }
    }
    return undefined;
  }

  async health(): Promise<{ ok: boolean; status: string }> {
    const answer = (await this.client.healthCheck()) as unknown as Record<string, unknown>;
    const data = (answer.data ?? answer) as Record<string, unknown>;
    const status = String(data.status ?? answer.status ?? "unknown");
    const ok = data.ready === true || status.toLowerCase() === "healthy" || status.toLowerCase() === "ok";
    return { ok, status };
  }
}
