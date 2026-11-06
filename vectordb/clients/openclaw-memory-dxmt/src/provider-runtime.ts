/**
 * The host-facing memory provider: what `openclaw memory status`, Gateway search and other
 * plugins reach through `api.registerMemoryCapability({ providerRuntime })`. Reads are gated
 * the way the bundled memory plugin gates them: an operator may read, a session only under an
 * owner-private audience for the same agent, host operations never.
 */
import type { MemoryPluginCapability } from "openclaw/plugin-sdk/memory-host-core";
import { normalizeAgentId } from "openclaw/plugin-sdk/routing";
import type { MemoryConfig } from "./config.js";
import { formatRecalledMemoryForModel, normalizeRecallQuery, truncateText } from "./policy.js";
import type { MemoryStore } from "./store.js";

export type MemoryProviderRuntime = NonNullable<MemoryPluginCapability["providerRuntime"]>;
type OpenParams = Parameters<MemoryProviderRuntime["open"]>[0];
type ProviderHandle = NonNullable<Awaited<ReturnType<MemoryProviderRuntime["open"]>>["provider"]>;
type CallerAuthority = OpenParams["context"]["authority"];
type SearchHit = Awaited<ReturnType<ProviderHandle["search"]>>["hits"][number];
type Health = Awaited<ReturnType<ProviderHandle["health"]>>;

const DEFAULT_RESULTS = 5;
const MAX_RESULTS = 50;

export type ProviderDeps = {
  providerId: string;
  store: MemoryStore;
  resolveConfig(): MemoryConfig;
  /** The normalized agent id while memory is enabled for it, else undefined. */
  resolveEnabledAgentId(rawAgentId: string | undefined): string | undefined;
  logger: { warn?: (message: string) => void };
};

function resolveReadDenial(authority: CallerAuthority, agentId: string): string | undefined {
  if (authority.kind === "operator") {
    return undefined;
  }
  if (authority.kind === "host") {
    return "host operations cannot read stored memories";
  }
  const audience = authority.audience;
  if (audience?.kind !== "owner-private") {
    return "this session has no owner-private memory audience";
  }
  if (normalizeAgentId(audience.agentId) !== agentId) {
    return "the memory audience belongs to another agent";
  }
  return undefined;
}

function sliceLines(text: string, from: number | undefined, lines: number | undefined) {
  const all = text.split("\n");
  const start = Math.max(1, Math.floor(from ?? 1));
  const count = lines === undefined ? all.length : Math.max(1, Math.floor(lines));
  const selected = all.slice(start - 1, start - 1 + count);
  const end = start - 1 + selected.length;
  const truncated = end < all.length;
  return {
    text: selected.join("\n"),
    from: start,
    lines: selected.length,
    truncated,
    ...(truncated ? { nextFrom: end + 1 } : {}),
  };
}

export function createDxmtMemoryProviderRuntime(deps: ProviderDeps): MemoryProviderRuntime {
  return {
    async open(params: OpenParams) {
      const agentId = deps.resolveEnabledAgentId(params.agentId);
      if (!agentId) {
        return { provider: null, error: "Memory is disabled for this agent." };
      }
      const { context } = params;
      const readDenial = resolveReadDenial(context.authority, agentId);
      const assertCurrent = () => {
        context.assertCurrent();
        context.signal?.throwIfAborted();
        if (!deps.resolveEnabledAgentId(agentId)) {
          throw new Error("Memory is disabled for this agent.");
        }
      };
      const assertReadAllowed = () => {
        if (readDenial) {
          throw new Error(`${deps.providerId}: memory access denied: ${readDenial}`);
        }
        assertCurrent();
      };
      const reference = (id: string) => ({ providerId: deps.providerId, id });

      const provider: ProviderHandle = {
        capabilities: { sources: ["memory"], pagination: false, candidates: [], projectFilter: false },
        async search(request) {
          if (readDenial) {
            assertCurrent();
            return { hits: [], warning: `${deps.providerId}: stored memories were not searched: ${readDenial}` };
          }
          assertCurrent();
          const cfg = deps.resolveConfig();
          const query = normalizeRecallQuery(request.query, cfg.recallMaxChars);
          if (!query) {
            return { hits: [], coverage: "complete" };
          }
          const maxResults = Math.min(MAX_RESULTS, Math.max(1, Math.floor(request.maxResults ?? DEFAULT_RESULTS)));
          const minScore = request.minScore ?? cfg.minScore;
          const results = await deps.store.search(agentId, query, maxResults, minScore);
          assertCurrent();
          const hits = results.map((m): SearchHit => {
            const ref = reference(m.id);
            return {
              reference: ref,
              excerpt: truncateText(m.text, cfg.recallMaxChars),
              score: m.score,
              source: "memory",
              citations: [{ label: `${deps.providerId}:${m.id}`, reference: ref }],
            };
          });
          return { hits, coverage: "complete" };
        },
        async get(request) {
          assertReadAllowed();
          const { id, revision } = request.reference;
          if (revision !== undefined || !id) {
            return { status: "not_found" };
          }
          const memory = await deps.store.get(agentId, id);
          assertCurrent();
          if (!memory) {
            return { status: "not_found" };
          }
          const ref = reference(id);
          return {
            status: "ok",
            reference: ref,
            citations: [{ label: `${deps.providerId}:${id}`, reference: ref }],
            ...sliceLines(memory.text, request.from, request.lines),
          };
        },
        async health(): Promise<Health> {
          assertCurrent();
          const cfg = deps.resolveConfig();
          // Health reaches every authenticated caller: the server's address, never memory text.
          const details: Record<string, unknown> = {
            backend: "dxmt",
            url: cfg.url,
            scope: cfg.scope,
            autoCapture: cfg.autoCapture,
            autoRecall: cfg.autoRecall,
          };
          try {
            const health = await deps.store.health();
            assertCurrent();
            details.server = health.status;
            if (!health.ok) {
              return { status: "degraded", message: `The DXMT server reports ${health.status}.`, details };
            }
          } catch (error) {
            assertCurrent();
            deps.logger.warn?.(
              `${deps.providerId}: health could not reach the server: ${error instanceof Error ? error.message : String(error)}`,
            );
            return {
              status: "unavailable",
              message: `The DXMT server at ${cfg.url} did not answer. Start it (vectordb serve, or docker run -p 127.0.0.1:9999:9999 -v dxmt:/data dxmtai/dxmt) and retry.`,
              details,
            };
          }
          return { status: "ready", details };
        },
        async close() {},
      };
      return { provider };
    },
  };
}

export { formatRecalledMemoryForModel };
