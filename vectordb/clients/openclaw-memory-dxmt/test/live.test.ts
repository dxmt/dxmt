/** The same loop against a real DXMT server with an embedding model: `DXMT_TEST_URL=http://127.0.0.1:9999`. */
import { afterAll, describe, expect, test } from "vitest";
import { createMemoryPlugin } from "../src/index.js";
import { allowRecallAuthority, createFakeApi } from "./fake-api.js";

const url = process.env.DXMT_TEST_URL;
const agentId = `live-${Math.random().toString(16).slice(2, 10)}`;
const created: string[] = [];
type ToolResult = { content: Array<{ text: string }>; details: Record<string, unknown> };

describe.skipIf(!url)("live", () => {
  const fake = createFakeApi({ url });
  createMemoryPlugin().register(fake.api);
  const tool = (name: string) => {
    const made = fake.tools.get(name)!({ agentId, sessionKey: `agent:${agentId}:main` }) as { execute(id: string, p: unknown): Promise<ToolResult> };
    return (p: unknown) => made.execute("c", p);
  };
  afterAll(async () => {
    for (const id of created) await tool("memory_forget")({ memoryId: id });
  });

  test("store, recall by meaning, supersede, provider health", async () => {
    const stored = await tool("memory_store")({ text: "The user keeps bees on a rooftop in Lisbon and sells the honey at a Saturday market.", category: "fact" });
    expect(stored.details.action).toBe("created");
    created.push(stored.details.id as string);
    expect((await tool("memory_store")({ text: "The user keeps bees on a rooftop in Lisbon and sells the honey at a Saturday market.", category: "fact" })).details.action).toBe("already_present");

    const recalled = await tool("memory_recall")({ query: "apiculture and weekend trading", limit: 3 });
    expect((recalled.details.memories as Array<{ id: string; score: number }>)[0]).toMatchObject({ id: stored.details.id });

    const hook = fake.hooks.get("before_prompt_build")!.handler as (e: unknown, c: unknown) => Promise<{ prependContext?: string } | undefined>;
    const out = await hook({ prompt: "what does the user sell at the market?", messages: [] }, { agentId, sessionKey: `agent:${agentId}:main`, toolAuthority: allowRecallAuthority });
    expect(out?.prependContext).toContain("honey");

    const moved = await tool("memory_store")({ text: "The user moved the beehives to Porto.", category: "fact", supersedes: [stored.details.id] });
    created.push(moved.details.id as string);
    expect(moved.details.superseded).toEqual([stored.details.id]);
    const ids = ((await tool("memory_recall")({ query: "where are the beehives" })).details.memories as Array<{ id: string }>).map((m) => m.id);
    expect(ids).toContain(moved.details.id);
    expect(ids).not.toContain(stored.details.id);

    const runtime = fake.capability!.providerRuntime as { open(p: unknown): Promise<{ provider: { health(): Promise<{ status: string }> } | null }> };
    const opened = await runtime.open({ cfg: {}, agentId, context: { authority: { kind: "operator", scopes: [] }, assertCurrent: () => undefined } });
    expect((await opened.provider!.health()).status).toBe("ready");
  });
});
