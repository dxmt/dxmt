/** The slice of `OpenClawPluginApi` the plugin touches, recording what it registers. */
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";

type AnyFn = (...args: never[]) => unknown;

export type FakeApi = {
  api: OpenClawPluginApi;
  hooks: Map<string, { handler: AnyFn; opts: unknown }>;
  tools: Map<string, (ctx: Record<string, unknown>) => unknown>;
  services: Array<{ id: string; start: () => unknown; stop?: () => unknown }>;
  capability: Record<string, unknown> | undefined;
  logs: string[];
  runtimeConfig: Record<string, unknown>;
};

export function createFakeApi(pluginConfig: Record<string, unknown> | undefined, runtimeConfig: Record<string, unknown> = {}): FakeApi {
  const fake: FakeApi = {
    hooks: new Map(),
    tools: new Map(),
    services: [],
    capability: undefined,
    logs: [],
    runtimeConfig,
    api: undefined as unknown as OpenClawPluginApi,
  };
  const log = (level: string) => (message: string) => void fake.logs.push(`${level}: ${message}`);
  fake.api = {
    id: "memory-dxmt",
    name: "Memory (DXMT)",
    pluginConfig,
    logger: { info: log("info"), warn: log("warn"), error: log("error"), debug: log("debug") },
    runtime: { config: { current: () => fake.runtimeConfig } },
    registerTool: (factory: unknown, opts?: { name?: string }) => {
      fake.tools.set(opts?.name ?? "?", factory as (ctx: Record<string, unknown>) => unknown);
    },
    on: (name: string, handler: AnyFn, opts?: unknown) => {
      fake.hooks.set(name, { handler, opts });
    },
    registerService: (service: { id: string; start: () => unknown; stop?: () => unknown }) => {
      fake.services.push(service);
    },
    registerMemoryCapability: (capability: Record<string, unknown>) => {
      fake.capability = capability;
    },
  } as unknown as OpenClawPluginApi;
  return fake;
}

export const allowRecallAuthority = {
  fingerprint: "test",
  allows: (name: string) => name === "memory_recall",
  assertActive: () => undefined,
};
