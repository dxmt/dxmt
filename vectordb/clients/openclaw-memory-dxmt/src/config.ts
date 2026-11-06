/** Plugin configuration: `plugins.entries["memory-dxmt"].config`, validated here. */

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

export const DEFAULT_URL = "http://localhost:9999";
export const DEFAULT_RECALL_LIMIT = 3;
export const DEFAULT_MIN_SCORE = 0.3;
export const DEFAULT_CAPTURE_MAX_CHARS = 500;
export const DEFAULT_RECALL_MAX_CHARS = 1000;

export type MemoryScope = "agent" | "shared";

export type MemoryConfig = {
  url: string;
  apiKey?: string;
  scope: MemoryScope;
  autoCapture: boolean;
  autoRecall: boolean;
  recallLimit: number;
  minScore: number;
  captureMaxChars: number;
  recallMaxChars: number;
  customTriggers: string[];
};

const KNOWN_KEYS = new Set([
  "url",
  "apiKey",
  "scope",
  "autoCapture",
  "autoRecall",
  "recallLimit",
  "minScore",
  "captureMaxChars",
  "recallMaxChars",
  "customTriggers",
]);

/** `${ENV_VAR}` in a string value is read from the environment; unset is an error. */
export function resolveEnvVars(value: string, env: NodeJS.ProcessEnv = process.env): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, name: string) => {
    const resolved = env[name];
    if (!resolved) {
      throw new Error(`Environment variable ${name} is not set`);
    }
    return resolved;
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readBoolean(raw: Record<string, unknown>, key: string, fallback: boolean): boolean {
  const value = raw[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "boolean") {
    throw new Error(`${key} must be a boolean`);
  }
  return value;
}

function readNumber(
  raw: Record<string, unknown>,
  key: string,
  fallback: number,
  { min, max, integer }: { min: number; max: number; integer: boolean },
): number {
  const value = raw[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new Error(`${key} must be ${integer ? "an integer" : "a number"}`);
  }
  if (value < min || value > max) {
    throw new Error(`${key} must be between ${min} and ${max}`);
  }
  return value;
}

/** Parse the raw plugin config. Throws with the offending key named; the plugin then registers disabled. */
export function parseMemoryConfig(input: unknown, env: NodeJS.ProcessEnv = process.env): MemoryConfig {
  const raw = input === undefined || input === null ? {} : input;
  if (!isRecord(raw)) {
    throw new Error("memory-dxmt config must be an object");
  }
  const unknown = Object.keys(raw).filter((key) => !KNOWN_KEYS.has(key));
  if (unknown.length > 0) {
    throw new Error(`memory-dxmt config has unknown keys: ${unknown.join(", ")}`);
  }
  const urlRaw = raw.url === undefined ? DEFAULT_URL : raw.url;
  if (typeof urlRaw !== "string" || !urlRaw.trim()) {
    throw new Error("url must be a non-empty string");
  }
  const url = resolveEnvVars(urlRaw.trim(), env);
  parseServerUrl(url); // validates the shape
  let apiKey: string | undefined;
  if (raw.apiKey !== undefined) {
    if (typeof raw.apiKey !== "string") {
      throw new Error("apiKey must be a string");
    }
    apiKey = resolveEnvVars(raw.apiKey, env) || undefined;
  }
  const scope = raw.scope === undefined ? "agent" : raw.scope;
  if (scope !== "agent" && scope !== "shared") {
    throw new Error('scope must be "agent" or "shared"');
  }
  let customTriggers: string[] = [];
  if (raw.customTriggers !== undefined) {
    if (!Array.isArray(raw.customTriggers) || raw.customTriggers.some((t) => typeof t !== "string" || !t.trim())) {
      throw new Error("customTriggers must be an array of non-empty strings");
    }
    customTriggers = (raw.customTriggers as string[]).map((t) => t.trim());
  }
  return {
    url,
    apiKey,
    scope,
    autoCapture: readBoolean(raw, "autoCapture", true),
    autoRecall: readBoolean(raw, "autoRecall", true),
    recallLimit: readNumber(raw, "recallLimit", DEFAULT_RECALL_LIMIT, { min: 1, max: 10, integer: true }),
    minScore: readNumber(raw, "minScore", DEFAULT_MIN_SCORE, { min: 0, max: 1, integer: false }),
    captureMaxChars: readNumber(raw, "captureMaxChars", DEFAULT_CAPTURE_MAX_CHARS, {
      min: 100,
      max: 10_000,
      integer: true,
    }),
    recallMaxChars: readNumber(raw, "recallMaxChars", DEFAULT_RECALL_MAX_CHARS, {
      min: 100,
      max: 10_000,
      integer: true,
    }),
    customTriggers,
  };
}

export type ServerAddress = { host: string; port: number; ssl: boolean };

/** `http[s]://host[:port]` to what the SDK client takes. */
export function parseServerUrl(url: string): ServerAddress {
  let parsed: URL;
  try {
    parsed = new URL(url.includes("://") ? url : `http://${url}`);
  } catch {
    throw new Error(`url must look like http://host:port, got ${JSON.stringify(url)}`);
  }
  if ((parsed.protocol !== "http:" && parsed.protocol !== "https:") || !parsed.hostname) {
    throw new Error(`url must look like http://host:port, got ${JSON.stringify(url)}`);
  }
  const ssl = parsed.protocol === "https:";
  const port = parsed.port ? Number(parsed.port) : ssl ? 443 : 9999;
  return { host: parsed.hostname, port, ssl };
}
