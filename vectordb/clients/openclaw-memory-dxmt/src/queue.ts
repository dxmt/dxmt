/** Serialise async work per key: a later task for the same key starts after the earlier one settles. */
export async function enqueueKeyedTask(params: {
  tails: Map<string, Promise<void>>;
  key: string;
  task: () => Promise<void>;
}): Promise<void> {
  const { tails, key, task } = params;
  const previous = tails.get(key) ?? Promise.resolve();
  const run = previous.catch(() => undefined).then(task);
  const settled = run.catch(() => undefined);
  tails.set(key, settled);
  try {
    await run;
  } finally {
    if (tails.get(key) === settled) {
      tails.delete(key);
    }
  }
}
