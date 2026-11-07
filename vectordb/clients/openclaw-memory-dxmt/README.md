# openclaw-memory-dxmt

[DXMT](https://github.com/dxmt/dxmt), the open-source memory database for personal agents, as
the long-term memory of [OpenClaw](https://github.com/openclaw/openclaw).

A `kind: "memory"` plugin for the memory slot. Before each turn it recalls from a DXMT server
by meaning *and* keyword and prepends the hits; after each turn it captures durable user
statements; the model gets `memory_recall`, `memory_store` and `memory_forget`. The server
does the embedding (a local model through ollama or LM Studio, or any OpenAI-compatible
endpoint), so OpenClaw holds no embedding key, no vectors and no local database, and the same
memories are what every other agent and tool on the machine sees over REST and MCP.

```bash
openclaw plugins install npm:openclaw-memory-dxmt
```

Then select it as the memory slot and restart the Gateway:

```json5
{
  plugins: {
    slots: { memory: "memory-dxmt" },
    entries: {
      "memory-dxmt": {
        enabled: true,
        config: {
          url: "http://localhost:9999",   // the DXMT server; the default
          // apiKey: "${DXMT_API_KEY}",   // only if the server has one
          scope: "agent",                 // or "shared"
        },
      },
    },
  },
}
```

A DXMT server with an embedding model has to be running. A local `ollama pull bge-m3` is found
on its own:

```bash
docker run -d -p 127.0.0.1:9999:9999 -v dxmt:/data \
  --add-host=host.docker.internal:host-gateway -e OLLAMA_HOST=host.docker.internal:11434 dxmtai/dxmt
```

`openclaw memory status` reports the provider's health, including whether the server answers.

## What is specific to DXMT

- **One store, many agents.** Memories are rows in a server you run. With `scope: "shared"`
  the agent recalls what Hermes, a LangChain app or `curl` wrote, and they recall what it
  wrote. With `scope: "agent"` (the default) each OpenClaw agent sees only its own.
- **Supersede, not overwrite.** `memory_store(text, supersedes: [id])` keeps the old fact as
  history and leaves it out of recall; the server deduplicates a fact said twice in the same
  category, so a repeated statement refreshes it instead of adding a copy.
- **Keyword and meaning together.** Recall is the server's hybrid search, so an exact name or
  error string is found as well as a paraphrase. Scores are cosine similarity (1.0 identical); a server before 1.7 reports only a rank, and then `memory_forget` never deletes without being given an id.
- **Nothing leaves the machine** unless `url` points somewhere else.

## What gets stored

| OpenClaw event | DXMT memory |
| --- | --- |
| `memory_store` | kind = category (`preference`, `fact`, `decision`, `entity`, `other`); `importance`, `origin: tool`, `session_key` in metadata |
| auto-capture after a successful turn | user messages that state a preference, fact or decision (the same triggers as the bundled LanceDB plugin); `origin: auto_capture` |

Every memory carries the tags `openclaw` and `openclaw:agent:<agentId>`. Incognito sessions
are never written. Memories are injected as untrusted data inside `<relevant-memories>`;
text that looks like prompt instructions is refused by `memory_store` and skipped by capture.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `url` | `http://localhost:9999` | the DXMT server; `${ENV_VAR}` is expanded |
| `apiKey` | unset | sent as a bearer token if the server has one |
| `scope` | `agent` | `agent`: recall this agent's memories; `shared`: everything on the server |
| `autoRecall` / `autoCapture` | `true` | the two hooks |
| `recallLimit` | `3` | memories injected per turn |
| `minScore` | `0.3` | similarity floor for recall |
| `captureMaxChars` / `recallMaxChars` | `500` / `1000` | size limits |
| `customTriggers` | `[]` | phrases that make auto-capture keep a message |

The toggles and limits follow a hot config reload; the server address does not.

## Development

```bash
npm install
npm run typecheck && npm test                    # against an in-process stand-in for the server
DXMT_TEST_URL=http://127.0.0.1:9999 npm test     # plus the live loop against a server with a model
npm run build && npm pack                        # then: openclaw plugins install npm-pack:./openclaw-memory-dxmt-*.tgz
```

## Disclosure

Conversation turns that match the capture triggers, memories the model stores and recall
queries are sent to the configured DXMT server, `http://localhost:9999` by default. Nothing is
sent anywhere else, and nothing is sent for incognito sessions.
