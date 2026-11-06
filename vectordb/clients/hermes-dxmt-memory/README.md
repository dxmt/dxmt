# hermes-dxmt-memory

[DXMT](https://github.com/dxmt/dxmt), the open-source memory database for personal agents, as a
memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Hermes recalls from a DXMT server before each turn, by meaning *and* keyword, and extracts
durable facts at session end and before context compression. The agent gets four tools:
`dxmt_recall`, `dxmt_remember`, `dxmt_read`, `dxmt_forget`. The server does the embedding
(a local model through ollama or LM Studio, or any OpenAI-compatible endpoint), so Hermes needs
no embedding key, and the same memories are what every other agent on the machine sees over
REST and MCP.

What is specific to DXMT, against the memory providers that keep a store inside Hermes's
process or send turns to a hosted service:

- **One store, many agents.** Memories are rows in a server you run. A fact Hermes keeps is
  recalled by anything else that talks to DXMT (OpenClaw, a LangChain app, `curl`), and a
  memory another tool wrote is recalled here when workspace isolation is off.
- **Supersede, not overwrite.** `dxmt_remember(supersedes=[id])` keeps the old fact as
  history and leaves it out of recall; the server deduplicates a fact said twice in the same
  category.
- **Keyword and meaning together.** Recall is the server's hybrid search, so an exact name or
  error string is found as well as a paraphrase.
- **Nothing leaves the machine** unless you point `DXMT_URL` somewhere else.

## Installation: users

1. Install Hermes Agent (see its README), then the plugin. The plugin lives in a subdirectory
   of the DXMT repository, which the installer understands:

   ```sh
   hermes plugins install dxmt/dxmt/vectordb/clients/hermes-dxmt-memory
   ```

2. Install the client into Hermes's interpreter (shared across profiles, so once):

   ```sh
   uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 dxmt-python-sdk pyyaml
   ```

3. Run a DXMT server with an embedding model. A local `ollama pull bge-m3` is found on its
   own; `DXMT_EMBEDDING_MODEL` and `DXMT_EMBEDDING_BASE_URL` name any OpenAI-compatible one.

   ```sh
   docker run -d -p 127.0.0.1:9999:9999 -v dxmt:/data \
     --add-host=host.docker.internal:host-gateway -e OLLAMA_HOST=host.docker.internal:11434 dxmtai/dxmt
   ```

4. Activate it and pick `dxmt`:

   ```sh
   hermes memory setup
   hermes memory status      # Provider: dxmt
   ```

The server is elsewhere? `DXMT_URL=http://host:port` in the environment, or
`plugins.dxmt.server.url` in `~/.hermes/config.yaml`. An API key, if the server has one, is
read from `DXMT_API_KEY`. Try it in an isolated profile first with `hermes profile create
dxmt-demo` and `-p dxmt-demo` on every command.

## What gets stored

| Hermes event | DXMT memory |
| --- | --- |
| `dxmt_remember`, extraction, built-in memory tool mirror | kind = category (`preference`, `entity`, `event`, `case`, `pattern`, `general`); `abstract`, `provenance`, `origin`, `session_id` in metadata |
| each completed turn (`turns.enabled`) | kind `turn`, id derived from (session, index, role, text) so a replay writes nothing twice |

Every row carries the tags `hermes`, `hermes:ws:<workspace>`, `hermes:user:<id>`,
`hermes:agent:<identity>`. Recall filters on the workspace tag while
`retrieval.isolate_workspace` is true (the default); set it false to recall across workspaces
and across the other tools writing to the same server.

## Configuration

Defaults are in [`src/default_config.yaml`](src/default_config.yaml); override under
`plugins.dxmt` in `~/.hermes/config.yaml`:

```yaml
plugins:
  dxmt:
    server:
      url: http://localhost:9999   # DXMT_URL wins over this
      api_key_env: DXMT_API_KEY
    retrieval:
      top_k: 10                    # dxmt_recall default
      prefetch_k: 5                # memories injected before a turn
      min_score: 0.0               # cosine similarity floor (1.0 = identical)
      isolate_workspace: true
    turns:
      enabled: true
    extraction:
      enabled: true                # LLM extraction at pre-compress and session end
      min_turns: 3
```

Extraction uses Hermes's auxiliary LLM (`call_llm`, task `dxmt_extraction`), so it follows
the model and limits configured there.

## Development

```sh
cd vectordb/clients/hermes-dxmt-memory
ln -sf "$PWD" ~/.hermes/plugins/dxmt          # Hermes loads the directory; edits apply next session
HERMES_AGENT_DIR=/path/to/hermes-agent pytest  # offline: the server is a stand-in
DXMT_TEST_URL=http://127.0.0.1:9999 HERMES_AGENT_DIR=... pytest -m live   # against a server with a model
```

The offline suite covers the full tool loop, scoping, idempotent turns, extraction and
provenance against an in-memory stand-in for the server's memory API; the live suite drives the
same loop through Hermes's own `MemoryManager`.

## Disclosure

Conversation turns, extracted facts and the built-in memory tool's writes are sent to the
configured DXMT server, `http://localhost:9999` by default. Nothing is sent anywhere else.
