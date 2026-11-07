---
name: dxmt-setup
description: Install and run a DXMT server, give it an embedding model, and connect it to Claude Code or another MCP client. Use when the user wants agent memory or a local vector/hybrid search database set up, when the dxmt MCP server will not start or its tools fail with an embedding or connection error, or when they ask how to point an agent at DXMT.
---

# Setting up DXMT

DXMT is one binary: a database with dense, keyword (BM25) and hybrid search, and a memory API
that agents reach over MCP or REST. Memory needs an **embedding model** the server can reach;
everything else works without one.

## 1. Run the server

Pick one. The server listens on `http://127.0.0.1:9999` (console at `/dashboard`) and keeps its
data in `~/.dxmt/wal` (`DXMT_WAL_DIR` to move it).

```bash
# Prebuilt binary (Linux x86_64/aarch64, macOS): https://github.com/dxmt/dxmt/releases
v=1.6.0; arch=$(uname -m)
curl -L https://github.com/dxmt/dxmt/releases/download/v$v/dxmt-vectordb-$v-$arch-linux-gnu.tar.gz | tar xz
./dxmt-vectordb-$v-$arch-linux-gnu/vectordb serve

# Docker (data in the `dxmt` volume, reaching an ollama on the host)
docker run -d --name dxmt -p 127.0.0.1:9999:9999 -v dxmt:/data \
  --add-host=host.docker.internal:host-gateway -e OLLAMA_HOST=host.docker.internal:11434 dxmtai/dxmt:latest

# From source
git clone https://github.com/dxmt/dxmt && cd dxmt/vectordb && cargo build --release
```

Put `vectordb` on the PATH: the MCP configuration below starts it by that name.

## 2. Give it an embedding model

The server decides at startup:

- **`DXMT_EMBEDDING_MODEL` set** -- that model, at `DXMT_EMBEDDING_BASE_URL` (any
  OpenAI-compatible `/embeddings`: LM Studio, vLLM, text-embeddings-inference, OpenAI itself).
  For OpenAI set `DXMT_EMBEDDING_MODEL=text-embedding-3-small` and `OPENAI_API_KEY`; another
  provider's key goes in the variable `DXMT_EMBEDDING_API_KEY_ENV` names.
- **Otherwise, a local ollama** that already has an embedding model: `ollama pull bge-m3` and
  nothing to configure. `DXMT_EMBEDDING_AUTODETECT=0` turns this off.

Without one, `remember` and `recall` fail with an error that says the server has no embedding
model. That message is the cue for this step; restart the server after setting the variables.

## 3. Connect the agent

The plugin's `.mcp.json` already starts `vectordb mcp` for Claude Code. By hand:

```bash
claude mcp add dxmt -- vectordb mcp                                  # stdio
claude mcp add --transport http dxmt http://127.0.0.1:9999/mcp       # a running server
```

`vectordb mcp` opens the same store `serve` does. When a server already has it open, it relays
to that server instead of failing on the lock -- so every agent on the machine shares one memory.
Claude Desktop and Cursor take the same command in their MCP configuration.

## 4. Check it

```bash
curl -s http://127.0.0.1:9999/health
curl -s -X POST http://127.0.0.1:9999/api/v1/memories -H 'content-type: application/json' \
  -d '{"text":"setup check","kind":"note","tags":["setup-check"]}'
curl -s -X POST http://127.0.0.1:9999/api/v1/memories/recall -H 'content-type: application/json' \
  -d '{"query":"setup check","tags":["setup-check"],"limit":1}'
```

Delete the check memory afterwards with `DELETE /api/v1/memories/{id}` (the id the first call
returned).

## Options worth knowing

| Variable | Effect |
| --- | --- |
| `DXMT_API_KEY` | Require this key on every request (`api-key` header, or `Authorization: Bearer`). MCP clients pass it in their headers; `vectordb mcp` relaying to a server sends it itself |
| `DXMT_PORT`, `DXMT_HOST` | Where `serve` listens (default 127.0.0.1:9999) |
| `DXMT_MCP_READ_ONLY=1` | Newer servers: MCP offers only the tools that read (search, list, recall). For an agent that should consult memory but never change it |
| `DXMT_MEMORY_LLM_MODEL` | Newer servers: a chat model for `extract_memories` and `update_profile` (base URL defaults to the embedding provider's) |

If a tool in this skill is not in the server's tool list, the server is older than the feature.

## Troubleshooting

- **"no embedding model"** -- step 2, then restart `serve`.
- **MCP server fails to start** -- `vectordb` is not on the PATH, or (without a running server)
  the data directory is unwritable. `vectordb mcp` writes JSON-RPC to stdout; run it from a
  terminal only to see the error, it waits for input otherwise.
- **Docker: the server cannot reach ollama** -- `OLLAMA_HOST=host.docker.internal:11434` with
  `--add-host=host.docker.internal:host-gateway`, as above.
- **Another app on the machine reaches DXMT from a container** (MaxKB, Dify, ...) -- put both on
  one Docker network and use the service name; some platforms' sandboxes refuse `localhost` and
  `host.docker.internal`.
