# DXMT as memory for MaxKB agents (MCP)

MaxKB's AI chat node and its MCP tools can call an MCP server over Streamable HTTP. DXMT serves
one at `POST /mcp`, so a MaxKB agent can keep and recall memories in DXMT with no code on either
side: `remember`, `recall`, `forget`, `list_memories`, and -- on a server with a memory model --
`extract_memories` (facts out of a conversation) and `update_profile` (a person's profile).

What this adds to MaxKB: its built-in long-term memory is one text per application and user,
rewritten by a model and pasted whole into the prompt. Through DXMT the agent recalls only what
the question needs, by meaning and by keyword, and the same memories are shared with every other
agent on the server -- another MaxKB application, Hermes, Claude Code.

## 1. Put DXMT where MaxKB is allowed to reach it

**This is the step that goes wrong.** The official MaxKB image runs MCP connections inside a
sandbox that refuses a list of addresses, `MAXKB_SANDBOX_PYTHON_BANNED_HOSTS`, which by default
is:

    127.0.0.0/8, localhost, host.docker.internal, 172.17.0.0/16, maxkb, pgsql, redis,
    172.31.250.192/26, 0.0.0.0/32, ::/0  (and the MaxKB container's own IP)

So `http://localhost:9999/mcp`, `http://host.docker.internal:9999/mcp` and an address on
Docker's default bridge all fail, even though the server answers them. Run DXMT on a Docker
network shared with MaxKB and use its service name:

```yaml
# docker-compose.yml
services:
  maxkb:
    image: 1panel/maxkb:latest
    ports: ["8080:8080"]
    volumes: ["maxkb:/opt/maxkb"]
    networks: [ai]
  dxmt:
    image: dxmtai/dxmt:latest
    environment:
      DXMT_EMBEDDING_BASE_URL: http://your-model-server:1234/v1   # any OpenAI-compatible /embeddings
      DXMT_EMBEDDING_MODEL: bge-m3
      # DXMT_API_KEY: change-me                                  # then send it, see below
      # DXMT_MEMORY_LLM_MODEL: qwen3                             # enables extract_memories / update_profile
    volumes: ["dxmt:/data"]
    networks: [ai]
networks:
  ai: {}
volumes:
  maxkb: {}
  dxmt: {}
```

MaxKB then reaches DXMT at `http://dxmt:9999/mcp`. A DXMT on another machine works the same way at
any address outside the list. Narrowing the list itself (setting
`MAXKB_SANDBOX_PYTHON_BANNED_HOSTS` on the MaxKB container) also works, and loosens a protection
MaxKB put there on purpose; prefer the shared network.

## 2. Add the MCP server in MaxKB

In the AI chat node's MCP settings (or under Tools, as an MCP tool), give:

```json
{
  "dxmt": {
    "transport": "streamable_http",
    "url": "http://dxmt:9999/mcp"
  }
}
```

With `DXMT_API_KEY` set on the server, add `"headers": {"api-key": "change-me"}` (or
`"Authorization": "Bearer change-me"`).

## 3. Tell the model when to use it

The tools describe themselves, but a model uses memory consistently only when the prompt says
so. A system prompt along these lines works:

> At the start of each conversation, call `recall` with the user's request to see what you
> already know about them. When the user states something durable -- a preference, a fact about
> themselves, a decision -- call `remember` with it, kind `preference` or `fact`. Tag every memory
> with `user:<the user's id>` and pass the same tag to `recall`, so one user never sees another's.

For the tag, insert MaxKB's per-user variable (the start node's `chat_user_id`) into the prompt.
On a server with `DXMT_MEMORY_LLM_MODEL`, an end-of-conversation step can call
`extract_memories` with the transcript instead of relying on the model to remember facts one by
one, and `update_profile` to keep a profile that `recall` with kind `profile` returns.

## What was tested

On 2026-11-07, with the official `1panel/maxkb:latest` image (MaxKB v2.10.8-lts, sandbox on) and
`dxmtai/dxmt:latest` (1.6.0) on one Docker network, MaxKB's own MCP client
(`common.mcp.client.create_mcp_client`, the path a chat node takes, through the sandbox) listed
DXMT's tools and ran `remember` and `recall` against `http://dxmt:9999/mcp`. The same call to
DXMT's address on the default bridge failed inside the sandbox while a plain `curl` from the same
container reached it -- the banned list at work. The configuration was exercised through that
client, not by clicking through MaxKB's UI. `extract_memories` and `update_profile` arrived after
1.6.0 and were tested with the same client library (`langchain-mcp-adapters` 0.3.0, `mcp`
1.30.0) against a newer server.
