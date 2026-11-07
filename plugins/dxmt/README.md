# DXMT plugin for Claude Code

Long-term memory and hybrid search for Claude Code, backed by a local
[DXMT](https://github.com/dxmt/dxmt) server.

```bash
claude plugin marketplace add dxmt/dxmt
claude plugin install dxmt@dxmt
```

(In a session: `/plugin marketplace add dxmt/dxmt`, then `/plugin install dxmt@dxmt`.)

What it adds:

- **The `dxmt` MCP server** (`.mcp.json`): starts `vectordb mcp`, so `vectordb` must be on the
  PATH -- a release binary, `cargo build --release`, or relaying to a running server. It opens
  the same store as `vectordb serve`, and relays to that server when one is running, so every
  agent on the machine shares one memory. To use a server elsewhere instead, replace the entry
  with `{"type": "http", "url": "http://host:9999/mcp"}`.
- **Skills**, which Claude loads when they apply:
  - `dxmt-setup` -- install and run the server, give it an embedding model, connect clients,
    troubleshoot.
  - `dxmt-memory` -- when to recall, what to remember and how (kinds, tags, supersede), and how
    to forget safely.
  - `dxmt-search` -- tables, keyword/vector/hybrid search, filters, from MCP, REST or code.

The skills are plain Agent Skills (`skills/*/SKILL.md`) and work in any agent that reads that
format.
