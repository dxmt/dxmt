---
name: dxmt-memory
description: Use DXMT's memory tools (recall, remember, forget, list_memories) well, so what the user tells you in one session is there in the next. Use at the start of a task that may depend on earlier sessions, when the user states a durable preference, fact, decision or instruction, when something they told you before has changed, or when they ask you to forget something.
---

# Long-term memory with DXMT

The `dxmt` MCP server keeps memories that outlive the session and are shared with every other
agent on the same server. A memory is a short text with a **kind**, **tags**, a **source** and a
time (`at`). The server embeds it, and `recall` finds it by meaning and by keyword at once.

## At the start of a task: recall

Before answering anything that could depend on what the user said before, call `recall` with the
user's request in their own words:

```json
{"query": "how should I format the release notes", "limit": 10}
```

- Each hit has `id`, `text`, `kind`, `tags`, `at`, and a `score`. `score` is a **rank** -- the
  best hit is near 1.0 however weak the match. Newer servers also return `similarity`, the cosine
  to the query (1.0 identical): judge relevance by `similarity` when it is there, by reading the
  text when it is not.
- Narrow with `kind` (one kind), `tags` (any of), `since` (Unix seconds or RFC 3339).
- Newer servers page with `offset`: if the answer may be beyond the first page, ask the same
  query with `offset` equal to what you have seen.
- If nothing relevant comes back, say so rather than guessing what you were told.

## When the user says something durable: remember

Keep what will still be true and useful next week; skip what only matters for this task.

| kind | for | example |
| --- | --- | --- |
| `preference` | how they want things done | "Prefers answers in Chinese, with code in English" |
| `fact` | about them, their work, their world | "Works on a Rust vector database called DXMT" |
| `instruction` | standing rules for agents | "Never push to main without asking" |
| `event` | something that happened, with when | "Shipped 1.6.0 on 4 November 2026" |
| `note` | anything else (the default) | |

```json
{"text": "Prefers answers in Chinese, with code in English", "kind": "preference",
 "tags": ["user:richard"], "source": "claude-code"}
```

- **One fact per memory, self-contained.** Name the person or project; no "he", no "this repo".
  Write dates as dates, not "yesterday".
- **Tags scope it.** Use stable tags such as `user:<name>`, `project:<repo>`, and pass the same
  tags to `recall` so one project's memories do not leak into another's.
- Saying the same thing twice keeps it once: the server refreshes the memory it already has
  (`duplicate: true`), so there is no need to check first.

## When something changed: supersede, do not delete

"I moved to Berlin" replaces "I live in Shanghai" -- it does not erase it. Recall the old memory,
then remember the new one with `supersedes`:

```json
{"text": "Lives in Berlin (moved in October 2026)", "kind": "fact", "supersedes": ["mem-…"]}
```

The old memory is kept as history, marked `superseded_by`, and left out of recall unless
`include_superseded` is set. To correct a typo in place, remember again with the same `id`.

## Forgetting

Delete only when the user asks you to forget something, or a memory is plainly wrong.

1. Find it: `recall` (or `get_memory` with an id, on newer servers).
2. Show the user what you found and confirm which one.
3. `forget` with the exact `ids`. Deletion is permanent.

Never delete because a query "matched": a loose query ranks *something* first.

## Reviewing what is kept

`list_memories` (newest first, optional `kind`) is for the user to audit what you remember about
them. Offer it when they ask what you know.

## Newer servers: derived memory

If the tool list has them (servers with `DXMT_MEMORY_LLM_MODEL` set):

- `extract_memories` -- at the end of a session, hand it the conversation (`text`, one line per
  turn, `when` the date) and it keeps the facts it establishes, relative times resolved. Pass an
  `id_prefix` per conversation so extracting it again replaces the set.
- `update_profile` -- condenses everything known about a person (`name`, `tags`) into one memory
  of kind `profile`. At the start of a task, `recall` with `kind: "profile"` gives the whole
  picture in one hit.

A server started with `DXMT_MCP_READ_ONLY=1` offers only the reading tools; do not try to work
around it.
