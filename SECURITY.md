# Security Policy

## Reporting a vulnerability

**Do not open a public issue.**

Report privately through GitHub: go to the **Security** tab of this repository and choose
**Report a vulnerability**. That opens a private advisory visible only to you and the
maintainers, and it is the channel we watch.

Useful in a report, roughly in order:

- the commit you built from, and the feature flags you built with
- the subcommand the server was running, and the interface flags it was given
- a request sequence that reproduces it, ideally as `curl`
- what you expected, and what happened

You will get an acknowledgement within a week. If a report turns out to be valid, we will agree
a disclosure timeline with you before publishing anything.

## What is in scope

The supported surface — what a deployment is expected to run — is the single-node server:
`vectordb serve`, the local write-ahead log and snapshots, and the
documented HTTP routes. Reports against that are in scope.

**Out of scope**, because they are not claimed to work and the README says so:

- object-storage mirroring, which is experimental
- indexes the support matrix lists as not selectable
- anything reached only by building with a non-default feature flag that the README marks
  experimental

If you are unsure which side of that line something falls on, report it and we will say.

## Deployment expectations

Three properties of the current release shape are worth stating plainly, because they are design
decisions rather than vulnerabilities:

- **Authentication is one optional shared key, and there is no authorization.** Without
  `DXMT_API_KEY`, anyone who can reach the port can read and write every table and trigger
  administrative operations such as snapshots and backups. With it, every caller holding the
  key can do all of that; there are no users, roles or per-table permissions. Run it on a
  private network or behind a proxy that authenticates — `vectordb/examples/nginx-jwt.conf` is
  a sample of the latter.
- **Payload data is stored and returned as given.** There is no encryption at rest; the write-
  ahead log, snapshots and row files hold plaintext vectors and payloads.
- **The web console and the API reference are served by default, and both are read-only; the
  port they are served on is not.** The console issues only reading requests and the reference
  is a document, but neither puts the server into any kind of read-only mode — the same port
  still accepts writes, snapshots and backups from anything that can reach it. What they change
  is discoverability: they turn an exposed port into a browsable view of the data and a
  readable map of the API. Both sit behind `DXMT_API_KEY` when one is set. On a port that is
  reachable from anywhere you do not control, start with `--no-dashboard --no-swagger`, or put
  the port on a private interface or behind an authenticating proxy — which is the advice for
  the data routes too, since those were never optional.

Reports that the server is unauthenticated when no key is set, or that a key holder can do
everything, are not treated as vulnerabilities. Reports that a
route bypasses a boundary the documentation does claim — a filter that returns records it
should exclude, a path that escapes the configured data directory, a payload that can crash or
hang the process — are.

### Embedding provider keys

A table that embeds text names its provider by `base_url` and the *name* of a server
environment variable holding the key (`api_key_env`). Both come from whoever can create or
alter a table, so the server limits what they can reach: the variable must be a known
provider's (`OPENAI_API_KEY`, `COHERE_API_KEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, ...), the one
named in `DXMT_EMBEDDING_API_KEY_ENV`, or listed in `DXMT_ALLOWED_KEY_ENVS` — never
`DXMT_API_KEY`; the key is sent only to a known provider's API host, a loopback address, the
host of `DXMT_EMBEDDING_BASE_URL`, or a host listed in `DXMT_ALLOWED_PROVIDER_HOSTS`; and a
provider URL pointing at a link-local or cloud metadata address is refused whether or not a
key is involved. The rule is applied where the request is made, so it covers tables restored
from a backup or imported from a file as well as new ones. Widen the two lists only with names
you control. A way to make the server send a key outside these rules is in scope.

## Supported versions

Only the latest release and the `main` branch are supported; fixes land on `main` and ship in
the next release. There is no backport policy yet.
