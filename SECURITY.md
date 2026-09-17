# Security Policy

## Reporting a vulnerability

**Do not open a public issue.**

Report privately through GitHub: go to the **Security** tab of this repository and choose
**Report a vulnerability**. That opens a private advisory visible only to you and the
maintainers, and it is the channel we watch.

Useful in a report, roughly in order:

- the commit you built from, and the feature flags you built with
- whether the server was running `serve` or `serve --simple`
- a request sequence that reproduces it, ideally as `curl`
- what you expected, and what happened

You will get an acknowledgement within a week. If a report turns out to be valid, we will agree
a disclosure timeline with you before publishing anything.

## What is in scope

The supported surface — what a deployment is expected to run — is the single-node server:
`vectordb serve` and `vectordb serve --simple`, the local write-ahead log and snapshots, and the
documented HTTP routes. Reports against that are in scope.

**Out of scope**, because they are not claimed to work and the README says so:

- cluster and consensus code, which is scaffolding with no supported deployment
- object-storage mirroring, which is experimental
- the multimodal modules, which return placeholder vectors
- indexes the support matrix lists as not selectable
- anything reached only by building with a non-default feature flag that the README marks
  experimental

If you are unsure which side of that line something falls on, report it and we will say.

## Deployment expectations

Two properties of the current release shape are worth stating plainly, because they are design
decisions rather than vulnerabilities:

- **The server has no authentication or authorization of its own.** Anyone who can reach the
  port can read and write every table, and can trigger administrative operations such as
  snapshots and backups. Run it on a private network or behind a proxy that authenticates —
  `vectordb/examples/nginx-jwt.conf` is a sample of the latter.
- **Payload data is stored and returned as given.** There is no encryption at rest; the write-
  ahead log, snapshots and row files hold plaintext vectors and payloads.
- **The web console (`--dashboard`) is read-only; the port it is served on is not.** The console
  issues only reading requests, but `--dashboard` does not put the server into any kind of
  read-only mode — the same port still accepts writes, snapshots and backups from anything that
  can reach it. What the console changes is discoverability: it turns an exposed port into a
  browsable, screenshot-ready view of the data. It is off unless both compiled in
  (`--features api-dashboard`) and asked for (`--dashboard`); keep it that way unless the port
  is on a private interface or behind an authenticating proxy.

Reports that the server is unauthenticated are not treated as vulnerabilities. Reports that a
route bypasses a boundary the documentation does claim — a filter that returns records it
should exclude, a path that escapes the configured data directory, a payload that can crash or
hang the process — are.

## Supported versions

Pre-1.0 and in alpha: only the `main` branch is supported. Fixes land there. There is no
backport policy yet, and the `v0.2.0-*` tags in this repository are measurement checkpoints,
not releases.
