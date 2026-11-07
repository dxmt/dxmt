# Maintainers and governance

## Who maintains DXMT

| Maintainer | GitHub | Areas |
| --- | --- | --- |
| Eric | [@eric-epsilla](https://github.com/eric-epsilla) | Everything: the server (`vectordb/src`), the clients under `vectordb/clients/`, releases |

`.github/CODEOWNERS` routes review requests from this table. When someone else takes on an
area, they are added here and to `CODEOWNERS` in the same pull request.

## How decisions are made

DXMT is maintainer-led. A change lands when a maintainer approves it and CI passes; there is no
vote. A change to the HTTP API, the storage format or a published client's interface is
discussed in an issue first, because a release cannot take it back: `vectordb/docs/SUPPORT_MATRIX.md`
records what is promised, and its "Promotion Rules" are what a feature needs before it is.

Disagreements are settled in the issue or pull request, in public. If the project grows past one
maintainer, this section will describe how maintainers are added and how they decide together.

## What to expect

| What | Target |
| --- | --- |
| A security report ([SECURITY.md](SECURITY.md)) | Acknowledged within a week |
| A new issue | First response within a week |
| A pull request | First review within a week |
| A release | When `main` has something worth shipping; the history is in `vectordb/CHANGELOG.md` |

These are targets, not guarantees. If something has waited longer, a comment on it is welcome.

## Supported versions

The latest release and `main`. Fixes land on `main` and ship in the next release; there is no
backport branch yet. See [SECURITY.md](SECURITY.md#supported-versions).
