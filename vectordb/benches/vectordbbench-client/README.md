# VectorDBBench client for DXMT

VectorDBBench discovers database clients from its own `DB` enum and imports them from its
own package, so this code cannot run from here — it has to sit inside the installed
VectorDBBench tree:

```
<venv>/lib/python3.X/site-packages/vectordb_bench/backend/clients/dxmt/
```

This directory is the version-controlled copy. Keep them in sync by hand, in both
directions: edit here and copy over, or edit there and copy back before the change is lost.

```bash
VDB=<venv>/lib/python3.11/site-packages/vectordb_bench/backend/clients/dxmt
cp benches/vectordbbench-client/{dxmt.py,config.py,__init__.py} "$VDB"/
```

## Why this copy exists

Every DXMT number VectorDBBench had ever produced was taken with the server's default
search parameters, because `DXMTAutoConfig.search_param()` returned `{}` and the client
never passed anything — while the Qdrant client in the same suite pins `hnsw_ef=256`. The
comparison described itself as parameter-aligned and was not. Fixing that meant editing
files that lived only inside a virtualenv, where a reinstall would have silently reverted
it and the same wrong comparison would have come back looking authoritative.

## What is in here beyond the stock client

- `search_mode` plumbed from `DXMTIndexConfig` through to `DXMTClient.search`, so a case
  can ask for `fast` / `normal` / `accurate` (ef_search x1 / x1 / x4) the way the Qdrant
  case pins `hnsw_ef`.
- `DXMT_BENCH_SKIP_REBUILD=1` makes `optimize()` return without forcing an index rebuild.
  That is a measurement switch, not a feature: it answers "is the graph insert already
  built good enough to serve?" without the rebuild masking the answer. It is what found
  that a table built through the create-table path scored recall@100 0.874 against 0.958
  after a rebuild — an `ef_construction` divergence between two selection paths that a
  comment claimed did not exist.

Both defaults are the stock behaviour, so a run that sets neither measures what it
measured before.
