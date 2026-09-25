# scripts/

What each script here is, and whether anything depends on it. **None of these is the quality
gate** — that is `make verify` (fmt-check, clippy `-D warnings`, the unit tests and every
integration target) plus `make test-api` for the HTTP suite. These are the tools around it.

They were 35 files with no index, so the only way to find out what one did was to run it — and
two of them called endpoints the server stopped serving, which is not something running them
tells you quickly.

## Part of the documented workflow

| script | what it does | named in |
| --- | --- | --- |
| `api_tests.sh` | starts a server per storage mode and runs `tests/api` against each | `make test-api`, CI, both CLAUDE.md files |
| `run_wal_tests.sh` | the WAL recovery targets | `vectordb/CLAUDE.md` |
| `make_compat_fixture.py` | writes a data directory for `tests/compat` to open with a later build | `CONTRIBUTING.md`, `tests/compat/test_compat.py` |
| `soak.py` | end-to-end soak against a real `vectordb serve` | the 24-hour soak in the design notes |
| `soak_long.py` | the same, over hours to a day | |
| `http_ingest_rate.py` | ingest rate over HTTP, plus recall and the quantization that lands after the load | `README.md`'s measured-performance section |

## Benchmarks (need a running server unless stated)

`benchmark_phase1.py`, `comprehensive_benchmark.py`, `gist_benchmark.py`,
`gist_quick_benchmark.py`, `test_index_comparison.py`, `test_performance.sh`,
`validate_performance.py` (thresholds, no server), `multimodal_benchmark.py` (needs the
`multimodal` feature), `test_vectordbbench_integration.py` and `verify_vectordbbench.sh`
(VectorDBBench harness).

The published numbers in `README.md` came from the measurement described beside them, not from
these; treat a benchmark here as a starting point rather than as the number.

## Diagnostics and one-off probes

`diagnose_insert.py`, `diagnose_storage.sh`, `force_rebuild_under_load.py`, `test_500_error.py`,
`test_insert_api.py`, `test_insert_fixed.py`, `test_error_handling_fix.sh`,
`test_upsert_fix.sh`, `test_std.sh`, `test_json_as_f64.rs` (a standalone probe of
`serde_json::Value::as_f64`, compiled by nothing).

These were written to chase a specific problem and kept because the shape of the probe is
useful. What they assert is covered by `tests/api/`; when one disagrees with the suite, the
suite is right.

## Environment and deployment

`check-docker-setup.sh`, `gpu-env.sh` (source it), `run-gpu.sh`, `deploy-phase3.sh`,
`run_working_tests.sh` (three test targets, not the gate), `run_all_haystack_tests.sh`.

## Removed

`test_index_status.py` called `/tables/{t}/index_status`, which answers 404: the router serves
`/tables/{t}/index` and `/tables/{t}/index/rebuild`, and nothing reports index status. `test_sync_insert.py`
compared "sync" against "async" inserts by sending `?sync=true`, a query parameter no handler
has ever read, so it timed two runs of the same path and reported the difference as a finding.
