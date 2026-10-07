# scripts/

What each script here is, and whether anything depends on it. **None of these is the quality
gate** — that is `make verify` (fmt-check, clippy `-D warnings`, the unit tests and every
integration target) plus `make test-api` for the HTTP suite. These are the tools around it.

## Part of the documented workflow

| script | what it does | named in |
| --- | --- | --- |
| `api_tests.sh` | starts a server per storage mode and runs `tests/api` against each | `make test-api`, and three CI jobs |
| `run_examples.py` | runs `examples/`, naming every one it could not run and why | `make examples`, `make examples-list` |
| `make_compat_fixture.py` | writes a data directory for `tests/compat` to open with a later build | `CONTRIBUTING.md`, `tests/compat/test_compat.py` |
| `ann_to_raw.py` | converts an ann-benchmarks `.hdf5` file to the raw arrays the calibration test reads | `tests/recall/calibration_transfer_test.rs` |
| `http_ingest_rate.py` | ingest rate over HTTP, plus recall and the quantization that lands after the load | `README.md`'s measured-performance section |

## Load and soak (need a release build; each starts its own server)

| script | what it does |
| --- | --- |
| `soak.py` | end-to-end soak against a real `vectordb serve` |
| `soak_long.py` | the same, over hours to a day |
| `force_rebuild_under_load.py` | forces an index rebuild on a large table while readers and writers keep going, to show queries are not stalled by it |

## VectorDBBench

`test_vectordbbench_integration.py` and `verify_vectordbbench.sh` check that the DXMT client
is installed in a VectorDBBench environment. The client itself is
[`benches/vectordbbench-client/`](../benches/vectordbbench-client/), and its `install.sh` puts
it there.

The published numbers in `README.md` came from the measurement described beside them, not from
these scripts; treat a run of one as a starting point rather than as the number.

## Removed

Until 2026-10-07 this directory also held about thirty one-off probes and environment scripts:
checks for bugs since fixed and covered by `tests/api/`, benchmarks of features that are no
longer in the crate (`gpu`, `simd`, `multimodal`), scripts that called routes the server no
longer serves (`/tables` without `/api/v1`, `/collections`), and an internal deployment
script. They were taken out before the repository was published; `git log -- scripts/` has
them.
