# Changelog

Notable changes to DXMT VectorDB. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**What 1.0 promises, and what it does not.** The compatibility guarantee covers the HTTP API
under `/api/v1` and the six clients' public surfaces: a breaking change there needs a major
version. It does not cover anything `docs/SUPPORT_MATRIX.md` marks experimental, the cargo
features that are off by default, the on-disk formats (which are versioned separately and
migrate themselves), or the internals of the `dxmt-vectordb` crate as a library. Read that
matrix before depending on a feature; it is kept current on purpose and is the answer to
"can I use this", which is a different question from "does this exist".

The `v0.2.0-*` tags in this repository are **not** releases -- they are measurement
checkpoints from benchmarking runs, taken before anything was published. `v1.0.0` is the first
release, and the first commit that was ever pushed anywhere.

## [Unreleased]

### Security

- **A table's provider settings can no longer send the server's environment to a URL.** A
  table names its embedding and rerank provider by `base_url` and the name of an environment
  variable holding the key, both from the caller; the server read whatever variable was named
  and sent its value to whatever URL was named, so one table and one insert could hand any
  server secret to anyone who could reach the port. The variable must now be a provider key
  (a known provider's, `DXMT_EMBEDDING_API_KEY_ENV`'s, or one in `DXMT_ALLOWED_KEY_ENVS`;
  never `DXMT_API_KEY`), a key goes only to a known provider host, loopback, the host of
  `DXMT_EMBEDDING_BASE_URL` or one in `DXMT_ALLOWED_PROVIDER_HOSTS`, and link-local and cloud
  metadata URLs are refused. A keyless local provider is unaffected.
- **One request could stop the server.** `/search/text` (and the text half of
  `/search/hybrid`) with `limit: 0` or a huge `limit` panicked in Tantivy, and highlighting text
  whose lowercase is longer than itself (`İ`) sliced outside it; the release build aborts on a
  panic. Both are 400s or handled now. A query of the wrong width for a named vector space
  reached SIMD kernels that read past the shorter vector; it is a 400, and the kernels assert
  equal lengths in every build.
- **An export could overwrite the database's own files.** It was kept inside
  `DXMT_BACKUP_ROOT`, which defaults to the parent of the data directory, but not outside the
  store: an export to `<store>/wal.log` replaced it. Refused now, as backups already were.
- **`serve` listens on 127.0.0.1 by default** (was 0.0.0.0). Every route is open unless
  `DXMT_API_KEY` is set; `--host 0.0.0.0` serves the network. The Docker image is unchanged.

### Added

- **A table's text index can be changed after it holds data.** `PATCH /tables/{t}` takes
  `text_fields` -- which payload fields are indexed and with which analyzer -- and rebuilds the
  index from the records without stopping reads or writes: the current index keeps answering,
  writes made meanwhile reach both, and the request returns once the new index has taken over.
  It also turns full-text search on for a table created without it; `enable_full_text_search:
  true` on its own used to set the flag and build nothing. Indexes now live in generations
  (`fulltext/{table_id}/gN`, `CURRENT`), each recording the config it was built for, so an
  index built for another config -- a rebuild cut short by a crash -- is rebuilt at the next
  start. Indexes from before keep working where they are.
- **Read replicas (experimental).** `serve --replica` (or `DXMT_REPLICA=1`) follows the object
  storage bucket a primary mirrors its WAL and snapshots to: it recovers from the bucket,
  applies the primary's new WAL chunks every `DXMT_REPLICA_POLL_MS` (default 1,000), serves
  every read, and refuses every write with `403 READ_ONLY_REPLICA`. Nothing on a replica writes
  the bucket -- uploads and deletes are refused in the storage layer -- and its object reads
  skip the caches, which would otherwise serve another process's MANIFEST stale. A replica that
  falls behind a snapshot that pruned chunks it had not applied reports not ready until
  restarted. `/metrics`: `dxmt_replica_applied_next_sequence`, `dxmt_replica_stale`.
- **Query admission.** At most `DXMT_MAX_CONCURRENT_QUERIES` searches run at once (default
  twice the cores), `DXMT_QUERY_QUEUE` more wait up to `DXMT_QUERY_QUEUE_TIMEOUT_MS`, and the
  rest are answered `429 TOO_MANY_REQUESTS` with `Retry-After: 1` instead of slowing every query
  down together. Reads only; `/metrics` exports `dxmt_queries_running`, `_limit`, `_waiting`
  and `dxmt_queries_rejected_total`.
- **A disk floor.** Writes are refused (503) while the data directory's disk has less free
  space than `DXMT_MIN_FREE_DISK_MB` (default a twentieth of the disk, 256 MB to 10 GB), so a
  full disk cannot fail the WAL mid-write; deletes and snapshots still run.

- **Documents: a long text in, overlapping chunks out.** `PUT /tables/{t}/documents/{doc_id}`
  splits a text on paragraph and sentence boundaries into chunks (500 characters, 50 shared, by
  default; `chunk_size`, `chunk_overlap`), stores each as a record `{doc_id}#{n}` carrying
  `doc_id`, `chunk_index`, `chunk_count` and the caller's `metadata`, and embeds it with the
  table's model; writing the id again replaces the document and deletes the chunks it no longer
  has. `GET` returns the chunks in order, `DELETE` removes them all. A record has one vector,
  and a long text used to be represented by its beginning alone: on a 20,530-character note of
  40 encyclopedia paragraphs (52 chunks, bge-m3), questions about its second half now find the
  chunk holding the answer first 98% of the time with hybrid search, 91% by vector alone, and
  in the top three every time. In the Python and JavaScript SDKs as
  `put_document`/`get_document`/`delete_document` (`putDocument`, ...), and over MCP as
  `add_document`/`delete_document`.
- **The MCP `hybrid_search` tool fuses text and vectors.** It ran a plain vector search under
  the name and answered text alone with an empty page; a table `create_table` made with an
  embedding model now gets keyword search over its text, as `kb` does.

- **A first run needs no configuration to store and search text.** `serve` prints where
  everything is (API, console, data directory, the embedding model in use) on stdout, with the
  two requests that store and find a note; it printed nothing at all before unless `RUST_LOG`
  was set, and now logs warnings by default. A default embedding model comes from
  `DXMT_EMBEDDING_MODEL` (with `DXMT_EMBEDDING_BASE_URL`, `DXMT_EMBEDDING_API_KEY_ENV`), or
  from a local ollama, preferring a multilingual model and warning when it can only find an
  English-only one; a table created with neither a dimension nor a model takes it. The `kb`
  table (`DXMT_KB_TABLE`) is made by its first write or search, with the default model and
  Chinese-aware full-text search, so `/search/hybrid` has both halves.


- **A clean restart keeps the payload and sparse indexes.** They used to be rebuilt after
  every restart -- 23-27 s of every core at 10M rows. A clean shutdown now writes them to
  `DXMT_WAL_DIR/derived/`, with the record store's row and live counts; the next start reads
  them back if both still match, before serving anything, and deletes the file whether or not
  it was used. A crash, or any write in between, means a rebuild as before. At 1M rows:
  read back in 0.3 s, against 2.5 s to rebuild; 68 MB on disk, 0.2 s to write.
- **Scheduled backups and a restore command.** `DXMT_BACKUP_INTERVAL_SECS` (off by default)
  writes a backup into `DXMT_BACKUP_ROOT/auto-<UTC time>` on a timer and keeps the newest
  `DXMT_BACKUP_KEEP` (default 7); backups taken by hand are never pruned. `vectordb restore
  <dir>` restores one with the server stopped: it checks the backup against its manifest,
  refuses while the store is open, moves the store's previous contents aside instead of
  deleting them, and opens the result to list what came back. `/metrics` has
  `dxmt_last_backup_timestamp_seconds` and `dxmt_backup_failures_total`.

- **Windows.** The server builds natively with MSVC and the whole suite passes there (82 test
  binaries, 767 tests; the four that measure Unix-only behaviour -- hole punching, permission
  bits, symlinks -- are compiled out). jemalloc is not used on Windows. Not yet a release
  target: see `docs/SUPPORT_MATRIX.md`.
- **One process per store.** Opening a store takes an exclusive lock on `{DXMT_WAL_DIR}/LOCK`;
  a second `serve`, or `vectordb mcp` beside a running server, is refused with the directory
  it found in use, instead of both appending to one log. The OS releases the lock however the
  process exits.
- **A storage fixture written by 1.3.2** (`tests/compat/fixtures/1.3.2/`), which every later
  build has to open, write to and recover after a crash. The only fixture until now was
  0.2.0's, so nothing checked that a release could read what 1.x wrote. The generator now also
  writes a table with the 1.x table settings -- a payload-index policy, fast-scan off, a rerank
  provider and a recall target -- and alters it after the snapshot, so the log has to replay
  those fields; the test checks each one, and that a filter on the unindexed field is refused.

### Changed

- **Hybrid search fuses with `weighted_sum` at `vector_weight` 0.6 when `fusion_method` is
  omitted** (it was `dbsf` with equal weights). Measured with `eval/fusion_sweep.py` over three
  sets, bge-m3 against BM25: mean MRR@10 0.877 against 0.873, and on DuReader-retrieval, whose
  questions are phrased as people search, 0.971 against 0.958 and Recall@10 0.962 against
  0.938 -- equal-weight `dbsf` there was below the vector side alone. A request carrying a
  sparse vector still defaults to `dbsf`, which normalises all three lists. The default
  `vector_weight` for `weighted_sum` and `reciprocal_rank_fusion` is 0.6 (was 0.7).

### Fixed

- **A snapshot that did not reach object storage no longer prunes it.** After a snapshot the
  local log was trimmed and the bucket's WAL chunks below it deleted whether or not the
  snapshot had been uploaded, so a failed upload left a bucket holding neither: a machine
  recovering from it came back without the table. The chunks are pruned only once the bucket
  holds the snapshot, and until then the local log keeps the entries the bucket lacks.
- **A WAL upload to object storage that failed is retried, in order.** Each write uploaded its
  own entries and, on failure, logged a warning and moved on; nothing uploaded them later, so
  the bucket kept a hole the manifest's own check could not see, and a machine recovering from
  object storage alone came back without those writes. The next write -- or the next start --
  now uploads everything from the first entry the bucket lacks, so its chunks stay contiguous.
  `/metrics` exports `dxmt_wal_mirrored_next_sequence`; its distance to
  `dxmt_wal_next_sequence` is what losing the machine would lose.
- **PQ parameters a one-byte code cannot hold are refused.** `quantization.num_centroids`
  above 256 wrapped centroid indices silently and degraded search; it is a 400 at creation,
  as is `num_subvectors: 0`.
- **A slow embedding model no longer fails a large write.** A batch the provider could not
  embed within the timeout was sent again, three times, and the write failed: bge-m3 on a
  CPU took over 60 s for 32 paragraphs. A batch that times out is now split in half until it
  fits, and the size that fit is used for that model from then on; one text too slow on its own
  fails with a message naming `DXMT_EMBEDDING_TIMEOUT_SECS`, the new setting for the timeout
  (default 60). Measured: 128 CMRC paragraphs in one request against bge-m3 on 8 CPU cores with
  a 20 s timeout went in, the batch settling from 64 to 16.
- **A question with punctuation is searched, not refused.** `/search/text` parsed the query
  strictly as Tantivy syntax, so prose with an apostrophe (`Crohn's Disease`) or a colon read
  as a field (`proteins : UreD`) got a 400 -- 7 of BEIR SciFact's 300 questions -- and a hybrid
  search carrying one dropped its text half without a word. A query that does not parse is now
  parsed leniently; real syntax still parses strictly first.
- **A Chinese question finds its paragraph by keyword.** One unspaced Chinese term that jieba
  split into several words became a phrase query, which a question's words almost never
  match: on CMRC2018 the paragraph was in the top ten for 1.8% of questions, now 100%.
- **Keyword search over Chinese text no longer crashes the server.** The snippet was cut at a
  byte offset inside a three-byte character, and the release build aborts on a panic.
- **A text longer than the embedding model takes is embedded from its beginning** instead of
  failing its whole batch; the record keeps its full text.
- **A replayed run the graph refused left its records one version ahead.** The batched
  replay of 60d76ab sends a run of logged records through the insert path; when the graph
  refused the run -- an id still in the snapshot's graph whose row the row files had deleted --
  the record map had already taken it, bumping every record's version, and the one-at-a-time
  retry checked versions again and kept the bump. Found by the stateful walk (`r2 version 10
  != 9` after an update and a SIGKILL); the retry no longer re-checks.

- **A text-only hybrid search ran only the keyword half** on a table with a model:
  `/search` embeds `text`, `/search/hybrid` did not, so a question sharing no word with the
  note that answers it came back empty. It embeds the text now when there is no `vector`.

- **Ingest into a quantized table spent a third of its CPU in `memcpy`.** Decoding a PQ code
  copies one centroid per subspace, and the copy for a four-float subspace -- the width of
  every 768-, 1024- and 1536-dimension table -- was a call to `memcpy` each time: 192 calls of
  16 bytes per decode. Widths 1-4, 6, 8 and 16 now copy as fixed-size arrays. Steady ingest
  into a quantized 768-d table, over HTTP: 1,905 -> 2,448 rows/s with one writer, 1,615 ->
  2,369 with four.
- **The first filtered query after a restart could stop the whole server.** It built the
  payload and sparse indexes inside the table map's read lock -- 23 s at 10M rows -- and a write
  waiting for that lock queued every later request to every table behind it. It reached that
  path whenever a write landed during the background warm-up, because a write voided the
  warm-up: on a server taking writes, after every restart. Writes during the build are now
  caught up into it, and every read that needs the indexes builds them outside the lock.
- **Recovery replayed the log one record at a time, about 80x slower than it needs to.** Each
  logged write was its own trip through the insert path and two fsynced record-store appends:
  6 ms an entry, so 20,000 entries took 120 s, and the tail five minutes of ingest at 3,000
  rows/s leaves behind would have taken over an hour to come back from a crash. Consecutive
  writes to one table are now replayed in runs of up to 1,000: 1.5 s for the same 20,000.
  Table import goes the same way.
- **A crash between a record's data and its log line corrupted every later record of that
  table.** Data is written first; extra bytes past the last row the log names were left in
  place, the next append landed after them while its log line placed it before, and every row
  from there decoded another row's bytes. A backup copying the two files while a write went in
  did the same. They are now cut on open. Found on Windows, where the slower fsync made a
  backup under writes lose records on every run.
- **A vector file with a partial last row refused to open**, so a crash or a full disk midway
  through an append left a server that would not start. The partial row is cut instead.
- **On Windows:** an alias could not be set ("Access is denied": the alias file was synced
  through a read-only handle); data went under the working directory when `HOME` was unset,
  which a plain PowerShell does not set (`USERPROFILE` is used now); and the free-memory floor
  never refused a write, there or on macOS, because only `/proc/meminfo` was read.
- **`/dashboard/../…` answered 200 with the app page.** Nothing outside the bundle was ever
  readable, but a path that climbs is now a 404. Its test had never sent one: it used curl,
  which removes `..` before sending.

- **A backup taken while the periodic snapshot ran could restore without a delete, an update
  or a table created in between.** `backup_to` took its snapshot and then copied, with nothing
  stopping another snapshot from compacting the log in the meantime, so the backup's `wal.log`
  could start after its snapshot's replay point; recovery replayed from that point without
  noticing the log began later. A backup now holds the snapshot lock until its last file is
  copied, and **recovery refuses to start on a log that begins after the snapshot's replay
  point**, naming both files, rather than coming up without the entries in between.
- **A write whose WAL entry never landed is no longer half there after a restart.** A write
  reaches the row file before its log entry; when the entry is lost -- a crash in between, a
  full disk tearing the frame, a backup copying the two at different moments -- `GET` and the
  exact scan returned the row's state while the graph held the log's. Restarts with no
  snapshot never indexed such an insert or update at all, and a lost delete stayed in the
  graph. Recovery now reconciles each table's graph with its rows in both directions.
- **An update lost from the log after a snapshot no longer leaves the old vector in the
  graph.** The snapshot's graph came back with the previous vector, the row file had the
  new one, and the record counts agreed, so nothing noticed: `GET` returned the new vector and
  an ordinary search found the record by the old one. A snapshot now records, per table, the
  row below which its graph is current (writes still between their row and the graph are
  accounted for), and recovery re-indexes every record at or above it that replay did not.
  Snapshots written before this carry no such row and are not re-checked; the next snapshot
  writes one.

- **`DXMT_HOST` and `DXMT_PORT` set the address `serve` binds.** The README and
  `.env.example` listed both, and no release read them: `serve` took `--host` and `--port`
  alone, whose defaults are the same values, so a setting looked like it worked until it was
  anything but the default. A flag still wins over the variable. `serve` also reads `.env`
  before the flags are parsed, so `DXMT_GRPC_PORT` there now applies too. The legacy
  `DB_HOST` and `DB_PORT` are *not* accepted: no release read them either, and they are what
  a compose file calls its Postgres.
- **`DXMT_WORKER_THREADS` sizes the runtime.** It was parsed and sized nothing; the runtime
  always ran one worker per core. `0` is refused at startup.
- **`.env.example` lists only variables the server reads**, in English, with everything
  commented out. Copied as it was, it stopped a default build from starting: it set
  `DXMT_OBJECT_STORAGE_PROVIDER=s3`, which needs the `s3-storage` feature. It also listed
  eight variables that did nothing (`DXMT_COMPRESSION`, `DXMT_LOG_LEVEL`,
  `DXMT_LOG_FORMAT`, `DXMT_MAX_CONNECTIONS`, `DXMT_QUERY_TIMEOUT`,
  `DXMT_DEFAULT_DIMENSION`/`_DISTANCE_METRIC`, `DXMT_MAX_RECORDS_PER_TABLE`). Logging is
  `RUST_LOG`.

- **A start no longer warns about a file nobody wrote.** Every start logged "Failed to load
  config file, using default configuration", twice, and every stop of a server without a WAL
  logged "Shutdown snapshot failed". Neither was a failure.

### Removed

- **Reading `config.toml` from the working directory.** It was never documented. When the file
  parsed, it replaced every environment setting at once, and a file with that name was used
  even when nobody meant it for this server (a Hugo site has one). When it did not parse, the
  server fell back to the environment with only that warning. Configuration is the environment
  and `.env`, as the README says. The `config` and `toml` crates, which only that path used,
  are gone from the dependencies.
- **`DXMT_COMPRESSION`**, and the `compression` field of `config::SsdConfig` it filled. It
  selected nothing; the WAL and row files are not compressed.

## [1.3.2] - 2026-10-07

The pre-publication pass: the Docker images, the README and the CI toolchain. Nothing changes
for a default build, on the wire or on disk. Every package carries this version. 1.3.1 was
tagged before these changes were merged, so they ship here rather than in it.

### Fixed

- **The Docker images build.** `docker/Dockerfile` and `Dockerfile.multiarch` cached
  dependencies by building against placeholder sources that stubbed two of the crate's five
  binaries, so cargo refused the manifest and `make docker-build` and docker-compose failed.
  Both now build the real crate, include the web console's bundle (a container served none),
  and build on the same Debian release they run on.
- **The README's request examples work as written.** Every `curl -d` example lacked
  `Content-Type: application/json`, which the server requires; the Quick Start now also
  creates, fills and searches a table rather than stopping at `/health`.
  `examples/filter_example.md` and `examples/nginx-jwt.conf` route the paths the server
  serves.
- **The `config::layout` permission tests pass under root.** They checked that a read-only
  directory is refused, which root's writes ignore -- containers, and the Dockerfile's `test`
  stage. They now ask the directory whether its bits bind the process and return early when
  they do not.

### Changed

- **The Rust toolchain is pinned to 1.99.0** (`vectordb/rust-toolchain.toml` and every
  workflow step) instead of whatever `stable` was that day, so a new Rust release can no
  longer turn CI red on code nobody changed -- which is how 1.0.0 to 1.3.0 came to be
  released on a red CI. Moving to a newer Rust is a commit of its own.
- **Documentation brought up to date with 1.1 to 1.3:** the default build (full-text search,
  Swagger and the console are on), index selection (HNSW-PQ at every size), the background
  index warm-up after a restart, the downgrade notes, and seven configuration variables that
  were read but undocumented. `DXMT_COMPRESSION` and `DXMT_WORKER_THREADS` are documented as
  what they are: parsed and unused. `SECURITY.md` describes the optional API key and the
  current release line.

### Removed

- About thirty one-off and dead scripts from `scripts/`, `docs/autogen_integration.md` (for
  a module that does not exist), an unused `static/swagger-ui.html`, an internal progress
  report, and references to files that are not in the repository.

## [1.3.1] - 2026-10-07

A build fix for 1.3.0; nothing changes for a default build, on the wire or on disk.

### Fixed

- **1.3.0 did not compile with `--features api-grpc` or without `text-bm25`.** The reranker
  added a field to the protobuf `SearchHit` literal that the generated type does not have,
  and left the inner half of the split full-text search without its `text-bm25` gate. The
  default build, and every SDK, were unaffected.
- **Clean under Rust 1.99's clippy.** CI installs the newest stable on every run, and 1.99
  (2026-09-28) added two lints the tree failed: `double_must_use`, raised on code
  `async_trait` and tonic generate (allowed on those items), and `chunks_exact` with a
  constant size (now `as_chunks`).
- **`langchain-dxmt` could not be installed next to the current SDK.** It required
  `dxmt-python-sdk>=0.1.0,<1.0.0` while the SDK has been 1.x since 1.0.0, so `pip install`
  failed to resolve wherever the current SDK was present. Both adapters now require
  `>=1.3.0,<2.0.0`. Three of its test files also failed to collect, importing a helper by a
  path pytest does not put on `sys.path`; with that fixed, 122 tests run where only the unit
  tests had.
- **CI runs the tests again.** Its clippy step had failed on every push to main since 1.0.0,
  under each new stable's lints, and every job that depends on it -- the integration tests,
  the SDK live suites, the web console, the full-text API tests -- was skipped. 1.0.0 to
  1.3.0 were released on that red. Once clippy passed, those jobs ran for the first time in
  weeks and failed: they installed `pytest httpx` but the OpenAPI checks import `jsonschema`
  (now `tests/api/requirements.txt`), and the LangChain step hit the dependency above. Both
  are fixed here, and the run on the commit before this release is the first fully green one
  since 1.0.0.

## [1.3.0] - 2026-10-07

Every package carries this version. The HTTP API gains `rerank` on `POST /tables`,
`PATCH /tables/{t}` and the three search routes, and `rerank_score` on a hit; nothing existing
changes shape. **Upgrading reads everything 1.2 wrote; downgrading does not.** The WAL logs
a table's config and every `PATCH` as a MessagePack array, which now has one more element, so
a 1.2 binary refuses a log in which 1.3 created or altered any table, `rerank` set or not:
take a snapshot (`POST /snapshot`) before going back, as for 1.1. Found while preparing this
release, not after it: `rerank` had first been added in the middle of the alter request, where
a 1.2 log's `recall_target` would have decoded into it and stopped the replay;
`alter_request_logged_by_1_2_replays` pins the order.

### Added

- **Reranking.** A table may name a cross-encoder provider -- `rerank` on `POST /tables` and
  `PATCH /tables/{t}`: `model`, `base_url`, `format` (`cohere` or `tei`), `api_key_env`,
  `field`, `top_n` -- and `/search`, `/search/text` and `/search/hybrid` take
  `rerank: {top_n?, field?}` to have the page re-scored by it. Each hit gains `rerank_score`;
  `score` stays. A request with no query text, or against a table with no provider, is a 400;
  a provider that will not answer fails the request rather than returning the page unranked.
  The Python and JavaScript clients take `rerank=` on `create_table` and on every search
  method. Nothing existing changes shape.

### Fixed

- **The Python SDK's examples say when they did not run.** `memory_optimization_demo.py`
  called `batch_insert`, which the client does not have, and caught the error in all three
  scenarios; it is rewritten to report the index each setting builds, without the recall and
  memory figures it printed and nobody had measured. An example missing a library or a key now
  exits 77 naming it (it printed a line and exited 0), and the handlers that turned a failed
  full-text, hybrid or filtered search into a printed warning are gone. Every example reads
  `DXMT_HOST`/`DXMT_PORT`/`DXMT_API_KEY`; `pip install 'dxmt[examples]'` installs what they
  import; `tests/test_examples.py` runs all of them against a live server, and CI runs it.

## [1.2.0] - 2026-10-06

Every package carries this version. The HTTP API gains one request field
(`quantization.fast_scan`), one stats field (`fast_scan`) and one metric family
(`dxmt_filter_path_seconds_total`), and loses nothing. **One compatibility note:** a WAL
written by 1.2 holds entries (`UpsertRecordInRowFile`) that a 1.1 binary cannot read, so
take a snapshot (`POST /snapshot`) before downgrading; a 1.2 binary reads every earlier log
and index file, and the new `.fs` sidecar beside an index is ignored by older binaries.
The filtered-search planner, the WAL's row references and the background warm-up are two
to three days old and measured on one machine; `docs/SUPPORT_MATRIX.md` carries the numbers.

### Added

- **Fast-scan codes for filtered search.** A quantized table keeps 4-bit PQ codes in
  blocks of 32 beside its 8-bit ones (same bytes per vector; a sidecar `.fs` file next to
  the index), and a filter matching more than one row in 64 is scored from them 32 codes
  at a time -- 4.6 ns a code against 69 for the 8-bit gather. On the big-ann filtered
  track at 10M rows the 100K-500K match band went from 27 ms to 17 ms p50 (p95 44 to 23) at
  `normal`, recall unchanged, and the 16-thread throughput cost 9%. Tables quantized before this get the codes from
  `ensure_fast_scan` (built from the row file; 45 s at 10M); `GET /tables/{t}/stats`
  reports `fast_scan`. `quantization.fast_scan: false` at creation leaves the codes out,
  for a table that is never filtered or a server whose cores are all spoken for.

- **The payload index is built in the background after a restart, on every core.** It was
  built on the first filtered query, one row at a time: 33 s on a 10M-row table, paid by
  whichever caller came first. The server now warms each table's payload and sparse indexes
  and its node map as soon as recovery is done, decoding rows in parallel under short read
  locks and installing the result only if no write reached the table meanwhile; the lazy
  build remains for a table that was written to first, and decodes in parallel too.

### Changed

- **Filtered search is planned by the real match count and runs on bitmaps.** A filter's
  matching records are a roaring bitmap of payload ordinals (string postings already were;
  a number held by 1,024 or more records keeps one too), resolved to graph node positions
  through a per-table map that is filled once per node rather than per query, and the
  planner chooses between scoring the matched nodes' codes and walking the graph with the
  filter gating its results by the measured match count. The previous planner turned
  every matching id into an owned string and hashed it twice per query, and its in-graph
  branch was unreachable on the default index. On the big-ann filtered track at 2M rows,
  `normal` p50 went from 6.4 ms to 1.8 ms with recall unchanged at 1.0; `GET /metrics`
  now also reports seconds per planner branch (`dxmt_filter_path_seconds_total`).

### Fixed

- **A hybrid query with `text` against a table that declared no text fields is refused**
  (`FULL_TEXT_NOT_ENABLED`, as `/search/text` already did) instead of running the dense half
  and returning it shaped like a fusion. The Python SDK's live suite had asserted this and
  failed on every default build since `text-bm25` became default in 1.1.0.
- **A numeric payload field no longer slows every insert down as the table grows.** The
  payload index's numeric half folded its unsorted tail into one sorted array every 4,096
  entries by re-sorting the whole array, so the cost of each fold grew with the table: on
  the big-ann filtered track (10.8 integer tags a row) the load ran at 2,732 rows/s over
  the first million rows and 1,000 over the second. It is three levels now (an unsorted
  tail, a sorted run it is linearly merged into, and the main array that run is merged into
  once it reaches a 64th of it); the same load holds 4,400 rows/s through two million.

### Changed

- **The WAL no longer carries a vector the row file already holds.** An insert or update on
  a table with a row file logs the record and the row its vector is at
  (`UpsertRecordInRowFile`), and replay reads the row. A table whose vectors are in memory,
  and every table when the log is mirrored to object storage, still log the vector. Logs
  written by earlier versions replay unchanged; a log written by this version is not
  readable by an earlier binary. Three replay defects that the copy in the log was hiding
  are fixed with it: a replayed `CREATE TABLE` wiped the table's row files; a replayed
  `DROP TABLE` removed the vector file that the entries after it (the same name created
  again) point into; and a replayed record was re-stored without its row, which on a
  disk-backed table kept a second, inline copy of every replayed vector in the records
  file. Replay also no longer appends a replayed vector to the row file again: an entry that
  names its row is indexed from that row, so a restart leaves the file the size it was.
- **A quantized graph builds 1.8-1.9x faster.** Node-to-node distances during neighbour
  selection and pruning came from a table of every centroid pair per subspace (6.5 MB at
  25 dimensions, 50 MB at 768) whose every lookup missed cache; both codes are decoded and
  compared with one SIMD pass instead, and the fixed node of a pruning loop is decoded once.
  The exact graph's euclidean distance is resolved once at startup rather than dispatched
  per call. 200K rows, 16 threads: 25-d storage path 8,669 -> 16,572 rows/s, 100-d
  4,086 -> 7,237; the 100-d graph alone now matches hnswlib (8,993 against 8,531).
- **PQ codebook training is k-means with linear seeding** instead of a quadratic one, the
  training sample is capped at 128 points per centroid, and a one-dimension subspace
  assigns by binary search. The low-dimension storage path spent most of its CPU there.

## [1.1.0] - 2026-10-03

Every package carries this version. The HTTP API gains fields and routes and loses none;
the Python, JavaScript, Rust and Go clients each had defects found by running every method
against a live server, and `docs/SUPPORT_MATRIX.md` was re-measured where it was stale.

### Added

- **Recall calibration.** Every table has a `recall_target` (default 0.95; `PATCH /tables/{t}`
  sets it, `0` opts out). From 20,000 rows, and again whenever a table doubles, the index
  sweep makes queries from pairs of the table's own rows, scores them with the exact scan,
  and raises the `normal` search budget by doublings until recall@10 meets the target.
  `POST /tables/{t}/recall/calibrate` runs it on demand; `GET /tables/{t}/stats` reports the
  budget in use, the recall measured and the curve. On glove-100 the graph measures the same
  recall per budget as hnswlib at the same parameters, so the budget is what a target buys.
- **Reachability repair.** The same sweep gives every graph node that nothing points at an
  in-edge back (`POST /tables/{t}/index/repair` runs it now). On GIST-960 at 100K rows one
  pass repaired 1,563 nodes in 80 ms and took stored vectors not returned by their own query
  from 2.25% to 0.45%, recall unchanged.
- **`on_error: "skip"` on `POST /tables/{t}/records`** writes the valid rows and lists the
  rest; the default refusal now lists every invalid row (`error.details.failed`), not the
  first.
- **Full-text search in the default build** (`text-bm25`), and a `"analyzer": "chinese"`
  text field is now actually tokenized with jieba -- the setting was parsed and read by
  nothing, so Chinese full-text search found a sentence only by the whole sentence.
- A per-stage insert timer (`insert stages` log line), an in-process build profiler
  (`tests/recall/build_profile_test.rs`), and live tests in CI for every route the Python
  and JavaScript clients wrap, including export/import.

### Changed

- **PQ code size follows the table's width**: one dimension per subspace below 64, two below
  256, four from there. glove-25 `normal` recall@10 0.76 -> 1.00 for +0.7 ms; glove-100 and
  glove-200 +3 points. Tables built before keep their layout.
- **`search_mode: "exact"` is allowed to 10M rows** (was 1M): 1M x 960 measures 300 ms with
  the row file in page cache.
- **The graph builds 1.7x faster** at 100 dimensions (3,365 -> 5,841 rows/s in-process,
  hnswlib 7,963 on the same rows): the next neighbour's vector is prefetched during the walk
  and the prune, and parallel planning starts at 1,024 rows instead of 10,000, with the rows
  of a chunk linked to their nearest siblings afterwards. At 960 dimensions the server was
  already at parity. The AVX2 distance kernels use four FMA accumulators.
- Every description that said "horizontally scalable", "98% cost savings" or `serve --simple`
  now says what the server is: a single-node database. The matrix's filter-latency cliff
  (160-800 ms at 12-50% selectivity) was re-measured at 4-8 ms and the entry corrected.

### Fixed

- **A row the WAL never saw is indexed on restart.** A write reaches the row file before its
  log entry; a crash in that window, or a backup whose `records/` copy was taken after a write
  and whose `wal.log` copy before it, left a record `GET` returned and no ordinary search found.
  A restore whose index holds fewer rows than the row files now inserts the difference.
- **Sparse search answered nothing after deletes of records without a sparse vector**: the
  index's count went down on every delete whether or not the id had a posting, and reached
  zero, at which point `search` returned an empty page without reading a posting.
- JavaScript client: a record read back could not be written back (`created_at` etc. were
  sent and refused, and `upload()` did not throw); error codes were replaced by the class's;
  the LangChain store ignored `Document.id`, could not delete by filter, and sent filters the
  server refuses. Rust and Go clients: `dot`/`manhattan` could not create a table;
  `list_tables` was cut at the server's default page. Python client: the live suite's hybrid
  tests had been failing on CI's default build since 2026-09-28; four examples could not run.
- Python and JavaScript READMEs, `SUPPORT_MATRIX.md` and the Python SDK's examples directory
  no longer describe methods, defaults, files or claims that do not exist.

## [1.0.1] - 2026-10-02

### Fixed

- **A table's embedding model survives a restart.** In 1.0.0, every clean restart turned a
  table that embeds text into one that refuses it: the rows came back intact, and `/search` or
  an insert carrying `text` answered `INVALID_INPUT: table has no embedding model`. `main`
  snapshots on SIGTERM, and `restore_table` rebuilt each table from the snapshot head copying
  every field but `embedding`, which it set to `None`. WAL replay carried the config and was
  never affected -- which is why `the_model_survives_a_reopen`, whose comment said the config
  "is in the snapshot and in the WAL's table record", stayed green: it never took a snapshot.
  `import_table` and `branch_table` had the same `embedding: None`, so an exported-then-imported
  table and a branch lost the model too. All three now copy it; three tests cover the three
  paths, and each failed before the change. Found by restarting a 1M-row table while measuring
  DXMT against vectorgraph-rs's benchmark corpus.

## [1.0.0] - 2026-09-29

First public release. Everything below was developed before any of it was published, so this
entry is the whole history rather than a delta against a previous version.

Every package in the repository carries this version: the server crate, the web console, and
the six clients (Python, JavaScript, Rust, Go, LangChain, LlamaIndex). They were at four
different numbers -- the JavaScript client was at `2.0.0`, ahead of the server -- and none had
been published, so unifying them costs nothing now and removes the question of which client
version goes with which server.

`vectordb --version` exists as of this release. It answered `error: unexpected argument
'--version' found` until the day of the tag, and it is the first thing anyone runs against a
binary they were handed.

### Added

- **The LlamaIndex examples run without an API key.** All eight opened with
  `OpenAIEmbedding()`, so the first thing anyone evaluating DXMT through LlamaIndex met was a
  missing key. `examples/embedding.py` picks OpenAI when `OPENAI_API_KEY` is set,
  sentence-transformers when it is installed, and otherwise a hashed bag of words, printing
  which -- and the table's dimension follows the model instead of a hardcoded 1536. The three
  that need no LLM run end to end; the rest still say so, in one line, for the generation step
  alone. They also clear their table first, so a second run no longer shows every document
  twice.

- **The JS SDK has an eslint configuration**, so `npm run lint` runs instead of failing with
  "looked for configuration files and found none". Seven source files, clean.

- **A snapshot no longer stops every write for most of its duration.** Measured over HTTP with
  four tables of 40,000 rows and writers running throughout: a snapshot took 1.55 s and the
  worst write waited 1.28 s of it — on every table, whichever one was being serialized, while
  searches were never affected. The roadmap blamed the table map's read lock; it was the WAL
  mutex, held by the compaction at the end.

  Compaction drops the prefix a verified snapshot covers. It did that by decoding **every**
  frame in the log — 160,000 entries — to re-encode the twenty that arrived while the snapshot
  ran, because `replay_from` deliberately decodes frames below its start so a corrupt one
  cannot be skipped silently. That reasoning is for recovery; the frames being dropped here are
  covered by a snapshot that has just been read back and verified, which is the same argument
  the existing "nothing survives" shortcut already rests on. The surviving tail is now found by
  reading frame *headers* and copied verbatim, with the decoding path kept as the fallback for
  a header that does not make sense.

  The first version of that scan read 13 bytes and seeked per frame, which is 320,000 syscalls
  on this log and cost 1.23 s of its own. Buffered, with `seek_relative`, the same measurement
  is: snapshot **0.32 s**, worst write **90 ms**. At twice the data, 1.00 s and 313 ms.

- **The index sweep climbs the size ladder and does not come back down.** A rebuild is O(n)
  whichever direction it goes, and a growing table sampled at a low point paid it twice: a soak
  watched one table go HNSW → brute force at 1,000 rows and brute force → HNSW at 38,400 ten
  seconds later, two full rebuilds for a table that only ever got bigger. The periodic
  re-evaluation now refuses a demotion — an oversized graph on a table that shrank costs memory
  nobody notices, while flapping across a threshold costs the table twice per crossing.
  `POST /tables/{t}/index/rebuild` still re-selects from scratch for an operator who wants the
  smaller index back. Both halves are tested: the sweep must still *upgrade*, or a test of the
  refusal alone would pass with the sweep doing nothing at all.

- **A batch into a small table no longer blocks every search on it.** Below 10,000 rows a
  batch is inserted one row at a time, and the whole batch went in under a single write lock:
  measured over HTTP, a 5,000-row insert into a small table blocked searches on that table for
  the full 8.0 s it took. `GET /tables` and searches on other tables were unaffected, so the
  "bulk writes make the database unreadable" note in the roadmap was half stale — it is one
  table's index lock, not the map.

  That path now takes the lock per 128 rows and yields between chunks. Chunking alone was not
  enough (8.0 s to 3.8 s): tokio's `RwLock` is write-preferring, so the next chunk queues ahead
  of the waiting reader and starves it. With the yield, the same measurement is **302 ms**, and
  the insert takes the same time it did.

  The parallel path, which a table over 10,000 rows uses, is unchanged and still has a shorter
  version of the problem: 1.86 s for a 5,000-row batch, 478 ms if the same rows are sent as
  five batches of 1,000 — an 8% ingest cost for a 4x better worst case. README's limitations
  section now carries both numbers.

- **The LangChain examples run without an API key**, the way the LlamaIndex ones already did.
  Fifteen of the seventeen opened with `OpenAIEmbeddings()`, so the first thing anyone
  evaluating DXMT through LangChain met was a missing key — for examples whose subject is a
  vector database, not an embedding provider. `examples/embedding.py` picks OpenAI when
  `OPENAI_API_KEY` is set, sentence-transformers when it is installed, and otherwise a hashed
  bag of words, printing which; the table's dimension follows the model instead of a hardcoded
  1536. Seven now run end to end with no key; the rest do their retrieval half and print one
  line where the generation step would be, and their optional imports moved inside that branch,
  because an example that dies at `import` cannot tell you what it needed.

  Running all seventeen turned up three things that had nothing to do with keys. The
  configuration example constructed a store against `my-dxmt-server.example.com` to *illustrate*
  a setting — but the constructor creates the table, so it really connected, and failed with a
  DNS error under a line reading "✓ Would connect to". It listed `dot_product` among the
  distance metrics, which the server answers `INVALID_DISTANCE_METRIC` to; the name is `dot`.
  And it imported `langchain_dxmt.client`, a module that has never existed in that package.
  The search engine example crashed with `EOFError` when run without a terminal, at the end of
  a run that had otherwise worked.

- **`total` on a search result says what it is.** It is the number of hits in the page, which is
  what the code always returned, and the name invites the other reading: a caller taking it for
  "how many records match this filter" gets a number that is right only when the page happens to
  hold all of them. `records/count` answers that question; the field now says so, and
  `tests/api/test_search.py` pins which of the two it is.

- **The support matrix says what each client covers.** Six clients ship from this repository
  and they are not equivalent: Python and JavaScript follow the whole HTTP surface, Rust and Go
  wrap the table lifecycle, the record routes and `search`, and LangChain and LlamaIndex go
  through the Python one. Nothing said so, and the contract test's own note still said the
  answer to "who breaks if this route is renamed" was three callers when it was five. Picking
  the Go client for something it does not wrap is one request of your own, not a missing server
  feature — but that is only obvious once someone writes it down.

- **Recall of the parallel build is now in the gate.** Every storage-level recall assertion
  here ran on at most 6,000 rows, and the batch planner that production tables are built with
  only engages above 10,000 — so the build path the server actually uses had no end-to-end
  accuracy guard. `tests/parallel_build_recall_test.rs` loads 15,000 rows in batches, with the
  records and vectors on disk, and measures recall@10 against this server's own
  `search_mode: "exact"`: no external dataset, no published ground truth, 3.4 seconds.

  It uses a deliberately lean graph (`m: 4`, `ef_construction: 16`), because the shipped
  defaults answer this corpus at recall 1.000 and a test whose answer is 1.000 cannot show a
  build getting *worse* — only one going catastrophic. At m=4 it measures 0.984, with the floor
  at 0.95. It also asserts that `fast` scores below `normal` (0.780 against 0.984): if the two
  ever match, the corpus has stopped telling a weaker search from a stronger one and could not
  tell a worse graph from a better one either, which is a failure of the test rather than a
  pass of the code.

- **Diversity reranking (`mmr`) on a search, done by the server.** A chunked corpus answers a
  query with several near-copies of one passage; they really are the nearest neighbours, and
  together they say one thing. `mmr: {"lambda_mult": 0.5, "fetch_k": 20}` picks each next hit by
  how near it is to the query minus how near it already is to what has been picked. 1.0 is the
  plain ranking, 0.0 ignores the query. Unlike `search/groups` it needs no field saying two
  records belong together — the vectors say it.

  **The LangChain integration was re-embedding to do this.** `max_marginal_relevance_search`
  fetched `fetch_k` documents and called `embed_documents` on their text to get vectors to
  diversify with — "not optimal but works", as the comment said. It is an embedding call per
  search, paid in latency and in money, and it diversifies over vectors that are *not* the ones
  in the index: a document whose stored vector came from different text, or from an earlier
  version of the model, was being compared against something the database had never seen. That
  method now calls the server, which has the vectors, and 71 lines of client-side numpy went
  with it.

  The cost is `fetch_k`: diversity can only come from candidates that were fetched, and with
  `fetch_k` at the page size the answer is the plain ranking — which the tests pin, because
  silently returning the undiversified page is the failure mode this would otherwise have.
  Pages of a diversified search tile, since the offset applies to the reranked order.

- **`offset` on a search.** Page two without fetching page one twice: paging meant asking for
  `2 * top_k` and throwing half of it away on the client, over the wire, with the payloads
  attached. Both SDKs take it.

  It is not free, and the field says so: the index is asked for `offset + top_k` and the prefix
  is dropped, because an approximate index cannot start in the middle of a ranking it has not
  computed. For walking a whole table, `records/list` with a cursor is the paging that stays
  flat. The offset is taken *after* the filter and the score floor — skipping rows that were
  never going to be in the answer would make page two start somewhere other than where page one
  ended — and `tests/api/test_search_offset.py` checks that the pages tile the unpaged ranking
  exactly, on the exact mode as well as the approximate one.

- **Table aliases.** `PUT /aliases/{alias}` points a name at a table, `DELETE` removes it,
  `GET /aliases` lists them, and an alias can be used anywhere a table name can. Both SDKs have
  them.

  What they are for is the swap. Rebuilding a table with different index parameters means
  building a second one and moving traffic across, and without an alias that move is a client
  deploy — every caller has to learn the new name at the same moment. With the export and
  import from the previous change it is now: export, import under a new name, top it up, move
  the alias. Moving an alias is one map write, so a reader arriving mid-swap gets one table or
  the other, never neither.

  Aliases live in `{DXMT_WAL_DIR}/aliases`, written atomically on every change, rather than in
  the WAL — they change rarely, and a WAL entry would mean the snapshot has to carry them too,
  which is the bincode manifest and the break `src/storage/snapshot_v1.rs` exists to document.
  A missing file is no aliases, which is what every database written before this has.

  An alias and a table cannot share a name in either direction, because which one a name meant
  would otherwise depend on the order of two maps; dropping a table drops the aliases pointing
  at it, or they would resolve to a name nothing holds and the error would talk about a table
  the caller never mentioned. `tests/table_alias_test.rs` walks *every* table-scoped route with
  an alias in the path: one that resolved on `search` and not on `records/count` would work
  until the one call that does not.

- **Sparse vectors are searchable.** `POST /tables/{t}/search/sparse` retrieves by sparse
  vector alone, ranked by the dot product the WAND-pruned sparse index already computed, with
  `filter`, `score_threshold`, `payload_fields` and `with_vector` behaving as they do on a dense
  search. In the default build: the sparse index is pure Rust and needs no full-text engine.

  They could be written and were indexed the whole time. The only path into that index was
  `POST /search/hybrid`, which refuses a body with no dense vector and no text — "At least one
  of 'vector' or 'text' must be provided" — and needs `text-bm25` besides. So a table of SPLADE
  vectors was stored, indexed and unreachable: accepted and ignored, which is the failure this
  repository keeps finding in its own surface.

  The JS SDK could not write them either — `Record` had no `sparse_vector` — so it gains the
  type as well as `sparseSearch`. Python gains `sparse_search`. And because the sparse index is
  derived state rebuilt on demand rather than restored, `tests/sparse_search_test.rs` checks it
  after a restart: the payload index had that exact shape once and every filter silently matched
  nothing.

- **`IsEmpty` and `IsNull`.** "Which records is this field missing from" had no expression at
  all: `NotEqual` answers a different question, because a record holding some other value
  satisfies it too, so finding the rows a pipeline failed to enrich meant paging the whole
  table and checking by hand. `{"type": "IsEmpty"}` matches a field that is absent, `null`, or
  an empty array; `{"type": "IsNull"}` matches one that is present and null. Both carry no
  `value`. Python has `Field("x").is_empty()` / `.is_null()` and the `$isempty` / `$isnull`
  aliases.

  `IsEmpty` is the complement of "has something here", so it is computed at query time out of
  the field's own postings rather than kept as a bitmap per field maintained on every write —
  that bitmap was measured at two seconds of a 10M-row rebuild when `NotIn` wanted it, and
  charging every table for an operator few use is the wrong trade. What *is* kept is a mark for
  the records whose value is null or `[]`, because those are the two cases the index otherwise
  cannot tell from "has a value": a null is filed like any other scalar and an empty array is
  marked present with no element to file. `tests/filter_agreement_test.rs` carries all four
  shapes — null, `[]`, `[null]`, absent — in memory and on disk.

- **Filters reach into a nested payload.** `meta.lang` reads the `lang` of the object under
  `meta`, at any depth, on `search`, `records/list`, `records/count` and delete-by-filter. A
  payload is arbitrary JSON and people nest it; until now only the top-level keys were
  filterable, `meta` held an object, an object matches nothing scalar, and a filter on
  `meta.lang` matched **nothing at all** — an empty page, not an error.

  A key that really contains a dot still wins over the path reading of the same string, so a
  payload written as `{"user.name": "..."}` is still filterable under the name it was written
  with. A record holding both that and `{"user": {"name": ...}}` files only the literal one, so
  the index and the record-by-record matcher cannot disagree about which of the two a filter
  means — `tests/filter_agreement_test.rs` carries that payload now, in memory and on disk.

  Paths are indexed path by path, which is what makes them answerable at all here, and they show
  up in `GET /tables/{t}/payload_index` with what each one costs. `payload_index` names match
  subtrees, so `except: ["blob"]` stops indexing everything under `blob` — otherwise the
  exclusion would miss the field that actually costs something. An array under a path behaves
  like one at the top level: any element matching is a match. An array *of objects* is not
  walked into, because `items.price` over `[{"price": 1}]` would have to mean "any element's
  price", and that deserves an operator that says so rather than a path that quietly means two
  things.

- **Per-table export and import.** `POST /tables/{t}/export` writes one table — its
  configuration and every record, vectors included — to a file on the server inside
  `DXMT_BACKUP_ROOT`, and `POST /tables/import` reads it back, here or on another machine,
  optionally under a different name. Both SDKs have it. `POST /backup` is the whole instance
  and restoring it means stopping the server and replacing its directory; moving one table had
  no answer short of paging it out through the API and writing the insert loop yourself.

  Two things it deliberately does not do. It does not restamp records: they are applied the way
  WAL replay applies them, so ids, payloads, versions and `created_at` survive and an import
  does not silently reorder `records/list`. And it does not carry the graph — that is a pure
  function of the records, the import rebuilds it, and shipping it would tie the file to an
  index format as well as to a record format. The file *is* the snapshot format, which
  `tests/compat/` already guards, so an export taken today opens on a later build.

  The first version of it wrote empty files and said so in the wrong way: it built the record
  list the way a snapshot does, and `snapshot_records` returns nothing for a disk-backed table
  on purpose, because the row files are the durable copy on that machine. It reported
  `"records": 0` rather than failing. An export now walks the store and reads each vector back
  out of the row file, and reports `records_without_vectors` — zero on a healthy table.

- **Grouped search.** `POST /tables/{t}/search/groups` takes a plain search body plus
  `group_by`, `groups` and `group_size`, and answers with groups instead of hits. A document
  chunked for retrieval takes a whole top-10 by itself — the ten nearest vectors are ten pieces
  of the same text — and grouping by its id with `group_size: 1` returns ten documents. Both
  SDKs have it (`search_groups`, `searchGroups`).

  It groups a *window*, not the table: there is no index over (field, distance), so the groups
  are filled by scoring neighbours until enough distinct values appear. The window starts at
  `groups × group_size × 4` and doubles while it is short, capped at 10,000 candidates, and the
  answer reports `candidates_fetched` — otherwise a caller cannot tell "there are only three
  documents near this query" from "the fourth one's chunks are further away than we looked".
  Records with no value under `group_by` are counted as `ungrouped` rather than dropped
  silently, and each group reports `matched`, how many of the fetched candidates fell in it.

- **Explicit payload index management.** `GET /tables/{t}/payload_index` reports what the
  filter index holds per field — distinct values, postings, estimated bytes, worst field first
  — and `PUT`/`DELETE /tables/{t}/payload_index/{field}` turn a field's index on and off;
  `payload_index` on `POST /tables` and `PATCH /tables/{t}` takes the whole policy (`only`,
  `except`). Every field stays indexed by default, which is what makes a filter fast here with
  nothing declared — Qdrant and Weaviate require the declaration — but that cost is uneven: at
  10M rows a field holding a distinct string per row took 11 s of a 32 s index rebuild against
  3.5 s for a four-valued one, and a field nobody filters on was paying it for nothing.

  A filter naming an excluded field is **refused with a 400 that names the field and the way
  back**, on every entry point — search, list, scroll, count, delete-by-filter. This engine
  answers a filter only from the index: a field with no postings yields an empty set, not a
  scan, and an empty page is indistinguishable from "nothing matched". The report does not
  build the index in order to describe it, so asking what it costs does not cost a pass over
  every record.

- **Per-table index parameters.** `hnsw` (`m`, `ef_construction`, `ef_search`) and
  `quantization` (`num_subvectors`, `num_centroids`, `quantize_at_rows`) on `POST /tables`.
  `m` was hardcoded at 16 and `ef_construction` came from an environment variable, so one
  server built a 5,000-row lookup table and a 10M-row corpus the same way — and these are the
  parameters that cannot be recovered afterwards, because searching harder does not add edges
  the build never made. They are persisted with the table and applied on rebuilds and after a
  restart, or the first re-evaluation would quietly revert a tuned table to the defaults.
  `quantize_at_rows` above a table's eventual size is how to say "keep this one exact".

  Putting them in the table also broke every snapshot on disk, which is the interesting part.
  A `tN.meta` was one bincode record, and bincode stores no field names, so two new fields in
  `Table` shifted every later byte: `tag for enum is not valid, found 16`, and a server that
  refuses to start rather than come up with an empty database. Meta files now carry a
  MessagePack head — the table, the algorithm, `has_index` — in front of the bincode records,
  so a future field costs nothing, and the records keep the compact encoding because they are
  the 30 GB. Snapshots written before this are read through
  `src/storage/snapshot_v1.rs` and rewritten in the new shape at the next snapshot.
  `tests/compat/` is what caught it and what keeps it caught.

- **`GET /tables/{t}/stats` reports `quantized`.** An HNSW-PQ index is an exact graph until it
  has enough rows to train codebooks on, and everything about recall and memory changes when
  it crosses. Which side of that a table is on was not reported anywhere, so "why did recall
  change at some size" had no answer short of reading the server log.

- **`score_threshold` on a search.** A floor on the similarity the caller sees — higher is
  nearer — not on the internal distance, which runs the other way: a threshold that silently
  meant the distance would return exactly the hits it was asked to remove. It cannot make a
  search find more; the index is still asked for `top_k` and the floor only removes, so a short
  page means that was all there was above it.

- **`payload_fields` on a search.** Return two keys instead of the whole payload. A payload is
  often mostly the chunk of text the embedding came from, so a page of a hundred hits can be a
  megabyte of JSON to deliver two fields. `with_payload` stays a boolean: making it polymorphic
  the way Qdrant does would have touched 30 call sites across 13 files including the gRPC,
  GraphQL and Python surfaces, for a projection.

- **Recall after quantization is under test.** Every recall assertion in this repository ran on
  a few thousand rows, and an HNSW-PQ index is an *exact* graph until 65,536 — so the path every
  production table above that size takes had no accuracy guard anywhere, and a commit that broke
  the codebooks, the encoding or the rerank went through green. `tests/quantized_recall_test.rs`
  crosses the threshold at a lowered setting and measures both halves: **0.706 on the codes
  alone, 1.000 after the exact rerank**. That gap is the claim the quantized design rests on,
  and it is now the thing that fails if it stops being true.

- **`GET /metrics`, in Prometheus text exposition format.** At the root, where a scrape
  config expects it, in the default build — being observable should not need a special build.
  It carries `dxmt_up`, the table and record counts, in-flight operations by kind, the
  operation and snapshot-failure counters, and the durability frontier (WAL sequence, snapshot
  sequence, last sync and last snapshot as Unix seconds). `tests/prometheus_metrics_test.rs`
  parses every sample, which is how the first version's `dxmt_index_building false` — valid
  JSON, rejected by every scraper — was caught. The `metrics` cargo feature pulled in the
  `prometheus` crate and was used by no code at all; it is gone.

- **Delete by filter.** `DELETE /tables/{t}/records` takes `filter` instead of `ids`, so
  "remove everything belonging to tenant X" is one request. It used to mean paging the table,
  collecting ids and sending them back — a round trip per page, at the mercy of anything
  written meanwhile. Exactly one selector: both would have to mean intersection or union, and
  neither reading is obvious; neither means "delete the table", which is not something to do
  by omission. The JS SDK already took a `filter` and threw "the server API currently supports
  delete by record_ids only" — a parameter that could only ever produce an error.

- **`POST /tables/{t}/records/count`.** Exact, with or without a filter, from the same index
  that answers the filter. Table stats carry a count for the table and nothing for a
  condition, so counting matches meant putting every matching record on the wire to produce
  one integer. Both SDKs have `count`.

- **A LangGraph example, and the LangGraph and LlamaIndex suites now run.** `langgraph` and
  `llama-index-core` are installed in CI's live job, `dxmt-python-sdk/examples/langgraph_memory.py`
  is an agent whose long-term memory is a DXMT namespace, and the LlamaIndex integration tests
  embed locally instead of calling OpenAI -- they had skipped themselves everywhere, so that
  package's live path had never run at all.

- **An unknown query parameter is an error too.** `?limitt=1` was accepted and dropped, so a
  caller who misspelled `limit` got the default page of 50 while believing they had asked for
  one row -- the same silence the request bodies had, in the place where a typo is easiest to
  make. `ApiQuery` refuses it in the same envelope, and the source-reading test now fails on a
  handler that takes a plain `Query<T>` as well as a plain `Json<T>`.

- **A field the server does not have is now an error.** Every request body carries
  `deny_unknown_fields` and goes through `ApiJson` (`src/api/extract.rs`), so a misspelled or
  invented field is a 400 naming it and listing the ones that exist. serde's default -- ignore
  it -- had cost this API real behaviour three times: `mode` on an insert (ten names for one
  code path, removed from the server while every SDK went on sending it), `alpha` and `top_k`
  on a hybrid search (in this crate's own Swagger example, tuning nothing), `named_vectors` on
  a table that then built no named index and searched the default vector instead. None of them
  failed; they were accepted, dropped, and answered wrongly. `ApiJson` also replaces axum's
  rejection -- a 422 with a `text/plain` body -- with the normal error envelope, so a client
  reading `error.code` no longer chokes on the most ordinary mistake there is.
  `tests/unknown_fields_test.rs` checks every body endpoint, a nested filter condition, and --
  by reading the source -- that no handler has gone back to a plain `Json<T>` body or a request
  type without the attribute.

- **`Field`, a filter builder for Python.** `(Field("lang") == "en") & Field("year").between(2020,
  2024) & ~Field("tags").is_in(["spam"])`, accepted wherever a filter is. Python's `and`/`or`/`not`
  and chained comparisons raise `TypeError`, since each would silently keep one side.

- **Paging by cursor.** `records/list` returns `next_cursor`, and takes it back as `cursor`. An
  offset shifts under writes — the list is newest first, so each insert pushes the rest back and
  the next page repeats a record — where a cursor names a place in the order: every record
  present for a whole scroll is returned exactly once. The Python and JavaScript SDKs' `scroll`
  pages by it, and their `upload` loads any iterable in batches, in parallel, retrying what a
  retry can fix and reporting what it could not.

- **Every client can send the API key.** `DXMT_API_KEY` (22fc06a) left the Python SDK, the
  LangChain and LlamaIndex packages, and the Rust and Go clients with no way to send it — against
  a keyed server they could not be used at all. Each takes a key now (`api_key=`,
  `with_api_key`, `WithAPIKey`; the JavaScript SDK already had `apiKey`), and none of them
  prints it: pydantic stores keep it off their fields, and the Rust and Go clients redact it
  from `Debug` and `%v`. CI runs every client's live suite a second time against a server that
  requires a key, after proving the server refuses a request without one.

- **Filtering on array fields.** A condition on a field holding an array is satisfied when any
  element satisfies it, so `tags == "rag"` finds `tags: ["rag", "llm"]` — Qdrant's rule, and
  what Chroma's `$contains` and Weaviate's `ContainsAny` are for. It used to compare the whole
  array with `"rag"` and match nothing. `NotEqual` and `NotIn` exclude a record when any element
  matches; Qdrant's `except` on an array instead keeps it if any element does not match, which
  lets `["spam", "x"]` through an exclusion of `"spam"`.

- **An optional API key.** `DXMT_API_KEY` puts a shared secret in front of every route but the
  health probes, sent as `api-key` (Qdrant's header name) or `Authorization: Bearer`. Off by
  default, as it is in Qdrant, Chroma and Weaviate. One static key, no rotation and no accounts
  — a seatbelt for a port that outlived its firewall rule, not a substitute for TLS and a
  private network.

- **Durability.** A write-ahead log on the write path, replayed at startup: `kill -9` and
  restart preserves acknowledged writes. Per-table snapshots of records and index carry recovery
  past the log they cover, taken periodically, on shutdown, and after a rebuild; the log is
  compacted once a snapshot reads back. `HnswIndex` and `HnswPqIndex` gained a versioned on-disk
  format so a restart does not rebuild the graph from scratch.
- **Records and vectors on disk.** With a WAL directory configured, a table's records and full
  vectors live in row files beside it and only the graph, the PQ codes and the id map stay
  resident — about 0.5 KB per 768-dimension row, against 8.3 KB when everything was in the map.
  `DXMT_RECORD_STORE=memory` / `DXMT_VECTOR_STORE=memory` put a new table back in memory.
- **Backup and restore**, replacing two earlier paths that never worked, with a test that proves
  a backup restores rather than that it reported success.
- **Object storage as the durable home for snapshots and WAL chunks**, plus table eviction:
  a table nothing has queried is dropped from memory entirely and rehydrated on the next query.
  Still experimental; the local WAL remains the readiness boundary.
- **Partitioned IVF** whose partitions load from object storage on demand, for a single table
  larger than memory. Experimental.
- **An MCP server** over stdio (`vectordb mcp`), backed by the same durable storage as the HTTP
  server.
- **Operational probes.** `/live` for liveness and `/ready` for readiness, the latter returning
  503 while local durability is degraded; `/health/metrics` exposes the WAL frontier, last fsync
  and snapshot times, and snapshot failure counts.
- **A functional HTTP test suite** (`tests/api/`) that runs the same assertions against a server
  with records in memory and again with them in row files, plus property invariants over a
  generated corpus.
- `DXMT_HNSW_EF_CONSTRUCTION` to set the build effort of new tables; `search_mode` so a caller can
  ask for more search effort without a rebuild.

### Changed

- **The API reference and the web console are served by default.** `api-swagger` and
  `api-dashboard` are default features, and `serve` mounts both without being asked:
  `/api/docs` and `/dashboard`. A database you cannot look at, and whose API you cannot read,
  is harder to evaluate than one you can — and both are read-only, and both sit behind
  `DXMT_API_KEY` when one is set, like every other route. `--no-swagger` and `--no-dashboard`
  turn them off; SECURITY.md says to use them on a port you do not control, which is the same
  advice the data routes have always carried.

  The console needs its bundle compiled in. A binary built without `dashboard/dist/` now warns
  once at startup and serves everything else, where before it refused to start — that refusal
  was right when the console was something you asked for, and would be wrong as a default,
  because a checkout without Node could not run the database at all. Asking by name restores
  the old strictness: `--dashboard` makes a missing bundle a refusal to start. `--swagger` and
  `--dashboard` are still accepted so existing scripts keep working.

- **The SDKs no longer send `mode` on an insert** (Python, JavaScript), and the parameter is
  gone from `insert_records`/`ainsert_records` and `InsertRecordsParams`. The server deleted the
  field; a call still passing it now fails loudly in the client rather than being ignored by the
  server. Nothing has been released, so this breaks no published API.

- **A filter comparing against an array or an object is refused** with `INVALID_INPUT`, from
  every endpoint. `tags == ["a", "b"]` used to return no records from `records/list`, a
  selective search or `search_mode: "exact"`, and the right ones from a broad or full-text
  search, depending on which path answered. `Equal`, `NotEqual`, `In` and `NotIn` take strings,
  numbers, booleans and null, which is what Qdrant's match accepts too.

- **A filtered page no longer costs the whole match.** `records/list` with a filter matching
  2.5M of 10M rows took 2.9 s a page, almost none of it filtering: every match became an owned
  id, was hashed back to its row to read a timestamp, and was freed, to return five. The index
  now keeps each record's `created_at` and chooses the page itself; 0.2 s. At 1M rows, 211 ms to
  13 ms.

- **The payload index rebuilds in half the time and memory.** The first filtered query after a
  restart rebuilds it, and at 10M rows that went from 40 s to 27 s end to end, with the server's
  peak memory from 8.7 GB to 7.2 GB and every answer unchanged. A value's records are held
  inline when there is one and in a roaring bitmap when there are many, where each distinct
  value used to get its own hash set; record ids are stored once rather than twice; and the
  index's maps use a keyed `ahash` instead of `SipHash`.

- **Every server-side environment variable is now `DXMT_*`.** `DB_` is not a namespace — it is
  the word "database", and `DB_HOST`, this server's bind address, is also what almost every
  compose file calls its Postgres. Qdrant namespaces its variables and so does Weaviate; this
  follows the convention while renaming is still free. The old `DB_*` name is still read, warns
  once at startup naming its replacement, and will be removed before the first release.

- **The default feature set is now the minimal supported surface**: `tokio-runtime` +
  `basic-storage`. Full-text search (`text-bm25`), Swagger, GraphQL, WebSocket, S3, LMDB,
  cluster and multimodal are opt-in. An interface that was not compiled in returns an explicit
  configuration error rather than mounting a route that does nothing.
- **Index building moved off the single core it was on.** Every node's edges sit behind their own
  lock, neighbour planning happens in parallel, queued insert requests merge into one index
  insert, and a bulk build makes one pass across all cores. Measured on Cohere 1M: total load
  time 7,478 s to 1,020 s.
- **HNSW-PQ is the default for a new table**: an exact graph below 65,536 rows, PQ codes above
  with an exact rerank from the row file. Accepted at 1M and 10M against exact HNSW within 0.2
  points of recall@100, at 1.9x the QPS and 4.66 GB resident against 66.9 GB.
- `ef_search` is read at search time and scales with table size; it used to be a hard-coded
  constant, so the configured value never reached a query.
- Snapshots, rebuilds and evictions no longer hold the table lock across their disk work, and a
  rebuild converges its catch-up outside the lock.
- jemalloc is the default allocator, with its background decay thread on.

### Fixed

- **The examples' fallback embedding has meaning in it.** Without `sentence-transformers` the
  SDK examples hashed the whole string into every coordinate, so "the cat sat on the mat" and
  "a cat is sitting on a mat" scored 0.0215 against each other and 0.0179 against a sentence
  about lattice gauge theory -- noise. Eleven examples whose subject is semantic search were
  demonstrating nothing whenever that package was absent, which is the default after a plain
  `pip install`. It is a hashed bag of words now: the same pair scores 0.667, and
  `retrieval_eval.py` reports Recall@1 0.37 and MRR 0.60 where it used to report 0.05 and 0.10.
  The two examples that print quality metrics say that the numbers measure the embedding rather
  than the database when the fallback is in use, and `agent_memory.py`'s consolidation step now
  adds a memory that actually crosses the deduplication threshold -- it announced a dedup demo
  and removed nothing.

- **`vector_weight` reached a fusion method that reads it.** The server's default fusion is
  `dbsf`, which normalises each side by its own distribution and ignores the weight; the Python
  and JavaScript SDKs sent a weight and never a `fusion_method`, so `vector_weight=0.0` and
  `vector_weight=1.0` returned the same rows with the same scores -- measured against a running
  server, not inferred. Both SDKs invented a default (0.7 and 0.5) for callers who set nothing,
  and the LlamaIndex and MetaGPT adapters invented 0.7 again, so the knob was in four APIs and
  three sets of docs and did nothing anywhere. A weight now selects `weighted_sum`, `dbsf`
  stays the default when no weight is given, and `fusion_method` is a parameter. The JS SDK's
  `text_weight` was never sent at all and is gone; the text side gets what the vector side does
  not take. The tests that passed a weight and asserted only the shape of the answer -- which
  is how this survived -- now assert the ranking changes.

- **Three Makefile targets that could not run**, and two that ran and did nothing useful.
  `release-patch`, `release-minor` and `release-major` began with `cargo bump`, which is not a
  cargo subcommand and which `make setup` does not install, and then `git commit` after
  `git add Cargo.toml Cargo.lock` -- which commits whatever else was staged. They are gone.
  `make docker-run` published port 8000 while the server listens on 9999 and the image
  `EXPOSE`s it, so the container came up healthy and unreachable, with no volume, so every
  restart was an empty database. `make help` listed about half the targets; `make check-help`
  now fails when one is missing, and runs as part of `verify-core`.

- **The JS SDK sent `ef_search` on every search, and the server has no such field.** It was
  accepted and dropped for as long as unknown fields were ignored — a caller tuning the
  traversal budget watched a number that did nothing — and once they started being refused, any
  search carrying it failed outright. It is replaced by `search_mode`, which the server does
  read, alongside the new `score_threshold` and `payload_fields`.

- **The Go client can clear a payload.** `Update.Payload` is `map[string]any` with
  `omitempty`, which drops an empty map as readily as a nil one -- so "clear this payload"
  never reached the wire, and the server, which keeps what an update does not mention, left the
  old payload in place. A nil map now means "leave it alone" and an empty one clears it.

- **`import dxmt` is quiet again.** `metagpt_vector_store.DXMTVectorStore` declared `client` as
  a pydantic field, shadowing the abstract property `BasePydanticVectorStore` declares, and
  carried a v1-style `class Config`: two warnings printed on every import once llama-index was
  installed. It now uses `PrivateAttr` plus the property the interface asks for, and
  `ConfigDict`, which is what the standalone LlamaIndex package always did.

- **A LangGraph namespace round-trips.** A namespace is a table, joined with `_` and split on
  `_`, so any component containing one came back wrong: `("agent", "memory", "session_123")`
  listed as four components, a prefix search for it matched nothing, and `("a_b",)` and
  `("a", "b")` quietly shared one table. An underscore inside a component is now doubled. A
  namespace whose table name would exceed the server's 63 characters says so, rather than
  failing on a name the caller never chose; and a store whose `embedding_dims` disagrees with
  the table the namespace already has is refused by name -- a store with no `embed_fn` writes a
  1-element placeholder, so it used to make every later put fail with the server's bare
  "expected 384, got 1". Seven tests of this store had been failing behind a skip.

- **The Python SDK translates a MongoDB-style filter exactly or not at all.** Four shapes ran as
  a different query and returned the wrong records: `{"$or": [{"a": 1, "b": 2}, {"c": 3}]}`
  became `a or b or c`; a key beside `$and`/`$or` was dropped; a nested `$and` became a
  condition on a field named "$and"; a native `must` filter lost any other key. A range with a
  string bound (`{"date": {"$gte": "2024-01-01"}}`, in the README's own example) matched
  nothing, since ranges compare numbers. Each now raises `ValueError`, as does anything else
  without an exact form in `must`/`should`/`must_not`. The LangGraph store's query-less search
  filters on the server instead of re-reading every record with a second, disagreeing
  implementation of the filter.

- **A filter now returns the same records from every endpoint.** `records/list`, a selective
  filtered search, `search_mode: "exact"` and the k-NN graph take the payload index's answer
  as final, while other searches check each record; the two disagreed in five ways. `NotEqual`
  dropped records lacking the field or holding an array; `NotIn` missed arrays; `id` was
  answered from a payload field of that name rather than the record's id; numbers were keyed
  at six decimal places and saturated at ±9.2e12, so every microsecond timestamp was one value;
  and `5` and `5.0` were equal to one and not the other. On a 400-row table, 12 of 14 filters
  returned wrong records through both `records/list` and exact search.
- **`5` now equals `5.0` in a filter**, everywhere. JSON has one kind of number, and a
  JavaScript client cannot send `5.0` at all.

- **Filtered search returned almost nothing below 5% selectivity.** The graph walk put the
  exploration queue and the result set behind the same filter test, so a non-matching entry point
  left nothing to explore: at 2% selectivity, 97.65% of queries returned zero rows.
- **String filters silently returned short pages.** `Contains` and `EndsWith` estimated
  selectivity from a hard-coded constant, so the over-fetch was too small and the retry the
  comments promised did not exist. All twelve filter operators now return a full page.
- **Hybrid search ranked the dense side backwards**, feeding distances into a fusion that scores
  higher-is-better; the nearest vector sorted last.
- **Concurrent snapshots corrupted each other** through a shared staging directory, and an index
  write failure was folded into "this index type has no disk format" — so a snapshot reported
  success with the graph silently dropped.
- **Row files grew with the number of writes rather than with the amount of data.** Both the
  vector row file and the record row file are append-only: an update appends a new row and
  repoints the id, and the row it left behind kept its blocks. Nothing reclaimed them -- the
  only thing that claimed to was a comment describing a rewrite no code performed, and the
  sweep that rebuilds an index fires on a change of algorithm, which a table of constant size
  never has. Measured in a soak at 200 rows/s over a fixed 400,000-row key space: 5.2 GB of
  vector file holding 1.2 GB of live rows after 2.4 hours, growing at about 52 GB a day
  whatever the table's size. A new sweep (`DXMT_ROW_COMPACTION_INTERVAL_SECS`, default 600)
  releases the blocks behind dead rows. Row numbers are unchanged, so nothing above the file
  layer moves; the file's apparent length still grows, and its allocated size no longer does.
- **Quantizing a table stopped every insert in the process for the length of the training.**
  Training the PQ codebooks at 65,536 rows was taken out of the index write lock in an earlier
  change, which fixed what searches saw and left writers where they were: measured in a soak at
  200 rows/s on eight cores, acknowledged writes stopped dead for the 256 s the training ran,
  the WAL sequence did not move, and eight batches timed out at the client and were answered on
  arrival by `Failed to send insert response: insertion result was dropped`. Two things
  serialised them, and removing either alone was not enough. Training ran on the storage layer's
  single insertion task, so inserts queued behind the `await`; it now runs in a spawned blocking
  task. And it ran on rayon's global pool as one unyielding chunk per worker, which is the same
  pool `HnswIndex::plan_batch` submits to above `PARALLEL_PLAN_MIN_NODES`, so an insert could
  not be picked up until a worker finished its whole share of the subspaces; it now trains in a
  pool of its own, sized to half the cores.
- **An upsert grew the index for ever.** A graph node cannot move -- where it sits is decided
  by the vector it had when it was linked in -- so an update tombstones the old slot and
  appends a new one, and nothing gave the slot back. A table of constant size under steady
  upserts therefore grew one slot per write. Measured in a 17-hour soak at 200 rows/s over a
  400,000-row key space: 11.1 million slots holding 400,000 records, `index_size_bytes` 5.3 GB
  where the live rows need about 200 MB, resident memory climbing 8.4 GB a day with the row
  count flat for sixteen of those hours, snapshots grown from 1.2 s to 18.9 s, and 196 s to
  come back from `kill -9`. The re-evaluation sweep now compares live entries against slots and
  rebuilds when the graph carries twice the slots it needs; `rebuild_index` no longer counts
  such a graph as equivalent to the one it would build.
- **The two append-only sidecars grew for ever as well**, for the same reason and with no dead
  regions to release: the vector file's `(id, row)` list and the record file's append log take
  a line per write, and an upsert writes. 214 MB and 303 MB respectively over the same 17
  hours. Both are now rewritten by the reclamation sweep -- the row list keeping each id's
  latest row, the record log keeping the live rows and replacing each run of dead ones with a
  `gap` that advances the offset accumulator without creating a row, so the surviving rows
  still name their own bytes and the data file needs no rewrite. Each commits with a single
  `rename`.
- **A restored table could come back as the wrong index.** Recovery ignored the algorithm the
  snapshot recorded, rebuilding a 94,000-row brute-force table as HNSW before binding the port.
- `update_record` never wrote to the WAL, so updates did not survive a restart. Upserts left
  stale sparse postings, payload index entries and named vectors behind.
- Non-finite vectors are refused at the one path every writer goes through.
- Score ties are broken by id before any cut to `k`, so the same query returns the same page.
- A dropped table stays dropped; an evicted table is no longer erased by the next snapshot.

### Removed

- **Simple mode (`serve --simple`, `/collections/*`).** It was a second data model over the
  same storage — its own wire shapes, its own scoring conventions, 942 lines and 12 routes, and
  until this week not one test touched it. What it cost was visible in the shapes themselves:
  its `distance` field carried a *similarity*, ordered descending, where Chroma — the API it
  imitated — returns a distance where lower is better, so a client written against Chroma read
  our results backwards. Its "auto-embedding" was a hashed bag of words: lexical retrieval
  wearing a dense vector's shape, where "car" and "automobile" land in unrelated buckets.

  The decision behind removing it rather than fixing it: **Qdrant, Weaviate, Chroma and LanceDB
  each ship one data model and reach it through several protocols — Weaviate through four —
  and none of them serves a competitor's API shape.** Compatibility, where it exists, lives in
  the client layer: the LangChain and LlamaIndex adapters, and this SDK's MongoDB-style filter
  translation. That is where ours stays. `--memory-limit`, which configured only simple mode,
  goes with it, and the four interface flags no longer need to explain that they require
  "standard mode", because there is one mode.

- **Two scripts that tested things the server no longer has.** `scripts/test_index_status.py`
  called `/tables/{t}/index_status`, which answers 404, and `scripts/test_sync_insert.py`
  compared "sync" against "async" inserts by sending `?sync=true` -- a query parameter no
  handler has ever read -- so it timed two runs of the same path and reported the difference as
  a finding. `scripts/README.md` now says what each of the remaining 33 is for and that none of
  them is the gate; `test_upsert_fix.sh` read `.data.results`, which the search endpoint stopped
  returning, so it had been failing on every run against a working server.

- **A second, dead `UpdateRecordRequest`.** `types::record` had one with `sparse_vector` and
  an `expected_version` for optimistic locking; nothing constructed or deserialised it, and
  `PUT /records/{id}` takes the one in `api::handlers`, which has neither field and has
  `merge_mode` and `upsert` instead. Two live-looking types under one name, one of them
  describing a version check the endpoint has never had.

- **25 files of leftover debugging from the LlamaIndex package's root**, plus three from its
  examples: seven progress-report markdown files (`FIXES_APPLIED.md`, `TEST_RESULTS_CN.md`,
  `RAGAS_QUICKFIX.md`, …) and sixteen ad-hoc `test_*.py` scripts that printed rather than
  asserted, hardcoded `/localdisk/...` paths from another machine, and were collected by
  nothing -- everything they probed is covered by `tests/`. The one analysis worth keeping
  moved to `design/`, which is not distributed.

- 51 source files that were never part of the build, five modules superseded by the code that is
  actually wired up, and every example that did not run.
- `api/grpc.rs`, which was never declared by `api/mod.rs`. There is no gRPC interface.

### Known limitations

See the "Known Limitations" section of [README.md](README.md). The short version: the graph is
built inside the write request, so ingest and queries contend and a rebuild is always a whole
table; object-storage mirroring is experimental; several index names fall back to brute force;
and the multimodal modules return placeholder vectors.
