# Backup and restore

Chinese version: [backup-restore.zh.md](backup-restore.zh.md). Where the two differ, this one
is authoritative.

**Remember one thing first: `$DXMT_WAL_DIR/snapshot/` (by default `~/.dxmt/store/snapshot/`)
is data, not a cache.**

The WAL is truncated: once a snapshot has been verified, the part of the log it covers is
deleted. From that moment the snapshot is the **only copy** of that data. Cleaning up
`snapshot/` as if it were a cache deletes data.

---

## What to back up

| Path | Contents | Must back up |
|---|---|---|
| `$DXMT_WAL_DIR/snapshot/` | Table metadata and index segments; the records of in-memory tables | **Yes** |
| `$DXMT_WAL_DIR/records/` | Records of disk-backed tables (`<table>.rec` + `.rec.log`); every table is disk-backed by default | **Yes** |
| `$DXMT_WAL_DIR/vectors/` | Full vectors of disk-backed tables (`<table>.f32` or `.f16`, plus `.rows`) | **Yes** |
| `$DXMT_WAL_DIR/wal.log` | Writes since the last snapshot | **Yes** |
| `$DXMT_WAL_DIR/aliases` | Table aliases | **Yes** |
| `$DXMT_WAL_DIR/fulltext/` | Full-text (Tantivy) indexes | No (rebuilt from the records after a restore) |
| `$DXMT_WAL_DIR/diskann/` | EnhancedDiskANN index directories | No (rebuildable from the records) |
| `$DXMT_WAL_DIR/derived/` | Filter and sparse indexes written at a clean shutdown, read once at the next start and then deleted | No (rebuildable from the records) |
| The `snapshot/`, `wal/` and `partitions/` prefixes in object storage | The durable home of the data when object storage is configured | **Yes** (equivalent to the rows above: with object storage configured, a disk-backed table's snapshot carries its records and vectors, so a new machine with no row files can recover from it; the cost is that a snapshot again takes time proportional to the record count) |

Everything the server writes lives under one root, `DXMT_ROOT_DIR`, which defaults to
`~/.dxmt` (`$HOME/.dxmt`):

```text
~/.dxmt/
├── store/      the data itself; this is what "backup" means        DXMT_WAL_DIR
├── backups/    the only place POST /backup may write               DXMT_BACKUP_ROOT
└── cache/      derived; deleting it only costs time                DXMT_CACHE_DIR
```

The three children have different lifetimes, which is why they are three directories: the name
says which one can be deleted. Each can be moved on its own; setting the root moves all of them.
`store/` and `backups/` are created at startup and probed for writability, so an unwritable
`backups/` is a refusal to start rather than a surprise for whoever takes the first backup.
`cache/` is created by whatever first writes to it.

The default is under the home directory rather than the working directory; otherwise starting
the server from a different directory would open a different, empty database.

**In production, set `DXMT_ROOT_DIR` (or `DXMT_WAL_DIR`) explicitly**, for two reasons, both
about where the home directory lives:

- The home directory is usually on the root partition. This database measured 44.7 GB at 100
  million rows; filling it fills `/`, not only the database.
- The home directory is often on NFS. Row files are read through `mmap` and the WAL is
  `fsync`ed after every batch, and neither has reliable semantics on NFS. That is a correctness
  problem, not a performance one.

The startup log has a line `data directory: …` naming the directory this process actually
opened.

**For deployments with object storage configured**, snapshot segments and WAL chunks are
already in the bucket: back up the bucket, and treat the local directory as a cache. The
`partitions/` prefix holds the segment files of partitioned indexes and must be backed up as
well: a partitioned index keeps only its centroids in memory.

---

## Taking a backup

```bash
curl -sX POST http://localhost:9999/api/v1/backup \
     -H 'Content-Type: application/json' \
     -d '{"path":"/backups/dxmt-2026-08-15"}'
```

It returns a manifest:

```json
{
  "created_at": "2026-08-15T03:14:07.123456789+00:00",
  "tables": 3,
  "files": ["snapshot/manifest", "snapshot/t0.meta", "snapshot/t0.idx",
            "records/docs.rec.log", "records/docs.rec",
            "vectors/docs.f32.rows", "vectors/docs.f32", "wal.log"],
  "bytes": 5387421,
  "restore": "Stop the server, replace the contents of DXMT_WAL_DIR with this directory, and start it again."
}
```

**The endpoint forces a snapshot first**, and the order matters: the snapshot pins writes not
yet on disk and compacts the log to the snapshot point. Copying the other way round yields an
old snapshot paired with a log that does not match it.

The destination must be inside `DXMT_BACKUP_ROOT`. It is written by the **server process**, so
the path must be writable by the server, not by the caller.

---

## Scheduled backups

```bash
DXMT_BACKUP_INTERVAL_SECS=86400   # once a day; the default is 0, off
DXMT_BACKUP_KEEP=7                # keep the latest 7; the default is 7
```

Each is written to `$DXMT_BACKUP_ROOT/auto-<UTC time>` and is the same thing a manual
`POST /backup` produces. Beyond the retention count the oldest are deleted, **but only
directories whose name starts with `auto-` and which hold a `BACKUP.json`**: manual backups,
half-written backups and anything else you keep there are never touched. Off by default,
because every backup is a full copy, and turning it on by default would quietly consume
several times the disk.

Success shows on `/metrics` as `dxmt_last_backup_timestamp_seconds` and
`dxmt_backup_failures_total` (manual backups count too). A failure is logged and counted and
retried at the next interval; it does not affect serving.

---

## Restoring

Restore is an offline operation; stop the server first:

```bash
systemctl stop dxmt-vectordb          # or kill -TERM
vectordb restore /backups/dxmt-2026-08-15            # into the data directory serve would open
vectordb restore /backups/dxmt-2026-08-15 --wal-dir /var/lib/dxmt/store   # or name it
systemctl start dxmt-vectordb
```

It:

1. checks that every file listed in the backup's `BACKUP.json` is present, and refuses,
   changing nothing, if one is missing;
2. takes the data directory's `LOCK`, and refuses if the server is still running;
3. **moves** the data directory's existing contents to `<directory>.before-restore-<time>`
   rather than deleting them, so a wrong restore can be moved back;
4. copies the backup in, then opens it once and runs recovery, printing each table and its
   record count.

Copying by hand still works (`snapshot/`, `records/`, `vectors/`, `wal.log` and `aliases` into
the data directory), but it is easy to miss something: an earlier version of this document
left out `vectors/`.

**Restore is deliberately not an API.** Loading restored data into a running server would mean
stopping writes, clearing in-memory state and replaying, which is a far more complex and
error-prone path than stopping and copying files, and all it would save is one restart.

---

## When snapshots happen

| Trigger | Configuration |
|---|---|
| Periodically | `DXMT_SNAPSHOT_INTERVAL_SECS`, default 300 seconds, 0 disables |
| Graceful shutdown | SIGTERM / Ctrl-C, after draining |
| After `POST /api/v1/tables/{name}/index/rebuild` | Automatic |
| `POST /api/v1/snapshot` | Manual |
| Before a cold table is evicted | `DXMT_EVICTION_INTERVAL_SECS`, default 600 seconds |

The snapshot interval bounds **how much WAL has to be replayed in the worst case**. Without
object storage it does not change what a crash can lose (no acknowledged write is lost), only
how long recovery takes.

---

## When a snapshot cannot be read

**The server refuses to start**, and prints something like:

```
snapshot at /home/you/.dxmt/store/snapshot is unreadable (…). The WAL no longer covers the entries it
holds, so starting would lose data. Restore it from backup, or delete the directory to accept
the loss and start from whatever the WAL still has.
```

This is intentional. The earlier behaviour was to log a warning and start with an empty
database, which, with the log already truncated, was silent data loss.

Two choices:

1. **Restore from a backup** (recommended), as described above.
2. **Accept the loss**: `rm -rf $DXMT_WAL_DIR/snapshot`, then start. Whatever the remaining log
   holds is replayed.

A snapshot is read back in full and verified as it is written (the manifest, every `.meta`,
and every `.idx` it claims to have); if verification fails it does not replace the previous
snapshot and the log is not truncated, so in normal operation this error should not occur.

---

## Deliberately not provided

- **Incremental backups.** There is no backup catalogue and no backup id.
- **Listing, deleting or querying the status of backups.**
- **Restore through the API.**

Seven endpoints — `/backup/full`, `/backup/incremental`, `/backup/manual`, `/backup/status`,
`/backup/list`, `/backup/restore` and `DELETE /backup/{id}` — used to exist and **returned 500
unconditionally**: `backup_manager` was hardcoded to `None` in `server.rs`. And the
`BackupManager` behind them collected objects under a `tables/` prefix that **no code ever
wrote to**, so even wired up it would have produced empty backups, which is worse than an
error. They were removed in favour of the one working endpoint above.
