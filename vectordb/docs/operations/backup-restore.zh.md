# 备份与恢复

英文版：[backup-restore.md](backup-restore.md)。两者内容相同，以英文版为准。

**先记住一件事：`$DXMT_WAL_DIR/snapshot/`（默认 `~/.dxmt/store/snapshot/`）是数据，不是缓存。**

WAL 会被截断——快照校验通过后，它所覆盖的那段日志会被删掉。所以从那一刻起，
快照是那段数据的**唯一副本**。把 `snapshot/` 当缓存清理掉，等于删数据。

---

## 该备份什么

| 路径 | 内容 | 必须备份 |
|---|---|---|
| `$DXMT_WAL_DIR/snapshot/` | 表元数据、索引段;内存表的记录 | **是** |
| `$DXMT_WAL_DIR/records/` | 盘表的记录(`<表>.rec` + `.rec.log`),默认每张表都是盘表 | **是** |
| `$DXMT_WAL_DIR/vectors/` | 盘表的全量向量(`<表>.f32` 或 `.f16` + `.rows`) | **是** |
| `$DXMT_WAL_DIR/wal.log` | 上次快照之后的写入 | **是** |
| `$DXMT_WAL_DIR/aliases` | 表别名 | **是** |
| `$DXMT_WAL_DIR/fulltext/` | 全文索引 | 否（恢复后从记录重建） |
| `$DXMT_WAL_DIR/diskann/` | EnhancedDiskANN 的索引目录 | 否（可从记录重建） |
| `$DXMT_WAL_DIR/derived/` | 正常关机时写下的筛选/稀疏索引，下次启动读一次即删 | 否（可从记录重建） |
| 对象存储的 `snapshot/` `wal/` `partitions/` 前缀 | 配了对象存储时的持久归宿 | **是**(等价于上面几项:配了对象存储时,盘表的快照会把记录连同向量一起带上,所以一台没有行文件的新机器也能从它恢复;代价是快照又回到按记录数计时) |

服务器写的所有东西都在一个根下面，`DXMT_ROOT_DIR`，默认 `~/.dxmt`：

```text
~/.dxmt/
├── store/      数据本身。备份指的就是这个目录        DXMT_WAL_DIR
├── backups/    POST /backup 只能写这里面             DXMT_BACKUP_ROOT
└── cache/      派生物，整个删掉只会变慢              DXMT_CACHE_DIR
```

三个子目录的生命周期不同，所以是三个目录 —— 看名字就知道哪个能删。每个可以单独搬，设根则一次搬完。
`store/` 和 `backups/` 在启动时创建并做一次可写性探测，所以"`backups/` 不可写"是一次拒绝启动，而不是留给第一个做备份的人去发现；`cache/` 由第一次往里写的东西创建。

放在家目录下而不是工作目录下，否则换个目录启动就会打开另一个空数据库。

**生产环境请显式设 `DXMT_WAL_DIR`**，有两个理由，都和"家目录在哪"有关：

- 家目录通常在根分区上。这个库在 1 亿行时实测 44.7 GB，撑满的是 `/` 而不只是数据库。
- 家目录常常是 NFS。行文件走 `mmap`、WAL 每批 `fsync`，这两样在 NFS 上的语义都不可靠——
  是正确性问题，不是快慢问题。

启动日志里有一行 `data directory: …`，说的就是这个进程实际打开的是哪一个。

**配了对象存储的部署**，快照段和 WAL 分块都已经在桶里了——备份桶即可，本地目录是缓存。
`partitions/` 前缀下是分区索引的段文件，同样要备份：分区索引只在内存里保留质心。

---

## 备份

```bash
curl -sX POST http://localhost:9999/api/v1/backup \
     -H 'Content-Type: application/json' \
     -d '{"path":"/backups/dxmt-2026-08-15"}'
```

返回一份清单：

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

**这个端点会先强制做一次快照**，顺序不能反：快照把内存里尚未落盘的写入固定下来，并把日志
压到快照点。反过来拷的话，拿到的是一份旧快照配一段与之不匹配的日志。

目录由**服务进程**写，所以路径要对服务端可写，不是对调用方。

---

## 定时备份

```bash
DXMT_BACKUP_INTERVAL_SECS=86400   # 每天一次；默认 0，关闭
DXMT_BACKUP_KEEP=7                # 保留最近 7 份，默认 7
```

写到 `$DXMT_BACKUP_ROOT/auto-<UTC 时间>`，和手动 `POST /backup` 是同一份东西。超过保留份数时删最旧的——
**只删名字以 `auto-` 开头、且有 `BACKUP.json` 的目录**：手动备份、写到一半的备份、你放在那里的
其他东西都不会碰。默认关闭，因为每份都是完整拷贝，默认打开会在不知情时占掉几倍的磁盘。

是否成功看 `/metrics` 的 `dxmt_last_backup_timestamp_seconds` 和 `dxmt_backup_failures_total`
（手动备份也计入）。失败只记日志和计数，下一轮再试，不影响服务。

---

## 恢复

恢复是离线操作，服务要先停：

```bash
systemctl stop dxmt-vectordb          # 或 kill -TERM
vectordb restore /backups/dxmt-2026-08-15            # 恢复到 serve 会打开的那个数据目录
vectordb restore /backups/dxmt-2026-08-15 --wal-dir /var/lib/dxmt/store   # 或明确指定
systemctl start dxmt-vectordb
```

它会：

1. 按备份里的 `BACKUP.json` 核对文件都在，缺一个就拒绝，什么都不动；
2. 拿数据目录的 `LOCK`——服务没停就拒绝；
3. 把数据目录原有的内容**移到** `<目录名>.before-restore-<时间>`，不删除——恢复错了可以搬回来；
4. 拷入备份，然后实际打开一次、跑完恢复，打印每张表和记录数。

手工拷贝也仍然可以（`snapshot/`、`records/`、`vectors/`、`wal.log`、`aliases` 拷进数据目录），
但容易漏——这份文档的旧版本就漏过 `vectors/`。

**故意不做成 API**：往一个正在跑的服务里灌恢复数据，需要先停掉写入、清空内存状态、
再重放——那是一个比"停机拷文件"复杂得多、也更容易出错的路径，而收益只是省一次重启。

---

## 快照什么时候发生

| 触发 | 配置 |
|---|---|
| 定期 | `DXMT_SNAPSHOT_INTERVAL_SECS`，默认 300 秒，0 关闭 |
| 优雅停机 | SIGTERM / Ctrl-C，drain 之后 |
| `POST /api/v1/tables/{name}/index/rebuild` 之后 | 自动 |
| `POST /api/v1/snapshot` | 手动 |
| 冷表驱逐前 | `DXMT_EVICTION_INTERVAL_SECS`，默认 600 秒 |

快照间隔决定了**最坏情况下需要回放多少 WAL**，也决定了单机部署（没配对象存储时）
一次崩溃可能丢多少——不会丢已确认的写入，但恢复要多花时间。

---

## 快照读不出来会怎样

**服务拒绝启动**，并打印类似：

```
snapshot at /home/you/.dxmt/store/snapshot is unreadable (…). The WAL no longer covers the entries it
holds, so starting would lose data. Restore it from backup, or delete the directory to accept
the loss and start from whatever the WAL still has.
```

这是有意的。原来的行为是告警一句然后空库启动——在日志已被截断的前提下，那等于静默丢数据。

两个选择：

1. **从备份恢复**（推荐）——按上面的步骤。
2. **接受损失**：`rm -rf $DXMT_WAL_DIR/snapshot`，然后启动。剩下的日志能回放多少算多少。

写快照时会先完整读回来校验（manifest、每个 `.meta`、每个声称有索引的 `.idx`），
校验不过就不替换、也不截断日志，所以正常情况下不该出现这个错误。

---

## 有意不提供的东西

- **增量备份**。没有备份目录（catalogue），也没有备份 ID。
- **备份列表 / 删除 / 状态查询**。
- **通过 API 恢复**。

`/backup/full`、`/backup/incremental`、`/backup/manual`、`/backup/status`、`/backup/list`、
`/backup/restore`、`DELETE /backup/{id}` 这七个端点曾经存在，但**无条件返回 500**：
`backup_manager` 在 `server.rs` 里写死 `None`。而它们背后的 `BackupManager` 收集的是
`tables/` 前缀下的对象——**没有任何代码写过那个前缀**，所以就算接上，产出的也是空备份。
比报错更糟。已经删掉，换成上面这一个能用的端点。
