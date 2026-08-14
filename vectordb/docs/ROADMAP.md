# 工程路线图

这份文档记录 DXMT VectorDB **当前真实的形态**、下一步做什么、以及树里已有但未接线的东西。
产品/算法层面的竞品对标路线在 `TODO.md`，两者不重叠。

本文替代了原 `docs/TODO_PRIORITY.md`——那份文档把若干"已实现"条目指向了从不参与编译的
文件（例如 `search/hybrid_indexing.rs`），结论"没有阻塞单节点生产"不成立。

最后更新：2026-08-14

---

## 1. 当前实际形态

服务端只有一条链路，两种启动模式都收敛到同一个存储实现：

```
main.rs
 ├─ serve --simple  →  api/simple_core.rs      /collections/*（根路径）
 └─ serve           →  api/server.rs:76        /api/v1/*
                            │
                            └─ IntelligentIndexedStorage
                                 ├─ tables: Arc<RwLock<HashMap<..>>>     ← 全部数据在此
                                 ├─ index:  HNSW / HnswPQ / BruteForce / EnhancedDiskANN
                                 └─ storage: Arc<TieredStorage>  #[allow(dead_code)]
```

必须知道的三件事：

1. **持久化靠 WAL，不靠分层存储（W1 之后）。** 服务端走
   `new_with_wal`，日志默认写 `./data/wal`（`DB_WAL_DIR` 可覆盖），启动时全量回放。
   `IntelligentIndexedStorage` 仍然没有一次 `self.storage` 调用——分层存储依旧未接线，
   数据在内存 + WAL，不在对象存储。另有一个手工 `backup()`
   （`storage/intelligent_indexed_storage.rs`）写 JSON 快照。
   注意 `IntelligentIndexedStorage::new()`（不带 WAL 的那个构造函数）不持久化，
   库使用者要自己选 `new_with_wal`。
2. **`DB_STORAGE_TYPE` 无效。** 它在 `config/mod.rs:221` 被读入配置，但服务端在
   `api/server.rs:76` 写死 `IntelligentIndexedStorage`，`StorageFactory` 从不被调用。
3. **索引选择在建表时不看数据量。** `intelligent_indexed_storage.rs:474` 写死
   `estimated_size = 10_000`。自动重评估函数在 `:1608` 起整段被注释。只有手工
   `POST /api/v1/tables/{t}/index/rebuild` 会按真实记录数重选。

---

## 2. P0 — 让它成为一个数据库

按依赖顺序，前一项不做完后一项没有意义。

### 2.1 WAL 接入主写路径

**现状**：`wal.rs` 本身可用（`replay_from` 有实现，单测通过），但唯一的使用者是
`storage/atomic_ops.rs`，而 `AtomicStorageManager` 只被 `cluster/shard_router.rs` 引用——
一个按 `CLAUDE.md` 自述"不是当前重点"的模块。`tests/integration.rs` 里 WAL 恢复测试是注释掉的，
注释写着 "WAL tests disabled - WAL configuration has been restructured"。

**要做**：
- `IntelligentIndexedStorage` 的 channel-based 插入任务在写入 `tables` 之前先写 WAL
- 启动时 `replay_from(0)` 重建 `tables`
- 恢复 `tests/integration/wal_recovery_test.rs` 的注册（该文件仍在，530 行，未删）
- 考虑直接用 `async_wal.rs`（见 3.3），避免在插入路径上做同步 fsync

**验收**：建表 → 插入 N 条 → kill -9 → 重启 → N 条全部可查。这是当前唯一没有的能力。

### 2.2 索引持久化

**现状**：`SerializableIndex` trait 在 `search/index_traits.rs:38` 定义，全仓库只有
`search/ivf.rs:273` 一处实现。HNSW 和 HNSW-PQ 没有 save/load。

**影响**：即使 2.1 做完，重启也要把所有向量重新插入索引——百万级数据集意味着分钟级到小时级的
启动时间。

**要做**：给 `HnswIndex` 和 `HnswPqIndex` 实现 `SerializableIndex`，启动时优先 load，
load 失败再从 WAL 重建。

### 2.3 TieredStorage 接入，或者删掉它

**现状**：`storage/object/tiered.rs` 实现完整、单测通过、`tier_promotion.rs` 的热度评分也是真的，
但 `IntelligentIndexedStorage` 只在构造函数里 `TieredStorage::new_from_config(config)` 建了个对象
放进 `#[allow(dead_code)]` 字段，此后再不碰它。

**两条路，必须选一条**：
- 接上——这样 README 的 "object storage as the source of truth" 才成立；
- 或者删掉，并把 README / CLAUDE.md 的架构描述改成实话。

在做出选择之前，这是全仓库最大的一处名实不符。

---

## 3. 已在树内、但没有接线

这些代码编译通过、部分有测试，只是没有任何入口能到达。保留在树里，按下表处置。

### 3.1 MCP Server（`src/mcp/` 3,519 行）

`CLAUDE.md` 标注为 P0，实际上没有任何启动方式——`main.rs` 只有 `serve` 一个子命令。
六个文件包含 server / tools / prompts / resources / transport，`transport.rs` 有 2 个测试。

**要做**：加 `serve --mcp` 子命令或独立 bin，把 `mcp::server` 挂上 stdio transport。
这是投入产出比最高的一项——代码基本齐了，缺的是十几行接线。

### 3.2 索引生命周期管理（`src/index_management/` 3,236 行 + `src/adaptive_index/` 576 行）

`adaptive_index` 只被 `index_management` 引用，后者没有任何使用者，两者一起悬空。
里面有 migrator、transaction_log、record_iterator、profiling、benchmark_harness。

**价值**：这正是 2.3 之后"真正的自适应索引"所缺的部分——按实际数据量在线迁移索引算法，
而不是像现在这样靠手工 rebuild。`intelligent_indexed_storage.rs:1608` 那段被注释的
`reevaluate_index_selection` 就是它的调用点。

**注意**：现在的 `rebuild_index`（`:1260`）虽然会按真实记录数重选算法，但把
`memory_budget_mb: 512`、`has_avx2: false`、`supports_simd: false` 写死了，没有用构造时
已经检测好的 `self.hardware_capabilities`。接线时一并修掉。

### 3.3 `src/async_wal.rs`（391 行）

后台 writer + mpsc channel 的非阻塞 WAL。2.1 应该优先用它而不是同步的 `wal.rs`。

### 3.4 `src/multi_tenant/`（760 行，7 个测试）

配额、用量限制、shard router、query router。单节点阶段用不上，多租户 SaaS 形态需要。

### 3.5 `src/worker_pool/`（329 行，3 个测试）、`src/memory_pool.rs`（178 行，3 个测试）

体量小、有测试、无使用者。接线成本低，等有实测瓶颈再接。

### 3.6 已接线（2026-08-14）：Swagger / GraphQL / WebSocket

原先 `create_app()` 写死 `create_app_with_features(false, false, false)`，而
`create_app_with_features` 在全仓库没有任何传 `true` 的调用者——三个 API 层都编译进了二进制
但运行时 404（curl 实测）。`api-swagger` / `api-graphql` 这两个 cargo feature 只控制编译，
不控制暴露，README 里"默认可用 Swagger UI"是错的。

现在 `main.rs` 加了 `--swagger` / `--graphql` / `--websocket` 三个开关，**默认仍为关闭**：
每一个都扩大暴露面，要不要默认打开是产品决定，不该由这次清理顺手改掉。

实测路径：Swagger `GET /api/docs`（spec 在 `/api/docs/openapi.json`）、GraphQL
`POST /graphql`、WebSocket `GET /ws`。

---

## 4. 从已删代码中回收的设计

commit `ff53809` 删掉了 51 个从不参与编译的文件（29,144 行）。这些文件**无法编译**——17 个
import 了不存在的 crate `asynctrait`（真名是 `async-trait`），`integration/mod.rs` 还
re-export 了 `_external_systems` 等根本不存在的模块。下划线的破坏是双向的，看得出是一次批量
改名之后从未编译过。

所以它们的价值在设计意图，不在实现。要用的时候从 `git show ff53809^:vectordb/src/...` 取出来
当草图读，别指望能直接编译：

| 想法 | 原位置 | 备注 |
|---|---|---|
| RBAC / 策略引擎 | `src/auth/` 1,976 行 | Resource / Action / Permission / PolicyRule / 时间窗限制，结构相当完整，多租户 SaaS 形态需要 |
| 信封加密 + 密钥轮换 | `src/crypto/` 1,828 行 | KeyManager、KeyRotationScheduler、静态加密存储；合规场景（等保、金融）会要 |
| 数据生命周期规则 | `src/lifecycle/` 764 行 | 按表模式匹配的 TTL / 降冷 / 压缩规则，和 2.3 的分层存储天然配套 |
| OpenTelemetry 指标与链路 | `src/observability/` 4,975 行 | metrics / tracing / 告警阈值；当前可观测性只有 `/health` 和 `/health/metrics` 两个端点 |
| 插件注册表 | `src/plugins/` 370 行 | Plugin trait + PluginManager + 能力声明 |

commit `25400ce` 删掉的 5 个模块（5,764 行）是被现有实现取代的重复品，不建议回收，理由见该
commit 的说明。

---

## 5. 已知的名实不符

修 bug 之前先知道哪些地方在说谎：

| 位置 | 问题 |
|---|---|
| `search/auto_tuned.rs` | `hnsw_search` / `pq_fastscan_search` / `lsh_search` / `enhanced_pq_search` 四个函数全部直接调用 `Self::brute_force_search` |
| `search/quantization/product_quantizer.rs` | 52 行的假 PQ：`train()` 只置了个 bool，`encode_batch()` 返回全 0 码字，`get_centroid()` 返回 `&[0.0]`。同目录的 `product_quantization.rs`（558 行）才是真实现，两个同名类型并存 |
| `create_index_instance` `:1553/:1595/:1600` | `DiskANN`、`ScaNN`、`QuantizedIVF` 三个分支直接退化成 `BruteForceIndex` |
| `select_memory_optimized_index` `:1444` | 两条分支带着 `// HierarchicalClustering not yet available` / `// HybridRaBitQClustering not yet available` 的回退注释。`DB_MEMORY_OPTIMIZATION` 默认为 **true**，所以这是默认路径 |
| 建表日志 | 0 条记录的表会打印 "Medium dataset detected (10K-100K), using Hierarchical Clustering"，紧接着又打印 "Selected HNSW"，而 API 返回的表元数据写着 `"index_type":"bruteforce"`。三者互不一致 |
| `EnhancedDiskANN` `:1570` | 索引目录用 `TempDir::new()...keep()`，重启即成孤儿目录。注释自己写着 "In production, this would use a persistent directory" |

---

## 6. 构建与验证的注意事项

- **`make verify` 通不过，从来没通过过。** 它是 fmt + `clippy -D warnings` + test，而
  `cargo clippy --all-targets -- -D warnings` 在原始导入 commit `1668a8b` 上报 **161 个错误**，
  当前是 **154 个**。所以它不能当门禁用。目前实际可用的门禁是 `cargo test --lib`
  （413 passed / 0 failed）。把 clippy 清零是一件独立的机械工作，值得单独做一个 commit。
- **`make verify` 会改你的文件。** 它的第一步是 `make fmt`，跑的是 `cargo fmt`（写入，不是
  `--check`）。仓库从未 fmt 干净过，所以这一步会一次性重排 **208 个文件**
  （+10,530 / −7,333），把任何真实改动淹掉。审阅前先 `cargo fmt -- --check` 看一眼，
  或者单独提一个纯格式化 commit。
- `.github/workflows/` 在 `vectordb/` 下，而 GitHub Actions 只读真实仓库根的 `.github/`，
  **CI 不会触发**。所以上面两条不会有任何自动化替你发现。
- `make test` / `make test-integ` / `make bench` 各自追加了一个 `python3 scripts/...` 步骤，
  脚本已不在树里，且被 `|| true` 吞掉，静默 no-op。等同于 `cargo test` / `cargo bench`。
- 唯一注册的 bench 目标是 `cargo bench --bench filter_performance_bench --features text-bm25`。
- `examples/` 下 41 个文件里只有 3 个 Rust example 在 `Cargo.toml` 注册，其余 `.rs`
  （含 `examples/legacy/`）不参与编译，别拿它们当可运行示例。

---

## 7. 工作项清单

每一项都写成自包含的：换个会话、上下文丢了，照着这一节也能接着做。**动手前先读 §1 和 §5。**

约定：
- 每项完成后把 `[ ]` 改成 `[x]`，并在标题后追加 `— 完成于 <commit sha>`。
- 每项独立一个 commit，不要合并。
- 验收命令必须真的跑过，把输出贴进 commit message，不要写"应该可以"。
- 门禁是 `cargo test --lib`（当前 413 passed）。**不要用 `make verify`**，理由见 §6。

---

### [x] W1 — WAL 接入主写路径 + 启动恢复 — 完成于 ecadbc6

**目标**：建表 → 插入 N 条 → `kill -9` → 重启 → N 条全部可查。这是当前唯一缺失的核心能力。

**前置**：无。这是 P0 的第一项，后面几项都依赖它。

> **已完成。** 实测：3 条记录 + euclidean 距离的表，`kill -9` 后重启，表元数据、
> `record_count`、payload 全部恢复，三个 score 与重启前逐位相同，日志打出
> `WAL replay finished: 4 entries -> 1 tables, 3 records, 0 skipped`。
>
> 实现要点，与原计划的出入：
> - **没有改 `StorageConfig`**。它有 36 处字面量构造点，加字段要全改。改成
>   `IntelligentIndexedStorage::new_with_wal(config, Option<PathBuf>)`，`new()` 行为不变
>   （不持久化），只有 `api/server.rs` 走新构造函数，WAL 目录由 `DB_WAL_DIR` 决定，
>   默认 `./data/wal`。这样 46 处 `::new` 调用点一个没动，单测也不会互相回放污染。
> - **日志写在应用之后、返回之前**，不是严格的 write-ahead。因为内存状态本来就是从日志
>   重建的，"先应用后记录再确认"同样满足"已确认的写入不丢"，而且避免了处理器失败
>   （例如维度不符）时在日志里留下永远重放不成功的幽灵条目。
> - **`wal.rs` 的两个持久性缺陷一并修了**：`write_entry` 原本攒够 256KB 才落盘，
>   `kill -9` 会丢缓冲区；`flush_buffer` 用的 `Write::flush` 对 `File` 是空操作，
>   数据只到 page cache。现在每次 `wal_append` 都 `force_flush`，且 `flush_buffer`
>   改用 `sync_data()`。代价是每批一次 fsync，性能优化留给以后的组提交。
> - **`tests/integration/wal_recovery_test.rs` 无法复活**：它对着的 `StorageConfig` 有
>   `enable_wal` / `wal_dir` / `wal_batch_size` / `wal_sync_interval_ms` /
>   `wal_consistency_mode` 五个字段和一个 `WalConsistencyMode` 类型，现在一个都不存在。
>   已删除，改写为 `tests/wal_recovery_test.rs`（7 个测试全绿），其中
>   `replay_is_idempotent` 专门盯着"回放把读到的条目又写回日志"这个会让启动时间
>   逐次累积到永远起不来的坑。

**要改的文件**
- `src/storage/intelligent_indexed_storage.rs` — 主要工作量
- `src/wal.rs` — 需要扩展 `WalEntry`，见下面的坑
- `src/config/mod.rs` — 加 WAL 目录配置项
- `tests/integration.rs` — 取消 `wal_recovery_test` 的注释

**步骤**
1. `WalEntry::CreateTable` 目前只有 `{ table, dimension }`（`wal.rs:25`），**不足以重建一张表**——
   丢了 `distance_metric`、`named_vectors`、`payload_schema`、`memory_optimization`、
   `enable_full_text_search`。改成携带序列化后的完整 `TableConfig`，或新增一个
   `CreateTableV2 { config: TableConfig }` 变体并在 replay 时兼容旧变体。
   `TableConfig` 已经是 `Serialize`/`Deserialize`，直接用。
2. `IntelligentIndexedStorage` 加字段 `wal: Arc<Mutex<WalManager>>`（用 `std::sync::Mutex`：
   `write_entry` 是同步 I/O，不跨 await 持锁）。构造函数里按配置的 WAL 目录建好。
3. `insertion_processor`（`:212`）多收一个 `wal` 参数。**在写入 `tables` 之前**先
   `write_entry(WalEntry::Insert{..})`，写失败就直接返回错误、不改内存状态。
   注意它现在是自由函数，`tokio::spawn` 在 `:410` 拉起。
4. `create_table` / `delete_table` / `delete_records` 三个路径同样补 WAL 写入。
   注意 `insert_records` 里有 UPSERT 逻辑（`:245` 起），更新走的也是 Insert 条目，replay 时
   靠同样的 UPSERT 语义自然收敛，不需要额外的 Update 变体。
5. 构造函数末尾做 replay：`replay_from(0)` 拿到 `Vec<(u64, WalEntry)>`，**按序号顺序**先重建表
   再灌记录。灌记录要走真正的索引插入路径，不能只塞 `records` map，否则索引是空的。
6. `tests/integration.rs` 里 `wal_recovery_test` 那段注释（写着 "WAL tests disabled - WAL
   configuration has been restructured"）取消掉，`tests/integration/wal_recovery_test.rs`
   530 行还在，按新 API 调整。

**验收**
```bash
cargo test --lib                      # 不低于 413 passed
cargo test --test integration         # wal_recovery_test 全绿
# 端到端，必须手动跑一遍：
./target/debug/vectordb serve --port 19999 &
curl -sX POST localhost:19999/api/v1/tables -H 'Content-Type: application/json' \
     -d '{"name":"t","dimension":4,"distance_metric":"cosine"}'
curl -sX POST localhost:19999/api/v1/tables/t/records -H 'Content-Type: application/json' \
     -d '{"records":[{"id":"a","vector":[1,0,0,0]},{"id":"b","vector":[0,1,0,0]}]}'
kill -9 <pid>                         # 必须是 -9，不能给它优雅退出的机会
./target/debug/vectordb serve --port 19999 &
curl -s localhost:19999/api/v1/tables                    # 期望 total=1
curl -sX POST localhost:19999/api/v1/tables/t/search -H 'Content-Type: application/json' \
     -d '{"vector":[1,0,0,0],"top_k":2}'                 # 期望 2 条命中，score 与重启前一致
```

**坑**
- 别用 `atomic_ops.rs` 那条路。它的 `AtomicStorageManager` 只被 `cluster/shard_router.rs`
  引用，绑着 `_Storage` trait 和 LMDB/RocksDB 后端，和主路径的 `tables` map 是两套东西。
- `async_wal.rs`（§3.3）看着更合适，但它没有 replay，且零调用者零测试。先用同步的
  `wal.rs` 把语义跑通，性能问题留给 W4。
- WAL 目录默认值要落在工作目录下（例如 `./data/wal`），并加进 `.gitignore`。

---

### [x] W2 — 索引持久化（HNSW / HNSW-PQ 的 save / load）— 完成于 77c43db

**目标**：重启不需要把所有向量重新插一遍索引。

**前置**：W1。没有 W1 就没有可恢复的数据，这一项无从谈起。

> **已完成。** 2 万条 32 维记录，release 构建实测：
>
> | | 恢复耗时 |
> |---|---|
> | 全量回放 WAL | **151.56 秒** |
> | 从快照恢复 | **0.349 秒** |
>
> 434 倍。两次的 top-5 搜索结果与崩溃前逐位相同。快照本身耗时 0.6 秒，55MB
> （索引 52MB + 记录 5MB）。
>
> 增量正确性也验证了：快照后再插一条、删一条，`kill -9` 重启后新增的在、删掉的是 404、
> 总数正确，耗时仍是 0.355 秒。
>
> 实现要点：
> - **快照 = 每表 `.meta`（表元数据 + 记录 + 算法）+ `.idx`（索引本体）+ 一个 manifest
>   记录当时 WAL 的下一个序号**。恢复时先加载快照，再只回放该序号之后的日志。
>   只存索引不够——payload、版本号、时间戳都在记录里，光靠索引恢复出来 payload 全空。
> - **顺序是先取序号再落盘**。反过来的话，取序号之前的写入会既在快照里又被当尾部重放；
>   因为插入是 upsert 所以不出错，只是白做功。
> - **`SerializableIndex::load` 是静态构造函数，没法放进 `VectorIndex` trait**（调用方
>   必须先知道具体类型）。存是没这个问题的，所以只给 trait 加了
>   `snapshot_to(&self, path) -> Result<bool>`，默认返回 false 表示"这个索引类型没有磁盘
>   格式"（BruteForce 没什么可存的），与"存失败了"区分开。恢复时按 `.meta` 里记的
>   `IndexAlgorithm` 选类型加载。
> - **快照是优化，永远不是唯一副本**。manifest 读不出来就退回全量回放，只告警不报错——
>   有测试专门把 manifest 写坏来验证这条。
> - 触发时机：`POST /api/v1/snapshot`，以及 `index/rebuild` 之后自动做一次。
>
> **尚未做**：优雅退出时自动快照、定期快照。需要把 storage 的 `Arc` 传到 `main.rs`
> 才能挂 shutdown hook，而 `create_app_with_features` 现在只返回 `Router`。
> 另外 WAL 至今不会截断——快照之后旧日志仍然全留着，磁盘会一直涨。这两条见 W7。

**背景**：`SerializableIndex` trait 在 `src/search/index_traits.rs:38` 定义好了
（`save` / `load` / `stats`），全仓库只有 `src/search/ivf.rs:273` 一处实现。

**要改的文件**
- `src/search/hnsw.rs`、`src/search/hnsw_pq.rs` — 实现 `SerializableIndex`
- `src/storage/intelligent_indexed_storage.rs` — 启动时先试 load，失败再回退到 W1 的 replay

**步骤**
1. 给 `HnswIndex` 实现 `SerializableIndex`。图结构（`nodes`、各层邻接表、entry point）用
   bincode 序列化；注意 `DistanceMetric` 和 `HnswConfig` 也要一并存，load 时校验一致。
2. `HnswPqIndex` 同理，额外要存 PQ 码本（`product_quantization.rs` 里的 centroids）。
   码本不存的话 load 回来的 PQ code 无法解码。
3. 存盘时机：先做最简单的——`POST /tables/{n}/index/rebuild` 之后存一次，以及进程收到
   SIGTERM 时存一次。增量/定期 checkpoint 留给以后。
4. 启动顺序：load 索引成功 → 只 replay WAL 里 checkpoint 之后的条目；load 失败 → 全量 replay。

**验收**
```bash
cargo test --lib
# 插入 10 万条后重启，启动耗时应从"分钟级"降到"秒级"，把两次的实测数字贴进 commit
```

**坑**
- `IndexWrapper` 是 `Arc<RwLock<Box<dyn VectorIndex>>>`，而 `SerializableIndex` 不在
  `VectorIndex` trait 里。需要么给 `VectorIndex` 加一个 `fn as_serializable(&self) ->
  Option<&dyn SerializableIndex>`，么在 `IndexAlgorithm` 上分派。别为此把两个 trait 合并——
  `BruteForceIndex` 没有可存的东西。
- 版本兼容：存盘格式带一个 magic + version 头，否则改了结构之后老索引文件会静默解析错。

---

### [ ] W3 — TieredStorage 接线（已定：对标 Turbopuffer）

**目标**：消除全仓库最大的一处名实不符。**二选一，不允许继续放着。**

**前置**：W1、W2。

**现状**：`storage/object/tiered.rs` 实现完整、单测通过，`tier_promotion.rs` 的热度评分是真的，
但 `IntelligentIndexedStorage` 只在构造函数里建了个 `TieredStorage` 放进
`#[allow(dead_code)]` 字段（`:43-44`），全文 `self.storage` 出现 0 次。

**选项 A — 接上**：读路径按 `tier_promotion` 的 `hotness_score` 决定从内存/SSD/对象存储取，
写路径异步下沉到对象存储。这样 README 的 "object storage as the source of truth" 才成立。
工作量大，但这是产品差异化所在。

**选项 B — 删掉**：删 `storage/object/`、`storage/tier_promotion.rs`、`storage/factory.rs`，
把 README / CLAUDE.md 的架构描述改成"单节点内存 + WAL 持久化"。诚实，但放弃了定位。

**决策要点**：这不是技术问题，是产品定位问题。选 B 的话，产品就是"带持久化的高性能单机向量库"，
对标 Chroma；选 A 才是对标 Turbopuffer / LanceDB 的路子。

> **已决策（2026-08-14）：选 A，对标 Turbopuffer。** 接上 TieredStorage，对象存储为准。
>
> 这个定位决定了后面几件事的优先级：
> - W2 的快照格式就是分层存储的下沉单位——`.idx` + `.meta` 可以直接作为对象存储上的
>   不可变段文件，不用另起一套格式。
> - `tier_promotion.rs` 的 `hotness_score`（阈值 0.7 / 0.5 / 0.4 / 0.2，3600 秒窗口）
>   已经实现且有单测，缺的是读路径按它决定从哪一层取。
> - Turbopuffer 的关键在"对象存储为准 + 本地只做缓存"，所以写路径要先落对象存储再更新
>   本地层，不能反过来。这一点会影响 W2 快照的触发时机：下沉即快照。

---

### [x] W4 — MCP Server 接线 — 完成于 <SHA4>

**目标**：`vectordb serve --mcp`（或独立 bin）能被 Claude Desktop / Cline 之类的客户端连上。

**前置**：无，可以和 W1 并行。**投入产出比最高的一项**——代码基本齐了，缺的是十几行接线。

**现状**：`src/mcp/` 3,519 行，六个文件 server / tools / prompts / resources / transport，
`transport.rs` 有 2 个测试。`main.rs` 只有 `serve` 一个子命令，没有任何方式启动它。

**步骤**
1. 读 `src/mcp/server.rs`（948 行）和 `src/mcp/transport.rs`，确认 stdio transport 是否可用。
2. `main.rs` 加 `--mcp` 开关或 `Commands::Mcp` 子命令，把 `IntelligentIndexedStorage`
   注入 `mcp::server`。
3. `src/mcp/tools.rs:154` 有个 `"rebuild_index"` 工具定义，核对它调的是不是主路径上的
   `rebuild_index`。

**验收**：`vectordb mcp` 启动 stdio 服务，实测 initialize → tools/list（7 个工具）→
create_table → insert_vector → vector_search → list_tables 全部成功。

> **已完成。** 接线本身确实只有十几行，但顺带暴露了三个协议 bug，不修的话任何真实客户端
> 都连不上：
>
> 1. **`initialize` 直接失败**，报 `missing field 'protocol_version'`。MCP 线上格式是
>    camelCase，而这些结构体用 serde 默认的 snake_case。25 个协议结构体全部补了
>    `#[serde(rename_all = "camelCase")]`。同理 `input_schema` 也得是 `inputSchema`。
> 2. **`MCP_VERSION` 是 `"1.0.0"`**。MCP 的协议版本是日期串，改成 `"2024-11-05"`。
> 3. **`tools/list` 只返回 1 个工具。** `McpServer::new` 里 `tokio::spawn` 了一个任务往
>    catalogue 里塞工具，跟所有调用方赛跑；而真正定义了 7 个工具的 `initialize_tools()`
>    从来没人调用。改成 `pub async fn initialize()`，两个 transport 都在服务前先调。
>
> 另外 JSON-RPC 响应原本同时带 `result` 和 `error: null`，严格的客户端会拒。
> 两个字段都加了 `skip_serializing_if`。
>
> **stdout 是 JSON-RPC 通道**，所以 MCP 模式下 tracing 必须写 stderr，否则一行日志插进
> 消息流里，客户端报一个指向不明的解析错误就断开。`main.rs` 里按子命令切换 writer。
>
> `tests/mcp_stdio_test.rs` 直接 spawn 真实二进制来测——这里最容易坏的东西
> （日志串到 stdout、忘了 flush、字段大小写）全都在 handler 之外，直接调
> `handle_message` 的测试会在服务完全不可用的情况下通过。4 个测试，其中一个验证
> MCP 与 HTTP 共用同一份持久化存储：换个进程用同一个 WAL 目录，向量还在。

**客户端配置**（Claude Desktop / Cline 的 `mcpServers`）：

```json
{
  "mcpServers": {
    "dxmt-vectordb": {
      "command": "/path/to/vectordb",
      "args": ["mcp", "--wal-dir", "/path/to/data/wal"]
    }
  }
}
```

---

### [ ] W5 — clippy 清零

**目标**：让 `make verify` 第一次变成一个能用的门禁。

**前置**：无，随时可做，但**必须单独一个 commit**，别和功能改动混在一起。

**现状**：`cargo clippy --all-targets -- -D warnings` 报 154 个错误
（原始导入 commit `1668a8b` 上是 161 个）。分布最集中的：
`benchmarks/metrics/{map,ndcg,precision,evaluator}.rs` 共 56 个、
`index_management/` 26 个、`storage/intelligent_indexed_storage.rs` 6 个。

大部分是 `new_without_default`、`should_implement_trait` 这类机械问题。

**同时要处理**：`make fmt` 跑的是 `cargo fmt`（写入，不是 `--check`），仓库从未 fmt 干净过，
跑一次重排 208 个文件。建议把 Makefile 的 `verify` 目标改成用 `cargo fmt -- --check`，
再单独提一个纯格式化 commit。

---

### [ ] W6 — 修掉 §5 里的名实不符

**目标**：让函数名不撒谎。每一条都很小，可以分批做。

- [ ] `search/auto_tuned.rs` 四个函数全部直接调 `brute_force_search`——要么真的实现，
      要么删掉这个文件（它没有任何调用者）。
- [ ] `search/quantization/product_quantizer.rs` 52 行的假 PQ（`encode_batch` 返回全 0 码字）
      与 `product_quantization.rs` 的真实现同名并存。删掉假的，把 `fastscan/` 指向真的。
- [ ] `create_index_instance` 里 `DiskANN` / `ScaNN` / `QuantizedIVF` 三个分支静默退化成
      `BruteForceIndex`（`:1553` / `:1595` / `:1600`）——改成显式报错，或从
      `IndexAlgorithm` 枚举里去掉。静默退化最坏：用户以为在用 DiskANN。
- [ ] 建表日志三处自相矛盾：0 条记录的表打印 "Medium dataset detected (10K-100K)"，
      紧接着打印 "Selected HNSW"，而 API 返回的元数据是 `"index_type":"bruteforce"`。
      表元数据要反映真实选中的算法。
- [ ] `rebuild_index`（`:1260`）写死 `memory_budget_mb: 512` / `has_avx2: false` /
      `supports_simd: false`，没用构造时已检测好的 `self.hardware_capabilities`。
- [ ] `EnhancedDiskANN` 用 `TempDir::new()...keep()`（`:1570`）建索引目录，重启即孤儿。
- [ ] `POST /tables/{n}/hybrid_search` 静默丢弃 sparse 和 text 分量，只返回 dense 结果，
      却照收 `alpha`/`beta`/`gamma` 权重（`api/handlers.rs:961-966`）。要么实现，要么对
      带 sparse/text 的请求直接报 400。
- [ ] **批量写入期间整个库对读不可用。** `insertion_processor` 取一次
      `tables.write().await`，然后在锁内把整批记录逐条插进 HNSW。一次 5000 条的请求会把
      写锁攥到全部插完，期间连 `GET /tables` 都阻塞——debug 构建下实测好几分钟。
      改法：分片持锁，或者把索引插入挪到锁外（记录 map 和索引分两阶段）。
- [ ] `/tables/{n}/records` 有 10000 条的批次上限，超了返回
      `{"error":"Validation failed","details":"Batch size 20000 exceeds maximum of 10000"}`，
      README 和 SDK 文档都没写。

---

### [ ] W7 — 快照的运维缺口

**前置**：W2（已完成）。这几条是 W2 交付后剩下的口子，不做的话快照在生产里不好用。

- [ ] **WAL 永不截断。** 快照之后旧日志仍然全留着，磁盘只涨不降。`wal.rs` 有
      `checkpoint()` 但没人调。快照成功后应该以 manifest 里的序号为界截断。
- [ ] **优雅退出不快照。** 现在只有 `POST /snapshot` 和 `index/rebuild` 会触发，
      正常 `SIGTERM` 停机不会存，下次启动仍要全量回放。需要把 storage 的 `Arc` 传到
      `main.rs` 挂 shutdown hook，而 `create_app_with_features` 目前只返回 `Router`。
- [ ] **没有定期快照。** 长期只写不重建索引的库，WAL 会一直长。
- [ ] **快照期间持 `tables` 读锁遍历所有表**，大库上会阻塞写入。
