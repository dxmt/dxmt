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

## 6.5 与其他向量数据库的对照

`~/code/dxmt_vectordb/reference/db` 下有 chroma / lancedb / qdrant / weaviate 的源码
（milvus 目录是空的，turbopuffer 只有客户端）。以下是按**我们实际做过的决策**逐条对照的结果，
定向阅读关键模块，不是全量审计。

**WAL 放对象存储 —— chroma 的 `wal3` 比我们完整得多。** 它**不要本地日志**：`Fragment`
不可变、`Manifest` 可变（靠 S3 `If-Match` 做 CAS，因此不需要任何额外的锁或协调）、
`Snapshot` 作为树的内部节点收拢老 fragment 以免 manifest 无限增长、`Setsum`（可交换可结合的
密码学哈希）支持增量完整性校验、`Cursor` 钉住位置防 GC、fragment 按 4096 一组分前缀
**因为 S3 按前缀限流**。

原本四个缺口，现在补了三个：**清单**（`wal/MANIFEST`，恢复不再 LIST）、**完整性不变式**
（`live + pruned == total`，对不上就重扫重建）、**前缀分桶**（`wal/{seq/4096}/…`，因为 S3
按前缀限流）。

**没补的是多写者安全。** wal3 靠 `If-Match` 条件写做 CAS，而我们的 `ObjectStorage` trait
根本没有条件写，所以清单更新是读-改-写。**这意味着单写者**：两个进程共用同一个桶会互相覆盖清单。
现在的架构本来就是单进程持有 WAL，但这条限制要写在明处，不能假装没有。

同样没有 GC 钉扎（wal3 的 `Cursor`）——我们的裁剪由快照序号驱动，没有第二个读者需要保护。

**重建不阻塞 —— qdrant 的 `ProxySegment` 比我们的 catch-up 更彻底。** 优化期间把只读段包起来，
新写入落到临时 segment，全程不持全局写锁。我们的 catch-up 已经改成在锁外收敛（2d83af4），
只有最后一次 delta 在写锁内应用——但那次 delta 的大小没有上界，量大时仍然是全局停顿；
qdrant 的做法根本不需要那一下。

> 曾经有个 `a_rebuild_under_write_load_does_not_stall_other_tables` 测试守这条，**已删除**。
> 两个原因，都是致命的：它对着"整个 catch-up 都在锁内"的旧版本也通过（这个规模下重建太短，
> 落进来的写太少，delta 两种写法都很小），所以它抓不到它要防的 bug；而且它拿
> `with_rebuild < baseline * 3` 做判据，baseline 只有几十毫秒且方差极大，机器一忙就 2/3 失败。
> 一个抓不到目标 bug、又会随机变红的计时测试，比没有更糟。空闲机器上 6/6 通过、繁忙时 2/3
> 失败这件事，我一度当成自己引入的回归去二分了半天。理由留在 `rebuild_index` 的注释里。
他们那个文件里还记录了一个同类坑：`deleted_mask` 在段冻结前拍快照，中间的 upsert 会让过滤
KNN 静默丢点——修法是**类型状态**，`new()` 返回不可用类型，只有持锁调 `finalize` 才拿得到可用的。

**"自适应索引"—— 我先前说"没有一家自动切换算法"，这是错的。**（只看了 qdrant 和 chroma
就下了结论。）

**weaviate 有专门的 `dynamic` 索引类型**（`adapters/repos/db/vector/dynamic`）：起始是
flat（暴力），对象数超过 `DefaultThreshold = 10_000` 自动升级为 HNSW；升级后还会向内层索引
继续问 `ShouldUpgrade()`，第二级是量化（SQ/RQ）。

不过看清楚层次之后，结论要改成更准确的说法：**大家自动的都是"何时建/何时压"，不是"选哪一族
算法"**。

| | 自动的部分 | 配置的部分 |
|---|---|---|
| weaviate | flat → HNSW（10,000 对象）、量化的时机 | 用不用量化、用哪种 |
| qdrant | 不建索引 → HNSW（`indexing_threshold_kb`） | 量化 |
| chroma | — | HNSW 还是 SPANN |
| lancedb | — | 索引类型；连 `optimize()` 都要应用自己调 |
| **我们** | **五选一**（BruteForce / HNSW / HNSW-PQ / EnhancedDiskANN / PartitionedIvf） | — |

所以我们的五路自动选择确实比任何一家都激进。这不必然是错的，但**没有先例**这件事本身是信息。

**"先建好再换上"weaviate 独立走到了同一处。** 它 `doUpgrade()` 里的注释写着：
*"The read lock keeps the current index alive (not dropped/closed) while the new one is built;
searches continue throughout."* ——和我们那次改动一字不差地是同一个性质。

**阈值单位并没有业界共识。** weaviate 用对象数（10,000），qdrant 用字节
（`indexing_threshold_kb`）。我们改成按字节的理由（不同维度下同样行数内存差几十倍）站得住，
但不能说成"大家都这么做"。

**冷数据卸载：weaviate 是显式的，而且卸载后不可查。** 租户有 `ACTIVE` / `INACTIVE` /
`OFFLOADED` 三态（外加 `OFFLOADING` / `ONLOADING` 两个只读中间态），转换由用户通过 API 驱动，
**`OFFLOADED` 的租户直接不可查询**，必须显式激活。

我们是热度驱动自动驱逐 + 查询时透明回填。更方便，但**藏了一个延迟悬崖**：调用方无从知道
下一次查询要付一次对象存储拉取的代价。weaviate 把它变成一个必须显式处理的状态。
折中做法已经实现：`GET /tables/{name}/stats` 现在返回 `resident` 字段。

**维护的调度方式四家四样**：qdrant 后台 optimizer、chroma 独立 compactor 服务、
weaviate 自动升级 + 显式租户生命周期、lancedb **完全交给应用**（文档原话：改动超过 10 万条
或 20 次修改操作就该调一次 `optimize()`）。我们是后台定时器。

**近 3 个月各家在改什么**：qdrant 的 WAL 提交几乎全是**回放与截断交互**的 bug（其中
`Clamp WAL replay target to the truncated prefix` 与我们 W7 做的完全同类）；chroma 的 wal3
全在加**条件写与 admission predicate**，外加一串 GC frontier 的 bug。
结论：**对象存储 WAL 的难点不在写，在 frontier / GC 边界的交互。**

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

### [x] W3 — TieredStorage 接线（对标 Turbopuffer）— 完成于 2eeba00 + 938f35e + 351c041 + 0d4f3ef + aec6779

**目标**：消除全仓库最大的一处名实不符。**二选一，不允许继续放着。**

**前置**：W1、W2。

**现状**：`storage/object/tiered.rs` 实现完整、单测通过，`tier_promotion.rs` 的热度评分是真的，
但 `IntelligentIndexedStorage` 只在构造函数里建了个 `TieredStorage` 放进
`#[allow(dead_code)]` 字段（`:43-44`），全文 `self.storage` 出现 0 次。

**选项 A — 接上**：读路径按 `tier_promotion` 的 `hotness_score` 决定从内存/SSD/对象存储取，
写路径异步下沉到对象存储。这样 README 的 "object storage as the source of truth" 才成立。
工作量大，但这是产品差异化所在。

> **第一阶段已完成：快照段以对象存储为持久归宿。**
>
> `storage: Arc<TieredStorage>` 上的 `#[allow(dead_code)]` 去掉了——它现在真的在用。
> 快照校验通过后，`manifest` / `tN.meta` / `tN.idx` 逐个 `put_object` 到 `snapshot/` 前缀；
> 启动时若本地快照目录不存在，就先从对象存储拉回来再走原有的恢复流程。
>
> 验收测试 `a_fresh_machine_recovers_from_object_storage_alone`：A 机写数据+快照，
> B 机用**另一个空的 WAL 目录**启动，只共享对象存储 —— 表、记录数、索引全部恢复，
> 搜索结果正确。测试还直接断言 `<root>/<bucket>/snapshot/manifest` 和 `t0.meta`
> 确实落在了对象存储的文件布局里，免得靠某个共享的本地缓存蒙混过关。
>
> 几个设计点：
> - **只有真配了 `object_storage` 才启用卸载**。`TieredStorage::new_from_config` 在没配时会
>   退回到临时目录下的 FilesystemStorage——往临时目录传快照比不传更糟，看着像有异地副本，
>   其实没有。
> - **段文件不可变**：一次快照写一组新键，没有读改写，也没有并发覆盖问题。这正是
>   W2 的快照格式能直接当段格式用的原因。
> - **上传在截断日志之前**。上传失败只告警：本地快照仍在且已校验，压缩依然安全，
>   但"能扛住这台机器没了"的副本是远端那份，所以要说出来。
> - `.idx` 是可选的（BruteForce 没有磁盘格式），下载时缺失不算错误。
>
> **已知的坑（`new_from_config` 的设计问题）**：`provider = "filesystem"` 时，对象存储的根路径
> 直接复用了 `ssd_cache_dir`，于是 SSD 缓存层和对象存储层落在同一棵目录树下（靠 bucket
> 子目录区分）。S3 provider 没这个问题。要修的话得给 filesystem provider 单独一个路径配置。
>
> **第二阶段已完成：冷表整体驱逐 + 查询时回填。**
>
> Turbopuffer 的分层单位是命名空间，不是单条向量——所以这里做的是表级：一张表长时间没人
> 访问就整个放出内存（记录 + 索引），只留对象存储里的快照；下一次查询自动拉回来。
> 内存占用从此由**工作集**决定，而不是数据总量。
>
> - `ensure_resident()` 挂在 `search` / `get_table` / `get_record` / `list_records` /
>   `insert_records` 五个入口。回填对调用方完全透明。
> - `evict_cold_tables()` **先快照再驱逐，且只驱逐这次快照覆盖到的表**。反过来会丢数据：
>   一张表若有快照之后的写入，驱逐等于把它们扔了——WAL 里虽然还在，但回填只读快照那份。
>   有测试 `eviction_never_drops_writes_made_after_the_last_snapshot` 专门盯这条。
> - `list_tables()` 会把被驱逐的表从快照元数据里补回来。只看内存的话，一张表会因为没人访问
>   而从列表里消失，对调用方来说和被删掉没区别。补的时候只读 `.meta` 不读记录——
>   驱逐的意义就是别为了列个表把数据全拉回来。
> - 后台扫描：`DB_EVICTION_INTERVAL_SECS`，默认 600 秒，设 0 关闭。
>
> **判定用的是 `should_demote_to_cold` 而不是 `get_recommended_tier`。** 后者把任何低于
> *warm* 阈值（0.4）的都叫 Cold，而刚写完的表只有约 0.31 分——频率项是按 100 次访问归一化的，
> 几次访问几乎不计分，近期性最多贡献 0.3。第一版用了 `get_recommended_tier`，测试立刻抓到
> "刚写完的表被驱逐"。改用降级判定（阈值 0.2）后，语义变成"约三分之一个窗口没人碰"。
> 常驻表会被标为 Warm，因为降级规则只对 Warm 生效。
> - 写路径的增量下沉：目前只有快照粒度（默认 300 秒一次），不是连续下沉。
> - FastScan：W6 删掉的那个实现是假的（PQ 编码全 0、`search_simd` 里没有一条 SIMD 指令），
>   但对象存储上的扫描确实需要它，要重写。

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

### [x] W4 — MCP Server 接线 — 完成于 6d82be3

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

### [x] W5 — clippy 清零 — 完成于 ff8981e

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

> **已完成。`make verify` 现在真的能过了**（"All checks passed!"，exit 0）。
>
> - clippy `--all-targets -D warnings`：**154 → 0**。约三分之二由 `cargo clippy --fix`
>   自动完成，其余手工。
> - **`benches/filter_performance_bench.rs` 自导入以来就编译不过**——它 import 的是
>   crate `vectordb`，而 lib 名是 `dxmt_vectordb`，且 `TableConfig` / `CreateRecordRequest`
>   / `SearchQuery` / `StorageConfig` 四个字面量全是旧结构。所以 CLAUDE.md 里那句
>   "唯一注册的 bench 目标"从来没跑起来过。已按当前 API 修好。
> - `make verify` 的 `fmt` 步骤换成新的 `fmt-check`（`cargo fmt -- --check`）。**会改你文件的
>   东西不能当门禁。** `lint` 也补上了 `--all-targets`，否则 tests/benches/examples 全不检查。
> - Makefile 里三处指向已不存在的 `scripts/` 的 `python3 ... || true` 步骤删掉了——
>   它们静默 no-op，让 target 看起来做了比实际更多的事。
> - 纯格式化单独一个 commit（212 文件）。注意 `cargo fmt` 在本仓库**不是幂等的**：
>   第一遍之后 `--check` 仍报 `src/storage/mod.rs` 一处行尾注释对齐，需要再跑一遍才收敛。
>
> 两个 `assert!(true)` 是删掉而不是改写的。一个在 GPU 测试里，有没有 GPU 都是合法结果，
> 那个测试的意义是证明 `try_new` 不 panic，注释改成了这么说；另一个守着一条需要 live
> factory 的 `IndexManager` 路径，改成 `#[ignore]` 并写明原因——通过的同义反复不算诚实。

---

### [x] W6 — 修掉 §5 里的名实不符 — 完成于 5716888 + fa7f6db

**目标**：让函数名不撒谎。每一条都很小，可以分批做。

- [x] `search/auto_tuned.rs` 四个函数全部直接调 `brute_force_search`。**已删除**——零调用者，
      459 行，带 3 个测试（掉的 3 个测试即出自此处）。
- [x] 假 PQ **已删除**，连同它唯一的使用者 `search/fastscan/`。原计划是把 fastscan 指向真
      实现，但真 PQ 的 API 完全不同（`encode` vs `encode_batch`、没有 `get_centroid`），
      而且 `fastscan` 的 `search_simd` 里**一条 SIMD 指令都没有**（`_mm*` 出现 0 次），
      只是个标了 `#[target_feature(enable = "avx2")]` 的标量循环。这不是修 lint，是重写一个
      特性——记进 W3，FastScan 对对象存储上的扫描确实有价值。
- [x] 三个分支**改成显式报错**并指明替代品。`index_management/factory.rs` 里同样的
      `Lsh`/`IvfPq` 静默退化也一并改了。
- [x] 日志改成陈述实际选中的算法并注明未实现的回退；`IndexType` 补了 `HnswPq` 和
      `EnhancedDiskAnn` 两个变体，建表时按选中的算法回填 `table.index_type`——此前它一律来自
      调用方配置、默认 `BruteForce`，所以跑着 HNSW 的表对外永远自称 brute_force。
- [x] `rebuild_index` **改用真实检测到的硬件能力**和构造时的性能/成本约束。原先写死的值
      会让同一台机器上"建表"和"重建"选出不同的算法。
- [x] `EnhancedDiskANN` 的索引目录移到 `<wal_dir>/diskann/<uuid>`。原来的
      `TempDir::new()...keep()` 是**故意**泄漏临时目录好让它活过 handle，于是每次重启都在
      /tmp 下多一份索引，且无人回收。没有 WAL 时仍退回临时目录——那种配置下本来就没有持久状态。
- [x] `POST /tables/{n}/hybrid_search` 带 sparse/text 时**直接返回 NOT_IMPLEMENTED** 并指明
      `/search/text` 和 `/search/hybrid` 两个能用的端点。此前它跑完 dense 搜索就把另外两路
      丢掉、当作融合完成返回，调用方的 alpha/beta/gamma 权重不影响任何东西。
- [ ] **批量写入期间整个库对读不可用**（尚未修，已写进 README 和 CLAUDE.md 提醒）。 `insertion_processor` 取一次
      `tables.write().await`，然后在锁内把整批记录逐条插进 HNSW。一次 5000 条的请求会把
      写锁攥到全部插完，期间连 `GET /tables` 都阻塞——debug 构建下实测好几分钟。
      改法：分片持锁，或者把索引插入挪到锁外（记录 map 和索引分两阶段）。
- [x] `/tables/{n}/records` 的 10000 条批次上限已写进 README。原文：超了返回
      `{"error":"Validation failed","details":"Batch size 20000 exceeds maximum of 10000"}`，
      README 和 SDK 文档都没写。

---

### [x] W7 — 快照的运维缺口 — 完成于 474a986

**前置**：W2（已完成）。这几条是 W2 交付后剩下的口子，不做的话快照在生产里不好用。

- [x] **WAL 现在会截断。** 新增 `WalManager::truncate_before(seq)`，快照成功后按 manifest
      的序号截断。实测优雅停机后 `wal.log` 从 155 字节压到 0。
      两个坑：写临时文件再 rename（中途中断保留旧日志——丢日志比不压缩糟得多）；
      以及**截断后不能让序号倒退**——`open_current_file` 是从文件扫出下一个序号的，
      日志被清空时会重新从 1 开始，与快照 manifest 已引用的序号冲突。有测试专门盯这条。
- [x] **优雅退出会快照。** 加了 `create_app_with_storage`（返回 `(Router, Arc<Storage>)`），
      `main.rs` 用 `with_graceful_shutdown` 接 SIGTERM/Ctrl-C，drain 之后做一次快照。
      实测：SIGTERM → 快照落盘 → WAL 归零 → 重启 `Snapshot loaded: 1 tables` +
      `No WAL entries after sequence 3`。
- [x] **定期快照**：`DB_SNAPSHOT_INTERVAL_SECS`，默认 300 秒，设 0 关闭。
      失败只告警不致命——WAL 里还有全部内容，代价只是下次启动慢一点。
- [ ] **快照期间持 `tables` 读锁遍历所有表**，大库上会阻塞写入。（未修）

**这一项改变了一条安全属性，必须知道：**

W2 里写的"快照是优化，永远不是唯一副本"**在截断之后不再成立**。日志一旦按快照序号压缩，
那部分写入就只存在于快照里了。所以配套加了两道防线：

1. **落盘后先校验再截断**（`verify_snapshot`）：manifest、每个 `.meta`、以及每个声称有索引的
   `.idx` 都要真的读得回来。"写完了"不等于"读得回来"。
2. **快照存在却读不出来时硬失败，不再静默全量回放。** 原来的行为是告警一句然后当无事发生，
   在截断之后那等于"空库启动 + 一条没人看的日志" —— 静默数据丢失。现在直接拒绝启动，
   错误信息里写明后果和两个选择（从备份恢复，或删掉快照目录、接受损失后用剩下的日志启动）。

原来那个 `a_corrupt_snapshot_falls_back_to_full_replay` 测试因此不再成立，
改写成 `a_corrupt_snapshot_is_refused_rather_than_silently_ignored`，
另加 `a_snapshot_that_cannot_be_read_back_does_not_compact_the_log` 验证压缩确实发生。

---

### [x] W8 — 拿真二进制做一次 soak，修掉它暴露的问题

W1–W7 的验证全是进程内的：直接调 `snapshot()` / `evict_cold_tables()` / `reevaluate_indexes()`。
真实进程里这三个是**三个后台定时器**，和读写流量并发跑。这个组合从没跑过一次。

跑法（脚本在 `scripts/soak.py`）：真 `vectordb serve`，间隔取互质的 5/7/11 秒让三个 sweep
错峰重叠，两个写线程 + 三个读线程打 120 秒，然后 `kill -9`、重启、逐条校验每一条服务端
返回过 200 的记录。

**结果：写读全绿（96,000 条 acked，0 失败），重启起不来**——600 秒没有绑定端口。
两个 bug，都只有真进程能看见。修完之后重启 **5.8 秒 / 27.3 秒**（两次跑），0 条丢失。

- [x] **四个调用点共用一个 staging 目录。** 定期快照、淘汰前快照、重建后快照、`POST /snapshot`
      都能并发。`snapshot()` 第一句是 `remove_dir_all(&staging)`，后来者删掉前一个写了一半的
      文件。**默认配置下这不是罕见竞争**：快照 300 秒、淘汰 600 秒，每 600 秒必然同刻触发一次。

      让它变静默的是 `let has_index = ....unwrap_or(false)`。`Ok(false)` 的合法含义是"这类索引
      没有磁盘格式"（BruteForce 就是），`Err` 被折叠成同一个值。于是索引写失败 → 记成"没有磁盘
      格式" → 目录被重新创建、元数据赶上了 → **快照报告成功，HNSW 图被静默丢掉**。恢复路径无从
      察觉，只是从原始行重建，正确且慢到没有上界。

      修法：`snapshotting: Arc<tokio::sync::Mutex<()>>`，在**取 WAL 序号之前**上锁（放在之后
      仍然可能两个快照各自采到序号，然后把日志截断到较晚那个位置，而落盘的是较早那个的
      manifest）；`unwrap_or(false)` 改成 `?`。

- [x] **恢复时忽略快照记录的算法。** BruteForce 不写索引文件，恢复要重新索引——但索引进
      *哪个* 索引？进的是 `create_table` 按那个 10,000 条固定估计选的，也就是 HNSW。于是
      **94,000 条的 BruteForce 表被还原成 HNSW**，要在绑定端口之前先建完一整张图，600 秒没起来。
      快照如实记了 `algorithm`，恢复路径只在"有索引文件"时才用它。

- [x] `restore_table` 里 `.ok()` 吞掉索引加载失败，与上面同一个模式：快照说写了图，图读不回来时
      静默退化成全量重建，唯一症状是启动变慢。改成告警。回填改用 `bulk_build`——分区索引一条条
      喂会全积在溢出缓冲里。

- [x] **日志可诊断性。** 重选算法那行打的是 `std::mem::discriminant(&current)`，渲染成
      `Discriminant(0)`，读者得知道 `IndexAlgorithm` 的变体顺序才能解出表原来用的是什么。
      重建完成行原本**有条件**打印（只在 delta 非零时），于是"重建还在跑"和"重建已完成"
      在日志里长得一模一样——soak 里我因此分不清 47 秒的 HNSW 重建是哪种。改成无条件并带耗时。

**没修，但 soak 量出来了，是下一个真问题：**

- [ ] **HNSW 重建在写入负载下跑不完。** 三次 soak，36,000–38,400 条的 HNSW 重建**一次都没有
      打印过 `swapped in`**——120 秒跑完全程它都没换入过。所以表全程停在 BruteForce，
      查询代价随行数线性上涨（读延迟从 0.10s 涨到 1.15s，而这正是该换索引的理由）。
      `bulk_build` 对 HNSW 是 `index_traits` 里的默认实现——一个 `add` 循环，没有真正的批量构建；
      再叠加 `MAX_CONVERGENCE_PASSES = 8` 的追赶，每一趟要补的量取决于上一趟耗时，写入快过
      插入时这个量是**递增**的。**写入负载下"自适应索引"实际上升级不成功。**
      这是 `TODO.md` 竞品对比里的核心指标，该单独立项。

- [ ] **整进程停摆，三次里出现一次，未诊断。** 同一份代码同一个负载跑三次：

      | 时间 | acked | 失败 | 最大读延迟 |
      |---|---|---|---|
      | 18:44 | 96,000 | 0 | 1.07s |
      | 19:25 | 42,400 | **15 次 30 秒超时** | 30.0s |
      | 19:30 | 87,400 | 0 | 1.15s |

      中间那次：HNSW 重建开始后**日志完全空白 98 秒**——连每 5 秒一次的定期快照都没打印，
      然后 20 个积压的快照在两秒内一起冲出来。不是某张表慢，是整个进程停了。
      定时器积压说明 tokio 的 worker 全被占住：搜索、HNSW 插入、快照的文件 IO 都是
      **不让出的 async fn**，CPU 密集的活没有走 `spawn_blocking`，协作式调度就此失效。
      这是**假说**，不是结论——健康那次的逐线程 CPU 采样（重建期间 84%→199%，4 核占 2 核）
      对停摆没有解释力，而停摆本身没能在采样窗口里复现。先复现再修。

**参考实现怎么解这两条**（qdrant 和 weaviate 都把它当一等问题处理，我们完全没做）：

- **qdrant：优化任务要向一个信号量 CPU 预算申请配额才能开跑。**
  `lib/common/common/src/budget.rs` 的 `ResourceBudget` 持有 `Semaphore::new(cpu_budget)`；
  `get_cpu_budget` 的默认策略按机器规模给服务**留核**（`default_cpu_budget_unallocated`：
  3–32 核留 1，33–48 留 2，129 核以上留 1/16）。申请时拿到多少算多少，但
  **低于期望值的一半就干脆不启动**（`min_cpu_permits` = `div_ceil(2)`），转而
  `notify_on_budget_available` 等着。Linux 上还给优化线程 `set_current_thread_priority` 降优先级。
  我们的重建是 900 秒 sweep 一到就无条件开跑，和查询在同一个 tokio worker 池里抢。

- **qdrant 的 HNSW 图是并行建的，线程数就等于申请到的配额。**
  `hnsw/build.rs:199` 用 `ThreadPoolBuilder::new().num_threads(permit.num_cpus)` 起专用 rayon 池，
  `pool.install(|| ids.into_par_iter().try_for_each(insert_point))` 并行插点。
  我们的 `bulk_build` 是 `index_traits` 里的默认实现——单线程 `add` 循环，跑在 tokio 上。
  `thread_count_for_hnsw` 还有条经验注释值得记：**超过 16 线程不再有收益，反而更容易建出断连的图**。

- **weaviate：索引构建整体挪到队列后面**（`adapters/repos/db/queue/`）。worker 数是
  `max(1, GOMAXPROCS(0)-1)`——同样是显式留一个核。关键是待索引的向量先落
  **DiskQueue**（`vector_index_queue.go:172`），崩了不丢；而且**异步索引是 opt-in，
  默认仍是同步的 `AddVectorsToIndex`**，因为它拿读的新鲜度换吞吐。

  这条比 qdrant 那条更彻底，但代价也更大——我们要先有 W8 那个停摆的诊断，才知道该不该走这么远。

- [ ] **算法在阈值附近抖动。** soak 里 `hot` 在 1,000 条时被从 HNSW 降级成 BruteForce，10 秒后
      到 38,400 条又升回 HNSW，各付一次全量重建。weaviate 的 `dynamic` 索引是**单向**的：
      flat 起步，过 10,000 升 HNSW，不再降回。我们的 re-evaluation 是对称的，一张正在长大的表
      只要被某次 sweep 采样在低点就白建一次。

- [ ] **淘汰在 soak 里从未触发过。** 两次跑都没有 `Evicted` 日志。第一次是我的脚本有个读线程
      一直在查 `cold`，它永远热不下来；第二次去掉了读线程仍没触发，原因未查。
