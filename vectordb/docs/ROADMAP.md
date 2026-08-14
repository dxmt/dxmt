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

1. **没有持久化。** `IntelligentIndexedStorage` 全文没有一次 `self.storage` 调用，也没有任何
   `wal` 引用。实测：建表 → 插入 → 重启，表列表返回空，记录 404，不生成 `./cache` 或 `./data`。
   唯一的落盘路径是手工调用 `backup()`（`storage/intelligent_indexed_storage.rs:2189`）写一个
   JSON 快照。
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

- `.github/workflows/` 在 `vectordb/` 下，而 GitHub Actions 只读真实仓库根的 `.github/`，
  **CI 不会触发**。合并前必须本地 `make verify`。
- `make test` / `make test-integ` / `make bench` 各自追加了一个 `python3 scripts/...` 步骤，
  脚本已不在树里，且被 `|| true` 吞掉，静默 no-op。等同于 `cargo test` / `cargo bench`。
- 唯一注册的 bench 目标是 `cargo bench --bench filter_performance_bench --features text-bm25`。
- `examples/` 下 41 个文件里只有 3 个 Rust example 在 `Cargo.toml` 注册，其余 `.rs`
  （含 `examples/legacy/`）不参与编译，别拿它们当可运行示例。
