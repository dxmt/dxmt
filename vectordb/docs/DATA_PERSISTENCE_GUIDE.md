# DXMT VectorDB 数据持久化指南

**日期：** 2026-01-25
**版本：** 1.0

---

## 快速回答您的问题 💡

### Q1: 数据目录默认在哪里？

**答案：取决于存储模式**

| 存储模式 | 数据目录 | 持久化 |
|---------|---------|--------|
| **Memory (默认)** | 无（内存中） | ❌ 否 |
| **Tiered** | `./cache` | ⚠️ 部分 |
| **LMDB** | `./data/lmdb` | ✅ 是 |
| **S3 Object Storage** | S3 Bucket | ✅ 是 |

### Q2: 重启后数据会自动加载吗？

**答案：取决于存储模式**

| 存储模式 | 重启后 | 说明 |
|---------|--------|------|
| **Memory** | ❌ 数据丢失 | 内存存储，重启清空 |
| **Tiered (默认)** | ❌ 数据丢失 | 主要在内存，缓存丢失 |
| **LMDB** | ✅ 自动加载 | 从磁盘加载 |
| **S3** | ✅ 自动加载 | 从对象存储加载 |

---

## 详细说明

### 1. 默认存储配置

**从源代码分析（`src/config/mod.rs:217`）：**

```rust
let storage_type = std::env::var("DB_STORAGE_TYPE")
    .unwrap_or_else(|_| "tiered".to_string());
```

**默认存储类型：** `tiered`（三层存储）

**三层存储架构：**
```
┌─────────────────────┐
│ Memory (Hot Tier)   │  512MB LRU Cache
│ - 最快访问          │  <1ms latency
│ - 非持久化          │  ❌ 重启丢失
└─────────────────────┘
          ↓
┌─────────────────────┐
│ SSD (Warm Tier)     │  10GB Filesystem Cache
│ - 快速访问          │  <50ms latency
│ - ./cache 目录      │  ⚠️ 缓存，可清理
└─────────────────────┘
          ↓
┌─────────────────────┐
│ S3 (Cold Tier)      │  Unlimited Capacity
│ - 对象存储          │  <500ms latency
│ - 需要配置          │  ✅ 持久化
└─────────────────────┘
```

### 2. 各存储模式详解

#### 模式 1: Memory Storage（内存存储）

**配置：**
```bash
export DB_STORAGE_TYPE=memory
```

**特点：**
- ✅ 最快：所有数据在内存
- ✅ 适合：开发、测试、临时数据
- ❌ 非持久化：重启后数据全部丢失
- ❌ 容量限制：受物理内存限制

**数据目录：** 无（纯内存）

**重启行为：**
```
服务器停止 → 所有数据丢失 → 重启后空数据库
```

#### 模式 2: Tiered Storage（分层存储，默认）

**配置：**
```bash
export DB_STORAGE_TYPE=tiered
export DB_SSD_CACHE_DIR=./cache  # 默认值
```

**特点：**
- ✅ 平衡性能和成本
- ✅ 自动数据分层
- ⚠️ 主要数据在内存
- ⚠️ SSD 仅用作缓存

**数据目录：**
```
./cache/              # SSD 缓存目录
├── vectors/          # 向量缓存
├── indices/          # 索引缓存
└── metadata/         # 元数据缓存
```

**重启行为：**
```
服务器停止 → 内存数据丢失 → 重启后需要重新插入数据
            ↓
         缓存可能保留部分数据（但不完整）
```

**⚠️ 重要：** Tiered 模式虽然有 ./cache 目录，但这只是**缓存**，不是完整的持久化存储！

#### 模式 3: LMDB Storage（持久化存储）

**配置：**
```bash
# 需要编译时启用 storage-lmdb feature
cargo build --features storage-lmdb

# 运行时配置
export DB_STORAGE_TYPE=lmdb
export DB_LMDB_PATH=./data/lmdb    # 数据目录
export DB_LMDB_MAP_SIZE=10737418240  # 10GB
```

**特点：**
- ✅ 完全持久化到磁盘
- ✅ ACID 事务保证
- ✅ 重启自动加载
- ✅ 适合生产环境

**数据目录：**
```
./data/lmdb/         # LMDB 数据目录（默认）
├── data.mdb         # 主数据文件
└── lock.mdb         # 锁文件
```

**重启行为：**
```
服务器停止 → 数据保存在 data.mdb → 重启后自动加载 ✅
```

**文件说明：**
- `data.mdb`: 包含所有向量、元数据、索引
- `lock.mdb`: LMDB 锁文件
- 数据完整性由 LMDB 保证

#### 模式 4: S3 Object Storage（云存储）

**配置：**
```bash
export DB_STORAGE_TYPE=tiered
export DB_OBJECT_STORAGE_PROVIDER=s3
export DB_S3_BUCKET=my-vectordb-bucket
export DB_S3_REGION=us-east-1
export DB_AWS_ACCESS_KEY_ID=your-key
export DB_AWS_SECRET_ACCESS_KEY=your-secret
```

**特点：**
- ✅ 无限容量
- ✅ 完全持久化
- ✅ 跨服务器共享
- ⚠️ 延迟较高（首次访问）
- ✅ 适合大规模数据

**数据目录：**
```
S3 Bucket: my-vectordb-bucket
├── tables/
│   ├── table1/
│   │   ├── vectors/
│   │   ├── metadata/
│   │   └── indices/
│   └── table2/
└── snapshots/
```

**重启行为：**
```
服务器停止 → 数据在 S3 → 重启后从 S3 加载（5分钟内） ✅
```

---

## 如何实现数据持久化

### 方案 1: 使用 LMDB（推荐）

**1. 编译启用 LMDB：**
```bash
cd /localdisk/code/dxmt/vectordb
cargo build --release --features storage-lmdb
```

**2. 配置环境变量：**
```bash
# 创建 .env 文件
cat > .env << 'EOF'
DB_STORAGE_TYPE=lmdb
DB_LMDB_PATH=./data/lmdb
DB_LMDB_MAP_SIZE=10737418240  # 10GB
EOF
```

**3. 运行服务器：**
```bash
./target/release/vectordb serve --host 0.0.0.0 --port 9999
```

**4. 验证持久化：**
```bash
# 插入数据
curl -X POST http://localhost:9999/api/v1/tables \
  -H "Content-Type: application/json" \
  -d '{"name":"test","dimension":128}'

curl -X POST http://localhost:9999/api/v1/tables/test/records \
  -H "Content-Type: application/json" \
  -d '{"records":[{"id":"1","vector":[0.1,0.2,...]}]}'

# 查看数据文件
ls -lh ./data/lmdb/
# 输出: data.mdb  lock.mdb

# 重启服务器
pkill vectordb
./target/release/vectordb serve --host 0.0.0.0 --port 9999

# 验证数据仍然存在
curl http://localhost:9999/api/v1/tables
# 应该看到 "test" 表
```

### 方案 2: 使用 S3（大规模）

**1. 配置 S3：**
```bash
# 创建 .env 文件
cat > .env << 'EOF'
DB_STORAGE_TYPE=tiered
DB_OBJECT_STORAGE_PROVIDER=s3
DB_S3_BUCKET=my-vectordb
DB_S3_REGION=us-east-1
DB_AWS_ACCESS_KEY_ID=your-key
DB_AWS_SECRET_ACCESS_KEY=your-secret

# 仍然使用内存和 SSD 缓存
DB_MEMORY_CACHE_SIZE=512
DB_SSD_CACHE_DIR=./cache
DB_SSD_CACHE_SIZE=10
EOF
```

**2. 创建 S3 Bucket：**
```bash
aws s3 mb s3://my-vectordb --region us-east-1
```

**3. 运行服务器：**
```bash
./target/release/vectordb serve
```

**4. 数据自动同步到 S3：**
- 插入的数据自动上传到 S3
- 重启后自动从 S3 下载
- 本地缓存加速访问

### 方案 3: 使用 MinIO（自托管 S3）

**1. 启动 MinIO：**
```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=admin" \
  -e "MINIO_ROOT_PASSWORD=password" \
  -v /data/minio:/data \
  minio/minio server /data --console-address ":9001"
```

**2. 创建 Bucket：**
```bash
# 访问 http://localhost:9001
# 创建 bucket: vectordb
```

**3. 配置 DXMT：**
```bash
cat > .env << 'EOF'
DB_STORAGE_TYPE=tiered
DB_OBJECT_STORAGE_PROVIDER=s3
DB_S3_BUCKET=vectordb
DB_S3_ENDPOINT=http://localhost:9000
DB_AWS_ACCESS_KEY_ID=admin
DB_AWS_SECRET_ACCESS_KEY=password
EOF
```

---

## 当前运行服务器的存储状态

让我检查您当前运行的服务器：

**运行的进程：**
```bash
# 检查正在运行的 vectordb 进程
ps aux | grep vectordb

# 示例输出：
# ./target/release/vectordb serve --host 127.0.0.1 --port 9998
# ./target/release/vectordb serve  (端口 9999)
```

**当前配置（推测）：**
- 端口 9999: 默认配置（可能是 tiered 或 memory）
- 端口 9998: 测试实例

**检查数据目录：**
```bash
# 在 vectordb 运行目录
ls -la cache/      # 如果存在，说明使用 tiered
ls -la data/       # 如果存在，可能使用 LMDB
ls -la .dxmt/      # 其他可能的数据目录
```

**结论：**
如果您没有看到 `data/lmdb/` 目录，说明当前使用的是**内存或缓存模式**，重启后数据会丢失！

---

## 推荐的生产配置

### 小型部署（< 1M 向量）

```bash
# .env
DB_STORAGE_TYPE=lmdb
DB_LMDB_PATH=./data/lmdb
DB_LMDB_MAP_SIZE=10737418240  # 10GB
DB_MEMORY_CACHE_SIZE=512       # 512MB 内存缓存
```

### 中型部署（1M-10M 向量）

```bash
# .env
DB_STORAGE_TYPE=lmdb
DB_LMDB_PATH=/data/vectordb/lmdb  # 使用独立磁盘
DB_LMDB_MAP_SIZE=107374182400      # 100GB
DB_MEMORY_CACHE_SIZE=2048          # 2GB 内存缓存
DB_SSD_CACHE_DIR=/cache/vectordb   # SSD 缓存
DB_SSD_CACHE_SIZE=50               # 50GB SSD
```

### 大型部署（> 10M 向量）

```bash
# .env
DB_STORAGE_TYPE=tiered
DB_OBJECT_STORAGE_PROVIDER=s3
DB_S3_BUCKET=vectordb-production
DB_S3_REGION=us-east-1
DB_AWS_ACCESS_KEY_ID=xxx
DB_AWS_SECRET_ACCESS_KEY=xxx

# 本地缓存配置
DB_MEMORY_CACHE_SIZE=8192          # 8GB 内存
DB_SSD_CACHE_DIR=/nvme/cache       # NVMe SSD
DB_SSD_CACHE_SIZE=500              # 500GB SSD
```

---

## 数据迁移指南

### 从内存模式迁移到 LMDB

**1. 导出数据（在内存模式运行时）：**
```python
from dxmt import Client

client = Client(host="localhost", port=9999)

# 获取所有表
tables = client.list_tables()

# 导出每个表的数据
import json

for table in tables:
    table_name = table['name']

    # 获取所有记录（分批）
    all_records = []
    offset = 0
    batch_size = 1000

    while True:
        records = client.list_records(table_name, limit=batch_size, offset=offset)
        if not records:
            break
        all_records.extend(records)
        offset += batch_size

    # 保存到文件
    with open(f"backup_{table_name}.json", "w") as f:
        json.dump({
            "table_config": table,
            "records": all_records
        }, f)
```

**2. 重新配置为 LMDB 模式：**
```bash
# 停止服务器
pkill vectordb

# 更新 .env
export DB_STORAGE_TYPE=lmdb
export DB_LMDB_PATH=./data/lmdb

# 重启服务器
./target/release/vectordb serve --features storage-lmdb
```

**3. 导入数据：**
```python
import json

for table_file in glob.glob("backup_*.json"):
    with open(table_file) as f:
        data = json.load(f)

    # 创建表
    client.create_table(**data['table_config'])

    # 插入记录（分批）
    table_name = data['table_config']['name']
    records = data['records']

    for i in range(0, len(records), 1000):
        batch = records[i:i+1000]
        client.insert_records(table_name, batch)
```

---

## 备份和恢复

### LMDB 备份

**方法 1: 文件复制（需要停止服务）：**
```bash
# 停止服务器
systemctl stop vectordb

# 备份
cp -r ./data/lmdb ./backup/lmdb_$(date +%Y%m%d)

# 重启服务器
systemctl start vectordb
```

**方法 2: 热备份（使用 LMDB 工具）：**
```bash
# 安装 lmdb-utils
apt-get install lmdb-utils

# 热备份（不需要停止服务）
mdb_copy -c ./data/lmdb ./backup/lmdb_$(date +%Y%m%d)
```

### S3 备份

**S3 数据自动持久化，无需额外备份**

但可以创建快照：
```bash
# 复制整个 bucket
aws s3 sync s3://vectordb s3://vectordb-backup-$(date +%Y%m%d)
```

---

## 常见问题

### Q: 如何查看当前使用的存储模式？

**A:** 查看日志或配置：
```bash
# 查看服务器启动日志
journalctl -u vectordb | grep storage

# 或查看环境变量
echo $DB_STORAGE_TYPE

# 或通过 API
curl http://localhost:9999/health | jq '.storage_type'
```

### Q: 数据存储在哪里，占用多少空间？

**A:** 取决于存储模式：

```bash
# LMDB 模式
du -sh ./data/lmdb

# Tiered 模式（缓存）
du -sh ./cache

# S3 模式
aws s3 ls s3://your-bucket --recursive --summarize
```

### Q: 如何清理缓存？

**A:** 安全清理：
```bash
# 停止服务器
systemctl stop vectordb

# 清理缓存（不影响持久化数据）
rm -rf ./cache/*

# 重启服务器（会重建缓存）
systemctl start vectordb
```

### Q: 重启后为什么数据丢失了？

**A:** 可能原因：
1. 使用了 **memory** 模式（非持久化）
2. 使用了 **tiered** 模式但没有配置 S3（主要数据在内存）
3. 数据目录被删除或不可访问

**解决方法：** 切换到 LMDB 或 S3 模式

### Q: 如何实现自动备份？

**A:** 使用 cron 定时备份：

```bash
# 编辑 crontab
crontab -e

# 添加每日备份任务（LMDB）
0 2 * * * mdb_copy -c /path/to/data/lmdb /backup/lmdb_$(date +\%Y\%m\%d) 2>&1 | logger -t vectordb-backup

# 或（S3）
0 2 * * * aws s3 sync s3://vectordb s3://vectordb-backup-$(date +\%Y\%m\%d) 2>&1 | logger -t vectordb-backup
```

---

## 总结

### 核心要点 🎯

1. **默认配置（tiered）不会持久化主数据**
   - 内存是主要存储
   - ./cache 只是缓存
   - 重启后数据丢失

2. **要实现持久化，需要明确配置：**
   - **LMDB**: 适合中小规模，完全本地
   - **S3**: 适合大规模，云原生

3. **推荐生产配置：**
   ```bash
   DB_STORAGE_TYPE=lmdb          # 或 tiered + S3
   DB_LMDB_PATH=./data/lmdb      # 持久化目录
   DB_LMDB_MAP_SIZE=10737418240  # 根据需求调整
   ```

4. **验证持久化：**
   ```bash
   # 插入数据 → 重启服务器 → 验证数据仍在
   ```

### 下一步行动 📋

**如果您想要数据持久化：**

1. ✅ 选择存储模式（LMDB 或 S3）
2. ✅ 更新配置文件
3. ✅ 重新编译（如果需要）
4. ✅ 导出现有数据（如果有）
5. ✅ 重启服务器
6. ✅ 导入数据
7. ✅ 验证持久化

**需要帮助？**
- 查看：`docs/DEPLOYMENT.md`
- 示例配置：`.env.example`
- 问题反馈：GitHub Issues

---

**文档版本：** 1.0
**最后更新：** 2026-01-25
**作者：** Claude Code (Sonnet 4.5)
