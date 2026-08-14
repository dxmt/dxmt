# DXMT Python SDK - Tests

这个目录包含DXMT Python SDK的所有测试。

## 测试文件

### 单元测试（无需DXMT服务器）

**`test_basic.py`** - 基本功能测试 (8个测试)
- 包导入验证
- 客户端初始化
- 异常类测试
- 方法存在性验证

**`test_client_complete.py`** - 完整客户端测试 (21个测试)
- 表管理操作
- 记录CRUD操作
- 搜索功能
- 错误处理
- 异步操作

### 集成测试（需要DXMT服务器）

**`test_integration.py`** - 集成测试套件 (50+个测试)
- 真实HTTP通信测试
- 数据持久性验证
- 并发操作测试
- 混合搜索测试
- 边缘情况测试

## 运行测试

### 快速开始

```bash
# 运行所有单元测试
./run_tests.sh unit

# 运行测试并查看覆盖率
./run_tests.sh coverage

# 运行集成测试（需要DXMT服务器）
./run_tests.sh integration
```

### 详细用法

```bash
# 从项目根目录

# 1. 确保已安装依赖
pip install -e ".[dev]"

# 2. 运行单元测试
pytest tests/test_basic.py tests/test_client_complete.py -v

# 3. 运行特定测试
pytest tests/test_basic.py::test_client_initialization -v

# 4. 运行带覆盖率
pytest tests/ --cov=dxmt --cov-report=html

# 5. 运行集成测试（需要服务器）
pytest tests/test_integration.py -v -m integration
```

## 测试标记

使用pytest标记来选择性运行测试：

```bash
# 只运行集成测试
pytest -m integration

# 跳过集成测试
pytest -m "not integration"

# 只运行异步测试
pytest -m asyncio
```

## 集成测试要求

集成测试需要：
1. DXMT服务器运行在 `localhost:9999`
2. 服务器可访问且响应正常

启动服务器：
```bash
cd /localdisk/code/dxmt/vectordb
cargo run -- serve --host 0.0.0.0 --port 9999
```

## 测试覆盖率

当前测试覆盖率：**51%**

查看详细报告：
```bash
# 生成HTML报告
pytest --cov=dxmt --cov-report=html

# 打开报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 编写新测试

### 单元测试示例

```python
from dxmt import Client

def test_my_feature():
    """Test description."""
    client = Client()
    # Your test code
    assert True
```

### 集成测试示例

```python
import pytest
from dxmt import Client

@pytest.mark.integration
def test_real_server():
    """Test with real server."""
    client = Client(host="localhost", port=9999)
    tables = client.list_tables()
    assert isinstance(tables, list)
```

### 异步测试示例

```python
import pytest
from dxmt import Client

@pytest.mark.asyncio
async def test_async_feature():
    """Test async feature."""
    async with Client() as client:
        result = await client.asearch(...)
        assert result is not None
```

## 测试组织

```
tests/
├── README.md                    # 本文件
├── test_basic.py               # 基本功能测试
├── test_client_complete.py     # 完整客户端测试
└── test_integration.py         # 集成测试
```

## 故障排除

### 问题: 测试失败 "Connection refused"

**原因:** DXMT服务器未运行
**解决:** 启动DXMT服务器或跳过集成测试

### 问题: Import错误

**原因:** SDK未安装或环境未激活
**解决:**
```bash
pip install -e ".[dev]"
# 或
source venv/bin/activate
```

### 问题: Async测试警告

**原因:** pytest-asyncio配置
**解决:** 已在pytest.ini配置，无需额外操作

## 更多信息

- 完整测试报告: `TEST_REPORT.md`
- 验证报告: `SDK_VERIFICATION_REPORT.md`
- 主文档: `README.md`
