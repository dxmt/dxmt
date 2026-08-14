# Docker Hub自动化配置检查清单

使用此清单确保所有配置都已正确完成。

## 📋 配置前准备

- [ ] 有Docker Hub账号
- [ ] 有GitHub账号并有repository的admin权限
- [ ] 本地安装了Docker
- [ ] 本地安装了GitHub CLI（推荐）

---

## 🔧 Docker Hub配置

### 账号设置

- [ ] 已登录Docker Hub: https://hub.docker.com/
- [ ] 用户名确认: ________________

### Access Token

- [ ] 已创建Access Token
- [ ] Token权限包含: Read, Write, Delete
- [ ] Token已保存（格式: `dckr_pat_xxxxx...`）

### Repository

- [ ] 已创建或计划创建repository: `dxmt/vectordb`
- [ ] Repository可见性设置: □ Public  □ Private

---

## 🔑 GitHub Secrets配置

进入: `https://github.com/YOUR_ORG/YOUR_REPO/settings/secrets/actions`

### 必需的Secrets

- [ ] `DOCKER_USERNAME` 已配置
  - 值: Docker Hub用户名
  - 验证: `gh secret list | grep DOCKER_USERNAME`

- [ ] `DOCKER_PASSWORD` 已配置
  - 值: Docker Hub Access Token
  - 验证: `gh secret list | grep DOCKER_PASSWORD`

### 可选的Secrets（按需配置）

- [ ] `CARGO_REGISTRY_TOKEN` - 用于发布到crates.io
- [ ] `AWS_ACCESS_KEY_ID` - 用于AWS部署
- [ ] `AWS_SECRET_ACCESS_KEY` - 用于AWS部署

---

## 📁 文件检查

### GitHub Actions工作流

- [ ] `.github/workflows/docker-publish.yml` 存在
- [ ] `.github/workflows/docker-nightly.yml` 存在（可选）
- [ ] 工作流文件语法正确（无YAML错误）

### Dockerfile

- [ ] `Dockerfile` 存在且可构建
- [ ] `Dockerfile.multiarch` 存在（可选，用于优化）
- [ ] Dockerfile包含HEALTHCHECK
- [ ] Dockerfile使用非root用户

### 文档

- [ ] `docs/DOCKER_HUB_SETUP.md` 已创建
- [ ] `docs/DOCKER_QUICKSTART.md` 已创建
- [ ] `docs/README_BADGES.md` 已创建

### 脚本

- [ ] `scripts/check-docker-setup.sh` 存在
- [ ] 脚本有执行权限: `chmod +x scripts/check-docker-setup.sh`

---

## ✅ 运行检查脚本

```bash
./scripts/check-docker-setup.sh
```

**预期结果**:
```
通过: [绿色数字]
失败: 0
警告: [黄色数字或0]

✓ 所有必需检查都已通过！
```

**如果有失败项**:
- [ ] 已根据脚本输出修复所有问题
- [ ] 重新运行脚本确认通过

---

## 🧪 本地测试（可选但推荐）

### Docker构建测试

```bash
# 构建测试镜像
docker build -t dxmt-vectordb:test .
```

- [ ] 构建成功完成
- [ ] 构建时间记录: ________ 分钟
- [ ] 镜像大小合理（<200MB）

### 容器运行测试

```bash
# 运行测试容器
docker run -d --name test-vectordb -p 9999:9999 dxmt-vectordb:test

# 等待启动
sleep 10

# 健康检查
curl http://localhost:9999/health

# 清理
docker stop test-vectordb
docker rm test-vectordb
```

- [ ] 容器启动成功
- [ ] 健康检查返回200 OK
- [ ] API响应正常

### 多架构测试（需要buildx）

```bash
# 创建buildx builder
docker buildx create --use

# 测试多架构构建
docker buildx build --platform linux/amd64,linux/arm64 .
```

- [ ] amd64架构构建成功
- [ ] arm64架构构建成功

---

## 🚀 首次触发构建

### 选择触发方式

**方式A: 推送到main分支** (推荐)

```bash
git add .
git commit -m "Add Docker Hub auto-build configuration"
git push origin main
```

- [ ] 代码已推送到main分支

**方式B: 创建版本tag**

```bash
git tag v1.0.0
git push origin v1.0.0
```

- [ ] Tag已创建并推送

**方式C: 手动触发**

```bash
gh workflow run docker-publish.yml
```

- [ ] 工作流已手动触发

---

## 📊 监控首次构建

### GitHub Actions

访问: `https://github.com/YOUR_ORG/YOUR_REPO/actions`

- [ ] 工作流已开始运行
- [ ] 没有立即失败的步骤
- [ ] 构建日志可以正常查看

### 实时监控

```bash
# 使用GitHub CLI监控
gh run watch
```

- [ ] 构建进度正常
- [ ] 预计完成时间: 15-20分钟（首次）

### 构建完成检查

- [ ] 所有步骤都显示绿色✓
- [ ] "Build and push" 步骤成功
- [ ] "Security Scan" 步骤完成
- [ ] "Test Docker Image" 步骤通过

---

## 🐳 验证Docker Hub

### 检查镜像

访问: `https://hub.docker.com/r/dxmt/vectordb`

- [ ] Repository已创建
- [ ] 至少有一个tag可见
- [ ] 镜像大小显示正常（~100MB）
- [ ] 支持的架构显示为: amd64, arm64

### 标签检查

根据触发方式，应该看到相应的标签：

**如果推送到main**:
- [ ] `main` 标签存在
- [ ] `latest` 标签存在
- [ ] `main-[sha]` 标签存在

**如果推送version tag (v1.0.0)**:
- [ ] `1.0.0` 标签存在
- [ ] `1.0` 标签存在
- [ ] `1` 标签存在
- [ ] `latest` 标签存在

### 元数据检查

点击任一tag查看详情：

- [ ] "Last pushed" 时间正确
- [ ] OS/Architecture 显示 linux/amd64 和 linux/arm64
- [ ] Image size 合理

---

## 🧪 拉取和测试镜像

### 拉取镜像

```bash
docker pull dxmt/vectordb:latest
```

- [ ] 镜像拉取成功
- [ ] 拉取速度正常

### 运行容器

```bash
docker run -d --name vectordb-verify -p 9999:9999 dxmt/vectordb:latest
```

- [ ] 容器启动成功
- [ ] 无错误日志

### 功能测试

```bash
# 健康检查
curl http://localhost:9999/health

# API测试
curl -X POST http://localhost:9999/api/v1/tables \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test",
    "dimension": 128,
    "distance_metric": "cosine"
  }'
```

- [ ] 健康检查返回OK
- [ ] API调用成功
- [ ] 基本功能正常

### 清理

```bash
docker stop vectordb-verify
docker rm vectordb-verify
```

- [ ] 测试容器已清理

---

## 📝 文档更新（推荐）

### 更新README.md

```markdown
## Quick Start with Docker

\`\`\`bash
docker run -d -p 9999:9999 dxmt/vectordb:latest
\`\`\`
```

- [ ] README已包含Docker使用说明
- [ ] 添加了Docker相关badges（参考`docs/README_BADGES.md`）

### 添加Badges

```markdown
[![Docker Build](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/YOUR_ORG/YOUR_REPO/actions/workflows/docker-publish.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/dxmt/vectordb)](https://hub.docker.com/r/dxmt/vectordb)
```

- [ ] Badges已添加到README
- [ ] Badges显示正常

---

## 🔔 配置通知（可选）

### GitHub通知

默认情况下，构建失败时会邮件通知commit作者。

如需自定义通知：
- [ ] 配置Slack webhook (如需要)
- [ ] 配置Discord webhook (如需要)
- [ ] 配置自定义webhook (如需要)

### Docker Hub Webhook

如需接收推送通知：
- [ ] 在Docker Hub配置webhook URL
- [ ] 选择触发事件: Push, Tag Push
- [ ] 测试webhook正常工作

---

## 📈 后续监控

### 定期检查

**每周**:
- [ ] 查看GitHub Actions运行历史
- [ ] 检查是否有失败的构建
- [ ] 查看Docker Hub下载统计

**每月**:
- [ ] 检查镜像大小是否合理
- [ ] 查看安全扫描报告
- [ ] 清理旧的未使用镜像

### 性能监控

- [ ] 记录平均构建时间
- [ ] 监控缓存命中率
- [ ] 跟踪镜像下载趋势

---

## 🔧 维护任务

### 定期更新

**每季度**:
- [ ] 更新base镜像版本
- [ ] 更新Rust版本
- [ ] 检查依赖安全更新

**按需**:
- [ ] 优化Dockerfile
- [ ] 调整构建配置
- [ ] 更新文档

### 安全更新

- [ ] 关注安全扫描警报
- [ ] 及时修复高危漏洞
- [ ] 更新依赖版本

---

## ✅ 最终确认

完成以上所有检查项后：

- [ ] 所有必需项都已完成
- [ ] Docker镜像可以正常拉取和使用
- [ ] 自动化构建流程正常工作
- [ ] 文档已更新
- [ ] 团队成员已通知

---

## 🎉 配置完成！

恭喜！Docker Hub自动化构建已配置完成。

**现在每次你**:
- 推送代码到main → 自动构建latest
- 推送代码到其他分支 → 自动构建分支镜像
- 创建version tag → 自动构建release镜像
- 每天UTC 2:00 → 自动构建nightly

**无需手动操作！**

---

## 📞 获取帮助

如果遇到问题：

1. **查看文档**
   - [完整配置指南](../docs/DOCKER_HUB_SETUP.md)
   - [快速开始指南](../docs/DOCKER_QUICKSTART.md)

2. **运行检查脚本**
   ```bash
   ./scripts/check-docker-setup.sh
   ```

3. **查看构建日志**
   ```bash
   gh run list --workflow=docker-publish.yml
   gh run view <run-id> --log
   ```

4. **提交Issue**
   - https://github.com/dxmt/vectordb/issues

---

**检查清单版本**: 1.0
**最后更新**: 2026-01-21
**维护者**: DXMT Team
