#!/bin/bash
# =============================================================================
# Docker Hub Setup Checker
# 检查GitHub Actions和Docker Hub配置是否正确
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 图标
CHECK_MARK="${GREEN}✓${NC}"
CROSS_MARK="${RED}✗${NC}"
WARNING="${YELLOW}⚠${NC}"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  DXMT VectorDB - Docker Setup Checker${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 检查计数
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

# 辅助函数
check_pass() {
    echo -e "${CHECK_MARK} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${CROSS_MARK} $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${WARNING} $1"
    ((CHECKS_WARNING++))
}

# =============================================================================
# 1. 检查必需文件
# =============================================================================
echo -e "\n${BLUE}[1] 检查必需文件...${NC}"

if [ -f ".github/workflows/docker-publish.yml" ]; then
    check_pass "docker-publish.yml 存在"
else
    check_fail "docker-publish.yml 不存在"
fi

if [ -f ".github/workflows/docker-nightly.yml" ]; then
    check_pass "docker-nightly.yml 存在"
else
    check_warn "docker-nightly.yml 不存在 (可选)"
fi

if [ -f "docker/Dockerfile" ]; then
    check_pass "Dockerfile 存在"
else
    check_fail "Dockerfile 不存在"
fi

if [ -f "docker/Dockerfile.multiarch" ]; then
    check_pass "Dockerfile.multiarch 存在"
else
    check_warn "Dockerfile.multiarch 不存在 (可选)"
fi

# =============================================================================
# 2. 检查GitHub CLI
# =============================================================================
echo -e "\n${BLUE}[2] 检查GitHub CLI...${NC}"

if command -v gh &> /dev/null; then
    check_pass "GitHub CLI (gh) 已安装"

    # 检查是否已登录
    if gh auth status &> /dev/null; then
        check_pass "GitHub CLI 已认证"
    else
        check_fail "GitHub CLI 未认证，请运行: gh auth login"
    fi
else
    check_fail "GitHub CLI 未安装，请访问: https://cli.github.com/"
fi

# =============================================================================
# 3. 检查GitHub Secrets
# =============================================================================
echo -e "\n${BLUE}[3] 检查GitHub Secrets...${NC}"

if command -v gh &> /dev/null && gh auth status &> /dev/null; then
    # 获取secrets列表
    SECRETS=$(gh secret list 2>/dev/null || echo "")

    if echo "$SECRETS" | grep -q "DOCKER_USERNAME"; then
        check_pass "DOCKER_USERNAME secret 已配置"
    else
        check_fail "DOCKER_USERNAME secret 未配置"
        echo -e "   ${YELLOW}配置方法: gh secret set DOCKER_USERNAME${NC}"
    fi

    if echo "$SECRETS" | grep -q "DOCKER_PASSWORD"; then
        check_pass "DOCKER_PASSWORD secret 已配置"
    else
        check_fail "DOCKER_PASSWORD secret 未配置"
        echo -e "   ${YELLOW}配置方法: gh secret set DOCKER_PASSWORD${NC}"
    fi

    # 可选secrets
    if echo "$SECRETS" | grep -q "CARGO_REGISTRY_TOKEN"; then
        check_pass "CARGO_REGISTRY_TOKEN secret 已配置 (可选)"
    else
        check_warn "CARGO_REGISTRY_TOKEN 未配置 (可选，用于发布到crates.io)"
    fi
else
    check_warn "无法检查GitHub Secrets (需要GitHub CLI)"
fi

# =============================================================================
# 4. 检查Docker
# =============================================================================
echo -e "\n${BLUE}[4] 检查Docker环境...${NC}"

if command -v docker &> /dev/null; then
    check_pass "Docker 已安装"

    # 检查Docker daemon
    if docker info &> /dev/null; then
        check_pass "Docker daemon 运行中"
    else
        check_fail "Docker daemon 未运行"
    fi

    # 检查buildx
    if docker buildx version &> /dev/null; then
        check_pass "Docker Buildx 可用"
    else
        check_warn "Docker Buildx 不可用 (推荐用于多架构构建)"
    fi
else
    check_fail "Docker 未安装"
fi

# =============================================================================
# 5. 检查工作流配置
# =============================================================================
echo -e "\n${BLUE}[5] 检查工作流配置...${NC}"

if [ -f ".github/workflows/docker-publish.yml" ]; then
    # 检查Docker Hub配置
    if grep -q "DOCKER_USERNAME" .github/workflows/docker-publish.yml; then
        check_pass "工作流引用 DOCKER_USERNAME secret"
    else
        check_fail "工作流未引用 DOCKER_USERNAME secret"
    fi

    if grep -q "DOCKER_PASSWORD" .github/workflows/docker-publish.yml; then
        check_pass "工作流引用 DOCKER_PASSWORD secret"
    else
        check_fail "工作流未引用 DOCKER_PASSWORD secret"
    fi

    # 检查多架构支持
    if grep -q "linux/amd64,linux/arm64" .github/workflows/docker-publish.yml; then
        check_pass "配置了多架构构建 (amd64, arm64)"
    else
        check_warn "未配置多架构构建"
    fi

    # 检查缓存配置
    if grep -q "cache-from.*gha" .github/workflows/docker-publish.yml; then
        check_pass "配置了GitHub Actions缓存"
    else
        check_warn "未配置缓存 (会导致构建较慢)"
    fi
fi

# =============================================================================
# 6. 检查Dockerfile
# =============================================================================
echo -e "\n${BLUE}[6] 检查Dockerfile...${NC}"

if [ -f "docker/Dockerfile" ]; then
    # 检查多阶段构建
    if grep -q "FROM.*AS builder" docker/Dockerfile; then
        check_pass "使用多阶段构建"
    else
        check_warn "未使用多阶段构建 (镜像可能较大)"
    fi

    # 检查健康检查
    if grep -q "HEALTHCHECK" docker/Dockerfile; then
        check_pass "配置了健康检查"
    else
        check_warn "未配置健康检查"
    fi

    # 检查非root用户
    if grep -q "USER" docker/Dockerfile; then
        check_pass "使用非root用户运行"
    else
        check_warn "以root用户运行 (安全风险)"
    fi
fi

# =============================================================================
# 7. 测试本地构建
# =============================================================================
echo -e "\n${BLUE}[7] 测试本地构建 (可选)...${NC}"

if [ -f "docker/Dockerfile" ] && command -v docker &> /dev/null && docker info &> /dev/null; then
    echo -e "${YELLOW}是否要测试本地Docker构建? (y/n)${NC}"
    read -r -p "> " response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}开始构建测试镜像...${NC}"

        if docker build -f docker/Dockerfile -t dxmt-vectordb:test .; then
            check_pass "本地构建成功"

            echo -e "${YELLOW}是否要运行测试容器? (y/n)${NC}"
            read -r -p "> " run_response

            if [[ "$run_response" =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}启动测试容器...${NC}"
                docker run -d --name vectordb-test-check -p 9999:9999 dxmt-vectordb:test

                echo "等待服务启动..."
                sleep 5

                if curl -f http://localhost:9999/health &> /dev/null; then
                    check_pass "容器启动成功并通过健康检查"
                else
                    check_fail "健康检查失败"
                fi

                echo -e "${BLUE}清理测试容器...${NC}"
                docker stop vectordb-test-check &> /dev/null
                docker rm vectordb-test-check &> /dev/null
            fi
        else
            check_fail "本地构建失败"
        fi
    else
        check_warn "跳过本地构建测试"
    fi
else
    check_warn "跳过本地构建测试 (Docker不可用)"
fi

# =============================================================================
# 8. 汇总结果
# =============================================================================
echo -e "\n${BLUE}============================================${NC}"
echo -e "${BLUE}  检查完成${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "通过: ${GREEN}${CHECKS_PASSED}${NC}"
echo -e "失败: ${RED}${CHECKS_FAILED}${NC}"
echo -e "警告: ${YELLOW}${CHECKS_WARNING}${NC}"
echo ""

# 给出建议
if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ 所有必需检查都已通过！${NC}"
    echo -e "${GREEN}✓ 可以开始使用GitHub Actions构建Docker镜像了${NC}"
    echo ""
    echo -e "${BLUE}下一步:${NC}"
    echo "1. 提交代码到GitHub: git push origin main"
    echo "2. 查看Actions: https://github.com/YOUR_ORG/YOUR_REPO/actions"
    echo "3. 等待构建完成 (~15-20分钟)"
    echo "4. 在Docker Hub查看镜像"
else
    echo -e "${RED}✗ 发现 ${CHECKS_FAILED} 个必需项未通过${NC}"
    echo ""
    echo -e "${YELLOW}请修复以下问题:${NC}"

    if ! command -v gh &> /dev/null; then
        echo "- 安装GitHub CLI: https://cli.github.com/"
    fi

    if command -v gh &> /dev/null && ! gh auth status &> /dev/null; then
        echo "- 认证GitHub CLI: gh auth login"
    fi

    if ! command -v docker &> /dev/null; then
        echo "- 安装Docker: https://docs.docker.com/get-docker/"
    fi

    echo ""
    echo "然后重新运行此脚本: ./scripts/check-docker-setup.sh"
fi

echo ""
echo -e "${BLUE}更多信息请查看: docs/DOCKER_HUB_SETUP.md${NC}"
echo ""

exit $CHECKS_FAILED
