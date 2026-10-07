# syntax=docker/dockerfile:1
# =============================================================================
# DXMT VectorDB - 通用多阶段 Docker 构建
# 支持多平台: linux/amd64, linux/arm64
# 解决所有编译依赖问题
# =============================================================================

# -----------------------------------------------------------------------------
# 第一阶段：基础构建环境
# -----------------------------------------------------------------------------
# Pinned to bookworm, the same Debian the runtime stage is: `rust:latest` follows Debian's
# newest release, and a binary linked against a newer glibc than the runtime image's does not
# start.
ARG RUST_VERSION=1
FROM rust:${RUST_VERSION}-bookworm AS base-builder

# Install only dependencies needed by the default single-node build. Optional LMDB/RocksDB,
# CUDA and ONNX integrations are built in dedicated images/jobs rather than silently ignored.
RUN apt-get update && apt-get install -y \
    # 基础构建工具
    build-essential \
    pkg-config \
    cmake \
    git \
    curl \
    # C/C++ 编译器和工具
    gcc \
    g++ \
    clang \
    libclang-dev \
    llvm-dev \
    # SSL 和加密库
    libssl-dev \
    # 压缩库
    libzstd-dev \
    zstd \
    liblz4-dev \
    # Other build tools
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量解决编译问题
ENV CC=clang \
    CXX=clang++ \
    CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    CARGO_BUILD_JOBS=1 \
    CARGO_INCREMENTAL=0 \
    CARGO_PROFILE_RELEASE_DEBUG=0 \
    RUST_BACKTRACE=1

# -----------------------------------------------------------------------------
# Stage 2: the web console's bundle
# -----------------------------------------------------------------------------
# `api-dashboard` is a default feature and embeds `dashboard/dist/` at compile time. Without
# this stage the image's binary had no bundle and served no console.
FROM node:22-bookworm-slim AS dashboard-builder

WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 3: the server
# -----------------------------------------------------------------------------
# This used to cache dependencies by building against placeholder sources first. The
# placeholders stubbed two of the five `[[bin]]` targets and none of the `[[test]]` or
# `[[example]]` ones, so cargo refused the manifest and `make docker-build` failed. BuildKit
# cache mounts keep the registry and `target/` between builds instead, with no copy of the
# manifest's target list to fall out of date.
FROM base-builder AS final-builder

WORKDIR /build

COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY benches ./benches
COPY tests ./tests
COPY examples ./examples
COPY proto ./proto
COPY --from=dashboard-builder /dashboard/dist ./dashboard/dist

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin vectordb && \
    strip target/release/vectordb && \
    cp target/release/vectordb /usr/local/bin/vectordb

# -----------------------------------------------------------------------------
# 第四阶段：最小运行时镜像
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libzstd1 \
    liblz4-1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r vectordb && \
    useradd -r -g vectordb -u 1000 -m -s /bin/bash vectordb && \
    mkdir -p /data /config /logs && \
    chown -R vectordb:vectordb /data /config /logs

# 复制二进制文件
COPY --from=final-builder /usr/local/bin/vectordb /usr/local/bin/vectordb

# 复制默认配置（如果存在）
COPY --chown=vectordb:vectordb config/ /config/

# 切换到非 root 用户
USER vectordb
WORKDIR /data

# 设置环境变量
# `VECTORDB_ROOT_DIR`, `VECTORDB_CONFIG_DIR` and `VECTORDB_LOG_DIR` used to be set here and
# were read by no code at all -- three variables that looked like configuration and were
# decoration. `DXMT_ROOT_DIR` is the one the server actually reads; everything else hangs off it.
ENV DXMT_ROOT_DIR=/data \
    RUST_LOG=info \
    PORT=9999

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/ready || exit 1

# 暴露端口
EXPOSE 9999

# 数据卷
VOLUME ["/data", "/config", "/logs"]

# Give the process a chance to flush its WAL/snapshot before the container is killed.
STOPSIGNAL SIGTERM

# 启动命令
ENTRYPOINT ["/usr/local/bin/vectordb"]
CMD ["serve", "--host", "0.0.0.0", "--port", "9999"]

# -----------------------------------------------------------------------------
# 开发阶段（可选）
# -----------------------------------------------------------------------------
FROM base-builder AS development

# 安装额外的开发工具
RUN cargo install \
    cargo-watch \
    cargo-edit \
    cargo-expand \
    cargo-outdated \
    cargo-audit

WORKDIR /workspace

# 开发环境变量
ENV RUST_BACKTRACE=full \
    RUST_LOG=debug \
    CARGO_WATCH_CLEAR=true

# 开发模式启动
CMD ["cargo", "watch", "-x", "run", "-w", "src"]

# -----------------------------------------------------------------------------
# 测试阶段（可选）
# -----------------------------------------------------------------------------
FROM final-builder AS test

WORKDIR /build

# 运行测试（仅在显式指定 --target test 时执行）
RUN cargo test --release

# -----------------------------------------------------------------------------
# 构建说明
# -----------------------------------------------------------------------------
# 生产镜像构建:
#   docker build -f docker/Dockerfile -t dxmt/vectordb:latest .
#
# 开发镜像构建:
#   docker build --target development -t dxmt/vectordb:dev .
#
# 测试镜像构建:
#   docker build --target test -t dxmt/vectordb:test .
#
# 多平台构建 (需要 docker buildx):
#   docker buildx create --use
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t dxmt/vectordb:latest --push .
#
# 运行容器:
#   docker run -d \
#     --name vectordb \
#     -p 9999:9999 \
#     -v $(pwd)/data:/data \
#     -v $(pwd)/config:/config \
#     -v $(pwd)/logs:/logs \
#     dxmt/vectordb:latest
#
# 使用 docker-compose:
#   docker-compose up -d
#
# 调试模式:
#   docker run -it --rm \
#     -p 9999:9999 \
#     -e RUST_LOG=debug \
#     dxmt/vectordb:latest
#
# -----------------------------------------------------------------------------
# 最终生产镜像（默认构建目标）
# -----------------------------------------------------------------------------
FROM runtime AS final
