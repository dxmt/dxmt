# syntax=docker/dockerfile:1
# =============================================================================
# DXMT VectorDB - multi-stage Docker build
# Platforms: linux/amd64, linux/arm64
# Build context: the `vectordb/` directory (see the build instructions at the end).
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: base build environment
# -----------------------------------------------------------------------------
# Pinned to bookworm, the same Debian the runtime stage is: `rust:latest` follows Debian's
# newest release, and a binary linked against a newer glibc than the runtime image's does not
# start.
ARG RUST_VERSION=1
FROM rust:${RUST_VERSION}-bookworm AS base-builder

# Install only dependencies needed by the default single-node build. Optional LMDB/RocksDB,
# CUDA and ONNX integrations are built in dedicated images/jobs rather than silently ignored.
RUN apt-get update && apt-get install -y \
    # Basic build tools
    build-essential \
    pkg-config \
    cmake \
    git \
    curl \
    # C/C++ compilers and tools
    gcc \
    g++ \
    clang \
    libclang-dev \
    llvm-dev \
    # SSL and crypto libraries
    libssl-dev \
    # Compression libraries
    libzstd-dev \
    zstd \
    liblz4-dev \
    # Other build tools
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Compiler and cargo settings for a reproducible, low-memory build
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
# Stage 4: minimal runtime image
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libzstd1 \
    liblz4-1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r vectordb && \
    useradd -r -g vectordb -u 1000 -m -s /bin/bash vectordb && \
    mkdir -p /data /config /logs && \
    chown -R vectordb:vectordb /data /config /logs

# The binary
COPY --from=final-builder /usr/local/bin/vectordb /usr/local/bin/vectordb

# Default configuration (if present)
COPY --chown=vectordb:vectordb config/ /config/

# Switch to the non-root user
USER vectordb
WORKDIR /data

# Environment
# `VECTORDB_ROOT_DIR`, `VECTORDB_CONFIG_DIR` and `VECTORDB_LOG_DIR` used to be set here and
# were read by no code at all -- three variables that looked like configuration and were
# decoration. `DXMT_ROOT_DIR` is the one the server actually reads; everything else hangs off it.
ENV DXMT_ROOT_DIR=/data \
    RUST_LOG=info \
    PORT=9999

# Health check: /ready answers 503 while durability is degraded
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/ready || exit 1

# Port
EXPOSE 9999

# Volumes
VOLUME ["/data", "/config", "/logs"]

# Give the process a chance to flush its WAL/snapshot before the container is killed.
STOPSIGNAL SIGTERM

# Start command. `--host 0.0.0.0` is needed inside a container; set DXMT_API_KEY before
# publishing the port anywhere you do not control.
ENTRYPOINT ["/usr/local/bin/vectordb"]
CMD ["serve", "--host", "0.0.0.0", "--port", "9999"]

# -----------------------------------------------------------------------------
# Development stage (optional)
# -----------------------------------------------------------------------------
FROM base-builder AS development

# Extra development tools
RUN cargo install \
    cargo-watch \
    cargo-edit \
    cargo-expand \
    cargo-outdated \
    cargo-audit

WORKDIR /workspace

# Development environment
ENV RUST_BACKTRACE=full \
    RUST_LOG=debug \
    CARGO_WATCH_CLEAR=true

# Start in watch mode
CMD ["cargo", "watch", "-x", "run", "-w", "src"]

# -----------------------------------------------------------------------------
# Test stage (optional)
# -----------------------------------------------------------------------------
FROM final-builder AS test

WORKDIR /build

# Runs the tests (only when built with --target test)
RUN cargo test --release

# -----------------------------------------------------------------------------
# Build instructions. Run every command from `vectordb/`, which is the build context.
# -----------------------------------------------------------------------------
# Production image:
#   docker build -f docker/Dockerfile -t dxmt/vectordb:latest .
#
# Development image:
#   docker build -f docker/Dockerfile --target development -t dxmt/vectordb:dev .
#
# Test image:
#   docker build -f docker/Dockerfile --target test -t dxmt/vectordb:test .
#
# Multi-platform (needs docker buildx):
#   docker buildx create --use
#   docker buildx build -f docker/Dockerfile --platform linux/amd64,linux/arm64 \
#     -t dxmt/vectordb:latest --push .
#
# Run a container (the data lives under /data/store):
#   docker run -d \
#     --name vectordb \
#     -p 127.0.0.1:9999:9999 \
#     -e DXMT_API_KEY="$DXMT_API_KEY" \
#     -v "$(pwd)/data:/data" \
#     dxmt/vectordb:latest
#
# With compose:
#   docker compose -f docker/docker-compose.yml up -d
#
# Debug logging:
#   docker run -it --rm \
#     -p 127.0.0.1:9999:9999 \
#     -e RUST_LOG=debug \
#     dxmt/vectordb:latest
#
# -----------------------------------------------------------------------------
# Final production image (the default build target)
# -----------------------------------------------------------------------------
FROM runtime AS final
