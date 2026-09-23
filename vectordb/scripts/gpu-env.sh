#!/usr/bin/env bash
# Source this to enable GPU (CUDA 11) acceleration for cargo build/test/run
# with the `gpu` feature.
#
#   source scripts/gpu-env.sh
#   cargo run --features gpu --bin vectordb -- serve
#   cargo test --features gpu --test gpu_acceleration_test
#
# Why this is needed: `gpu` is an opt-in cargo feature and cudarc is pinned to
# CUDA 11.5, but this host ships only the GPU *driver* (libcuda) -- no CUDA
# toolkit. We install the CUDA 11 nvrtc+runtime libs via pip wheels (userspace,
# no sudo) and point LD_LIBRARY_PATH at them so cudarc can dlopen() them.
set -u

SITE="$(python3 -c 'import site; print(site.getusersitepackages())')"
NVRTC_DIR="$SITE/nvidia/cuda_nvrtc/lib"
RT_DIR="$SITE/nvidia/cuda_runtime/lib"

# 1. Ensure the CUDA 11 nvrtc/runtime wheels are present (one-time).
if [ ! -d "$NVRTC_DIR" ] || ! ls "$NVRTC_DIR"/libnvrtc.so.11.* >/dev/null 2>&1; then
  echo "[gpu-env] installing CUDA 11 nvrtc/runtime pip wheels (one-time)..."
  python3 -m pip install --user --quiet nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11
fi

# 2. Create the exact sonames cudarc dlopen()s (libnvrtc.so.11, libcudart.so).
NVRTC_REAL="$(basename "$(ls "$NVRTC_DIR"/libnvrtc.so.11.* | head -1)")"
ln -sf "$NVRTC_REAL" "$NVRTC_DIR/libnvrtc.so.11"
ln -sf "$NVRTC_REAL" "$NVRTC_DIR/libnvrtc.so"
RT_REAL="$(basename "$(ls "$RT_DIR"/libcudart.so.11.* | head -1)")"
ln -sf "$RT_REAL" "$RT_DIR/libcudart.so"

# 3. Export the path (prepended so it wins over any other CUDA on the box).
export LD_LIBRARY_PATH="$NVRTC_DIR:$RT_DIR:${LD_LIBRARY_PATH:-}"
echo "[gpu-env] CUDA 11 libs ready; LD_LIBRARY_PATH set."
