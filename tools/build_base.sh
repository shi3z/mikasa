#!/bin/bash
# Build the "before" binary: kernels and forward_batch of the original commit (HEAD = 360dcc3) with only the host-side instrumentation
# (--time-b, timing prints) of the working tree.  Output: ./mikasa_base
cd "$(dirname "$0")/.." || exit 1
rm -rf .base_src && mkdir -p .base_src && cp -r src/* .base_src/
for f in batch_kernels.cuh batch_kernels2.cuh batch_host.inc; do git show HEAD:src/$f > .base_src/$f; done
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 -I .base_src -o mikasa_base .base_src/inkling_model.cu .base_src/gguf.cpp 2>&1 | grep -E "error"
rm -rf .base_src; echo "built mikasa_base"
