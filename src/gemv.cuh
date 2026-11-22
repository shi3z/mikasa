// Batch-1 GEMV kernels (SIMT). y[n] = sum_k W[n, k] * x[k], W stored as GGUF blocks.
// Row n of W is K/blck blocks, consecutive in memory.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Q8_0: block = half d + 32 x int8, 34 bytes per 32 weights.
__device__ __forceinline__ float q8_0_block_dot(const uint8_t * blk, const float * xb) {
    float d = __half2float(*reinterpret_cast<const __half *>(blk));
    const int8_t * q = reinterpret_cast<const int8_t *>(blk + 2);
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        acc += (float) q[i] * xb[i];
    }
    return d * acc;
}

// One warp per output row. Lanes take blocks round robin.
__global__ void gemv_q8_0_simt(const uint8_t * __restrict__ W, const float * __restrict__ x,
                               float * __restrict__ y, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int nb = K / 32;
    const uint8_t * row = W + (size_t) warp * nb * 34;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        acc += q8_0_block_dot(row + (size_t) b * 34, x + b * 32);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[warp] = acc;
}

// NVFP4: block = 4 x UE4M3 sub-scales + 32 bytes E2M1, 36 bytes per 64 weights.
// E2M1 value table is doubled, and the UE4M3 decode returns value * 0.5 (see ggml-impl.h).
__device__ __constant__ float kE2M1[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ float ue4m3_to_f32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF;
    int m = x & 0x7;
    float raw = (e == 0) ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}

__device__ __forceinline__ float nvfp4_block_dot(const uint8_t * blk, const float * xb) {
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float d = ue4m3_to_f32(blk[s]);
        const uint8_t * qs = blk + 4 + s * 8;
        float sub = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sub += kE2M1[qs[j] & 0xF] * xb[s * 16 + j];
            sub += kE2M1[qs[j] >> 4]  * xb[s * 16 + j + 8];
        }
        acc += d * sub;
    }
    return acc;
}

__global__ void gemv_nvfp4_simt(const uint8_t * __restrict__ W, const float * __restrict__ x,
                                float * __restrict__ y, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    const int nb = K / 64;
    const uint8_t * row = W + (size_t) warp * nb * 36;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        acc += nvfp4_block_dot(row + (size_t) b * 36, x + b * 64);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[warp] = acc;
}

// Host launch helpers: 8 warps per block.
inline void launch_gemv_q8_0(const uint8_t * W, const float * x, float * y, int K, int N, cudaStream_t s) {
    dim3 block(256), grid((N * 32 + 255) / 256);
    gemv_q8_0_simt<<<grid, block, 0, s>>>(W, x, y, K, N);
}

inline void launch_gemv_nvfp4(const uint8_t * W, const float * x, float * y, int K, int N, cudaStream_t s) {
    dim3 block(256), grid((N * 32 + 255) / 256);
    gemv_nvfp4_simt<<<grid, block, 0, s>>>(W, x, y, K, N);
}
