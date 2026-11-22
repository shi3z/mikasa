// GEMV variants on real Inkling 2 weights (RTX 3090, sm_86).
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo gemv_bench.cu gguf.cpp
// Run:   gemv_bench <model.gguf> [--ncu]
// Times are device time per launch from CUDA graphs (100 launches per graph).
#include "gguf.hpp"
#include "gemv.cuh"

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace nvcuda;

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

// ------------------------------------------------------------ device LUTs --

__device__ __forceinline__ float ue4m3_dev(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 0x7;
    float raw = (e == 0) ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;  // same convention as ggml_ue4m3_to_fp32
}

// Doubled E2M1 magnitude by bit ops (no table): 2*val.
__device__ __forceinline__ float e2m1x2_bits(uint32_t nib) {
    uint32_t e = (nib >> 1) & 3, m = nib & 1;
    float mag = (e == 0) ? (float) m : __int_as_float(((e + 127) << 23) | (m << 22));
    return (nib & 8) ? -mag : mag;
}

// ----------------------------------------------- SoA layouts for the engine --
// Q8: qs int8 [N][K], sc half [N][K/32]
// NVFP4: qs u8 [N][K/2] (32 bytes per 64 weights, same nibble order as GGUF),
//        sc u8 [N][K/16] (four UE4M3 sub-scales per block)

__device__ __constant__ float kE2M1d[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Warp-per-row AoS (baseline in gemv.cuh) is kept as the reference kernel.

// ---- V1: warp per row, SoA, 128-bit loads, x read from global.
__global__ void v1_q8_warp_row(const int8_t * __restrict__ qs, const __half * __restrict__ sc,
                               const float * __restrict__ x, float * __restrict__ y, int K, int N) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const int nb = K / 32;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float d = __half2float(sc[(size_t) row * nb + b]);
        uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
        const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        float s = 0.0f;
        const float * xb = x + b * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int v = (int8_t) (w[i] >> (8 * k));
                s += (float) v * xb[i * 4 + k];
            }
        }
        acc += d * s;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// ---- V2: block of 256 threads, WPR warps per row (split-K), x staged in shared memory.
template <int WPR>
__global__ void v2_q8_split(const int8_t * __restrict__ qs, const __half * __restrict__ sc,
                            const float * __restrict__ x, float * __restrict__ y, int K, int N) {
    extern __shared__ float xs[];
    constexpr int RPB = 8 / WPR;                     // rows per block
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rl = warp / WPR, part = warp % WPR;
    const int row = blockIdx.x * RPB + rl;
    const int nb = K / 32;
    float acc = 0.0f;
    if (row < N) {
        const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
        for (int b = part * 32 + lane; b < nb; b += 32 * WPR) {
            float d = __half2float(sc[(size_t) row * nb + b]);
            uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
            const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float s = 0.0f;
            const float * xb = xs + b * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int k = 0; k < 4; ++k) s += (float) (int8_t) (w[i] >> (8 * k)) * xb[i * 4 + k];
            }
            acc += d * s;
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    __shared__ float red[8];
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (part == 0 && lane == 0 && row < N) {
        float t = 0.0f;
        for (int p = 0; p < WPR; ++p) t += red[rl * WPR + p];
        y[row] = t;
    }
}

// ---- V3: block of 8 warps = 8 rows, x in shared memory, unrolled 2 blocks in flight.
__global__ void v3_q8_multirow(const int8_t * __restrict__ qs, const __half * __restrict__ sc,
                               const float * __restrict__ x, float * __restrict__ y, int K, int N) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= N) return;
    const int nb = K / 32;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
    float acc = 0.0f;
#pragma unroll 2
    for (int b = lane; b < nb; b += 32) {
        float d = __half2float(sc[(size_t) row * nb + b]);
        uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
        const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        float s = 0.0f;
        const float * xb = xs + b * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int k = 0; k < 4; ++k) s += (float) (int8_t) (w[i] >> (8 * k)) * xb[i * 4 + k];
        }
        acc += d * s;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// ---- NVFP4 kernels. Template: BITOPS selects E2M1 decode (true) or LUT (false).
// One block of 8 warps covers 8 rows (WPR = 1) ; SPLIT warps per row when WPR > 1.
template <int WPR, bool BITOPS>
__global__ void v4_nvfp4(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                         const float * __restrict__ x, float * __restrict__ y, int K, int N,
                         const int * __restrict__ ids, int64_t q_expert_stride, int64_t s_expert_stride,
                         int ystride) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    constexpr int RPB = 8 / WPR;
    const int slot = ids ? blockIdx.y : 0;
    const int expert = ids ? ids[slot] : 0;
    const uint8_t * qbase = qs + expert * q_expert_stride;
    const uint8_t * sbase = sc + expert * s_expert_stride;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    if (!BITOPS) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            lut_ue[i] = ue4m3_dev((uint8_t) i);
            lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]);
        }
    } else {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) lut_ue[i] = ue4m3_dev((uint8_t) i);
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rl = warp / WPR, part = warp % WPR;
    const int row = blockIdx.x * RPB + rl;
    const int nb = K / 64;
    float acc = 0.0f;
    if (row < N) {
        const uint4 * q = reinterpret_cast<const uint4 *>(qbase + (size_t) row * (K / 2));
        const uint32_t * s32 = reinterpret_cast<const uint32_t *>(sbase + (size_t) row * (K / 16));
        for (int b = part * 32 + lane; b < nb; b += 32 * WPR) {
            uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
            const uint32_t sw = __ldg(s32 + b);
            const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float * xb = xs + b * 64;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float d = lut_ue[(sw >> (8 * s)) & 0xFF];
                float sub = 0.0f;
                // byte j of sub-block s sits in word (s*8+j)/4
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t byte = (w[(s * 8 + j) >> 2] >> (8 * ((s * 8 + j) & 3))) & 0xFF;
                    float lo, hi;
                    if (BITOPS) {
                        lo = e2m1x2_bits(byte & 0xF);
                        hi = e2m1x2_bits(byte >> 4);
                    } else {
                        float2 v = lut2[byte];
                        lo = v.x;
                        hi = v.y;
                    }
                    sub += lo * xb[s * 16 + j] + hi * xb[s * 16 + j + 8];
                }
                acc += d * sub;
            }
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    __shared__ float red[8];
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (part == 0 && lane == 0 && row < N) {
        float t = 0.0f;
        for (int p = 0; p < WPR; ++p) t += red[rl * WPR + p];
        y[(size_t) slot * ystride + row] = t;
    }
}


// ---- V7: warp handles ROWS rows; x block loaded once as float4 and reused for all rows.
template <int ROWS>
__global__ void v7_q8_rows(const int8_t * __restrict__ qs, const __half * __restrict__ sc,
                           const float * __restrict__ x, float * __restrict__ y, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * ROWS;
    if (row0 >= N) return;
    const int nb = K / 32;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = __ldg(x4 + b * 8 + i);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
                const float d = __half2float(sc[(size_t) row * nb + b]);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float t = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = xv[i];
                    t += (float) (int8_t) w[i] * v.x + (float) (int8_t) (w[i] >> 8) * v.y
                       + (float) (int8_t) (w[i] >> 16) * v.z + (float) (int8_t) (w[i] >> 24) * v.w;
                }
                acc[r] += d * t;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = a;
    }
}

// ---- V8: NVFP4, ROWS rows per warp, x block (64 floats) as 16 float4 shared across rows.
// BITOPS: E2M1 by bit ops; otherwise byte-pair LUT in shared memory. Scales via 256-entry LUT.
template <int ROWS, bool BITOPS>
__global__ void v8_nvfp4_rows(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ sc,
                              const float * __restrict__ x, float * __restrict__ y, int K, int N) {
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        lut_ue[i] = ue4m3_dev((uint8_t) i);
        lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]);
    }
    __syncthreads();
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * ROWS;
    if (row0 >= N) return;
    const int nb = K / 64;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float4 xv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) xv[i] = __ldg(x4 + b * 16 + i);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2));
                const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float tot = 0.0f;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float d = lut_ue[(sw >> (8 * s)) & 0xFF];
                    float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = s * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        // x element index inside the block: s*16 + j (lo) and s*16 + j + 8 (hi)
                        const float xl = reinterpret_cast<const float *>(&xv[(s * 16 + j) >> 2])[(s * 16 + j) & 3];
                        const float xh = reinterpret_cast<const float *>(&xv[(s * 16 + j + 8) >> 2])[(s * 16 + j + 8) & 3];
                        float lo, hi;
                        if (BITOPS) {
                            lo = e2m1x2_bits(byte & 0xF);
                            hi = e2m1x2_bits(byte >> 4);
                        } else {
                            const float2 v = lut2[byte];
                            lo = v.x;
                            hi = v.y;
                        }
                        sub += lo * xl + hi * xh;
                    }
                    tot += d * sub;
                }
                acc[r] += tot;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = a;
    }
}


// ---- V8F: top-8 experts in one launch. grid.y = slot, expert = ids[slot] into the full table.
template <int ROWS, bool BITOPS>
__global__ void v8f_nvfp4_rows(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all,
                               const float * __restrict__ x, float * __restrict__ y, int K, int N,
                               const int * __restrict__ ids) {
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        lut_ue[i] = ue4m3_dev((uint8_t) i);
        lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]);
    }
    __syncthreads();
    const int slot = blockIdx.y;
    const int64_t e = ids[slot];
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * ROWS;
    if (row0 >= N) return;
    const int nb = K / 64;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float4 xv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) xv[i] = __ldg(x4 + b * 16 + i);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2));
                const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float tot = 0.0f;
#pragma unroll
                for (int s2 = 0; s2 < 4; ++s2) {
                    const float d = lut_ue[(sw >> (8 * s2)) & 0xFF];
                    float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = s2 * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        const float xl = reinterpret_cast<const float *>(&xv[(s2 * 16 + j) >> 2])[(s2 * 16 + j) & 3];
                        const float xh = reinterpret_cast<const float *>(&xv[(s2 * 16 + j + 8) >> 2])[(s2 * 16 + j + 8) & 3];
                        float lo, hi;
                        if (BITOPS) {
                            lo = e2m1x2_bits(byte & 0xF);
                            hi = e2m1x2_bits(byte >> 4);
                        } else {
                            const float2 v = lut2[byte];
                            lo = v.x;
                            hi = v.y;
                        }
                        sub += lo * xl + hi * xh;
                    }
                    tot += d * sub;
                }
                acc[r] += tot;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[(size_t) slot * N + row0 + r] = a;
    }
}


// ---- V9: NVFP4, x staged once per block in shared memory, shared by 8 warps.
// Layout: x block b (64 floats) at b*68 (4 pad floats), so float4 reads of 32 lanes spread over banks.
// grid = (ceil(N / (8*ROWS)), slots); ids (may be null) selects the expert for grid.y.
template <int ROWS, bool BITOPS>
__global__ void v9_nvfp4_smem(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all,
                              const float * __restrict__ x, float * __restrict__ y, int K, int N,
                              const int * __restrict__ ids) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    const int nb = K / 64;
    const int slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const int b = i >> 6, r = i & 63;
        xs[b * 68 + r] = x[i];
    }
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        lut_ue[i] = ue4m3_dev((uint8_t) i);
        lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]);
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * ROWS;
    if (row0 >= N) return;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68);
        float4 xv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) xv[i] = xb[i];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2));
                const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float tot = 0.0f;
#pragma unroll
                for (int s2 = 0; s2 < 4; ++s2) {
                    const float d = lut_ue[(sw >> (8 * s2)) & 0xFF];
                    float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = s2 * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        const float xl = reinterpret_cast<const float *>(&xv[(s2 * 16 + j) >> 2])[(s2 * 16 + j) & 3];
                        const float xh = reinterpret_cast<const float *>(&xv[(s2 * 16 + j + 8) >> 2])[(s2 * 16 + j + 8) & 3];
                        float lo, hi;
                        if (BITOPS) {
                            lo = e2m1x2_bits(byte & 0xF);
                            hi = e2m1x2_bits(byte >> 4);
                        } else {
                            const float2 v = lut2[byte];
                            lo = v.x;
                            hi = v.y;
                        }
                        sub += lo * xl + hi * xh;
                    }
                    tot += d * sub;
                }
                acc[r] += tot;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[(size_t) slot * N + row0 + r] = a;
    }
}


// ---- V10: like V9, but x is read per 16-float sub-block (4 float4), not per 64-float block.
// Cuts live x registers from 64 to 16, which keeps occupancy up for the bit-op decode.
template <int ROWS, bool BITOPS>
__global__ void v10_nvfp4_sub(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all,
                              const float * __restrict__ x, float * __restrict__ y, int K, int N,
                              const int * __restrict__ ids) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    const int nb = K / 64;
    const int slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const int b = i >> 6, r = i & 63;
        xs[b * 68 + r] = x[i];
    }
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        lut_ue[i] = ue4m3_dev((uint8_t) i);
        lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]);
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * ROWS;
    if (row0 >= N) return;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const float * xb = xs + b * 68;
        uint4 wq[ROWS][2];
        uint32_t sw[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2));
                sw[r] = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
                wq[r][0] = __ldg(q + b * 2);
                wq[r][1] = __ldg(q + b * 2 + 1);
            }
        }
#pragma unroll
        for (int s2 = 0; s2 < 4; ++s2) {
            const float4 * xs4 = reinterpret_cast<const float4 *>(xb + s2 * 16);
            const float4 v0 = xs4[0], v1 = xs4[1], v2 = xs4[2], v3 = xs4[3];
            const float xv[16] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
                                  v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w};
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = row0 + r;
                if (row < N) {
                    const uint32_t w[8] = {wq[r][0].x, wq[r][0].y, wq[r][0].z, wq[r][0].w,
                                           wq[r][1].x, wq[r][1].y, wq[r][1].z, wq[r][1].w};
                    const float d = lut_ue[(sw[r] >> (8 * s2)) & 0xFF];
                    float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = s2 * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        float lo, hi;
                        if (BITOPS) {
                            lo = e2m1x2_bits(byte & 0xF);
                            hi = e2m1x2_bits(byte >> 4);
                        } else {
                            const float2 v = lut2[byte];
                            lo = v.x;
                            hi = v.y;
                        }
                        sub += lo * xv[j] + hi * xv[j + 8];
                    }
                    acc[r] += d * sub;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[(size_t) slot * N + row0 + r] = a;
    }
}

// ---- V6: FP16 tensor cores (wmma 16x16x16). Weights pre-dequantized to half [N][K]. Batch 1: x in column 0.
__global__ void v6_fp16_wmma(const __half * __restrict__ W, const float * __restrict__ x, float * __restrict__ y,
                             int K, int N) {
    __shared__ __half bs[8][16 * 16];
    __shared__ float cs[8][16 * 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * 16;
    if (row0 >= N) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = lane; i < 256; i += 32) {
            int kk = i / 16, nn = i % 16;
            bs[warp][i] = (nn == 0) ? __float2half(x[k0 + kk]) : __float2half(0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(fa, W + (size_t) row0 * K + k0, K);
        wmma::load_matrix_sync(fb, bs[warp], 16);
        wmma::mma_sync(fc, fa, fb, fc);
        __syncwarp();
    }
    wmma::store_matrix_sync(cs[warp], fc, 16, wmma::mem_row_major);
    __syncwarp();
    if (lane < 16) y[row0 + lane] = cs[warp][lane * 16];
}

// ---- V7: fused top-8 expert launch for NVFP4 is V4 with ids (grid.y = 8).

// Dequantize helpers on the host (exact GGUF semantics).
static float ue4m3_host(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 7;
    float raw = (e == 0) ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return raw * 0.5f;
}

static const float kKV[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

struct Weights {
    int K, N;
    std::vector<int8_t> q8;        // SoA
    std::vector<__half> q8s;
    std::vector<uint8_t> nqs, nsc; // SoA NVFP4 (one expert)
    std::vector<uint8_t> nqs_all, nsc_all;  // 8 experts stacked
    std::vector<__half> fp16;      // dequantized Q8 for the tensor core path
    std::vector<uint8_t> aos_q8, aos_nv;    // GGUF AoS for the baseline
};

static void repack_q8(const uint8_t * src, int K, int N, Weights & w) {
    int nb = K / 32;
    w.q8.resize((size_t) N * K);
    w.q8s.resize((size_t) N * nb);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nb; ++b) {
            const uint8_t * blk = src + ((size_t) n * nb + b) * 34;
            uint16_t hbits;
            std::memcpy(&hbits, blk, 2);
            __half h;
            std::memcpy(&h, &hbits, 2);
            w.q8s[(size_t) n * nb + b] = h;
            std::memcpy(&w.q8[(size_t) n * K + b * 32], blk + 2, 32);
        }
}

static void repack_nv(const uint8_t * src, int K, int N, std::vector<uint8_t> & qs, std::vector<uint8_t> & sc) {
    int nb = K / 64;
    qs.resize((size_t) N * (K / 2));
    sc.resize((size_t) N * (K / 16));
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nb; ++b) {
            const uint8_t * blk = src + ((size_t) n * nb + b) * 36;
            std::memcpy(&sc[(size_t) n * (K / 16) + b * 4], blk, 4);
            std::memcpy(&qs[(size_t) n * (K / 2) + b * 32], blk + 4, 32);
        }
}


// Device time per launch using a graph of R launches.
template <typename F>
static float graph_time_us(F launch, cudaStream_t s, int R = 100) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < R; ++i) launch(s);
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaGraphLaunch(ge, s));  // warm
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaEventRecord(e0, s));
    for (int i = 0; i < 5; ++i) CHECK(cudaGraphLaunch(ge, s));
    CHECK(cudaEventRecord(e1, s));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms * 1000.0f / (5.0f * R);
}

static int regs_of(const void * f) {
    cudaFuncAttributes a;
    CHECK(cudaFuncGetAttributes(&a, f));
    return a.numRegs;
}

static void report(const char * name, const char * fmt, float us, double bytes, int regs,
                   const std::vector<float> & y, const std::vector<float> & ref) {
    double maxabs = 0, maxref = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        maxabs = std::max(maxabs, (double) std::fabs(y[i] - ref[i]));
        maxref = std::max(maxref, (double) std::fabs(ref[i]));
    }
    std::printf("%-30s %-6s %9.2f us %8.1f GB/s  regs %3d  rel.err %9.2e\n", name, fmt, us,
                bytes / (us * 1e-6) / 1e9, regs, maxabs / (maxref + 1e-30));
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gemv_bench model.gguf [--ncu]\n"); return 2; }
    const bool ncu_mode = argc > 2 && std::strcmp(argv[2], "--ncu") == 0;
    gguf::File f = gguf::open(argv[1]);
    auto find = [&](const char * name) -> const gguf::Tensor * {
        for (const auto & t : f.tensors) if (t.name == name) return &t;
        return nullptr;
    };
    const gguf::Tensor * tq = find("blk.0.attn_q.weight");        // Q8_0 [2304, 3072]
    const gguf::Tensor * tg = find("blk.0.ffn_gate_up_exps.weight"); // NVFP4 [2304, 2304, 128]
    if (!tq || !tg) { std::fprintf(stderr, "tensors missing\n"); return 1; }

    const int Kq = 2304, Nq = 3072;
    const int Kg = 2304, Ng = 2304;
    const int E = 8;
    const int ids_h[E] = {3, 17, 42, 60, 81, 99, 101, 127};

    Weights wq, wg;
    wq.K = Kq; wq.N = Nq;
    wg.K = Kg; wg.N = Ng;
    const uint8_t * src_q = f.base + f.data_start + tq->offset;
    const uint8_t * src_g = f.base + f.data_start + tg->offset;
    const size_t per_expert_g = (size_t) Ng * (Kg / 64) * 36;
    wq.aos_q8.assign(src_q, src_q + (size_t) Nq * (Kq / 32) * 34);
    wg.aos_nv.assign(src_g, src_g + per_expert_g);
    repack_q8(src_q, Kq, Nq, wq);
    repack_nv(src_g, Kg, Ng, wg.nqs, wg.nsc);
    // stacked experts for the fused kernel
    const int NE = 128;  // all experts resident, ids index this table
    wg.nqs_all.resize((size_t) NE * Ng * (Kg / 2));
    wg.nsc_all.resize((size_t) NE * Ng * (Kg / 16));
    std::vector<uint8_t> qe, se;
    for (int e = 0; e < NE; ++e) {
        repack_nv(f.base + f.data_start + tg->offset + (size_t) e * per_expert_g, Kg, Ng, qe, se);
        std::memcpy(&wg.nqs_all[(size_t) e * Ng * (Kg / 2)], qe.data(), qe.size());
        std::memcpy(&wg.nsc_all[(size_t) e * Ng * (Kg / 16)], se.data(), se.size());
    }
    // fp16 dequantized Q8 for the tensor core path
    wq.fp16.resize((size_t) Nq * Kq);
    for (int n = 0; n < Nq; ++n)
        for (int b = 0; b < Kq / 32; ++b) {
            float d = __half2float(wq.q8s[(size_t) n * (Kq / 32) + b]);
            for (int k = 0; k < 32; ++k)
                wq.fp16[(size_t) n * Kq + b * 32 + k] = __float2half(d * wq.q8[(size_t) n * Kq + b * 32 + k]);
        }

    // input vector
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> xq(Kq), xg(Kg);
    for (auto & v : xq) v = nd(rng);
    for (auto & v : xg) v = nd(rng);

    // device buffers
    int8_t * d_q8; __half * d_q8s; uint8_t * d_nqs; uint8_t * d_nsc; uint8_t * d_aos_q; uint8_t * d_aos_nv;
    uint8_t * d_nqs_all; uint8_t * d_nsc_all; __half * d_fp16; int * d_ids;
    float * d_xq; float * d_xg; float * d_y; float * d_yref;
    CHECK(cudaMalloc(&d_q8, wq.q8.size()));
    CHECK(cudaMalloc(&d_q8s, wq.q8s.size() * sizeof(__half)));
    CHECK(cudaMalloc(&d_aos_q, wq.aos_q8.size()));
    CHECK(cudaMalloc(&d_nqs, wg.nqs.size()));
    CHECK(cudaMalloc(&d_nsc, wg.nsc.size()));
    CHECK(cudaMalloc(&d_aos_nv, wg.aos_nv.size()));
    CHECK(cudaMalloc(&d_nqs_all, wg.nqs_all.size()));
    CHECK(cudaMalloc(&d_nsc_all, wg.nsc_all.size()));
    CHECK(cudaMalloc(&d_fp16, wq.fp16.size() * sizeof(__half)));
    CHECK(cudaMalloc(&d_ids, E * sizeof(int)));
    CHECK(cudaMalloc(&d_xq, Kq * sizeof(float)));
    CHECK(cudaMalloc(&d_xg, Kg * sizeof(float)));
    CHECK(cudaMalloc(&d_y, (size_t) E * Ng * sizeof(float)));
    CHECK(cudaMalloc(&d_yref, (size_t) E * Ng * sizeof(float)));
    CHECK(cudaMemcpy(d_q8, wq.q8.data(), wq.q8.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_q8s, wq.q8s.data(), wq.q8s.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_aos_q, wq.aos_q8.data(), wq.aos_q8.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nqs, wg.nqs.data(), wg.nqs.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nsc, wg.nsc.data(), wg.nsc.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_aos_nv, wg.aos_nv.data(), wg.aos_nv.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nqs_all, wg.nqs_all.data(), wg.nqs_all.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nsc_all, wg.nsc_all.data(), wg.nsc_all.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_fp16, wq.fp16.data(), wq.fp16.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ids, ids_h, E * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xq, xq.data(), Kq * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xg, xg.data(), Kg * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    const size_t smem_q = Kq * sizeof(float);
    const size_t smem_g = Kg * sizeof(float);

    auto fetch = [&](const float * dev, size_t n) {
        std::vector<float> h(n);
        CHECK(cudaMemcpy(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    };

    if (argc > 2 && std::strcmp(argv[2], "--ncu9") == 0) {
        // profiling entry: only the fused top-8 kernels, one launch each
        const size_t sm9 = (size_t) (Kg / 64) * 68 * sizeof(float);
        v9_nvfp4_smem<4, false><<<dim3((Ng + 31) / 32, E), 256, sm9, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
        v9_nvfp4_smem<4, true><<<dim3((Ng + 31) / 32, E), 256, sm9, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
        v9_nvfp4_smem<2, false><<<dim3((Ng + 15) / 16, E), 256, sm9, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
        CHECK(cudaStreamSynchronize(s));
        return 0;
    }
    // ---------------- Q8_0 ----------------
    // reference: baseline AoS kernel from gemv.cuh
    launch_gemv_q8_0(d_aos_q, d_xq, d_yref, Kq, Nq, s);
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> refq = fetch(d_yref, Nq);
    const double bytes_q8 = (double) Nq * Kq * 34.0 / 32.0;
    const double bytes_nv = (double) Ng * Kg * 36.0 / 64.0;
    const double bytes_fp16 = (double) Nq * Kq * 2.0;

    std::printf("== Q8_0  attn_q  K=%d N=%d (weight %.2f MB in GGUF)\n", Kq, Nq, bytes_q8 / 1e6);
    if (ncu_mode) {
        for (int it = 0; it < 1; ++it) {
            v1_q8_warp_row<<<(Nq * 32 + 255) / 256, 256, 0, s>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq);
            v2_q8_split<1><<<Nq / 8, 256, smem_q, s>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq);
            v2_q8_split<2><<<Nq / 4, 256, smem_q, s>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq);
            v2_q8_split<4><<<Nq / 2, 256, smem_q, s>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq);
            v3_q8_multirow<<<Nq / 8, 256, smem_q, s>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq);
            v6_fp16_wmma<<<Nq / 128, 256, 0, s>>>(d_fp16, d_xq, d_y, Kq, Nq);
            CHECK(cudaStreamSynchronize(s));
            std::printf("ncu mode done\n");
            return 0;
        }
    }

    struct Row { const char * name; const char * fmt; float us; double bytes; int regs; std::vector<float> y; };
    std::vector<Row> rows;
    auto add = [&](const char * name, const char * fmt, float us, double bytes, int regs, float * dev, size_t n) {
        rows.push_back({name, fmt, us, bytes, regs, fetch(dev, n)});
    };

    // baseline (AoS, one warp per row)
    add("v0 warp/row AoS (baseline)", "Q8", graph_time_us([&](cudaStream_t st) { launch_gemv_q8_0(d_aos_q, d_xq, d_y, Kq, Nq, st); }, s), bytes_q8,
        regs_of((const void *) gemv_q8_0_simt), d_y, Nq);
    add("v1 warp/row SoA vec", "Q8", graph_time_us([&](cudaStream_t st) {
            v1_q8_warp_row<<<(Nq * 32 + 255) / 256, 256, 0, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v1_q8_warp_row), d_y, Nq);
    add("v2 split WPR=1", "Q8", graph_time_us([&](cudaStream_t st) {
            v2_q8_split<1><<<Nq / 8, 256, smem_q, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v2_q8_split<1>), d_y, Nq);
    add("v2 split WPR=2", "Q8", graph_time_us([&](cudaStream_t st) {
            v2_q8_split<2><<<Nq / 4, 256, smem_q, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v2_q8_split<2>), d_y, Nq);
    add("v2 split WPR=4", "Q8", graph_time_us([&](cudaStream_t st) {
            v2_q8_split<4><<<Nq / 2, 256, smem_q, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v2_q8_split<4>), d_y, Nq);
    add("v3 multirow 8 rows, smem x", "Q8", graph_time_us([&](cudaStream_t st) {
            v3_q8_multirow<<<Nq / 8, 256, smem_q, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v3_q8_multirow), d_y, Nq);
    add("v7 Q8 rows/warp=1", "Q8", graph_time_us([&](cudaStream_t st) {
            v7_q8_rows<1><<<(Nq + 7) / 8, 256, 0, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v7_q8_rows<1>), d_y, Nq);
    add("v7 Q8 rows/warp=2", "Q8", graph_time_us([&](cudaStream_t st) {
            v7_q8_rows<2><<<(Nq / 2 + 7) / 8, 256, 0, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v7_q8_rows<2>), d_y, Nq);
    add("v7 Q8 rows/warp=4", "Q8", graph_time_us([&](cudaStream_t st) {
            v7_q8_rows<4><<<(Nq / 4 + 7) / 8, 256, 0, st>>>(d_q8, d_q8s, d_xq, d_y, Kq, Nq); }, s), bytes_q8,
        regs_of((const void *) v7_q8_rows<4>), d_y, Nq);
    add("v6 fp16 wmma (tensor core)", "FP16", graph_time_us([&](cudaStream_t st) {
            v6_fp16_wmma<<<Nq / 128, 256, 0, st>>>(d_fp16, d_xq, d_y, Kq, Nq); }, s), bytes_fp16,
        regs_of((const void *) v6_fp16_wmma), d_y, Nq);
    for (auto & r : rows) report(r.name, r.fmt, r.us, r.bytes, r.regs, r.y, refq);

    // ---------------- NVFP4 (one expert, K=2304 N=2304) ----------------
    launch_gemv_nvfp4(d_aos_nv, d_xg, d_yref, Kg, Ng, s);
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> refg = fetch(d_yref, Ng);
    std::printf("\n== NVFP4 gate_up expert  K=%d N=%d (weight %.2f MB)\n", Kg, Ng, bytes_nv / 1e6);
    rows.clear();
    add("v0 warp/row AoS (baseline)", "NV4", graph_time_us([&](cudaStream_t st) { launch_gemv_nvfp4(d_aos_nv, d_xg, d_y, Kg, Ng, st); }, s), bytes_nv,
        regs_of((const void *) gemv_nvfp4_simt), d_y, Ng);
    add("v8 NV4 rows/warp=1 bitops", "NV4", graph_time_us([&](cudaStream_t st) {
            v8_nvfp4_rows<1, true><<<(Ng + 7) / 8, 256, 0, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng); }, s), bytes_nv,
        regs_of((const void *) v8_nvfp4_rows<1, true>), d_y, Ng);
    add("v8 NV4 rows/warp=2 bitops", "NV4", graph_time_us([&](cudaStream_t st) {
            v8_nvfp4_rows<2, true><<<(Ng / 2 + 7) / 8, 256, 0, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng); }, s), bytes_nv,
        regs_of((const void *) v8_nvfp4_rows<2, true>), d_y, Ng);
    add("v8 NV4 rows/warp=4 bitops", "NV4", graph_time_us([&](cudaStream_t st) {
            v8_nvfp4_rows<4, true><<<(Ng / 4 + 7) / 8, 256, 0, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng); }, s), bytes_nv,
        regs_of((const void *) v8_nvfp4_rows<4, true>), d_y, Ng);
    add("v8 NV4 rows/warp=2 LUT", "NV4", graph_time_us([&](cudaStream_t st) {
            v8_nvfp4_rows<2, false><<<(Ng / 2 + 7) / 8, 256, 0, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng); }, s), bytes_nv,
        regs_of((const void *) v8_nvfp4_rows<2, false>), d_y, Ng);
    add("v4 LUT WPR=1", "NV4", graph_time_us([&](cudaStream_t st) {
            v4_nvfp4<1, false><<<Ng / 8, 256, smem_g, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng, nullptr, 0, 0, Ng); }, s), bytes_nv,
        regs_of((const void *) v4_nvfp4<1, false>), d_y, Ng);
    add("v4 bitops WPR=1", "NV4", graph_time_us([&](cudaStream_t st) {
            v4_nvfp4<1, true><<<Ng / 8, 256, smem_g, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng, nullptr, 0, 0, Ng); }, s), bytes_nv,
        regs_of((const void *) v4_nvfp4<1, true>), d_y, Ng);
    add("v4 LUT WPR=2", "NV4", graph_time_us([&](cudaStream_t st) {
            v4_nvfp4<2, false><<<Ng / 4, 256, smem_g, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng, nullptr, 0, 0, Ng); }, s), bytes_nv,
        regs_of((const void *) v4_nvfp4<2, false>), d_y, Ng);
    add("v4 LUT WPR=4", "NV4", graph_time_us([&](cudaStream_t st) {
            v4_nvfp4<4, false><<<Ng / 2, 256, smem_g, st>>>(d_nqs, d_nsc, d_xg, d_y, Kg, Ng, nullptr, 0, 0, Ng); }, s), bytes_nv,
        regs_of((const void *) v4_nvfp4<4, false>), d_y, Ng);
    for (auto & r : rows) report(r.name, r.fmt, r.us, r.bytes, r.regs, r.y, refg);

    // ---------------- V9 shared-memory x (single expert) ----------------
    {
        const size_t sm9 = (size_t) (Kg / 64) * 68 * sizeof(float);
        std::printf("\n== NVFP4 v9 shared-memory x (single expert, smem %zu B)\n", sm9);
        rows.clear();
        auto nv_grid = [&](int R) { return dim3((Ng + 8 * R - 1) / (8 * R), 1); };
        add("v9 smem x rows=1 bitops", "NV4", graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<1, true><<<nv_grid(1), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); }, s), bytes_nv,
            regs_of((const void *) v9_nvfp4_smem<1, true>), d_y, Ng);
        add("v9 smem x rows=2 bitops", "NV4", graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<2, true><<<nv_grid(2), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); }, s), bytes_nv,
            regs_of((const void *) v9_nvfp4_smem<2, true>), d_y, Ng);
        add("v9 smem x rows=2 LUT", "NV4", graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<2, false><<<nv_grid(2), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); }, s), bytes_nv,
            regs_of((const void *) v9_nvfp4_smem<2, false>), d_y, Ng);
        add("v9 smem x rows=4 LUT", "NV4", graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<4, false><<<nv_grid(4), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); }, s), bytes_nv,
            regs_of((const void *) v9_nvfp4_smem<4, false>), d_y, Ng);
        add("v9 smem x rows=4 bitops", "NV4", graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<4, true><<<nv_grid(4), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); }, s), bytes_nv,
            regs_of((const void *) v9_nvfp4_smem<4, true>), d_y, Ng);
        for (auto & r : rows) report(r.name, r.fmt, r.us, r.bytes, r.regs, r.y, refg);
    }

    // ---------------- Top-8 experts ----------------
    std::printf("\n== NVFP4 top-8 experts (gate_up, 8 x [%d x %d])\n", Ng, Kg);
    rows.clear();
    // reference: 8 launches of the baseline kernel
    std::vector<float> ref8(E * Ng);
    for (int e = 0; e < E; ++e) {
        launch_gemv_nvfp4(d_aos_nv, d_xg, d_yref + e * Ng, Kg, Ng, s);  // expert 0 weights for all slots
    }
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> one = fetch(d_yref, Ng);
    // per-expert reference on the host with the real expert data
    for (int e = 0; e < E; ++e) {
        std::vector<uint8_t> blob(per_expert_g);
        std::memcpy(blob.data(), f.base + f.data_start + tg->offset + (size_t) ids_h[e] * per_expert_g, per_expert_g);
        for (int n = 0; n < Ng; ++n) {
            double acc = 0;
            for (int b = 0; b < Kg / 64; ++b) {
                const uint8_t * blk = &blob[((size_t) n * (Kg / 64) + b) * 36];
                for (int sub = 0; sub < 4; ++sub) {
                    float d = ue4m3_host(blk[sub]);
                    for (int j = 0; j < 8; ++j) {
                        uint8_t by = blk[4 + sub * 8 + j];
                        acc += d * kKV[by & 0xF] * xg[b * 64 + sub * 16 + j];
                        acc += d * kKV[by >> 4] * xg[b * 64 + sub * 16 + j + 8];
                    }
                }
            }
            ref8[(size_t) e * Ng + n] = (float) acc;
        }
    }
    std::vector<float> ref8_dev(E * Ng);
    // launch fused kernel once for the reference check
    v4_nvfp4<1, false><<<dim3(Ng / 8, E), 256, smem_g, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids,
        (int64_t) Ng * (Kg / 2), (int64_t) Ng * (Kg / 16), Ng);
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> y8 = fetch(d_y, E * Ng);

    auto t_seq = graph_time_us([&](cudaStream_t st) {
        for (int e = 0; e < E; ++e) {
            v4_nvfp4<1, false><<<Ng / 8, 256, smem_g, st>>>(d_nqs_all + (size_t) ids_h[e] * Ng * (Kg / 2), d_nsc_all + (size_t) ids_h[e] * Ng * (Kg / 16),
                d_xg, d_y + e * Ng, Kg, Ng, nullptr, 0, 0, Ng);
        }
    }, s, 20);
    auto t_fused = graph_time_us([&](cudaStream_t st) {
        v4_nvfp4<1, false><<<dim3(Ng / 8, E), 256, smem_g, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids,
            (int64_t) Ng * (Kg / 2), (int64_t) Ng * (Kg / 16), Ng);
    }, s, 20);
    auto t_fused2 = graph_time_us([&](cudaStream_t st) {
        v4_nvfp4<2, false><<<dim3(Ng / 4, E), 256, smem_g, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids,
            (int64_t) Ng * (Kg / 2), (int64_t) Ng * (Kg / 16), Ng);
    }, s, 20);
    const double bytes8 = E * bytes_nv;
    auto t_f8r2 = graph_time_us([&](cudaStream_t st) {
        v8f_nvfp4_rows<2, false><<<dim3((Ng / 2 + 7) / 8, E), 256, 0, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
    }, s, 20);
    auto t_f8r4 = graph_time_us([&](cudaStream_t st) {
        v8f_nvfp4_rows<4, false><<<dim3((Ng / 4 + 7) / 8, E), 256, 0, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
    }, s, 20);
    auto t_f8r2b = graph_time_us([&](cudaStream_t st) {
        v8f_nvfp4_rows<2, true><<<dim3((Ng / 2 + 7) / 8, E), 256, 0, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
    }, s, 20);
    // numerical check of the v8f LUT rows=2 path against the host reference
    v8f_nvfp4_rows<2, false><<<dim3((Ng / 2 + 7) / 8, E), 256, 0, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> y8v8 = fetch(d_y, E * Ng);
    double e8v8 = 0, m8v8 = 0;
    for (size_t i = 0; i < ref8.size(); ++i) {
        e8v8 = std::max(e8v8, (double) std::fabs(y8v8[i] - ref8[i]));
        m8v8 = std::max(m8v8, (double) std::fabs(ref8[i]));
    }
    std::printf("%-30s %-6s %9.2f us %8.1f GB/s  regs %3d  rel.err %9.2e\n", "v8f fused rows=2 LUT", "NV4",
        t_f8r2, bytes8 / (t_f8r2 * 1e-6) / 1e9, regs_of((const void *) v8f_nvfp4_rows<2, false>), e8v8 / m8v8);
    std::printf("%-30s %-6s %9.2f us %8.1f GB/s  regs %3d\n", "v8f fused rows=4 LUT", "NV4",
        t_f8r4, bytes8 / (t_f8r4 * 1e-6) / 1e9, regs_of((const void *) v8f_nvfp4_rows<4, false>));
    std::printf("%-30s %-6s %9.2f us %8.1f GB/s  regs %3d\n", "v8f fused rows=2 bitops", "NV4",
        t_f8r2b, bytes8 / (t_f8r2b * 1e-6) / 1e9, regs_of((const void *) v8f_nvfp4_rows<2, true>));

    // errors vs host reference of the real expert data
    double maxabs = 0, maxref = 0;
    for (size_t i = 0; i < ref8.size(); ++i) {
        maxabs = std::max(maxabs, (double) std::fabs(y8[i] - ref8[i]));
        maxref = std::max(maxref, (double) std::fabs(ref8[i]));
    }
    std::printf("%-30s %-6s %9.2f us %8.1f GB/s  regs %3d  rel.err %9.2e\n", "8 launches, LUT WPR=1", "NV4",
        t_seq, bytes8 / (t_seq * 1e-6) / 1e9, regs_of((const void *) v4_nvfp4<1, false>), maxabs / maxref);
    std::printf("%-30s %-6s %9.2f us %8.1f GB/s  regs %3d  rel.err %9.2e\n", "fused 1 launch, LUT WPR=1", "NV4",
        t_fused, bytes8 / (t_fused * 1e-6) / 1e9, regs_of((const void *) v4_nvfp4<1, false>), maxabs / maxref);
    std::printf("%-30s %-6s %9.2f us %8.1f GB/s  regs %3d  rel.err %9.2e\n", "fused 1 launch, LUT WPR=2", "NV4",
        t_fused2, bytes8 / (t_fused2 * 1e-6) / 1e9, regs_of((const void *) v4_nvfp4<2, false>), maxabs / maxref);

    {
        const size_t sm9 = (size_t) (Kg / 64) * 68 * sizeof(float);
        std::vector<float> ref_top8 = ref8;
        std::printf("\n== top-8 fused, v9 shared-memory x\n");
        auto t9_2 = graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<2, false><<<dim3((Ng + 15) / 16, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); }, s, 20);
        auto t9_4 = graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<4, false><<<dim3((Ng + 31) / 32, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); }, s, 20);
        auto t9_4b = graph_time_us([&](cudaStream_t st) {
            v9_nvfp4_smem<4, true><<<dim3((Ng + 31) / 32, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); }, s, 20);
        v9_nvfp4_smem<4, false><<<dim3((Ng + 31) / 32, E), 256, sm9, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
        CHECK(cudaStreamSynchronize(s));
        std::vector<float> y9 = fetch(d_y, E * Ng);
        double e9 = 0, m9 = 0;
        for (size_t i = 0; i < ref_top8.size(); ++i) { e9 = std::max(e9, (double) std::fabs(y9[i] - ref_top8[i])); m9 = std::max(m9, (double) std::fabs(ref_top8[i])); }
        std::printf("%-30s %9.2f us %8.1f GB/s  rel.err %9.2e\n", "v9 fused rows=2 LUT", t9_2, bytes8 / (t9_2 * 1e-6) / 1e9, e9 / m9);
        std::printf("%-30s %9.2f us %8.1f GB/s\n", "v9 fused rows=4 LUT", t9_4, bytes8 / (t9_4 * 1e-6) / 1e9);
        std::printf("%-30s %9.2f us %8.1f GB/s\n", "v9 fused rows=4 bitops", t9_4b, bytes8 / (t9_4b * 1e-6) / 1e9);
    }

    {
        // V10 timing: single expert and fused top-8
        const size_t sm9 = (size_t) (Kg / 64) * 68 * sizeof(float);
        std::printf("\n== V10 sub-block x (regs = live x 16 instead of 64)\n");
        auto g10 = [&](int R) { return dim3((Ng + 8 * R - 1) / (8 * R), 1); };
        auto t10 = [&](auto launch) { return graph_time_us(launch, s, 100); };
        // single expert
        float t_s2 = t10([&](cudaStream_t st) { v10_nvfp4_sub<2, false><<<g10(2), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); });
        float t_s4 = t10([&](cudaStream_t st) { v10_nvfp4_sub<4, false><<<g10(4), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); });
        float t_s4b = t10([&](cudaStream_t st) { v10_nvfp4_sub<4, true><<<g10(4), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, nullptr); });
        std::printf("single rows=2 LUT %8.2f us %7.1f GB/s regs %3d | rows=4 LUT %8.2f us %7.1f GB/s regs %3d | rows=4 bitops %8.2f us %7.1f GB/s regs %3d\n",
            t_s2, bytes_nv / (t_s2 * 1e-6) / 1e9, regs_of((const void *) v10_nvfp4_sub<2, false>),
            t_s4, bytes_nv / (t_s4 * 1e-6) / 1e9, regs_of((const void *) v10_nvfp4_sub<4, false>),
            t_s4b, bytes_nv / (t_s4b * 1e-6) / 1e9, regs_of((const void *) v10_nvfp4_sub<4, true>));
        // fused top-8
        const double b8 = E * bytes_nv;
        float t_f2 = t10([&](cudaStream_t st) { v10_nvfp4_sub<2, false><<<dim3((Ng + 15) / 16, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); });
        float t_f4 = t10([&](cudaStream_t st) { v10_nvfp4_sub<4, false><<<dim3((Ng + 31) / 32, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); });
        float t_f4b = t10([&](cudaStream_t st) { v10_nvfp4_sub<4, true><<<dim3((Ng + 31) / 32, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); });
        // correctness vs host reference
        v10_nvfp4_sub<4, false><<<dim3((Ng + 31) / 32, E), 256, sm9, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
        CHECK(cudaStreamSynchronize(s));
        std::vector<float> y10 = fetch(d_y, E * Ng);
        double e10 = 0, m10 = 0;
        for (size_t i = 0; i < ref8.size(); ++i) { e10 = std::max(e10, (double) std::fabs(y10[i] - ref8[i])); m10 = std::max(m10, (double) std::fabs(ref8[i])); }
        std::printf("fused top-8 rows=2 LUT %8.2f us %7.1f GB/s | rows=4 LUT %8.2f us %7.1f GB/s (rel.err %.2e) | rows=4 bitops %8.2f us %7.1f GB/s regs %3d\n",
            t_f2, b8 / (t_f2 * 1e-6) / 1e9, t_f4, b8 / (t_f4 * 1e-6) / 1e9, e10 / m10, t_f4b, b8 / (t_f4b * 1e-6) / 1e9, regs_of((const void *) v10_nvfp4_sub<4, true>));
    }
    {
        // rows per warp = 8 and 16 (v9 kernel), fused top-8, LUT
        const size_t sm9 = (size_t) (Kg / 64) * 68 * sizeof(float);
        const double b8 = E * bytes_nv;
        float t_r4 = graph_time_us([&](cudaStream_t st) { v9_nvfp4_smem<4, false><<<dim3((Ng + 31) / 32, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); }, s, 100);
        float t_r8 = graph_time_us([&](cudaStream_t st) { v9_nvfp4_smem<8, false><<<dim3((Ng + 63) / 64, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); }, s, 100);
        float t_r16 = graph_time_us([&](cudaStream_t st) { v9_nvfp4_smem<16, false><<<dim3((Ng + 127) / 128, E), 256, sm9, st>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids); }, s, 100);
        v9_nvfp4_smem<8, false><<<dim3((Ng + 63) / 64, E), 256, sm9, s>>>(d_nqs_all, d_nsc_all, d_xg, d_y, Kg, Ng, d_ids);
        CHECK(cudaStreamSynchronize(s));
        std::vector<float> y8r = fetch(d_y, E * Ng);
        double e8r = 0, m8r = 0;
        for (size_t i = 0; i < ref8.size(); ++i) { e8r = std::max(e8r, (double) std::fabs(y8r[i] - ref8[i])); m8r = std::max(m8r, (double) std::fabs(ref8[i])); }
        std::printf("\n== fused top-8, rows per warp sweep (v9 smem x, LUT)\n");
        std::printf("rows=4 %8.2f us %7.1f GB/s regs %3d\n", t_r4, b8 / (t_r4 * 1e-6) / 1e9, regs_of((const void *) v9_nvfp4_smem<4, false>));
        std::printf("rows=8 %8.2f us %7.1f GB/s regs %3d rel.err %.2e\n", t_r8, b8 / (t_r8 * 1e-6) / 1e9, regs_of((const void *) v9_nvfp4_smem<8, false>), e8r / m8r);
        std::printf("rows=16 %8.2f us %7.1f GB/s regs %3d\n", t_r16, b8 / (t_r16 * 1e-6) / 1e9, regs_of((const void *) v9_nvfp4_smem<16, false>));
    }
    cudaFree(d_q8); cudaFree(d_q8s); cudaFree(d_aos_q); cudaFree(d_nqs); cudaFree(d_nsc); cudaFree(d_aos_nv);
    cudaFree(d_nqs_all); cudaFree(d_nsc_all); cudaFree(d_fp16); cudaFree(d_ids); cudaFree(d_xq); cudaFree(d_xg);
    cudaFree(d_y); cudaFree(d_yref);
    gguf::close(f);
    return 0;
}
