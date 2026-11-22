// Shared CUDA kernels for the Inkling 2 engine (sm_86). Frozen GEMV baseline:
// NVFP4 fused top-8 with shared-memory x, 4 rows per warp, byte-pair LUT (v9 rows=4 LUT).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ------------------------------------------------------------ helpers ----

__device__ __constant__ float kE2M1d[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ggml_ue4m3_to_fp32: unsigned E4M3 (bias 7), value * 0.5 (E2M1 table is doubled).
__device__ __forceinline__ float ue4m3_dev(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 0x7;
    float raw = (e == 0) ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}

// Block-wide sum for blockDim.x <= 1024. smem must hold 32 floats.
__device__ __forceinline__ float block_sum(float v, float * smem) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    float r = smem[0];
    __syncthreads();
    return r;
}

// bf16 bits -> float
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(((uint32_t) b) << 16);
}

// ------------------------------------------------------------ GEMV ----
// Q8_0 in SoA: qs int8 [N][K], sc half [N][K/32]. Two rows per warp, x read as float4.
__global__ void gemv_q8_soa_rows2(const int8_t * __restrict__ qs, const __half * __restrict__ sc,
                                  const float * __restrict__ x, float * __restrict__ y, int K, int N) {
    constexpr int ROWS = 2;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * ROWS;
    if (row0 >= N) return;
    const int nb = K / 32;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc[ROWS] = {0.0f, 0.0f};
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

// Q8_0 GEMV, template R rows per warp. Same per-row arithmetic as gemv_q8_soa_rows2 (bit-identical results).
template <int R>
__global__ void gemv_q8_rows(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                             float * __restrict__ y, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int row0 = warp * R;
    if (row0 >= N) return;
    const int nb = K / 32;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = __ldg(x4 + b * 8 + i);
#pragma unroll
        for (int r = 0; r < R; ++r) {
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
    for (int r = 0; r < R; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = a;
    }
}

// Q8_0 GEMV, four warps per row, warps split the K blocks; partials reduced in shared memory (small N).
__global__ void gemv_q8_split4(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                               float * __restrict__ y, int K, int N) {
    __shared__ float red[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rl = warp / 4, part = warp % 4;
    const int row = blockIdx.x * 2 + rl;
    const int nb = K / 32;
    float acc = 0.0f;
    if (row < N) {
        const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
        const __half * s = sc + (size_t) row * nb;
        for (int b = part * 32 + lane; b < nb; b += 128) {
            const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
            const float d = __half2float(s[b]);
            const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float4 * xb = reinterpret_cast<const float4 *>(x + b * 32);
            float t = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 v = __ldg(xb + j);
                t += (float) (int8_t) w[j] * v.x + (float) (int8_t) (w[j] >> 8) * v.y
                   + (float) (int8_t) (w[j] >> 16) * v.z + (float) (int8_t) (w[j] >> 24) * v.w;
            }
            acc += d * t;
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (part == 0 && lane == 0 && row < N) {
        float t = 0.0f;
        for (int p = 0; p < 4; ++p) t += red[rl * 4 + p];
        y[row] = t;
    }
}

// Dispatch by shape (measured best per shape on RTX 3090, see q8_bench).
inline void launch_gemv_q8(const int8_t * qs, const __half * sc, const float * x, float * y, int K, int N,
                           cudaStream_t s) {
    if (K == 2304 && N == 384) {                        // wrel: 4 warps per row
        gemv_q8_split4<<<(N + 1) / 2, 256, 0, s>>>(qs, sc, x, y, K, N);
    } else if (K == 2304 && N == 3072) {                // wq, wgate
        gemv_q8_rows<2><<<(N / 2 + 3) / 4, 128, 0, s>>>(qs, sc, x, y, K, N);
    } else if (K == 2304 && N == 1024) {                // wk, wv
        gemv_q8_rows<2><<<(N / 2 + 1) / 2, 64, 0, s>>>(qs, sc, x, y, K, N);
    } else if (K == 3072 && N == 2304) {                // wo
        gemv_q8_rows<2><<<(N / 2 + 1) / 2, 64, 0, s>>>(qs, sc, x, y, K, N);
    } else {                                            // head and others: 2 rows per warp, 128 threads
        gemv_q8_rows<2><<<((N + 1) / 2 * 32 + 127) / 128, 128, 0, s>>>(qs, sc, x, y, K, N);
    }
}

// F32 weights [N][K], warp per row.
__global__ void gemv_f32(const float * __restrict__ W, const float * __restrict__ x, float * __restrict__ y,
                         int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    float acc = 0.0f;
    const float * row = W + (size_t) warp * K;
    for (int k = lane; k < K; k += 32) acc += row[k] * x[k];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[warp] = acc;
}

// BF16 weights [N][K], warp per row.
__global__ void gemv_bf16(const uint16_t * __restrict__ W, const float * __restrict__ x, float * __restrict__ y,
                          int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= N) return;
    float acc = 0.0f;
    const uint16_t * row = W + (size_t) warp * K;
    for (int k = lane; k < K; k += 32) acc += bf16_to_f32(row[k]) * x[k];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[warp] = acc;
}

// NVFP4 SoA: qs u8 [E][N][K/2], sc u8 [E][N][K/16]. Fused over `slots` experts:
// expert = ids[slot] (ids may be null: expert 0), x for slot = x + slot * x_slot_stride.
// Output y[slot][row]. This is the frozen v9 rows=4 LUT kernel.
template <int ROWS>
__global__ void nvfp4_gemv_fused(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all,
                                 const float * __restrict__ x, int64_t x_slot_stride,
                                 float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    const int nb = K / 64;
    const int slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    const float * xin = x + slot * x_slot_stride;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const int b = i >> 6, r = i & 63;
        xs[b * 68 + r] = xin[i];
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
                        const float2 v = lut2[byte];
                        sub += v.x * xl + v.y * xh;
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

// Same math as nvfp4_gemv_fused<4>, block of 128 threads (4 warps x 4 rows).
__global__ void nvfp4_gemv_fused_t128(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all,
                                      const float * __restrict__ x, int64_t x_slot_stride,
                                      float * __restrict__ y, int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    constexpr int R = 4, T = 128;
    const int nb = K / 64, slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    const float * xin = x + slot * x_slot_stride;
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    for (int i = threadIdx.x; i < 256; i += T) { lut_ue[i] = ue4m3_dev((uint8_t) i); lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]); }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (T / 32) + warp) * R;
    if (row0 >= N) return;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68);
        float4 xv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) xv[i] = xb[i];
#pragma unroll
        for (int r = 0; r < R; ++r) {
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
                        const float2 v = lut2[byte];
                        sub += v.x * xl + v.y * xh;
                    }
                    tot += d * sub;
                }
                acc[r] += tot;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[(size_t) slot * N + row0 + r] = a;
    }
}

// Launch helper: smem = (K/64)*68 floats.
inline void launch_nvfp4_fused(const uint8_t * qs_all, const uint8_t * sc_all, const float * x, int64_t x_slot_stride,
                               float * y, int K, int N, const int * ids, int slots, cudaStream_t s) {
    const size_t smem = (size_t) (K / 64) * 68 * sizeof(float);
    // 4 rows per warp, 4 warps per block: measured best (nvfp4_sweep), same per-row math as the 256-thread variant
    dim3 grid((N + 15) / 16, slots);
    nvfp4_gemv_fused_t128<<<grid, 128, smem, s>>>(qs_all, sc_all, x, x_slot_stride, y, K, N, ids);
}

// ---------------------------------------------------------- norms, clip ----

// out[r][i] = in[r][i] * rsqrt(mean(in[r]^2) + eps) * (w ? w[i] : 1). One block per row.
__global__ void k_rms_rows(const float * __restrict__ in, const float * __restrict__ w, float * __restrict__ out,
                           int n, float eps) {
    __shared__ float red[32];
    const float * a = in + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += a[i] * a[i];
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = a[i] * inv;
        if (w) v *= w[i];
        out[(size_t) blockIdx.x * n + i] = v;
    }
}

// Per-head clip (inkling2 clip()): optional per-channel scale, then rows over L2 norm thr are shrunk.
// One block per head, blockDim >= dim.
__global__ void k_clip_heads(const float * __restrict__ in, const float * __restrict__ scale, float * __restrict__ out,
                             int dim, float thr) {
    __shared__ float red[32];
    const int h = blockIdx.x, d = threadIdx.x;
    float v = 0.0f;
    if (d < dim) {
        v = in[h * dim + d];
        if (scale) v *= scale[h * dim + d];
    }
    float ss = block_sum(d < dim ? v * v : 0.0f, red);
    float norm = sqrtf(ss) + 1e-6f;
    float f = fminf(1.0f, thr / norm);
    if (d < dim) out[h * dim + d] = v * f;
}

// Causal short conv with residual (convolve() with zero dynamic taps). state = last 3 inputs [3][C].
// out = (cur + sum_j w[c][j] * hist_j) / sqrt(5), then state shifts.
__global__ void k_conv4(const float * __restrict__ in, float * __restrict__ state, const float * __restrict__ w,
                        float * __restrict__ out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    const float cur = in[c];
    const float y = w[c * 4 + 0] * h0 + w[c * 4 + 1] * h1 + w[c * 4 + 2] * h2 + w[c * 4 + 3] * cur;
    out[c] = (cur + y) * 0.44721359549995794f;  // 1/sqrt(5)
    state[c] = h1;
    state[C + c] = h2;
    state[2 * C + c] = cur;
}

// ------------------------------------------------------- MoE, mix ----

// mix(blocks=[a, b], cur=b, query): rms-normalize both states, softmax over 2 dot products / sqrt(E),
// out = sum_s p_s * norm_s. One block of 256 threads.
__global__ void k_mix2(const float * __restrict__ a, const float * __restrict__ b, const float * __restrict__ q,
                       float * __restrict__ out, int E) {
    __shared__ float red[32];
    __shared__ float p[2];
    const float eps = 1e-6f;
    float ssa = 0.0f, ssb = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { ssa += a[i] * a[i]; ssb += b[i] * b[i]; }
    ssa = block_sum(ssa, red);
    ssb = block_sum(ssb, red);
    const float ia = rsqrtf(ssa / E + eps), ib = rsqrtf(ssb / E + eps);
    float da = 0.0f, db = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        da += a[i] * ia * q[i];
        db += b[i] * ib * q[i];
    }
    da = block_sum(da, red);
    db = block_sum(db, red);
    if (threadIdx.x == 0) {
        const float s0 = da / sqrtf((float) E), s1 = db / sqrtf((float) E);
        const float m = fmaxf(s0, s1);
        const float e0 = expf(s0 - m), e1 = expf(s1 - m);
        p[0] = e0 / (e0 + e1);
        p[1] = e1 / (e0 + e1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        out[i] = p[0] * (a[i] * ia) + p[1] * (b[i] * ib);
    }
}

// Router: probs = sigmoid(logits), top-8 on probs + bias (ties: smallest index), weights = probs / (sum + 1e-20) * scale.
// Single block of 128 threads (one per expert). Writes ids[8], wts[8].
__global__ void k_router_top8(const float * __restrict__ logits, const float * __restrict__ bias,
                              int * __restrict__ ids, float * __restrict__ wts, float scale) {
    __shared__ float probs[128];
    __shared__ float sel[128];
    __shared__ float wv[4];
    __shared__ int wi[4];
    __shared__ int chosen[8];
    const int e = threadIdx.x;
    const float pr = 1.0f / (1.0f + expf(-logits[e]));
    probs[e] = pr;
    sel[e] = pr + bias[e];
    __syncthreads();
    const int lane = e & 31, warp = e >> 5;
    for (int k = 0; k < 8; ++k) {
        bool used = false;
        for (int u = 0; u < k; ++u) used |= (chosen[u] == e);
        float v = used ? -INFINITY : sel[e];
        int idx = e;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, off);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { wv[warp] = v; wi[warp] = idx; }
        __syncthreads();
        if (e == 0) {
            float bv = wv[0];
            int bi = wi[0];
            for (int w = 1; w < 4; ++w)
                if (wv[w] > bv || (wv[w] == bv && wi[w] < bi)) { bv = wv[w]; bi = wi[w]; }
            chosen[k] = bi;
            ids[k] = bi;
        }
        __syncthreads();
    }
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k) sum += probs[chosen[k]];
        sum += 1e-20f;
        for (int k = 0; k < 8; ++k) wts[k] = probs[chosen[k]] / sum * scale;
    }
}

// SITU on gate_up of the 8 slots: gu[s][0:1152] = gate, gu[s][1152:2304] = up, both scaled by gu_scale[expert].
// h[s][i] = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb).
__global__ void k_situ(const float * __restrict__ gu, const float * __restrict__ gu_scale, const int * __restrict__ ids,
                       float * __restrict__ h, float beta, float lb) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1152) return;
    const float sc = gu_scale[ids[s]];
    const float g = gu[s * 2304 + i] * sc;
    const float u = gu[s * 2304 + 1152 + i] * sc;
    const float act = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
    const float up = lb * tanhf(u / lb);
    h[s * 1152 + i] = act * up;
}

// moe_out[i] = sum_s wts[s] * down_scale[ids[s]] * dn[s][i]
__global__ void k_moe_sum(const float * __restrict__ dn, const float * __restrict__ dn_scale, const int * __restrict__ ids,
                          const float * __restrict__ wts, float * __restrict__ out, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float acc = 0.0f;
    for (int s = 0; s < 8; ++s) acc += wts[s] * dn_scale[ids[s]] * dn[s * N + i];
    out[i] = acc;
}

// Gate: out[i] = o[i] * sigmoid(g[i]).
__global__ void k_gate_mul(const float * __restrict__ o, const float * __restrict__ g, float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = o[i] * (1.0f / (1.0f + expf(-g[i])));
}

// Attention for one query position (decode), one block of 128 threads per head.
// Q [24][128] (clipped), K/V cache [T][8][128], R [24][16] (clipped), relW [1024][16], sink [24].
// bias(h, j) = sum_r relW[pos - j][r] * R[h][r] - sink[h]; keys j in [max(0, pos-1023), pos].
// p = exp(s - m) / (exp(-m) + sum exp(s - m)) with the zero sink logit.
__global__ void k_attn_decode(const float * __restrict__ Q, const float * __restrict__ Kc, const float * __restrict__ Vc,
                              const float * __restrict__ R, const float * __restrict__ relW, const float * __restrict__ sink,
                              int pos, float * __restrict__ O) {
    __shared__ float q[128];
    __shared__ float r[16];
    __shared__ float sc[1024];
    __shared__ float denom;
    const int h = blockIdx.x, tid = threadIdx.x, kvh = h / 3;
    q[tid] = Q[h * 128 + tid];
    if (tid < 16) r[tid] = R[h * 16 + tid];
    __syncthreads();
    const int j0 = pos - 1023 > 0 ? pos - 1023 : 0;
    const int nk = pos - j0 + 1;
    const float inv_sqrt_d = 0.08838834764831845f;  // 1/sqrt(128)
    for (int j = j0 + tid; j <= pos; j += blockDim.x) {
        const int d = pos - j;
        float rb = 0.0f;
        for (int k = 0; k < 16; ++k) rb += relW[d * 16 + k] * r[k];
        const float bias = rb - sink[h];
        const float * kp = Kc + ((size_t) j * 8 + kvh) * 128;
        float dot = 0.0f;
        for (int dd = 0; dd < 128; ++dd) dot += q[dd] * kp[dd];
        sc[j - j0] = dot * inv_sqrt_d + bias;
    }
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;  // zero sink logit
        for (int i = 0; i < nk; ++i) m = fmaxf(m, sc[i]);
        float den = expf(-m);
        for (int i = 0; i < nk; ++i) {
            sc[i] = expf(sc[i] - m);
            den += sc[i];
        }
        denom = den;
    }
    __syncthreads();
    float o = 0.0f;
    for (int i = 0; i < nk; ++i) o += sc[i] * Vc[((size_t) (j0 + i) * 8 + kvh) * 128 + tid];
    O[h * 128 + tid] = o / denom;
}

__global__ void k_add(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// Token embedding row gather from Q8 SoA: out[i] = qs[tok][i] * sc[tok][i/32].
__global__ void k_embed_q8(const int8_t * __restrict__ qs, const __half * __restrict__ sc, int tok, int E,
                           float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    out[i] = (float) qs[(size_t) tok * E + i] * __half2float(sc[(size_t) tok * (E / 32) + i / 32]);
}

// ------------------------------------------------ inkling2 layer extras ----

// Causal conv with input-dependent taps (dynamic). Taps per KV head h (C/8 channels each): dyn[h*4 + lag].
// y = sum_j w[c][j] * x_{t+j-3} + sum_lag tap[h][lag] * x_{t-lag};  out = (cur + y) / 3.
__global__ void k_conv4_dyn(const float * __restrict__ in, float * __restrict__ state, const float * __restrict__ w,
                            const float * __restrict__ dyn, float * __restrict__ out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    const float cur = in[c];
    const int head = c / (C / 8);
    const float * tap = dyn + head * 4;
    const float y = w[c * 4 + 0] * h0 + w[c * 4 + 1] * h1 + w[c * 4 + 2] * h2 + w[c * 4 + 3] * cur
                  + tap[0] * cur + tap[1] * h2 + tap[2] * h1 + tap[3] * h0;
    out[c] = (cur + y) * (1.0f / 3.0f);
    state[c] = h1;
    state[C + c] = h2;
    state[2 * C + c] = cur;
}

__global__ void k_scale_inplace(float * __restrict__ v, int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= s;
}

// Multi-query mix, stage 1: scores[q][s] = <nst[s], query[q]> / sqrt(E). One block per (q, s).
__global__ void k_mix_scores(const float * __restrict__ nst, const float * __restrict__ q, float * __restrict__ scores,
                             int S, int E) {
    __shared__ float red[32];
    const int qi = blockIdx.x / S, si = blockIdx.x % S;
    float d = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) d += nst[si * E + i] * q[qi * E + i];
    d = block_sum(d, red);
    if (threadIdx.x == 0) scores[qi * S + si] = d / sqrtf((float) E);
}

// Stage 2: softmax over S per query row. One block of 32 threads per query.
__global__ void k_softmax_rows(float * __restrict__ scores, int S) {
    if (threadIdx.x != 0) return;
    float * r = scores + blockIdx.x * S;
    float m = r[0];
    for (int i = 1; i < S; ++i) m = fmaxf(m, r[i]);
    float s = 0.0f;
    for (int i = 0; i < S; ++i) { r[i] = expf(r[i] - m); s += r[i]; }
    for (int i = 0; i < S; ++i) r[i] /= s;
}

// Stage 3: out[q][i] = sum_s p[q][s] * nst[s][i].
__global__ void k_mix_out(const float * __restrict__ nst, const float * __restrict__ probs, float * __restrict__ out,
                          int S, int E) {
    const int qi = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    float a = 0.0f;
    for (int s = 0; s < S; ++s) a += probs[qi * S + s] * nst[s * E + i];
    out[qi * E + i] = a;
}

// loop control output: (1 - tanh(f)) * cur + tanh(g) * msg
__global__ void k_ctrl_out(const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ g,
                           const float * __restrict__ f, float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (1.0f - tanhf(f[i])) * cur[i] + tanhf(g[i]) * msg[i];
}

// Attention for one query position. window > 0: keys in [pos-window+1, pos] (SWA); window == 0: all keys.
// Relative bias uses row d = pos - j when d < 1024, else zero row. Requires pos < 1024 (smem scores).
__global__ void k_attn_decode_gen(const float * __restrict__ Q, const float * __restrict__ Kc, const float * __restrict__ Vc,
                                  const float * __restrict__ R, const float * __restrict__ relW, const float * __restrict__ sink,
                                  const int * __restrict__ pos_p, int window, float * __restrict__ O) {
    const int pos = *pos_p;
    __shared__ float q[128];
    __shared__ float r[16];
    __shared__ float sc[1024];
    __shared__ float denom;
    const int h = blockIdx.x, tid = threadIdx.x, kvh = h / 3;
    q[tid] = Q[h * 128 + tid];
    if (tid < 16) r[tid] = R[h * 16 + tid];
    __syncthreads();
    const int j0 = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int nk = pos - j0 + 1;
    const float inv_sqrt_d = 0.08838834764831845f;
    for (int j = j0 + tid; j <= pos; j += blockDim.x) {
        const int d = pos - j;
        float rb = 0.0f;
        if (d < 1024) {
            for (int k = 0; k < 16; ++k) rb += relW[d * 16 + k] * r[k];
        }
        const float bias = rb - sink[h];
        const float * kp = Kc + ((size_t) j * 8 + kvh) * 128;
        float dot = 0.0f;
        for (int dd = 0; dd < 128; ++dd) dot += q[dd] * kp[dd];
        sc[j - j0] = dot * inv_sqrt_d + bias;
    }
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (int i = 0; i < nk; ++i) m = fmaxf(m, sc[i]);
        float den = expf(-m);
        for (int i = 0; i < nk; ++i) { sc[i] = expf(sc[i] - m); den += sc[i]; }
        denom = den;
    }
    __syncthreads();
    float o = 0.0f;
    for (int i = 0; i < nk; ++i) o += sc[i] * Vc[((size_t) (j0 + i) * 8 + kvh) * 128 + tid];
    O[h * 128 + tid] = o / denom;
}

__global__ void k_softcap_cut(float * __restrict__ logits, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) logits[i] = cap * tanhf(logits[i] / cap);
}

// Q8_0 experts in SoA: qs int8 [E][N][K], sc half [E][N][K/32]. Two rows per warp, grid.y = slots.
__global__ void q8_experts_rows2(const int8_t * __restrict__ qs_all, const __half * __restrict__ sc_all,
                                 const float * __restrict__ x, int64_t x_slot_stride, float * __restrict__ y,
                                 int K, int N, const int * __restrict__ ids) {
    const int slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const int8_t * qs = qs_all + e * (int64_t) N * K;
    const __half * sc = sc_all + e * (int64_t) N * (K / 32);
    const float * xin = x + slot * x_slot_stride;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * 2;
    if (row0 >= N) return;
    const int nb = K / 32;
    const float4 * x4 = reinterpret_cast<const float4 *>(xin);
    float acc[2] = {0.0f, 0.0f};
    for (int b = lane; b < nb; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = __ldg(x4 + b * 8 + i);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
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
    for (int r = 0; r < 2; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[(size_t) slot * N + row0 + r] = a;
    }
}

inline void launch_q8_fused(const int8_t * qs_all, const __half * sc_all, const float * x, int64_t xs, float * y,
                            int K, int N, const int * ids, int slots, cudaStream_t s) {
    dim3 grid((N + 15) / 16, slots);
    q8_experts_rows2<<<grid, 256, 0, s>>>(qs_all, sc_all, x, xs, y, K, N, ids);
}


// ------------------------------------------- device-side position and token ----

// Token embedding with the token id read from device memory.
__global__ void k_embed_q8_dev(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const int * __restrict__ tok_p,
                               int E, float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const size_t tok = (size_t) (*tok_p);
    out[i] = (float) qs[tok * E + i] * __half2float(sc[tok * (E / 32) + i / 32]);
}

// Per-head clip writing into a cache row chosen by *pos_p. One block per head, row stride gridDim.x * dim.
__global__ void k_clip_heads_pos(const float * __restrict__ in, const float * __restrict__ scale, float * __restrict__ base,
                                 const int * __restrict__ pos_p, int dim, float thr) {
    __shared__ float red[32];
    const int h = blockIdx.x, d = threadIdx.x;
    float v = 0.0f;
    if (d < dim) {
        v = in[h * dim + d];
        if (scale) v *= scale[h * dim + d];
    }
    const float ss = block_sum(d < dim ? v * v : 0.0f, red);
    const float f = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
    if (d < dim) base[(size_t) (*pos_p) * gridDim.x * dim + h * dim + d] = v * f;
}

// RMS normalize per head into a cache row chosen by *pos_p (one block per head, 256 threads, dim <= 256).
__global__ void k_rms_heads_pos(const float * __restrict__ in, float * __restrict__ base, const int * __restrict__ pos_p,
                                int dim, float eps) {
    __shared__ float red[32];
    const int h = blockIdx.x, d = threadIdx.x;
    const float v = (d < dim) ? in[h * dim + d] : 0.0f;
    const float ss = block_sum(v * v, red);
    const float inv = rsqrtf(ss / dim + eps);
    if (d < dim) base[(size_t) (*pos_p) * gridDim.x * dim + h * dim + d] = v * inv;
}

// Scale by tau(pos) = 1 + alpha * log(max(1, (pos + 1) / floor)), pos from device.
__global__ void k_scale_tau_pos(float * __restrict__ v, int n, const int * __restrict__ pos_p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float tau = 1.0f + 0.10000000149011612f * logf(fmaxf(1.0f, ((float) (*pos_p) + 1.0f) / 8192.0f));
    if (i < n) v[i] *= tau;
}

// Argmax over n logits (first index on ties); single block of 1024 threads, writes out[0].
__global__ void k_argmax(const float * __restrict__ logits, int n, int * __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (logits[i] > v) { v = logits[i]; idx = i; }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, off);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bvv = bv[0];
        int bii = bi[0];
        for (int w = 1; w < (blockDim.x + 31) / 32; ++w)
            if (bv[w] > bvv || (bv[w] == bvv && bi[w] < bii)) { bvv = bv[w]; bii = bi[w]; }
        out[0] = bii;
    }
}

// pos += 1 on device.
__global__ void k_pos_inc(int * __restrict__ pos_p) { if (threadIdx.x == 0 && blockIdx.x == 0) *pos_p += 1; }

// Fused Mix over S states (contiguous [S][E], states[S-1] = current) and nq queries. One block (256 threads) per query.
// Each state is RMS-normalized by its inverse rms; scores = <state, q> * inv / sqrt(E); softmax over S;
// out = sum_s p_s * inv_s * state_s. No intermediate normalized copies are written.
__global__ void k_mix_fused(const float * __restrict__ states, int S, const float * __restrict__ query,
                            float * __restrict__ out, int E) {
    __shared__ float red[32];
    __shared__ float pw[16];
    __shared__ float inv_s[16];
    const int qi = blockIdx.x;
    const float * q = query + (size_t) qi * E;
    for (int s = 0; s < S; ++s) {
        const float * st = states + (size_t) s * E;
        float ss = 0.0f, dot = 0.0f;
        for (int i = threadIdx.x; i < E; i += blockDim.x) {
            const float v = st[i];
            ss += v * v;
            dot += v * q[i];
        }
        ss = block_sum(ss, red);
        dot = block_sum(dot, red);
        if (threadIdx.x == 0) {
            inv_s[s] = rsqrtf(ss / E + 1e-6f);
            pw[s] = dot * inv_s[s] / sqrtf((float) E);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = pw[0];
        for (int s = 1; s < S; ++s) m = fmaxf(m, pw[s]);
        float z = 0.0f;
        for (int s = 0; s < S; ++s) { pw[s] = expf(pw[s] - m); z += pw[s]; }
        for (int s = 0; s < S; ++s) pw[s] = pw[s] / z * inv_s[s];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < S; ++s) a += pw[s] * states[(size_t) s * E + i];
        out[(size_t) qi * E + i] = a;
    }
}

// Router: logits = W x (128 experts, K = E), sigmoid, top-8 on probs + bias, weights normalized and scaled.
// One block of 256 threads: warp w computes experts w, w+8, ... (lanes stride K). logits are also written
// to out_logits for validation dumps.
__global__ void k_router_fused(const float * __restrict__ W, const float * __restrict__ x, const float * __restrict__ bias,
                               float * __restrict__ out_logits, int * __restrict__ ids, float * __restrict__ wts,
                               int K, float scale) {
    __shared__ float probs[128];
    __shared__ float sel[128];
    __shared__ float wv[8];
    __shared__ int wi[8];
    __shared__ int chosen[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = warp; e < 128; e += 8) {
        const float * row = W + (size_t) e * K;
        float acc = 0.0f;
        for (int k = lane; k < K; k += 32) acc += row[k] * x[k];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            out_logits[e] = acc;
            const float pr = 1.0f / (1.0f + expf(-acc));
            probs[e] = pr;
            sel[e] = pr + bias[e];
        }
    }
    __syncthreads();
    for (int k = 0; k < 8; ++k) {
        // each warp finds its best unchosen expert among 16 candidates (e = warp*16 + j)
        float best = -INFINITY;
        int bi = 0;
        if (lane < 16) {
            const int e = warp * 16 + lane;
            bool used = false;
            for (int u = 0; u < k; ++u) used |= (chosen[u] == e);
            if (!used) { best = sel[e]; bi = e; }
        }
        for (int off = 8; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) { wv[warp] = best; wi[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = wv[0];
            int bb = wi[0];
            for (int w = 1; w < 8; ++w)
                if (wv[w] > bv || (wv[w] == bv && wi[w] < bb)) { bv = wv[w]; bb = wi[w]; }
            chosen[k] = bb;
            ids[k] = bb;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k) sum += probs[chosen[k]];
        sum += 1e-20f;
        for (int k = 0; k < 8; ++k) wts[k] = probs[chosen[k]] / sum * scale;
    }
}
