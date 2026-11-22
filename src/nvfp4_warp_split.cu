// NVFP4 expert GEMV: warp work decomposition study.
// Production = nvfp4_gemv_fused_t128 (1 warp x 4 rows, each lane walks 64-element blocks, 128 regs).
// Here: each lane owns one 16-byte weight chunk (32 elements = 2 sub-blocks, 128-bit coalesced load).
//   R  = rows per warp           (1, 2, 4)
//   KS = warps per row (K split)  (1, 2, 4, 8). Warps of one row cover contiguous chunk ranges and
//        combine partial sums via shared memory (KS > 1). KS == 1 reduces with warp shuffles only.
// Math per element is identical to production (same LUT, same scale rule); only the summation order differs.
#include "kernels_common.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// LM: 0 = 256-entry byte LUT (lut2) + scale LUT; 1 = 16-entry nibble LUT (conflict-free banks) + scale LUT;
//     2 = nibble LUT + arithmetic UE4M3 decode (no scale LDS).
__device__ __forceinline__ float ue4m3_arith(uint32_t x) {
    // exact: (1 + m/8) * 2^(e-8) for e>0, m * 2^-10 for e==0 (both = ue4m3_dev(x), the value * 0.5)
    const uint32_t e = (x >> 3) & 0xF, m = x & 0x7;
    const float norm = __uint_as_float((e + 119u) << 23 | (m << 20));
    const float sub = (float) m * (1.0f / 1024.0f);
    const float v = e ? norm : sub;
    return (x == 0x7F) ? 0.0f : v;
}

template <int R, int KS, int WARPS, int MB = 1, int LM = 0>
__global__ void __launch_bounds__(WARPS * 32, MB)
nv_chunk(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all,
         const float * __restrict__ x, int64_t x_slot_stride, float * __restrict__ y,
         int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[];
    __shared__ float lut_ue[256];
    __shared__ float2 lut2[256];
    __shared__ float kt[16];
    __shared__ float part[WARPS][R];
    constexpr int T = WARPS * 32, G = WARPS / KS, RPB = G * R;
    const int nch = K / 32, slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    const float * xin = x + slot * x_slot_stride;
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    if constexpr (LM == 0) {
        for (int i = threadIdx.x; i < 256; i += T) { lut_ue[i] = ue4m3_dev((uint8_t) i); lut2[i] = make_float2(kE2M1d[i & 15], kE2M1d[i >> 4]); }
    } else {
        if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
        if (LM == 1) for (int i = threadIdx.x; i < 256; i += T) lut_ue[i] = ue4m3_dev((uint8_t) i);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = warp / KS, kp = warp % KS;
    const int rowbase = blockIdx.x * RPB + g * R;
    const int cpk = (nch + KS - 1) / KS;
    const int c0 = kp * cpk, c1 = min(nch, c0 + cpk);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int c = c0 + lane; c < c1; c += 32) {
        const int b = c >> 1, half = c & 1;
        const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68 + half * 32);
        float4 xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = xb[i];
        const float * xf = reinterpret_cast<const float *>(xv);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = rowbase + r;
            if (row < N) {
                const uint4 a = __ldg(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c);
                const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
                const uint32_t w[4] = {a.x, a.y, a.z, a.w};
                float tot = 0.0f;
#pragma unroll
                for (int h = 0; h < 2; ++h) {          // the two 16-element sub-blocks of this chunk
                    const int s2 = 2 * half + h;
                    const uint32_t sb = (sw >> (8 * s2)) & 0xFF;
                    const float d = (LM == 2) ? ue4m3_arith(sb) : lut_ue[sb];
                    float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = h * 8 + j;      // byte within chunk
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        if constexpr (LM == 0) {
                            const float2 v = lut2[byte];
                            sub += v.x * xf[h * 16 + j] + v.y * xf[h * 16 + j + 8];
                        } else {
                            sub += kt[byte & 15] * xf[h * 16 + j] + kt[byte >> 4] * xf[h * 16 + j + 8];
                        }
                    }
                    tot += d * sub;
                }
                acc[r] += tot;
            }
        }
    }
    if constexpr (KS == 1) {
        // warp-internal reduction only, direct store by lane r
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float a = acc[r];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0 && rowbase + r < N) y[(size_t) slot * N + rowbase + r] = a;
        }
    } else {
        // warp reduction, then cross-warp sum through shared memory
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float a = acc[r];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0) part[warp][r] = a;
        }
        __syncthreads();
        if (kp == 0 && lane < R) {
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < KS; ++k) s += part[g * KS + k][lane];
            const int row = rowbase + lane;
            if (row < N) y[(size_t) slot * N + row] = s;
        }
    }
}

// Row-packed warp decomposition: one warp owns RP rows and iterates over RP*NCH (row, 16-byte chunk) units
// spread across its 32 lanes (unit u = lane + 32 t). Consecutive lanes read consecutive chunks of one row,
// so loads stay 128-bit and coalesced. Lane utilization = RP*NCH / (32*ceil(RP*NCH/32)).
// NCH = K/32 chunks per row. Uses the LM=2 decode (nibble LUT + arithmetic UE4M3). Single warp per row group.
// doubled E2M1 (kE2M1d) decoded arithmetically: mag = e ? 2^e * (1 + m/2) : m, sign from bit 3. Exact.
__device__ __forceinline__ float e2m1d_arith(uint32_t c) {
    const uint32_t e = (c >> 1) & 3, m = c & 1;
    const float mag = e ? __uint_as_float(((e + 127u) << 23) | (m << 22)) : (float) m;
    return (c & 8) ? -mag : mag;
}

template <int RP, int NCH, int LX = 2>   // LX: 2 = nibble LUT in smem, 3 = arithmetic E2M1 (no LDS)
__global__ void __launch_bounds__(128)
nv_pack(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all,
        const float * __restrict__ x, int64_t x_slot_stride, float * __restrict__ y,
        int K, int N, const int * __restrict__ ids) {
    extern __shared__ float xs[];
    __shared__ float kt[16];
    constexpr int T = 128, NB = NCH / 2, UNITS = RP * NCH;
    const int slot = blockIdx.y;
    const int64_t e = ids ? ids[slot] : 0;
    const uint8_t * qs = qs_all + e * (int64_t) N * (K / 2);
    const uint8_t * sc = sc_all + e * (int64_t) N * (K / 16);
    const float * xin = x + slot * x_slot_stride;
    for (int i = threadIdx.x; i < K; i += T) { const int b = i >> 6, r = i & 63; xs[b * 68 + r] = xin[i]; }
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rowbase = (blockIdx.x * 4 + warp) * RP;
    if (rowbase >= N) return;
    float acc[RP];
#pragma unroll
    for (int r = 0; r < RP; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int u0 = 0; u0 < UNITS; u0 += 32) {
        const int u = u0 + lane;
        if (u < UNITS) {
            const int rs = u / NCH, c = u - rs * NCH;
            const int row = rowbase + rs;
            const int b = c >> 1, half = c & 1;
            float tot = 0.0f;
            if (row < N) {
                const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 68 + half * 32);
                float4 xv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = xb[i];
                const float * xf = reinterpret_cast<const float *>(xv);
                const uint4 a = __ldg(reinterpret_cast<const uint4 *>(qs + (size_t) row * (K / 2)) + c);
                const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(sc + (size_t) row * (K / 16)) + b);
                const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t sb = (sw >> (8 * (2 * half + h))) & 0xFF;
                    const float d = ue4m3_arith(sb);
                    float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = h * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        const float vl = (LX == 2) ? kt[byte & 15] : e2m1d_arith(byte & 15);
                        const float vh = (LX == 2) ? kt[byte >> 4] : e2m1d_arith(byte >> 4);
                        sub += vl * xf[h * 16 + j] + vh * xf[h * 16 + j + 8];
                    }
                    tot += d * sub;
                }
            }
#pragma unroll
            for (int r = 0; r < RP; ++r) acc[r] += (rs == r) ? tot : 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < RP; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && rowbase + r < N) y[(size_t) slot * N + rowbase + r] = a;
    }
}

template <typename F> static float graph_us(F f, cudaStream_t s, int R_ = 100) {
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < R_; ++i) f(s);
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s); for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.f / (3.f * R_);
}

struct Variant {
    const char * name;
    void (*launch)(const uint8_t *, const uint8_t *, const float *, int64_t, float *, int, int, const int *, cudaStream_t);
    const void * fn;
    int threads;
    int rpb;   // rows per block
};

#define LAUNCH_CHUNK(R_, KS_, W_, MB_, LM_)                                                                   \
    static void l_##R_##_##KS_##_##W_##_##MB_##_##LM_(const uint8_t * q, const uint8_t * s_, const float * x, int64_t xs_, \
                                       float * y, int K, int N, const int * ids, cudaStream_t st) {          \
        const size_t smem = (size_t) (K / 64) * 68 * sizeof(float);                                          \
        constexpr int RPB = (W_ / KS_) * R_;                                                                  \
        dim3 grid((N + RPB - 1) / RPB, 8);                                                                    \
        nv_chunk<R_, KS_, W_, MB_, LM_><<<grid, W_ * 32, smem, st>>>(q, s_, x, xs_, y, K, N, ids);            \
    }                                                                                                         \
    static const void * f_##R_##_##KS_##_##W_##_##MB_##_##LM_ = (const void *) nv_chunk<R_, KS_, W_, MB_, LM_>;


LAUNCH_CHUNK(4, 1, 4, 1, 0)
LAUNCH_CHUNK(4, 1, 4, 1, 1)
LAUNCH_CHUNK(4, 1, 4, 1, 2)
LAUNCH_CHUNK(2, 1, 4, 1, 2)
LAUNCH_CHUNK(1, 1, 4, 1, 2)
LAUNCH_CHUNK(1, 2, 4, 1, 2)
LAUNCH_CHUNK(1, 8, 8, 1, 2)
LAUNCH_CHUNK(4, 1, 4, 8, 0)

#define LAUNCH_PACK(RP_, NCH_, LX_)                                                                           \
    static void lp_##RP_##_##NCH_##_##LX_(const uint8_t * q, const uint8_t * s_, const float * x, int64_t xs_, \
                                  float * y, int K, int N, const int * ids, cudaStream_t st) {          \
        const size_t smem = (size_t) (K / 64) * 68 * sizeof(float);                                          \
        dim3 grid((N + 4 * RP_ - 1) / (4 * RP_), 8);                                                          \
        nv_pack<RP_, NCH_, LX_><<<grid, 128, smem, st>>>(q, s_, x, xs_, y, K, N, ids);                        \
    }                                                                                                         \
    static const void * fp_##RP_##_##NCH_##_##LX_ = (const void *) nv_pack<RP_, NCH_, LX_>;

LAUNCH_PACK(2, 72, 2)
LAUNCH_PACK(4, 72, 2)
LAUNCH_PACK(4, 72, 3)
LAUNCH_PACK(2, 36, 2)
LAUNCH_PACK(4, 36, 2)
LAUNCH_PACK(4, 36, 3)
LAUNCH_PACK(8, 36, 2)

static void l_prod(const uint8_t * q, const uint8_t * s_, const float * x, int64_t xs_, float * y, int K, int N,
                   const int * ids, cudaStream_t st) {
    launch_nvfp4_fused(q, s_, x, xs_, y, K, N, ids, 8, st);
}

int main(int argc, char ** argv) {
    cudaStream_t s; cudaStreamCreate(&s);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    std::mt19937 rng(3); std::uniform_int_distribution<int> qd(0, 255);
    struct Cfg { const char * name; int K, N; int xstride; } cfgs[] = {{"gate_up", 2304, 2304, 0}, {"down", 1152, 2304, 1152}};
    struct V { const char * name; void (*fn)(const uint8_t *, const uint8_t *, const float *, int64_t, float *, int, int, const int *, cudaStream_t); const void * kern; int thr; int nch; };
    V vars[] = {
        {"prod 1w x4r (baseline, 128thr)", l_prod, (const void *) nvfp4_gemv_fused_t128, 128, 0},
        {"chunk 1w x4r LM2 (KS1 chunk)", l_4_1_4_1_2, f_4_1_4_1_2, 128, 0},
        {"chunk 8w x1r LM2 (KS8 smem)", l_1_8_8_1_2, f_1_8_8_1_2, 256, 0},
        {"pack 2 rows/warp LUT (72ch)", lp_2_72_2, fp_2_72_2, 128, 72},
        {"pack 4 rows/warp LUT (72ch)", lp_4_72_2, fp_4_72_2, 128, 72},
        {"pack 4 rows/warp ARITH (72ch)", lp_4_72_3, fp_4_72_3, 128, 72},
        {"pack 2 rows/warp LUT (36ch)", lp_2_36_2, fp_2_36_2, 128, 36},
        {"pack 4 rows/warp LUT (36ch)", lp_4_36_2, fp_4_36_2, 128, 36},
        {"pack 4 rows/warp ARITH (36ch)", lp_4_36_3, fp_4_36_3, 128, 36},
        {"pack 8 rows/warp LUT (36ch)", lp_8_36_2, fp_8_36_2, 128, 36},
    };



    const int only = argc > 1 ? atoi(argv[1]) : -1;
    const bool prof = getenv("NV_PROF") != nullptr;   // ncu mode: one direct launch per variant, no graph replays
    for (auto & c : cfgs) {
        const int NE = 128, slots = 8;
        const size_t qbytes = (size_t) NE * c.N * (c.K / 2), sbytes = (size_t) NE * c.N * (c.K / 16);
        std::vector<uint8_t> qs(qbytes), sc(sbytes);
        for (auto & v : qs) v = qd(rng);
        for (auto & v : sc) v = 0x38 + (qd(rng) & 7);
        std::vector<float> x((size_t) slots * (c.xstride ? c.xstride : c.K));
        std::uniform_real_distribution<float> ud(-1.f, 1.f);
        for (auto & v : x) v = ud(rng);
        std::vector<int> ids = {3, 17, 42, 60, 81, 99, 101, 127};
        uint8_t *dq, *ds; float *dx, *dref, *dy; int *di;
        cudaMalloc(&dq, qbytes); cudaMalloc(&ds, sbytes); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dref, 8 * c.N * 4);
        cudaMalloc(&dy, 8 * c.N * 4); cudaMalloc(&di, 32);
        cudaMemcpy(dq, qs.data(), qbytes, cudaMemcpyHostToDevice); cudaMemcpy(ds, sc.data(), sbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, ids.data(), 32, cudaMemcpyHostToDevice);
        launch_nvfp4_fused(dq, ds, dx, c.xstride, dref, c.K, c.N, di, 8, s); cudaStreamSynchronize(s);
        std::vector<float> ref(8 * c.N);
        cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost);
        double refmax = 0; for (float v : ref) refmax = std::max(refmax, (double) std::fabs(v));
        std::printf("== %s (K=%d N=%d)  ref max|y| %.3f\n", c.name, c.K, c.N, refmax);
        std::printf("   %-30s %9s %5s %7s %9s %10s %8s\n", "variant", "us", "regs", "spill", "occ(blk/SM)", "max|diff|", "rel");
        for (int vi = 0; vi < (int) (sizeof(vars) / sizeof(vars[0])); ++vi) {
            if (only >= 0 && vi != only) continue;
            if (vars[vi].nch && vars[vi].nch != c.K / 32) continue;   // pack kernels are specialized to one K
            auto & v = vars[vi];
            int regs = 0, spill = 0, occ = 0;
            if (v.kern) {
                cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, v.kern);
                regs = fa.numRegs; spill = (int) fa.localSizeBytes;
                const size_t smem = (size_t) (c.K / 64) * 68 * sizeof(float);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, v.kern, v.thr, smem);
            }
            cudaMemset(dy, 0, 8 * c.N * 4);
            v.fn(dq, ds, dx, c.xstride, dy, c.K, c.N, di, s); cudaStreamSynchronize(s);
            std::vector<float> h(8 * c.N);
            cudaMemcpy(h.data(), dy, h.size() * 4, cudaMemcpyDeviceToHost);
            double md = 0; for (size_t i = 0; i < h.size(); ++i) md = std::max(md, (double) std::fabs(h[i] - ref[i]));
            const float t = prof ? 0.f : graph_us([&](cudaStream_t st) { v.fn(dq, ds, dx, c.xstride, dy, c.K, c.N, di, st); }, s);
            std::printf("   %-30s %9.2f %5d %7d %11d %10.1e %8.1e\n", v.name, t, regs, spill, occ, md, md / refmax);
        }
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dref); cudaFree(dy); cudaFree(di);
    }
    (void) sms;
    return 0;
}
