#include <type_traits>
// Tensor-core GEMMs for the prefill path. Weights (Q8_0 or NVFP4) stay in the interleaved layout of the decode kernels
// (see batch_kernels2.cuh): the weight rows are the M operand (A) of mma.m16n8k16, the tokens the N operand (B).
//   Q8_0 : qsT[g][blk][h][lane] uint4 = bytes 16h..16h+15 of 32-k block blk of row 32g+lane ; scT[g][blk][lane] half
//   NVFP4: qsT[g][b][h][lane]   uint4 = bytes 16h..16h+15 of 64-k block b   of row 32g+lane ; scT[g][b][lane] u32 (4 ue4m3)
//          byte i of a 16-k sub-block: low nibble = k i, high nibble = k i+8 (E2M1 doubled table, ue4m3 scale incl. 0.5)
// Every thread loads 4-byte words of the weights straight from global memory (lane = (row r = lane/4, quarter q = lane%4)),
// converts them to FP16 in registers (Q8_0: exact integers, scale applied to the FP32 partial sum per 32-k block; NVFP4:
// exact products value*scale in FP16) and feeds the tensor core. The k order inside an mma step is permuted consistently
// for weights and activations (sum over k is order independent), see pf_kidx comments.
// Activations are FP16 [rows][K], staged by cp.async into a swizzled shared-memory tile (64 k per stage, 3 stages).
#pragma once
#include "kernels_common.cuh"
#include <cuda_fp16.h>
#include <cstdint>

#define PF_DEV __device__ __forceinline__

constexpr int PF_BK = 64;       // k per pipeline stage
constexpr int PF_NST = 3;       // cp.async stages
constexpr int PF_WARPS = 4;     // warps per block (32 weight rows each -> 128 rows per block)

enum PFFmt { PF_Q8 = 0, PF_NV4 = 1 };

PF_DEV void pf_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
PF_DEV void pf_cp16(void * dst, const void * src, bool valid) {
    const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(sz));
}
PF_DEV void pf_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> PF_DEV void pf_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
PF_DEV uint2 pf_lds64(const void * p) {
    uint2 r;
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "r"(a));
    return r;
}
PF_DEV uint32_t pf_ldg32(const uint32_t * p) { return __ldg(p); }

// 4 int8 -> 2 x half2 (bytes 0,1) / (2,3), exact
PF_DEV void pf_q8_to_h2(uint32_t w, uint32_t & lo, uint32_t & hi) {
    const uint32_t u = w ^ 0x80808080u;
    uint32_t a = __byte_perm(u, 0x64646464u, 0x5140), b = __byte_perm(u, 0x64646464u, 0x5342);
    const uint32_t m = 0x64806480u;   // 1152
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(lo) : "r"(a), "r"(m));
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(hi) : "r"(b), "r"(m));
}
// 8 E2M1 nibbles (word w: byte i = {lo: k i, hi: k i+8}) scaled by half2 s2 -> low-nibble regs (l01: bytes 0,1; l23: bytes 2,3) and
// high-nibble regs (h01, h23). Values are the doubled E2M1 table {0,1,2,3,4,6,8,12} times scale.
PF_DEV void pf_nv4_to_h2(uint32_t w, uint32_t s2, uint32_t & l01, uint32_t & l23, uint32_t & h01, uint32_t & h23) {
    const uint32_t LUT_LO = 0x42403C00u, LUT_HI = 0x4A484644u;
    const uint32_t slo = (w << 4) & 0x80808080u, shi = w & 0x80808080u;
    uint32_t t;
    t = __byte_perm(LUT_LO, LUT_HI, (w << 4) & 0x7070u) | __byte_perm(slo, 0u, 0x1404);
    asm("mul.f16x2 %0, %1, %2;\n" : "=r"(l01) : "r"(t), "r"(s2));
    t = __byte_perm(LUT_LO, LUT_HI, (w >> 12) & 0x7070u) | __byte_perm(slo, 0u, 0x3424);
    asm("mul.f16x2 %0, %1, %2;\n" : "=r"(l23) : "r"(t), "r"(s2));
    t = __byte_perm(LUT_LO, LUT_HI, w & 0x7070u) | __byte_perm(shi, 0u, 0x1404);
    asm("mul.f16x2 %0, %1, %2;\n" : "=r"(h01) : "r"(t), "r"(s2));
    t = __byte_perm(LUT_LO, LUT_HI, (w >> 16) & 0x7070u) | __byte_perm(shi, 0u, 0x3424);
    asm("mul.f16x2 %0, %1, %2;\n" : "=r"(h23) : "r"(t), "r"(s2));
}
// ue4m3 byte (value incl. the 0.5 factor) -> half2 {s, s}; exact in FP16
PF_DEV uint32_t pf_ue4m3_h2(uint32_t x) {
    const __half h = __float2half_rn(ue4m3_arith(x));
    const uint32_t b = __half_as_ushort(h);
    return b | (b << 16);
}

// swizzle of the 16-byte units of a 128-byte activation row (row n of the tile)
template <int FMT> PF_DEV int pf_swz(int n) { return FMT == PF_Q8 ? 2 * (n & 3) : ((n & 1) | (((n >> 1) & 1) << 2)); }

// ---------------------------------------------------------------------------------------------------------------
// One tile: TB tokens (activation rows given by xrow[]), 128 weight rows (this block), full K.
// wq/wsc: weight pointers for this warp's 32-row group G (word pointers already offset to the group), nb = number of
// weight blocks over K (Q8: K/32, NV4: K/64). out(row, tok, v) is called for each result.
// ---------------------------------------------------------------------------------------------------------------
template <int FMT, int TB>
struct PFTile {
    static constexpr int NT = TB / 8;
    float acc[2][NT][4];
};

template <int FMT, int TB, class OutFn>
PF_DEV void pf_tile(const uint32_t * __restrict__ wq, const void * __restrict__ wsc, int nb, int K, const __half * const (&xrow)[TB / 16],
                    int cnt, __half * smem, OutFn out) {
    constexpr int NT = TB / 8;
    constexpr int LD = TB * PF_BK;                       // halves per stage
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r = lane >> 2, q = lane & 3;
    const int nks = K / PF_BK;
    // activation staging: thread handles (row n, unit u): idx = tid + 128*i  ->  n = (tid>>3) + 16 i, u = tid & 7
    const int su = tid & 7;
    const int sn0 = tid >> 3;
    auto issue = [&](int ks, int buf) {
#pragma unroll
        for (int i = 0; i < TB / 16; ++i) {
            const int n = sn0 + 16 * i;
            __half * dst = smem + buf * LD + n * PF_BK + ((su ^ pf_swz<FMT>(n)) << 3);
            const __half * src = xrow[i] ? xrow[i] + ks * PF_BK + su * 8 : (const __half *) smem;
            pf_cp16(dst, src, xrow[i] != nullptr);
        }
    };
    // weight prefetch registers for one stage: [mt][rowhalf][word], scales
    struct WS { uint32_t w[2][2][4]; uint32_t s[2][2][2]; };
    auto loadw = [&](int ks, WS & ws) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                const int rl = 16 * mt + 8 * rh + r;
                if (FMT == PF_Q8) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const size_t b = (size_t) 2 * ks + j;
#pragma unroll
                        for (int h = 0; h < 2; ++h) ws.w[mt][rh][2 * j + h] = pf_ldg32(wq + (b * 2 + h) * 128 + rl * 4 + q);
                        ws.s[mt][rh][j] = __ldg(reinterpret_cast<const unsigned short *>(wsc) + b * 32 + rl);
                    }
                } else {
                    const size_t b = ks;
#pragma unroll
                    for (int h = 0; h < 2; ++h) ws.w[mt][rh][h] = pf_ldg32(wq + (b * 2 + h) * 128 + rl * 4 + q);
                    ws.s[mt][rh][0] = pf_ldg32(reinterpret_cast<const uint32_t *>(wsc) + b * 32 + rl);
                }
            }
    };
    float acc[2][NT][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NT; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.0f;
    const float zero4[4] = {0, 0, 0, 0};
    WS wcur, wnxt;
#pragma unroll
    for (int s = 0; s < PF_NST - 1; ++s) { if (s < nks) issue(s, s); pf_commit(); }
    loadw(0, wcur);
    const int ntact = (cnt + 7) >> 3;                       // active n8 tiles (uniform in the block)
    for (int ks = 0; ks < nks; ++ks) {
        pf_wait<PF_NST - 2>();
        __syncthreads();
        if (ks + PF_NST - 1 < nks) issue(ks + PF_NST - 1, (ks + PF_NST - 1) % PF_NST);
        pf_commit();
        if (ks + 1 < nks) loadw(ks + 1, wnxt);
        const __half * tile = smem + (ks % PF_NST) * LD;
        if (FMT == PF_Q8) {
            // two 32-k blocks j, two steps h each; step k offset = 32 j + 16 h + 4 q
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float tmp[2][NT][4];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t a[2][4];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        uint32_t r0l, r0h, r1l, r1h;
                        pf_q8_to_h2(wcur.w[mt][0][2 * j + h], r0l, r0h);   // row r   : bytes (0,1) (2,3)
                        pf_q8_to_h2(wcur.w[mt][1][2 * j + h], r1l, r1h);   // row r+8
                        a[mt][0] = r0l; a[mt][1] = r1l; a[mt][2] = r0h; a[mt][3] = r1h;
                    }
                    const int unit = 4 * j + 2 * h + (q >> 1);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if (nt < ntact) {
                            const int n = nt * 8 + r;
                            const uint2 bb = pf_lds64(tile + n * PF_BK + ((unit ^ pf_swz<FMT>(n)) << 3) + 4 * (q & 1));
#pragma unroll
                            for (int mt = 0; mt < 2; ++mt) {
                                if (h == 0) pf_mma(tmp[mt][nt], a[mt], bb.x, bb.y, zero4);
                                else        pf_mma(tmp[mt][nt], a[mt], bb.x, bb.y, tmp[mt][nt]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const float d0 = __half2float(__ushort_as_half((unsigned short) wcur.s[mt][0][j]));
                    const float d1 = __half2float(__ushort_as_half((unsigned short) wcur.s[mt][1][j]));
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if (nt < ntact) {
                            acc[mt][nt][0] += d0 * tmp[mt][nt][0]; acc[mt][nt][1] += d0 * tmp[mt][nt][1];
                            acc[mt][nt][2] += d1 * tmp[mt][nt][2]; acc[mt][nt][3] += d1 * tmp[mt][nt][3];
                        }
                    }
                }
            }
        } else {
            // 64-k block: word h (h = 0,1) holds sub-blocks 2h + (q>>1); step (h, lohi): k offset = 32 h + 16 (q>>1) + 4 (q&1) + 8 lohi
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t L[2][4], H[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int sb = 2 * h + (q >> 1);
                    uint32_t s2[2];
#pragma unroll
                    for (int rh = 0; rh < 2; ++rh) s2[rh] = pf_ue4m3_h2((wcur.s[mt][rh][0] >> (8 * sb)) & 0xFF);
                    uint32_t l01, l23, h01, h23, m01, m23, n01, n23;
                    pf_nv4_to_h2(wcur.w[mt][0][h], s2[0], l01, l23, h01, h23);   // row r
                    pf_nv4_to_h2(wcur.w[mt][1][h], s2[1], m01, m23, n01, n23);   // row r+8
                    L[mt][0] = l01; L[mt][1] = m01; L[mt][2] = l23; L[mt][3] = m23;
                    H[mt][0] = h01; H[mt][1] = n01; H[mt][2] = h23; H[mt][3] = n23;
                }
#pragma unroll
                for (int lohi = 0; lohi < 2; ++lohi) {
                    const int unit = 4 * h + lohi + 2 * (q >> 1);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if (nt < ntact) {
                            const int n = nt * 8 + r;
                            const uint2 bb = pf_lds64(tile + n * PF_BK + ((unit ^ pf_swz<FMT>(n)) << 3) + 4 * (q & 1));
#pragma unroll
                            for (int mt = 0; mt < 2; ++mt) pf_mma(acc[mt][nt], lohi ? H[mt] : L[mt], bb.x, bb.y, acc[mt][nt]);
                        }
                    }
                }
            }
        }
        if (ks + 1 < nks) wcur = wnxt;
    }
    pf_wait<0>();
    // results: c0,c1 = (row r, tokens 2q, 2q+1), c2,c3 = (row r+8, ...)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            if (nt < ntact) {
                const int t0 = nt * 8 + 2 * q;
                const int rowA = 16 * mt + r;
                if (t0 < cnt)     { out(rowA, t0, acc[mt][nt][0]); out(rowA + 8, t0, acc[mt][nt][2]); }
                if (t0 + 1 < cnt) { out(rowA, t0 + 1, acc[mt][nt][1]); out(rowA + 8, t0 + 1, acc[mt][nt][3]); }
            }
        }
}

// ------------------------------------------------------------------ dense multi-segment GEMM ----
struct PFSeg {
    const uint32_t * qs; const void * sc;   // interleaved weights of this matrix
    int rb0;                                 // first 128-row block of this segment in blockIdx.x
    int N;                                   // rows (multiple of 128)
    float * out; int ostride;                // out[tok * ostride + row]
    int xoff;                                // column offset of the activation row (halves)
};
struct PFDense { PFSeg seg[6]; int nseg; const __half * x; int xstride; int T; int K; };

template <int FMT, int TB>
__global__ void __launch_bounds__(128) pf_gemm_dense(const PFDense p) {
    extern __shared__ __align__(16) unsigned char pf_smem_raw[];
    __half * smem = reinterpret_cast<__half *>(pf_smem_raw);
    int si = 0;
    while (si + 1 < p.nseg && (int) blockIdx.x >= p.seg[si + 1].rb0) ++si;
    const PFSeg sg = p.seg[si];
    const int rb = blockIdx.x - sg.rb0;
    const int warp = threadIdx.x >> 5;
    const int tok0 = blockIdx.y * TB;
    const int cnt = min(TB, p.T - tok0);
    const int sn0 = threadIdx.x >> 3;
    const __half * xrow[TB / 16];
#pragma unroll
    for (int i = 0; i < TB / 16; ++i) {
        const int n = sn0 + 16 * i;
        xrow[i] = (n < cnt) ? p.x + (size_t) (tok0 + n) * p.xstride + sg.xoff : nullptr;
    }
    const int nb = FMT == PF_Q8 ? p.K / 32 : p.K / 64;
    const size_t G = (size_t) rb * PF_WARPS + warp;
    const uint32_t * wq = sg.qs + G * nb * 2 * 128;
    const void * wsc = FMT == PF_Q8 ? (const void *) ((const unsigned short *) sg.sc + G * nb * 32)
                                     : (const void *) ((const uint32_t *) sg.sc + G * nb * 32);
    const int rowbase = rb * 128 + warp * 32;
    pf_tile<FMT, TB>(wq, wsc, nb, p.K, xrow, cnt, smem, [&](int rl, int t, float v) {
        sg.out[(size_t) (tok0 + t) * sg.ostride + rowbase + rl] = v;
    });
}

// ------------------------------------------------------------------ grouped MoE GEMM ----
// tiles: expert tile_e[i], first sorted slot tile_p[i], count tile_c[i] (<= TB). rowmap[p] = activation row of sorted slot p
// (null: identity). y[p][N].
template <int FMT, int TB>
__global__ void __launch_bounds__(128) pf_gemm_moe(const uint32_t * __restrict__ qs, const void * __restrict__ sc, int N, int K,
                                                   const __half * __restrict__ x, int xstride, const int * __restrict__ rowmap,
                                                   float * __restrict__ y, const int * __restrict__ tile_e, const int * __restrict__ tile_p,
                                                   const int * __restrict__ tile_c, const int * __restrict__ ntiles) {
    extern __shared__ __align__(16) unsigned char pf_smem_raw[];
    __half * smem = reinterpret_cast<__half *>(pf_smem_raw);
    const int ti = blockIdx.y;
    if (ti >= *ntiles) return;
    const int e = tile_e[ti], p0 = tile_p[ti], cnt = tile_c[ti];
    const int warp = threadIdx.x >> 5;
    const int sn0 = threadIdx.x >> 3;
    const __half * xrow[TB / 16];
#pragma unroll
    for (int i = 0; i < TB / 16; ++i) {
        const int n = sn0 + 16 * i;
        xrow[i] = (n < cnt) ? x + (size_t) (rowmap ? rowmap[p0 + n] : p0 + n) * xstride : nullptr;
    }
    const int nb = FMT == PF_Q8 ? K / 32 : K / 64;
    const size_t G = (size_t) e * (N / 32) + (size_t) blockIdx.x * PF_WARPS + warp;
    const uint32_t * wq = qs + G * nb * 2 * 128;
    const void * wsc = FMT == PF_Q8 ? (const void *) ((const unsigned short *) sc + G * nb * 32)
                                     : (const void *) ((const uint32_t *) sc + G * nb * 32);
    const int rowbase = blockIdx.x * 128 + warp * 32;
    pf_tile<FMT, TB>(wq, wsc, nb, K, xrow, cnt, smem, [&](int rl, int t, float v) {
        y[(size_t) (p0 + t) * N + rowbase + rl] = v;
    });
}

// ------------------------------------------------------------------ MoE grouping ----
// One block of 1024 threads. ids [T*8] (expert per assignment). Outputs: perm[p] = assignment index (token*8+slot) of sorted
// slot p; inv[a] = p; rmap[p] = token (a >> 3); sexp[p] = expert; tiles (expert, first slot, count) with at most TB slots each;
// *ntiles. Slots of one expert are contiguous.
template <int TB>
__global__ void __launch_bounds__(1024) pf_group(const int * __restrict__ ids, int nass, int * __restrict__ perm, int * __restrict__ inv,
                                                 int * __restrict__ rmap, int * __restrict__ sexp, int * __restrict__ tile_e,
                                                 int * __restrict__ tile_p, int * __restrict__ tile_c, int * __restrict__ ntiles) {
    __shared__ int cnt[128], off[128], fill[128], toff[129];
    const int t = threadIdx.x;
    if (t < 128) { cnt[t] = 0; fill[t] = 0; }
    __syncthreads();
    for (int a = t; a < nass; a += 1024) atomicAdd(&cnt[ids[a]], 1);
    __syncthreads();
    if (t == 0) {
        int o = 0, to = 0;
        for (int e = 0; e < 128; ++e) { off[e] = o; toff[e] = to; o += cnt[e]; to += (cnt[e] + TB - 1) / TB; }
        toff[128] = to;
        *ntiles = to;
    }
    __syncthreads();
    for (int a = t; a < nass; a += 1024) {
        const int e = ids[a];
        const int p = off[e] + atomicAdd(&fill[e], 1);
        perm[p] = a; inv[a] = p; rmap[p] = a >> 3; sexp[p] = e;
    }
    if (t < 128) {
        const int nt = (cnt[t] + TB - 1) / TB;
        for (int i = 0; i < nt; ++i) {
            tile_e[toff[t] + i] = t; tile_p[toff[t] + i] = off[t] + i * TB; tile_c[toff[t] + i] = min(TB, cnt[t] - i * TB);
        }
    }
}
