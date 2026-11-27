// Tensor-core flash attention for a prefill chunk of ONE sequence (consecutive positions pos_arr[0] + t, t < T).
// Semantics = kb_attn_ref / kb_attn_part (long_kernels.cuh): scores s_j = q.k_j / sqrt(128) + relbias(pos-j) - sink[h],
// a zero sink logit joins the softmax, gated = O * sigmoid(G). Output gated as FP16 [T][3072] (feeds the Wo GEMM).
//
// Kernel: block = (QT-query tile, kv head, key split z). 3 query heads of the kv head share each K/V tile in smem:
// warp = (head hh, 16 queries). mma.sync m16n8k16 fp16 / fp32 accumulate, FlashAttention-2 online softmax (base-2 domain),
// key tile 32, K/V staged through registers (so FP32 caches are rounded to FP16 when staged; FP16 caches are exact).
// Relative bias: C[q][d] = R[q] . relW[d] for the 47 distances of a (16 queries x 32 keys) tile is one k16 mma per 8
// distances (fp16 operands), stored per warp in smem and gathered with index r - n + 31; skipped when all distances >= 1024.
// Key range split over gridDim.z = PF_NSPLIT blocks (partials in a scratch buffer), merged by pf_attn_combine.
// Positions must be consecutive: query t sits at pos_arr[0] + t, slot = slot_arr[0] for the whole chunk.
// Rounding: Q, K, V, P, R, relW to FP16 inside the mma (fp32 accumulate and softmax statistics).
#pragma once
#include "long_kernels.cuh"
#include <cuda_fp16.h>
#include <cstdint>

constexpr int PF_KT = 32;          // keys per tile
constexpr int PF_NSPLIT = 8;       // key splits (grid.z)
constexpr int PF_KSTR = 136;       // smem row stride (halves) of K / V tiles (128 + 8 padding)
constexpr int PF_CSTR = 49;        // smem row stride (floats) of the relative-bias scratch
constexpr float PF_LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void pf_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void pf_ldsm4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void pf_ldsm4t(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t pf_pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t *>(&h);
}

template <typename KT> struct PfRaw;
template <> struct PfRaw<float> { float4 a, b; };
template <> struct PfRaw<__half> { uint4 u; };
template <> struct PfRaw<fp8e4> { uint2 u; };
__device__ __forceinline__ void pf_raw_load(PfRaw<float> & r, const float * p) {
    r.a = __ldg(reinterpret_cast<const float4 *>(p)); r.b = __ldg(reinterpret_cast<const float4 *>(p) + 1);
}
__device__ __forceinline__ void pf_raw_load(PfRaw<__half> & r, const __half * p) { r.u = __ldg(reinterpret_cast<const uint4 *>(p)); }
__device__ __forceinline__ uint4 pf_raw_pack(const PfRaw<float> & r) {
    return make_uint4(pf_pack(r.a.x, r.a.y), pf_pack(r.a.z, r.a.w), pf_pack(r.b.x, r.b.y), pf_pack(r.b.z, r.b.w));
}
__device__ __forceinline__ uint4 pf_raw_pack(const PfRaw<__half> & r) { return r.u; }
__device__ __forceinline__ void pf_raw_load(PfRaw<fp8e4> & r, const fp8e4 * p) { r.u = __ldg(reinterpret_cast<const uint2 *>(p)); }
__device__ __forceinline__ uint4 pf_raw_pack(const PfRaw<fp8e4> & r) {
    __half2 h[4];
    fp8x4_to_h2(r.u.x, h[0], h[1]); fp8x4_to_h2(r.u.y, h[2], h[3]);
    return make_uint4(*reinterpret_cast<uint32_t *>(&h[0]), *reinterpret_cast<uint32_t *>(&h[1]), *reinterpret_cast<uint32_t *>(&h[2]), *reinterpret_cast<uint32_t *>(&h[3]));
}

// grid (ceil(T/QT), NKV, PF_NSPLIT), block 32 * 3 * QT / 16
template <typename KT, int QT>
__global__ void __launch_bounds__((3 * QT / 16) * 32)
pf_attn_kernel(const float * __restrict__ Q, const float * __restrict__ R, const float * __restrict__ relW, const float * __restrict__ sink,
               KVSlots kv, int ring, int window, const int * __restrict__ pos_arr, const int * __restrict__ slot_arr, int T,
               float * __restrict__ Opart, float * __restrict__ ML) {
    constexpr int NW = 3 * QT / 16, NT = NW * 32, QW = QT / 16;
    constexpr int NSLOT = (1024 + NT - 1) / NT;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * Ks[2], * Vs[2];
    __half * base = reinterpret_cast<__half *>(smem_raw);
    Ks[0] = base; Vs[0] = base + PF_KT * PF_KSTR; Ks[1] = base + 2 * PF_KT * PF_KSTR; Vs[1] = base + 3 * PF_KT * PF_KSTR;
    float * Cs_all = reinterpret_cast<float *>(smem_raw + 4 * PF_KT * PF_KSTR * 2);

    const int qt = blockIdx.x, kvh = blockIdx.y, z = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int hh = warp / QW, qh = warp % QW;
    const int head = kvh * 3 + hh;
    const int pos0 = pos_arr[0], slot = slot_arr[0];
    const int q_base = qt * QT;
    const int P0 = pos0 + q_base, P1 = pos0 + min(T - 1, q_base + QT - 1), Pmax = pos0 + T - 1;
    const int lo = window > 0 ? max(0, P0 - window + 1) : 0;
    const int t_lo = lo / PF_KT, t_hi = P1 / PF_KT, nt = t_hi - t_lo + 1;
    const int nsp = window > 0 ? 1 : min(PF_NSPLIT, (nt + 11) / 12);
    const int per = (nt + nsp - 1) / nsp;
    const int ts = t_lo + z * per, te = min(t_hi + 1, ts + per);
    const int tq0 = q_base + qh * 16;           // first query (token index) of this warp
    const int p0 = pos0 + tq0;
    const int r0 = tq0 + g, r1 = tq0 + g + 8;   // token index of the two fragment rows
    if (ts >= te) {   // empty split: m = -inf marks it
        for (int i = tid; i < QT * 3; i += NT) {
            const int hq = i / QT, tq = q_base + i % QT;
            if (tq < T) { float * ml = ML + (((size_t) z * T + tq) * 24 + kvh * 3 + hq) * 2; ml[0] = -INFINITY; ml[1] = 0.0f; }
        }
        return;
    }
    const KT * Kc = (const KT *) kv.k[slot];
    const KT * Vc = (const KT *) kv.v[slot];
    float * Cs = Cs_all + (size_t) warp * 16 * PF_CSTR;

    // ---- Q (scaled later in fp32), R fragments
    uint32_t qa[8][4], ra[4];
    {
        const float * q0 = Q + (size_t) min(r0, T - 1) * 3072 + head * 128 + 2 * t4;
        const float * q1 = Q + (size_t) min(r1, T - 1) * 3072 + head * 128 + 2 * t4;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const float2 a = *reinterpret_cast<const float2 *>(q0 + 16 * ks), b = *reinterpret_cast<const float2 *>(q1 + 16 * ks);
            const float2 c = *reinterpret_cast<const float2 *>(q0 + 16 * ks + 8), d = *reinterpret_cast<const float2 *>(q1 + 16 * ks + 8);
            qa[ks][0] = pf_pack(a.x, a.y); qa[ks][1] = pf_pack(b.x, b.y); qa[ks][2] = pf_pack(c.x, c.y); qa[ks][3] = pf_pack(d.x, d.y);
        }
        const float * rr0 = R + (size_t) min(r0, T - 1) * 384 + head * 16 + 2 * t4;
        const float * rr1 = R + (size_t) min(r1, T - 1) * 384 + head * 16 + 2 * t4;
        const float2 a = *reinterpret_cast<const float2 *>(rr0), b = *reinterpret_cast<const float2 *>(rr1);
        const float2 c = *reinterpret_cast<const float2 *>(rr0 + 8), d = *reinterpret_cast<const float2 *>(rr1 + 8);
        ra[0] = pf_pack(a.x, a.y); ra[1] = pf_pack(b.x, b.y); ra[2] = pf_pack(c.x, c.y); ra[3] = pf_pack(d.x, d.y);
    }
    const float sk0 = r0 < T ? sink[(size_t) r0 * 24 + head] : 0.0f, sk1 = r1 < T ? sink[(size_t) r1 * 24 + head] : 0.0f;
    const int pos_r0 = p0 + g, pos_r1 = p0 + g + 8;

    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    PfRaw<KT> raw[NSLOT];
    auto gload = [&](int jt) {
#pragma unroll
        for (int s = 0; s < NSLOT; ++s) {
            const int i = tid + s * NT;
            if (i < 1024) {
                const int isv = i >> 9, key = (i & 511) >> 4, c8 = i & 15;
                int j = min(max(jt + key, 0), Pmax);
                const size_t row = ring ? (size_t) (j % ring) : (size_t) j;
                pf_raw_load(raw[s], (isv ? Vc : Kc) + (row * 8 + kvh) * 128 + c8 * 8);
            }
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int s = 0; s < NSLOT; ++s) {
            const int i = tid + s * NT;
            if (i < 1024) {
                const int isv = i >> 9, key = (i & 511) >> 4, c8 = i & 15;
                *reinterpret_cast<uint4 *>((isv ? Vs[buf] : Ks[buf]) + key * PF_KSTR + c8 * 8) = pf_raw_pack(raw[s]);
            }
        }
    };
    const float scale = 0.08838834764831845f;
    gload(ts * PF_KT);
    sstore(0);
    __syncthreads();
    for (int it = ts; it < te; ++it) {
        const int buf = (it - ts) & 1;
        const int jt = it * PF_KT;
        if (it + 1 < te) gload(jt + PF_KT);
        const bool active = (jt <= p0 + 15) && (window == 0 || jt + PF_KT - 1 >= p0 - window + 1);
        if (active) {
            float s[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
            const __half * Kb = Ks[buf];
#pragma unroll
            for (int ks = 0; ks < 8; ++ks)
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    const int m = lane >> 3;
                    uint32_t b0, b1, b2, b3;
                    pf_ldsm4(b0, b1, b2, b3, Kb + (8 * (2 * np + (m >> 1)) + (lane & 7)) * PF_KSTR + 16 * ks + 8 * (m & 1));
                    pf_mma(s[2 * np], qa[ks], b0, b1);
                    pf_mma(s[2 * np + 1], qa[ks], b2, b3);
                }
            // relative bias
            const int dmin = p0 - jt - 31;
            const bool use_rel = dmin < 1024;
            if (use_rel) {
                __syncwarp();
#pragma unroll
                for (int dt = 0; dt < 6; ++dt) {
                    const int d = dmin + 8 * dt + g;
                    uint32_t b0 = 0, b1 = 0;
                    if (d >= 0 && d < 1024) {
                        const float2 w0 = *reinterpret_cast<const float2 *>(relW + (size_t) d * 16 + 2 * t4);
                        const float2 w1 = *reinterpret_cast<const float2 *>(relW + (size_t) d * 16 + 2 * t4 + 8);
                        b0 = pf_pack(w0.x, w0.y); b1 = pf_pack(w1.x, w1.y);
                    }
                    float c[4] = {0.f, 0.f, 0.f, 0.f};
                    pf_mma(c, ra, b0, b1);
                    Cs[g * PF_CSTR + 8 * dt + 2 * t4] = c[0]; Cs[g * PF_CSTR + 8 * dt + 2 * t4 + 1] = c[1];
                    Cs[(g + 8) * PF_CSTR + 8 * dt + 2 * t4] = c[2]; Cs[(g + 8) * PF_CSTR + 8 * dt + 2 * t4 + 1] = c[3];
                }
                __syncwarp();
            }
            // scores -> base-2 domain, masking
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int nt2 = 0; nt2 < 4; ++nt2)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r = g + 8 * (e >> 1), n = 8 * nt2 + 2 * t4 + (e & 1);
                    const int j = jt + n, pos = (e >> 1) ? pos_r1 : pos_r0;
                    const bool ok = (j <= pos) && (window == 0 || j > pos - window);
                    float v = -INFINITY;
                    if (ok) {
                        const float rb = use_rel ? Cs[r * PF_CSTR + r - n + 31] : 0.0f;
                        v = (s[nt2][e] * scale + rb - ((e >> 1) ? sk1 : sk0)) * PF_LOG2E;
                    }
                    s[nt2][e] = v;
                    if (e >> 1) mx1 = fmaxf(mx1, v); else mx0 = fmaxf(mx0, v);
                }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            const float mu0 = mn0 == -INFINITY ? 0.0f : mn0, mu1 = mn1 == -INFINITY ? 0.0f : mn1;
            const float al0 = exp2f(m0 - mu0), al1 = exp2f(m1 - mu1);
            m0 = mn0; m1 = mn1;
            float ls0 = 0.0f, ls1 = 0.0f;
#pragma unroll
            for (int nt2 = 0; nt2 < 4; ++nt2) {
                s[nt2][0] = exp2f(s[nt2][0] - mu0); s[nt2][1] = exp2f(s[nt2][1] - mu0);
                s[nt2][2] = exp2f(s[nt2][2] - mu1); s[nt2][3] = exp2f(s[nt2][3] - mu1);
                ls0 += s[nt2][0] + s[nt2][1]; ls1 += s[nt2][2] + s[nt2][3];
            }
            l0 = l0 * al0 + ls0; l1 = l1 * al1 + ls1;
#pragma unroll
            for (int i = 0; i < 16; ++i) { o[i][0] *= al0; o[i][1] *= al0; o[i][2] *= al1; o[i][3] *= al1; }
            const __half * Vb = Vs[buf];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                uint32_t a[4];
                a[0] = pf_pack(s[2 * kk][0], s[2 * kk][1]); a[1] = pf_pack(s[2 * kk][2], s[2 * kk][3]);
                a[2] = pf_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]); a[3] = pf_pack(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
                for (int dp = 0; dp < 8; ++dp) {
                    const int m = lane >> 3;
                    uint32_t b0, b1, b2, b3;
                    pf_ldsm4t(b0, b1, b2, b3, Vb + (16 * kk + 8 * (m & 1) + (lane & 7)) * PF_KSTR + 8 * (2 * dp + (m >> 1)));
                    pf_mma(o[2 * dp], a, b0, b1);
                    pf_mma(o[2 * dp + 1], a, b2, b3);
                }
            }
        }
        if (it + 1 < te) sstore((it - ts + 1) & 1);
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (r0 < T) {
        float * op = Opart + (((size_t) z * T + r0) * 24 + head) * 128;
#pragma unroll
        for (int i = 0; i < 16; ++i) *reinterpret_cast<float2 *>(op + 8 * i + 2 * t4) = make_float2(o[i][0], o[i][1]);
        if (t4 == 0) { float * ml = ML + (((size_t) z * T + r0) * 24 + head) * 2; ml[0] = m0; ml[1] = l0; }
    }
    if (r1 < T) {
        float * op = Opart + (((size_t) z * T + r1) * 24 + head) * 128;
#pragma unroll
        for (int i = 0; i < 16; ++i) *reinterpret_cast<float2 *>(op + 8 * i + 2 * t4) = make_float2(o[i][2], o[i][3]);
        if (t4 == 0) { float * ml = ML + (((size_t) z * T + r1) * 24 + head) * 2; ml[0] = m1; ml[1] = l1; }
    }
}

// grid (T, 24), block 128. Merge the splits (m in the base-2 domain, zero sink logit), gate, write FP16 (and optional FP32 O).
__global__ void pf_attn_combine(const float * __restrict__ Opart, const float * __restrict__ ML, const float * __restrict__ G, int T,
                                __half * __restrict__ out, float * __restrict__ Oout) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float M = 0.0f;
    float mz[PF_NSPLIT], lz[PF_NSPLIT];
#pragma unroll
    for (int z = 0; z < PF_NSPLIT; ++z) {
        const float * ml = ML + (((size_t) z * T + t) * 24 + h) * 2;
        mz[z] = ml[0]; lz[z] = ml[1];
        M = fmaxf(M, mz[z]);
    }
    float den = exp2f(-M), o = 0.0f;
#pragma unroll
    for (int z = 0; z < PF_NSPLIT; ++z) {
        if (mz[z] == -INFINITY) continue;
        const float w = exp2f(mz[z] - M);
        den += w * lz[z];
        o += w * Opart[(((size_t) z * T + t) * 24 + h) * 128 + d];
    }
    o /= den;
    const size_t idx = (size_t) t * 3072 + h * 128 + d;
    if (Oout) Oout[idx] = o;
    out[idx] = __float2half(o * (1.0f / (1.0f + expf(-G[idx]))));
}

// Launcher. Q [T][3072], R [T][384], sink [T][24], G [T][3072] fp32 device arrays; relW = rel_bias [1024][16];
// window = 1024 + ring cache (FP32, KT = float) for SWA, window = 0 + ring = 0 for full attention (KT = float or __half).
// pos_arr[0] = position of the first query (consecutive), slot_arr[0] = KV slot. Scratch is allocated lazily (grows with T).
template <typename KT, int QT = 32>
static void pf_attention(const float * Q, const float * R, const float * relW, const float * sink, const float * G, const KVSlots & kv,
                         int ring, int window, const int * pos_arr, const int * slot_arr, int T, __half * out, cudaStream_t s,
                         float * Oout = nullptr) {
    static float * Opart = nullptr, * ML = nullptr;
    static int cap = 0;
    if (T > cap) {
        if (Opart) { cudaFree(Opart); cudaFree(ML); }
        cap = T;
        if (cudaMalloc(&Opart, (size_t) PF_NSPLIT * cap * 24 * 128 * 4) != cudaSuccess || cudaMalloc(&ML, (size_t) PF_NSPLIT * cap * 24 * 2 * 4) != cudaSuccess) {
            std::fprintf(stderr, "pf_attention: out of memory\n"); std::exit(1);
        }
    }
    constexpr int NW = 3 * QT / 16;
    const size_t smem = (size_t) 4 * PF_KT * PF_KSTR * 2 + (size_t) NW * 16 * PF_CSTR * 4;
    static bool init = false;
    if (!init) { cudaFuncSetAttribute(pf_attn_kernel<KT, QT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem); init = true; }
    pf_attn_kernel<KT, QT><<<dim3((T + QT - 1) / QT, 8, PF_NSPLIT), NW * 32, smem, s>>>(Q, R, relW, sink, kv, ring, window, pos_arr, slot_arr, T, Opart, ML);
    pf_attn_combine<<<dim3(T, 24), 128, 0, s>>>(Opart, ML, G, T, out, Oout);
}
