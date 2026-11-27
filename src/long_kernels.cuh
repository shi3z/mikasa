// Long-context attention for forward_batch (up to 65536 positions per sequence slot).
//   * SWA applications: KV ring buffer of `ring` rows (>= window + MB - 1), row = pos % ring, FP32.
//   * Full applications: linear KV [maxctx][NKV*HD], FP32 or FP16 (template parameter), split-KV attention:
//     kb_attn_part (one block per (query, chunk of ATT_CH positions, kv head), online-softmax partials) + kb_attn_combine.
//   * kb_attn_ref: naive FP32 full softmax, one block per (head, query); the oracle for the validation of the above.
// All kernels keep the semantics of kb_attn2: a zero sink logit joins the softmax (scores carry "- sink[h]" and the
// denominator contains exp(-m)), relative bias row d = pos - j for d < 1024, causal limit pos_arr[b], slot by slot_arr[b].
#pragma once
#include "batch_kernels2.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>

constexpr int ATT_CH = 1024;                 // positions per split-KV chunk
constexpr int SWA_CH = 128;                  // positions per chunk of the split SWA attention (window 1024 -> 8 chunks)
constexpr int ATT_MAXCTX = 65536;            // hard upper bound of the position (scratch sizes)
constexpr int ATT_NCH = ATT_MAXCTX / ATT_CH; // 64

static int g_kv_rmask = 0;   // experiment only: MIKASA_KVROUND=1 (K) / 2 (V) / 3: FP32 storage rounded through FP16
struct KVSlots { void * k[4]; void * v[4]; };   // per application: K / V base pointers of the (up to 4) sequence slots

__device__ __forceinline__ void kv_store(float * p, float v) { *p = v; }
__device__ __forceinline__ float kv_round16(float v) { return __half2float(__float2half(v)); }
__device__ __forceinline__ void kv_store(__half * p, float v) { *p = __float2half(v); }

// FP8 (E4M3, no hardware conversion needed: sm_86 has none). Full-attention KV only. Stored value = x * 2^KV8_SH(K) (exact scaling).
struct fp8e4 { uint8_t b; };
__device__ __forceinline__ void kv_store(fp8e4 * p, float v) { p->b = (uint8_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }
// 4 E4M3 bytes -> 2 x half2 (exact): place the byte in the high half of a 16-bit lane, shift magnitude right by one, rebias with *256
__device__ __forceinline__ void fp8x4_to_h2(uint32_t w, __half2 & lo, __half2 & hi) {
    const uint32_t a = __byte_perm(w, 0, 0x1404), b = __byte_perm(w, 0, 0x3424);
    const uint32_t ra = ((a & 0x7fff7fffu) >> 1) | (a & 0x80008000u), rb = ((b & 0x7fff7fffu) >> 1) | (b & 0x80008000u);
    const __half2 k256 = __halves2half2(__ushort_as_half(0x5C00), __ushort_as_half(0x5C00));
    lo = __hmul2(*reinterpret_cast<const __half2 *>(&ra), k256);
    hi = __hmul2(*reinterpret_cast<const __half2 *>(&rb), k256);
}
__device__ __forceinline__ void kv_load8(const fp8e4 * p, float (&f)[8]) {
    const uint2 u = __ldg(reinterpret_cast<const uint2 *>(p));
    __half2 h[4];
    fp8x4_to_h2(u.x, h[0], h[1]); fp8x4_to_h2(u.y, h[2], h[3]);
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float2 t = __half22float2(h[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ int d_kv8sim = 0;     // experiment: FP16 storage, values rounded through E4M3 first (1 = K, 2 = V, 4 = K per-vector scaled, 8 = V per-vector scaled)
__device__ __forceinline__ float kv_dec8(uint8_t q) {
    const uint32_t a = (uint32_t) q << 8;
    const uint32_t r = ((a & 0x7fffu) >> 1) | (a & 0x8000u);
    return __half2float(__hmul(__ushort_as_half((unsigned short) r), __ushort_as_half(0x5C00)));
}
__device__ __forceinline__ float kv_r8(float v) { return kv_dec8((uint8_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)); }
__device__ __forceinline__ float kv_load1(const fp8e4 * p) {
    const uint32_t a = (uint32_t) __ldg(&p->b) << 8;
    const uint32_t r = ((a & 0x7fffu) >> 1) | (a & 0x8000u);
    return __half2float(__hmul(__ushort_as_half((unsigned short) r), __ushort_as_half(0x5C00)));
}

// 8 consecutive elements -> floats
__device__ __forceinline__ void kv_load8(const float * p, float (&f)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(p)), b = __ldg(reinterpret_cast<const float4 *>(p) + 1);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ void kv_load8(const __half * p, float (&f)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4 *>(p));
    const __half2 * h = reinterpret_cast<const __half2 *>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float2 t = __half22float2(h[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ __forceinline__ float kv_load1(const float * p) { return __ldg(p); }
__device__ __forceinline__ float kv_load1(const __half * p) { return __half2float(__ldg(p)); }

// ------------------------------------------------------------------ KV write ----
// grid (2*NKV, B), block 256. Row = pos (ring == 0) or pos % ring in the cache of sequence slot[b].
template <typename KT>
__global__ void kb_kv_write(const float * Kconv, const float * Vconv, float thr, const int * pos_arr, const int * slot, KVSlots kv,
                            int ring, int NKVh, int dim, float eps, int rmask = 0) {
    __shared__ float red[32];
    const int b = blockIdx.x, d = threadIdx.x, bb = blockIdx.y;
    const int pos = pos_arr[bb], sl = slot[bb];
    const size_t row = ring ? (size_t) (pos % ring) : (size_t) pos;
    if (b < NKVh) {
        KT * kc = (KT *) kv.k[sl];
        const float * in = Kconv + (size_t) bb * NKVh * dim + b * dim;
        float v = (d < dim) ? in[d] : 0.0f;
        const float ss = block_sum(d < dim ? v * v : 0.0f, red);
        const float f = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
        if (d < dim) { float w = v * f; if (sizeof(KT) == 2 && (d_kv8sim & 1)) w = kv_r8(w); kv_store(kc + row * NKVh * dim + b * dim + d, (rmask & 1) ? kv_round16(w) : w); }   // rmask: experiment, FP32 storage of FP16-rounded values
    } else {
        KT * vc = (KT *) kv.v[sl];
        const int h = b - NKVh;
        const float v = (d < dim) ? Vconv[(size_t) bb * NKVh * dim + h * dim + d] : 0.0f;
        const float ss = block_sum(v * v, red);
        const float inv = rsqrtf(ss / dim + eps);
        if (d < dim) { float w = v * inv; if (sizeof(KT) == 2 && (d_kv8sim & 2)) w = kv_r8(w); kv_store(vc + row * NKVh * dim + h * dim + d, (rmask & 2) ? kv_round16(w) : w); }
    }
}

// ------------------------------------------------------------- SWA attention ----
// kb_attn2 with ring addressing (FP32). Same arithmetic / summation order as kb_attn2.
__global__ void __launch_bounds__(512)
kb_attn2_ring(const float * __restrict__ Q, KVSlots kv, int ring,
              const float * __restrict__ R, const float * __restrict__ relW, const float * __restrict__ sink,
              const int * __restrict__ pos_arr, const int * __restrict__ slot, int window, float * __restrict__ O,
              const float * __restrict__ G, float * __restrict__ gated) {
    const int bb = blockIdx.y, kvh = blockIdx.x;
    const int pos = pos_arr[bb];
    const int sl = slot[bb];
    const float * Kc = (const float *) kv.k[sl];
    const float * Vc = (const float *) kv.v[sl];
    __shared__ __align__(16) float q[3][128];
    __shared__ float r[3][16];
    __shared__ float sc[3][1024];
    __shared__ float den[3];
    __shared__ float po[4][3][128];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid < 384) q[tid >> 7][tid & 127] = Q[(size_t) bb * 3072 + kvh * 384 + tid];
    if (tid < 48) r[tid >> 4][tid & 15] = R[bb * 384 + (kvh * 3) * 16 + tid];
    __syncthreads();
    const int j0 = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int nk = pos - j0 + 1;
    const int rs = j0 % ring;                       // ring row of position j0; row of j0 + i = rs + i (minus ring on wrap)
    const float inv_sqrt_d = 0.08838834764831845f;
    const float4 q0 = reinterpret_cast<const float4 *>(q[0])[lane], q1 = reinterpret_cast<const float4 *>(q[1])[lane],
                 q2 = reinterpret_cast<const float4 *>(q[2])[lane];
    for (int j = j0 + warp; j <= pos; j += 16) {
        int row = rs + (j - j0);
        if (row >= ring) row -= ring;
        const float4 kvv = __ldg(reinterpret_cast<const float4 *>(Kc + ((size_t) row * 8 + kvh) * 128) + lane);
        float d0 = q0.x * kvv.x + q0.y * kvv.y + q0.z * kvv.z + q0.w * kvv.w;
        float d1 = q1.x * kvv.x + q1.y * kvv.y + q1.z * kvv.z + q1.w * kvv.w;
        float d2 = q2.x * kvv.x + q2.y * kvv.y + q2.z * kvv.z + q2.w * kvv.w;
        for (int off = 16; off > 0; off >>= 1) {
            d0 += __shfl_xor_sync(0xffffffffu, d0, off);
            d1 += __shfl_xor_sync(0xffffffffu, d1, off);
            d2 += __shfl_xor_sync(0xffffffffu, d2, off);
        }
        if (lane < 3) {
            const int dd = pos - j;
            float rb = 0.0f;
            if (dd < 1024) {
#pragma unroll
                for (int k = 0; k < 16; ++k) rb += relW[dd * 16 + k] * r[lane][k];
            }
            const float dot = lane == 0 ? d0 : (lane == 1 ? d1 : d2);
            sc[lane][j - j0] = dot * inv_sqrt_d + (rb - sink[bb * 24 + kvh * 3 + lane]);
        }
    }
    __syncthreads();
    if (warp < 3) {
        float * s = sc[warp];
        float m = 0.0f;
        for (int i = lane; i < nk; i += 32) m = fmaxf(m, s[i]);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        float sum = 0.0f;
        for (int i = lane; i < nk; i += 32) { const float ev = expf(s[i] - m); s[i] = ev; sum += ev; }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) den[warp] = sum + expf(-m);
    }
    __syncthreads();
    const int dg = tid >> 7, d = tid & 127;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f;
    auto vrow = [&](int i) { int row = rs + i; if (row >= ring) row -= ring; return ((size_t) row * 8 + kvh) * 128 + d; };
    int i = dg;
    for (; i + 12 < nk; i += 16) {
        float v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = Vc[vrow(i + 4 * u)];
#pragma unroll
        for (int u = 0; u < 4; ++u) { o0 += sc[0][i + 4 * u] * v[u]; o1 += sc[1][i + 4 * u] * v[u]; o2 += sc[2][i + 4 * u] * v[u]; }
    }
    for (; i < nk; i += 4) {
        const float v = Vc[vrow(i)];
        o0 += sc[0][i] * v; o1 += sc[1][i] * v; o2 += sc[2][i] * v;
    }
    po[dg][0][d] = o0; po[dg][1][d] = o1; po[dg][2][d] = o2;
    __syncthreads();
    if (tid < 384) {
        const int hh = tid >> 7, dd2 = tid & 127;
        const float o = (po[0][hh][dd2] + po[1][hh][dd2] + po[2][hh][dd2] + po[3][hh][dd2]) / den[hh];
        const size_t idx = (size_t) bb * 3072 + (kvh * 3 + hh) * 128 + dd2;
        O[idx] = o;
        gated[idx] = o * (1.0f / (1.0f + expf(-G[idx])));
    }
}

// ------------------------------------------------- split-KV attention (full and SWA) ----
// grid (B, nchunk, NKV), block 512. Block = query b, chunk c = positions [j0 + c*ch, min(j0 + (c+1)*ch, pos+1)) where
// j0 = 0 (window == 0, full attention) or pos - window + 1 (SWA), all 3 query heads of the kv head. ring != 0: KV row =
// position % ring. Half-warp (16 lanes, 8 dims each, 16-byte loads) per key. Writes the unnormalised partial
//   Opart[b][h][c][128] = sum exp(s - m_c) v,   ML[b][h][c] = (m_c, sum exp(s - m_c)).
// b is the fastest grid index so that the queries of one same-sequence batch run together and share the chunk through L2.
template <typename KT>
__global__ void __launch_bounds__(512)
kb_attn_part(const float * __restrict__ Q, KVSlots kv, const float * __restrict__ R, const float * __restrict__ relW,
             const float * __restrict__ sink, const int * __restrict__ pos_arr, const int * __restrict__ slot,
             int window, int ring, int ch, float * __restrict__ Opart, float * __restrict__ ML) {
    const int bb = blockIdx.x, c = blockIdx.y, kvh = blockIdx.z;
    const int pos = pos_arr[bb];
    const int j0 = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int start = j0 + c * ch;
    if (start > pos) return;
    const int n = min(ch, pos - start + 1);
    const KT * Kc = (const KT *) kv.k[slot[bb]];
    const KT * Vc = (const KT *) kv.v[slot[bb]];
    __shared__ __align__(16) float q[3][128];
    __shared__ float r[3][16];
    __shared__ float sc[3][ATT_CH];
    __shared__ __align__(16) float po[16][3][128];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int sub = lane & 15, hf = lane >> 4;
    const unsigned hmask = hf ? 0xffff0000u : 0x0000ffffu;   // the two half-warps may leave the key loop at different times
    if (tid < 384) q[tid >> 7][tid & 127] = Q[(size_t) bb * 3072 + kvh * 384 + tid];
    if (tid < 48) r[tid >> 4][tid & 15] = R[bb * 384 + (kvh * 3) * 16 + tid];
    __syncthreads();
    const float inv_sqrt_d = 0.08838834764831845f;
    float qr[3][8];
#pragma unroll
    for (int h = 0; h < 3; ++h)
#pragma unroll
        for (int e = 0; e < 8; ++e) qr[h][e] = q[h][sub * 8 + e];
    // ---- scores
    for (int i = warp * 2 + hf; i < n; i += 32) {
        const int j = start + i;
        const size_t row = ring ? (size_t) (j % ring) : (size_t) j;
        float kf[8];
        kv_load8(Kc + (row * 8 + kvh) * 128 + sub * 8, kf);
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) { d0 += qr[0][e] * kf[e]; d1 += qr[1][e] * kf[e]; d2 += qr[2][e] * kf[e]; }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            d0 += __shfl_xor_sync(hmask, d0, off);
            d1 += __shfl_xor_sync(hmask, d1, off);
            d2 += __shfl_xor_sync(hmask, d2, off);
        }
        if (sub < 3) {
            const int dd = pos - j;
            float rb = 0.0f;
            if (dd < 1024) {
#pragma unroll
                for (int k = 0; k < 16; ++k) rb += relW[dd * 16 + k] * r[sub][k];
            }
            const float dot = sub == 0 ? d0 : (sub == 1 ? d1 : d2);
            sc[sub][i] = dot * inv_sqrt_d + (rb - sink[bb * 24 + kvh * 3 + sub]);
        }
    }
    __syncthreads();
    // ---- chunk softmax (without the sink logit; it is added in the combine)
    if (warp < 3) {
        float * s = sc[warp];
        float m = -INFINITY;
        for (int i = lane; i < n; i += 32) m = fmaxf(m, s[i]);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        float sum = 0.0f;
        for (int i = lane; i < n; i += 32) { const float ev = expf(s[i] - m); s[i] = ev; sum += ev; }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) {
            const size_t o = ((size_t) (bb * 24 + kvh * 3 + warp) * ATT_NCH + c) * 2;
            ML[o] = m; ML[o + 1] = sum;
        }
    }
    __syncthreads();
    // ---- PV: 32 groups of 16 lanes, group g takes keys g, g+32, ...
    const int g = tid >> 4;
    float acc[3][8];
#pragma unroll
    for (int h = 0; h < 3; ++h)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[h][e] = 0.0f;
    for (int i = g; i < n; i += 64) {
        float v0[8], v1[8];
        const bool has1 = (i + 32) < n;
        const size_t row0 = ring ? (size_t) ((start + i) % ring) : (size_t) (start + i);
        kv_load8(Vc + (row0 * 8 + kvh) * 128 + sub * 8, v0);
        if (has1) {
            const size_t row1 = ring ? (size_t) ((start + i + 32) % ring) : (size_t) (start + i + 32);
            kv_load8(Vc + (row1 * 8 + kvh) * 128 + sub * 8, v1);
        }
        const float a0 = sc[0][i], a1 = sc[1][i], a2 = sc[2][i];
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[0][e] += a0 * v0[e]; acc[1][e] += a1 * v0[e]; acc[2][e] += a2 * v0[e]; }
        if (has1) {
            const float b0 = sc[0][i + 32], b1 = sc[1][i + 32], b2 = sc[2][i + 32];
#pragma unroll
            for (int e = 0; e < 8; ++e) { acc[0][e] += b0 * v1[e]; acc[1][e] += b1 * v1[e]; acc[2][e] += b2 * v1[e]; }
        }
    }
#pragma unroll
    for (int h = 0; h < 3; ++h)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[h][e] += __shfl_xor_sync(0xffffffffu, acc[h][e], 16);
    if (hf == 0) {
#pragma unroll
        for (int h = 0; h < 3; ++h)
#pragma unroll
            for (int e = 0; e < 8; ++e) po[warp][h][sub * 8 + e] = acc[h][e];
    }
    __syncthreads();
    if (tid < 384) {
        const int hh = tid >> 7, d = tid & 127;
        float o = 0.0f;
#pragma unroll
        for (int w = 0; w < 16; ++w) o += po[w][hh][d];
        Opart[((size_t) (bb * 24 + kvh * 3 + hh) * ATT_NCH + c) * 128 + d] = o;
    }
}

// grid (NH, B), block 128 (one thread per head dim). Merges the chunk partials with the zero sink logit.
__global__ void kb_attn_combine(const float * __restrict__ Opart, const float * __restrict__ ML, const int * __restrict__ pos_arr,
                                int window, int ch, const float * __restrict__ G, float * __restrict__ O, float * __restrict__ gated) {
    const int h = blockIdx.x, bb = blockIdx.y, d = threadIdx.x;
    const int pos = pos_arr[bb];
    const int j0 = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int nch = (pos - j0) / ch + 1;
    const float * ml = ML + (size_t) (bb * 24 + h) * ATT_NCH * 2;
    float M = 0.0f;                              // the sink logit is 0
    for (int c = 0; c < nch; ++c) M = fmaxf(M, ml[2 * c]);
    float den = expf(-M), o = 0.0f;
    const float * op = Opart + (size_t) (bb * 24 + h) * ATT_NCH * 128 + d;
    for (int c = 0; c < nch; ++c) {
        const float w = expf(ml[2 * c] - M);
        den += w * ml[2 * c + 1];
        o += w * op[(size_t) c * 128];
    }
    o /= den;
    const size_t idx = (size_t) bb * 3072 + h * 128 + d;
    O[idx] = o;
    gated[idx] = o * (1.0f / (1.0f + expf(-G[idx])));
}

// ---------------------------------------------------- naive reference attention ----
// grid (NH, B), block 256. Plain FP32: all scores into global scratch S[b][h][maxctx], max, sum, weighted V sum.
// ring == 0: linear rows; otherwise row = j % ring. window == 0: all keys.
template <typename KT>
__global__ void kb_attn_ref(const float * __restrict__ Q, KVSlots kv, int ring, const float * __restrict__ R,
                            const float * __restrict__ relW, const float * __restrict__ sink, const int * __restrict__ pos_arr,
                            const int * __restrict__ slot, int window, float * __restrict__ S, size_t smax,
                            float * __restrict__ O, const float * __restrict__ G, float * __restrict__ gated) {
    const int h = blockIdx.x, bb = blockIdx.y, kvh = h / 3, tid = threadIdx.x;
    const int pos = pos_arr[bb];
    const KT * Kc = (const KT *) kv.k[slot[bb]];
    const KT * Vc = (const KT *) kv.v[slot[bb]];
    float * s = S + ((size_t) bb * 24 + h) * smax;
    __shared__ float q[128], r[16], red[256], part[256];
    if (tid < 128) q[tid] = Q[(size_t) bb * 3072 + h * 128 + tid];
    if (tid < 16) r[tid] = R[bb * 384 + h * 16 + tid];
    __syncthreads();
    const int j0 = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    float m = 0.0f;                                // includes the zero sink logit
    for (int j = j0 + tid; j <= pos; j += 256) {
        const size_t row = ring ? (size_t) (j % ring) : (size_t) j;
        const KT * kp = Kc + (row * 8 + kvh) * 128;
        float dot = 0.0f;
        for (int e = 0; e < 128; ++e) dot += q[e] * kv_load1(kp + e);
        float rb = 0.0f;
        const int dd = pos - j;
        if (dd < 1024) for (int k = 0; k < 16; ++k) rb += relW[dd * 16 + k] * r[k];
        const float v = dot * 0.08838834764831845f + (rb - sink[bb * 24 + h]);
        s[j - j0] = v;
        m = fmaxf(m, v);
    }
    red[tid] = m; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]); __syncthreads(); }
    m = red[0]; __syncthreads();
    float sum = 0.0f;
    for (int j = j0 + tid; j <= pos; j += 256) sum += expf(s[j - j0] - m);
    part[tid] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (tid < st) part[tid] += part[tid + st]; __syncthreads(); }
    const float den = part[0] + expf(-m);
    __syncthreads();
    const int d = tid & 127, half = tid >> 7;
    float acc = 0.0f;
    for (int j = j0 + half; j <= pos; j += 2) {
        const size_t row = ring ? (size_t) (j % ring) : (size_t) j;
        acc += expf(s[j - j0] - m) * kv_load1(Vc + (row * 8 + kvh) * 128 + d);
    }
    red[tid] = acc; __syncthreads();
    if (tid < 128) {
        const float o = (red[tid] + red[tid + 128]) / den;
        const size_t idx = (size_t) bb * 3072 + h * 128 + tid;
        O[idx] = o;
        gated[idx] = o * (1.0f / (1.0f + expf(-G[idx])));
    }
}
