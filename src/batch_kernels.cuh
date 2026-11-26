// Batched (multi-token, same sequence) kernels for forward_batch. Every kernel here is the B-token
// generalisation of a kernel in kernels_common.cuh; the per-token arithmetic order is kept the same, so a B-token
// pass reproduces B sequential forward_token steps up to float contraction noise.
#pragma once
#include "kernels_common.cuh"
#include <algorithm>

constexpr int MB = 16;            // maximum batch
constexpr int MS_STATES = 24;     // MAXSTATES (rows of blocks / bank per token)
constexpr int SC_STRIDE = 100;    // per-token scores stride (4 * (MAXSTATES + 1))
constexpr int INV_STRIDE = 25;

// -------------------------------------------------------------- reductions ----

// Sum of B per-thread values across a 256-thread block (same operation order as block_sum). Result valid in warp 0.
__device__ __forceinline__ void block_sum_multi(float (&v)[MB], int B, float * smem /* MB*8 */) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int bb = 0; bb < MB; ++bb) if (bb < B) {
        float t = v[bb];
        for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) smem[bb * 8 + warp] = t;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int bb = 0; bb < MB; ++bb) if (bb < B) {
            float t = (lane < 8) ? smem[bb * 8 + lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
            v[bb] = t;
        }
    }
}

// four block sums at once (same per-value operation order as block_sum). smem: 4 x 32 floats. Result in all threads.
__device__ __forceinline__ void block_sum4(float (&v)[4], float * smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        float t = v[q];
        for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) smem[q * 32 + warp] = t;
    }
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    if (warp == 0) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float t = (lane < nw) ? smem[q * 32 + lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
            if (lane == 0) smem[q * 32 + 31] = t;   // slot 31 never holds a warp partial (blockDim <= 992)
        }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = smem[q * 32 + 31];
    __syncthreads();
}

// ------------------------------------------------- small GEMVs (row per block) ----

__global__ void gemvb_bf16_row9(const uint16_t * __restrict__ W, const float * __restrict__ x, int xbs, float * __restrict__ y,
                                int ybs, int K, int N, int B) {
    __shared__ float red[MB * 8];
    const int row = blockIdx.x;
    const uint16_t * wr = W + (size_t) row * K;
    float w[RMS_NV], acc[MB];
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; w[j] = (i < K) ? bf16_to_f32(wr[i]) : 0.0f; }
#pragma unroll
    for (int bb = 0; bb < MB; ++bb) acc[bb] = 0.0f;
#pragma unroll
    for (int bb = 0; bb < MB; ++bb) if (bb < B) {
#pragma unroll
        for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; if (i < K) acc[bb] += w[j] * x[(size_t) bb * xbs + i]; }
    }
    block_sum_multi(acc, B, red);
    if (threadIdx.x == 0) for (int bb = 0; bb < B; ++bb) y[(size_t) bb * ybs + row] = acc[bb];
}

__global__ void gemvb_f32_row9(const float * __restrict__ W, const float * __restrict__ x, int xbs, float * __restrict__ y,
                               int ybs, int K, int N, int B) {
    __shared__ float red[MB * 8];
    const int row = blockIdx.x;
    const float * wr = W + (size_t) row * K;
    float w[RMS_NV], acc[MB];
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; w[j] = (i < K) ? wr[i] : 0.0f; }
#pragma unroll
    for (int bb = 0; bb < MB; ++bb) acc[bb] = 0.0f;
#pragma unroll
    for (int bb = 0; bb < MB; ++bb) if (bb < B) {
#pragma unroll
        for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; if (i < K) acc[bb] += w[j] * x[(size_t) bb * xbs + i]; }
    }
    block_sum_multi(acc, B, red);
    if (threadIdx.x == 0) for (int bb = 0; bb < B; ++bb) y[(size_t) bb * ybs + row] = acc[bb];
}

// loop-control down projection, B tokens: y[b][row] = W[row] . x[b]  (K = 4608, 256 threads, float4)
__device__ __forceinline__ void f32_row8_body(const float * __restrict__ W, const float * __restrict__ x, int xbs, float * __restrict__ y,
                               int ybs, int K, int B, int row, float * red) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float4 * w4 = reinterpret_cast<const float4 *>(W + (size_t) row * K);
    float acc[MB];
#pragma unroll
    for (int bb = 0; bb < MB; ++bb) acc[bb] = 0.0f;
    {
        constexpr int NIT = 5;                               // K / 4 = 1152 = 4.5 * 256: up to 5 iterations per thread
        float4 a[NIT];
#pragma unroll
        for (int it = 0; it < NIT; ++it) { const int k = warp * 32 + lane + it * 256; if (k < K / 4) a[it] = __ldg(w4 + k); }
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int k = warp * 32 + lane + it * 256;
            if (k < K / 4) {
#pragma unroll
                for (int bb = 0; bb < MB; ++bb) if (bb < B) {
                    const float4 b = __ldg(reinterpret_cast<const float4 *>(x + (size_t) bb * xbs) + k);
                    acc[bb] += a[it].x * b.x + a[it].y * b.y + a[it].z * b.z + a[it].w * b.w;
                }
            }
        }
    }
#pragma unroll
    for (int bb = 0; bb < MB; ++bb) if (bb < B) {
        float t = acc[bb];
        for (int off = 16; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) red[bb * 8 + warp] = t;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int bb = 0; bb < B; ++bb) {
            float t = 0.0f;
            for (int i = 0; i < 8; ++i) t += red[bb * 8 + i];
            y[(size_t) bb * ybs + row] = t;
        }
    }
}

__global__ void gemvb_f32_row8(const float * __restrict__ W, const float * __restrict__ x, int xbs, float * __restrict__ y,
                               int ybs, int K, int B) {
    __shared__ float red[MB * 8];
    f32_row8_body(W, x, xbs, y, ybs, K, B, blockIdx.x, red);
}

// out[b][i] = (1 - tanh(F_i . r[b])) * cur[b][i] + tanh(G_i . r[b]) * msg[b][i]; weights read once for all tokens.
__device__ __forceinline__ void ctrl_gf_body(const float * cur, int cur_bs, const float * __restrict__ msg, int msg_bs,
                                const float * __restrict__ G, const float * __restrict__ F, const float * __restrict__ r,
                                float * out, int out_bs, int E, int B, int i) {
    const int lane = threadIdx.x & 31;
    if (i >= E) return;
    const float * gr = G + (size_t) i * 64;
    const float * fr = F + (size_t) i * 64;
    const float g0 = gr[lane], g1 = gr[lane + 32], f0 = fr[lane], f1 = fr[lane + 32];
    for (int bb = 0; bb < B; ++bb) {
        const float r0 = r[bb * 64 + lane], r1 = r[bb * 64 + lane + 32];
        float g = g0 * r0 + g1 * r1;
        float f = f0 * r0 + f1 * r1;
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, off);
            f += __shfl_xor_sync(0xffffffffu, f, off);
        }
        if (lane == 0) out[(size_t) bb * out_bs + i] = (1.0f - tanhf(f)) * cur[(size_t) bb * cur_bs + i] + tanhf(g) * msg[(size_t) bb * msg_bs + i];
    }
}

__global__ void kb_ctrl_gf_warp(const float * cur, int cur_bs, const float * __restrict__ msg, int msg_bs,
                                const float * __restrict__ G, const float * __restrict__ F, const float * __restrict__ r,
                                float * out, int out_bs, int E, int B) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    ctrl_gf_body(cur, cur_bs, msg, msg_bs, G, F, r, out, out_bs, E, B, i);
}

// ------------------------------------------------------------- norms, copies ----

__global__ void kb_embed_q8(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const int * __restrict__ toks,
                            int E, float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const size_t tok = (size_t) toks[blockIdx.y];
    out[(size_t) blockIdx.y * E + i] = (float) qs[tok * E + i] * __half2float(sc[tok * (E / 32) + i / 32]);
}

// rows contiguous [B][n]; grid = B
__global__ void kb_rms_rows9(const float * __restrict__ in, const float * __restrict__ w, float * __restrict__ out, int n, float eps) {
    __shared__ float red[32];
    in += (size_t) blockIdx.x * n;
    out += (size_t) blockIdx.x * n;
    float v[RMS_NV], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; v[j] = (i < n) ? in[i] : 0.0f; ss += v[j] * v[j]; }
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / n + eps);
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) {
        const int i = threadIdx.x + 256 * j;
        if (i < n) out[i] = (v[j] * inv) * (w ? w[i] : 1.0f);
    }
}

// [B][3][n] rows, row r = blockIdx.x % 3 uses weight w_r. grid = 3 * B
__global__ void kb_rms_rows3_9(const float * __restrict__ in, const float * __restrict__ w0, const float * __restrict__ w1,
                               const float * __restrict__ w2, float * __restrict__ out, int n, float eps) {
    __shared__ float red[32];
    const int r = blockIdx.x % 3;
    const float * w = (r == 0) ? w0 : (r == 1 ? w1 : w2);
    const float * a = in + (size_t) blockIdx.x * n;
    float v[RMS_NV], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; v[j] = (i < n) ? a[i] : 0.0f; ss += v[j] * v[j]; }
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / n + eps);
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) {
        const int i = threadIdx.x + 256 * j;
        if (i < n) out[(size_t) blockIdx.x * n + i] = v[j] * inv * w[i];
    }
}

// out[b][0:n] = rms(a[b]), out[b][n:2n] = rms(m[b]). grid (2, B)
__device__ __forceinline__ void rms2_body(const float * __restrict__ a, int a_bs, const float * __restrict__ m, int m_bs,
                              float * __restrict__ out, int n, float eps, int part, int bb, float * red) {
    const float * in = part ? m + (size_t) bb * m_bs : a + (size_t) bb * a_bs;
    float * o = out + ((size_t) bb * 2 + part) * n;
    float v[RMS_NV], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; v[j] = (i < n) ? in[i] : 0.0f; ss += v[j] * v[j]; }
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / n + eps);
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) {
        const int i = threadIdx.x + 256 * j;
        if (i < n) o[i] = v[j] * inv;
    }
}

__global__ void kb_rms2_rows9(const float * __restrict__ a, int a_bs, const float * __restrict__ m, int m_bs,
                              float * __restrict__ out, int n, float eps) {
    __shared__ float red[32];
    rms2_body(a, a_bs, m, m_bs, out, n, eps, blockIdx.x, blockIdx.y, red);
}

// ---- the three loop controls of the attention inputs (independent) in one launch per stage; job = blockIdx.z
struct CtlJob { const float * cur; int cur_bs; const float * m; int m_bs; const float * Wd; const float * gain; const float * forget; float * out; int out_bs; };
struct CtlJobs { CtlJob j[3]; };
__global__ void kb_rms2_rows9_j(const CtlJobs J, float * hcat, float * dummy, int n, float eps) {
    __shared__ float red[32];
    const CtlJob & c = J.j[blockIdx.z];
    rms2_body(c.cur, c.cur_bs, c.m, c.m_bs, hcat + (size_t) blockIdx.z * MB * 2 * n, n, eps, blockIdx.x, blockIdx.y, red);
}
__global__ void gemvb_f32_row8_j(const CtlJobs J, const float * hcat, float * r64, int K, int B) {
    __shared__ float red[MB * 8];
    f32_row8_body(J.j[blockIdx.z].Wd, hcat + (size_t) blockIdx.z * MB * K, K, r64 + (size_t) blockIdx.z * MB * 64, 64, K, B, blockIdx.x, red);
}
__global__ void kb_ctrl_gf_warp_j(const CtlJobs J, const float * r64, int E, int B) {
    const CtlJob & c = J.j[blockIdx.z];
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    ctrl_gf_body(c.cur, c.cur_bs, c.m, c.m_bs, c.gain, c.forget, r64 + (size_t) blockIdx.z * MB * 64, c.out, c.out_bs, E, B, i);
}

// dst[b*dbs + i] = src[b*sbs + i], i < n. grid (ceil(n/256), B)
__global__ void kb_copy(const float * __restrict__ src, int sbs, float * __restrict__ dst, int dbs, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t) blockIdx.y * dbs + i] = src[(size_t) blockIdx.y * sbs + i];
}

// out[b][j][i] = in[b][i] for j < 3. grid (ceil(n/256), B)
__global__ void kb_rep3(const float * __restrict__ in, float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = in[(size_t) blockIdx.y * n + i];
    for (int j = 0; j < 3; ++j) out[((size_t) blockIdx.y * 3 + j) * n + i] = v;
}

// ------------------------------------------------------------------- mix ----

// grid (S, B). State s of token b: head + b*head_bs + s*E (s < nhead) else tail + b*tail_bs.
__device__ __forceinline__ void mix_scores_body(const float * __restrict__ head, int head_bs, int nhead, const float * __restrict__ tail, int tail_bs,
                               const float * __restrict__ q, int nq, float * __restrict__ scores, float * __restrict__ inv_out,
                               int S, int E, float eps, int st, int bb, float (*red)[32]) {
    const float * in = st < nhead ? head + (size_t) bb * head_bs + (size_t) st * E : tail + (size_t) bb * tail_bs;
    scores += bb * SC_STRIDE;
    inv_out += bb * INV_STRIDE;
    float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    float v[RMS_NV];
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; v[j] = (i < E) ? in[i] : 0.0f; }
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) {
        const int i = threadIdx.x + 256 * j;
        if (i < E) {
            ss += v[j] * v[j];
            d0 += v[j] * q[i];
            if (nq > 1) d1 += v[j] * q[E + i];
            if (nq > 2) d2 += v[j] * q[2 * E + i];
        }
    }
    float r4[4] = {ss, d0, d1, d2};
    block_sum4(r4, &red[0][0]);
    const float inv = rsqrtf(r4[0] / E + eps);
    const float sc = 1.0f / sqrtf((float) E);
    const float dd0 = r4[1], dd1 = r4[2], dd2 = r4[3];
    if (threadIdx.x == 0) {
        scores[st] = dd0 * inv * sc;
        if (nq > 1) scores[S + st] = dd1 * inv * sc;
        if (nq > 2) scores[2 * S + st] = dd2 * inv * sc;
        inv_out[st] = inv;
    }
}

__global__ void kb_mix_scores2(const float * __restrict__ head, int head_bs, int nhead, const float * __restrict__ tail, int tail_bs,
                               const float * __restrict__ q, int nq, float * __restrict__ scores, float * __restrict__ inv_out,
                               int S, int E, float eps) {
    __shared__ float red[4][32];
    mix_scores_body(head, head_bs, nhead, tail, tail_bs, q, nq, scores, inv_out, S, E, eps, blockIdx.x, blockIdx.y, red);
}

// grid (ceil(E/256), nq, B). rep == 3 (only with nq == 1): the result row is written to out rows 0, 1, 2.
__device__ __forceinline__ void mix_out_body(const float * __restrict__ head, int head_bs, int nhead, const float * __restrict__ tail, int tail_bs,
                            const float * __restrict__ scores, const float * __restrict__ inv, float * __restrict__ out, int out_bs,
                            int rep, int S, int E, int qi, int bb, int ib) {
    const float * sc = scores + bb * SC_STRIDE + qi * S;
    const float * iv = inv + bb * INV_STRIDE;
    float m = sc[0];
    for (int s = 1; s < S; ++s) m = fmaxf(m, sc[s]);
    float tot = 0.0f;
    for (int s = 0; s < S; ++s) tot += expf(sc[s] - m);
    const int i = ib * blockDim.x + threadIdx.x;
    if (i >= E) return;
    float a = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float p = expf(sc[s] - m) / tot;
        const float * in = s < nhead ? head + (size_t) bb * head_bs + (size_t) s * E : tail + (size_t) bb * tail_bs;
        a += p * iv[s] * in[i];
    }
    float * o = out + (size_t) bb * out_bs;
    if (rep == 3) { o[i] = a; o[E + i] = a; o[2 * E + i] = a; }
    else o[(size_t) qi * E + i] = a;
}

__global__ void kb_mix_out2(const float * __restrict__ head, int head_bs, int nhead, const float * __restrict__ tail, int tail_bs,
                            const float * __restrict__ scores, const float * __restrict__ inv, float * __restrict__ out, int out_bs,
                            int rep, int S, int E) {
    mix_out_body(head, head_bs, nhead, tail, tail_bs, scores, inv, out, out_bs, rep, S, E, blockIdx.y, blockIdx.z, blockIdx.x);
}

// up to 3 independent mixes in one launch (job = blockIdx.z; scores / inv buffers have one slab per job)
struct MixJob { const float * head; int head_bs, nhead; const float * tail; int tail_bs; const float * q; int nq; float * out; int out_bs, rep, S; };
struct MixJobs { MixJob j[3]; };
__global__ void kb_mix_scores_j(const MixJobs J, float * scores_base, float * inv_base, int E, float eps) {
    __shared__ float red[4][32];
    const MixJob & m = J.j[blockIdx.z];
    if ((int) blockIdx.x >= m.S) return;
    mix_scores_body(m.head, m.head_bs, m.nhead, m.tail, m.tail_bs, m.q, m.nq, scores_base + (size_t) blockIdx.z * MB * SC_STRIDE,
                    inv_base + (size_t) blockIdx.z * MB * INV_STRIDE, m.S, E, eps, blockIdx.x, blockIdx.y, red);
}
// grid (ceil(E/256), nqmax, B * njobs): blockIdx.z = job * B + bb
__global__ void kb_mix_out_j(const MixJobs J, const float * scores_base, const float * inv_base, int E, int B) {
    const int job = blockIdx.z / B, bb = blockIdx.z % B;
    const MixJob & m = J.j[job];
    if ((int) blockIdx.y >= m.nq) return;
    mix_out_body(m.head, m.head_bs, m.nhead, m.tail, m.tail_bs, scores_base + (size_t) job * MB * SC_STRIDE, inv_base + (size_t) job * MB * INV_STRIDE,
                 m.out, m.out_bs, m.rep, m.S, E, blockIdx.y, bb, blockIdx.x);
}

// ------------------------------------------------- convolutions (stateful) ----
// Per channel the thread walks the B tokens with a register window. H = [3 + MB][C] per-application history buffer:
// rows 0..2 receive the OLD committed state (so any rollback target j can be rebuilt), rows 3+b hold the inputs of the
// batch (written by the producing GEMV). S (committed 3-row state) ends up = last three inputs of the sequence.
__global__ void kb_conv4_kv(float * H0, float * S0, const float * __restrict__ w0, const float * __restrict__ dyn0, float * out0,
                            float * H1, float * S1, const float * __restrict__ w1, const float * __restrict__ dyn1, float * out1,
                            int C, int B, int dyn, int multi, const int * __restrict__ slot, float * Sx0, float * Sx1) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const bool vv = blockIdx.y;
    float * H = vv ? H1 : H0;
    float * S = vv ? S1 : S0;
    float * Sx = vv ? Sx1 : Sx0;
    const float * w = vv ? w1 : w0;
    const float * dy = vv ? dyn1 : dyn0;
    float * out = vv ? out1 : out0;
    const float w0c = w[c * 4 + 0], w1c = w[c * 4 + 1], w2c = w[c * 4 + 2], w3c = w[c * 4 + 3];
    const int head = c / (C / 8);
    const float oscale = dyn ? (1.0f / 3.0f) : 0.44721359549995794f;
    if (!multi) {
        float h0 = S[c], h1 = S[C + c], h2 = S[2 * C + c];
        H[c] = h0; H[C + c] = h1; H[2 * C + c] = h2;
        for (int bb = 0; bb < B; ++bb) {
            const float cur = H[(size_t) (3 + bb) * C + c];
            float y = w0c * h0 + w1c * h1 + w2c * h2 + w3c * cur;
            if (dyn) {
                const float * tap = dy + bb * 32 + head * 4;
                y += tap[0] * cur; y += tap[1] * h2; y += tap[2] * h1; y += tap[3] * h0;
            }
            out[(size_t) bb * C + c] = (cur + y) * oscale;
            h0 = h1; h1 = h2; h2 = cur;
        }
        S[c] = h0; S[C + c] = h1; S[2 * C + c] = h2;
    } else {   // every token belongs to a different sequence: state of its slot is the history
        for (int bb = 0; bb < B; ++bb) {
            const int sl = slot[bb];
            float * Sb = sl ? Sx + (size_t) (sl - 1) * 3 * C : S;
            const float h0 = Sb[c], h1 = Sb[C + c], h2 = Sb[2 * C + c];
            const float cur = H[(size_t) (3 + bb) * C + c];
            float y = w0c * h0 + w1c * h1 + w2c * h2 + w3c * cur;
            if (dyn) {
                const float * tap = dy + bb * 32 + head * 4;
                y += tap[0] * cur; y += tap[1] * h2; y += tap[2] * h1; y += tap[3] * h0;
            }
            out[(size_t) bb * C + c] = (cur + y) * oscale;
            Sb[c] = h1; Sb[C + c] = h2; Sb[2 * C + c] = cur;
        }
    }
}

// post conv with residual: out[b] = add ? res[b] + conv : conv. Input rows come from H (rows 3+b). res may alias out.
__global__ void kb_conv4_res(float * H, float * S, const float * __restrict__ w, const float * res, float * out, int C, int B, int add,
                             int multi, const int * __restrict__ slot, float * Sx, float * out2) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0c = w[c * 4 + 0], w1c = w[c * 4 + 1], w2c = w[c * 4 + 2], w3c = w[c * 4 + 3];
    if (!multi) {
        float h0 = S[c], h1 = S[C + c], h2 = S[2 * C + c];
        H[c] = h0; H[C + c] = h1; H[2 * C + c] = h2;
        for (int bb = 0; bb < B; ++bb) {
            const float cur = H[(size_t) (3 + bb) * C + c];
            const float y = w0c * h0 + w1c * h1 + w2c * h2 + w3c * cur;
            const float v = (cur + y) * 0.44721359549995794f;
            { const float o = add ? res[(size_t) bb * C + c] + v : v; out[(size_t) bb * C + c] = o; if (out2) out2[(size_t) bb * C + c] = o; }
            h0 = h1; h1 = h2; h2 = cur;
        }
        S[c] = h0; S[C + c] = h1; S[2 * C + c] = h2;
    } else {
        for (int bb = 0; bb < B; ++bb) {
            const int sl = slot[bb];
            float * Sb = sl ? Sx + (size_t) (sl - 1) * 3 * C : S;
            const float h0 = Sb[c], h1 = Sb[C + c], h2 = Sb[2 * C + c];
            const float cur = H[(size_t) (3 + bb) * C + c];
            const float y = w0c * h0 + w1c * h1 + w2c * h2 + w3c * cur;
            const float v = (cur + y) * 0.44721359549995794f;
            { const float o = add ? res[(size_t) bb * C + c] + v : v; out[(size_t) bb * C + c] = o; if (out2) out2[(size_t) bb * C + c] = o; }
            Sb[c] = h1; Sb[C + c] = h2; Sb[2 * C + c] = cur;
        }
    }
}

// MoE weighted sum of the 8 slots of each token, then the ffn post conv, residual add. dn [B*8][C].
__global__ void kb_moe_conv_res(const float * __restrict__ dn, const float * __restrict__ dn_scale, const int * __restrict__ ids,
                                const float * __restrict__ wts, float * H, float * S, const float * __restrict__ w,
                                const float * res, float * out, int C, int B, int multi, const int * __restrict__ slot, float * Sx) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = 0, h1 = 0, h2 = 0;
    if (!multi) {
        h0 = S[c]; h1 = S[C + c]; h2 = S[2 * C + c];
        H[c] = h0; H[C + c] = h1; H[2 * C + c] = h2;
    }
    const float w0c = w[c * 4 + 0], w1c = w[c * 4 + 1], w2c = w[c * 4 + 2], w3c = w[c * 4 + 3];
#pragma unroll 4
    for (int bb = 0; bb < B; ++bb) {
        float acc = 0.0f;
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            const int t = bb * 8 + s2;
            acc += wts[t] * dn_scale[ids[t]] * dn[(size_t) t * C + c];
        }
        H[(size_t) (3 + bb) * C + c] = acc;
        float * Sb = nullptr;
        if (multi) {
            const int sl = slot[bb];
            Sb = sl ? Sx + (size_t) (sl - 1) * 3 * C : S;
            h0 = Sb[c]; h1 = Sb[C + c]; h2 = Sb[2 * C + c];
        }
        const float y = w0c * h0 + w1c * h1 + w2c * h2 + w3c * acc;
        const float v = (acc + y) * 0.44721359549995794f;
        out[(size_t) bb * C + c] = res[(size_t) bb * C + c] + v;
        if (multi) { Sb[c] = h1; Sb[C + c] = h2; Sb[2 * C + c] = acc; }
        else { h0 = h1; h1 = h2; h2 = acc; }
    }
    if (!multi) { S[c] = h0; S[C + c] = h1; S[2 * C + c] = h2; }
}

// Rollback of the committed conv states to "after j tokens of the batch": S = H rows j, j+1, j+2.
struct RbEntry { float * S; const float * H; int C; };
__global__ void kb_rollback(const RbEntry * __restrict__ tab, const int * __restrict__ jp) {
    const RbEntry e = tab[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= e.C) return;
    const int j = *jp;
    if (j < 0) return;      // spec decoding: no rollback (all tokens accepted / first step)
    const float a = e.H[(size_t) j * e.C + c], b = e.H[(size_t) (j + 1) * e.C + c], d = e.H[(size_t) (j + 2) * e.C + c];
    e.S[c] = a; e.S[e.C + c] = b; e.S[2 * e.C + c] = d;
}

// ------------------------------------------------------- clip, norm, attention ----

// grid (2*NH, B), block 128
__global__ void kb_clip_qr(const float * Qv, const float * qscale, float * Qc, int qdim, float qthr,
                           const float * R0, const float * rscale, float * Rc, int rdim, float rthr, int NHh) {
    __shared__ float red[32];
    const int b = blockIdx.x, d = threadIdx.x, bb = blockIdx.y;
    if (b < NHh) clip_head(Qv + (size_t) bb * NHh * qdim + b * qdim, qscale + b * qdim, Qc + (size_t) bb * NHh * qdim + b * qdim, qdim, qthr, d, red);
    else { const int h = b - NHh; clip_head(R0 + (size_t) bb * NHh * rdim + h * rdim, rscale + h * rdim, Rc + (size_t) bb * NHh * rdim + h * rdim, rdim, rthr, d, red); }
}

// grid (2*NKV, B), block 256. Cache row = pos_arr[b] in the cache of sequence slot[b] (slot 0: kbase/vbase, else kx/vx + (slot-1)*slot_stride).
__global__ void kb_kclip_vrms_pos(const float * Kconv, float * kbase, float thr, const float * Vconv, float * vbase,
                                  const int * pos_arr, const int * slot, float * kx, float * vx, size_t slot_stride,
                                  int NKVh, int dim, float eps) {
    __shared__ float red[32];
    const int b = blockIdx.x, d = threadIdx.x, bb = blockIdx.y;
    const int pos = pos_arr[bb];
    const int sl = slot[bb];
    if (b < NKVh) {
        float * kc = sl ? kx + (size_t) (sl - 1) * slot_stride : kbase;
        const float * in = Kconv + (size_t) bb * NKVh * dim + b * dim;
        float v = (d < dim) ? in[d] : 0.0f;
        const float ss = block_sum(d < dim ? v * v : 0.0f, red);
        const float f = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
        if (d < dim) kc[(size_t) pos * NKVh * dim + b * dim + d] = v * f;
    } else {
        float * vc = sl ? vx + (size_t) (sl - 1) * slot_stride : vbase;
        const int h = b - NKVh;
        const float v = (d < dim) ? Vconv[(size_t) bb * NKVh * dim + h * dim + d] : 0.0f;
        const float ss = block_sum(v * v, red);
        const float inv = rsqrtf(ss / dim + eps);
        if (d < dim) vc[(size_t) pos * NKVh * dim + h * dim + d] = v * inv;
    }
}

// grid (ceil(n/256), B)
__global__ void kb_scale_tau(float * __restrict__ v, int n, const int * __restrict__ pos_arr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float tau = 1.0f + 0.10000000149011612f * logf(fmaxf(1.0f, ((float) pos_arr[blockIdx.y] + 1.0f) / 8192.0f));
    if (i < n) v[(size_t) blockIdx.y * n + i] *= tau;
}

// ----------------------------------------------------------- router, MoE ----

// grid B, block 128. Same as k_router_top8, per token.
__global__ void kb_router_top8(const float * __restrict__ logits_all, const float * __restrict__ bias,
                               int * __restrict__ ids_all, float * __restrict__ wts_all, float scale) {
    __shared__ float probs[128];
    __shared__ float sel[128];
    __shared__ float wv[4];
    __shared__ int wi[4];
    __shared__ int chosen[8];
    const float * logits = logits_all + blockIdx.x * 128;
    int * ids = ids_all + blockIdx.x * 8;
    float * wts = wts_all + blockIdx.x * 8;
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

// Expert -> tokens grouping. One block of 128 threads (one per expert). Unique experts, ascending id:
// grp_e[u] = expert, grp_n[u] = number of tokens, grp_t[u*16 + j] = flat slot t = b*8 + k of the j-th (b, k) choosing it.
__global__ void kb_group(const int * __restrict__ ids, int B, int * __restrict__ grp_e, int * __restrict__ grp_n,
                         int * __restrict__ grp_t, int * __restrict__ grp_nu) {
    __shared__ int sid[MB * 8];
    __shared__ int wtot[4];
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    if (e < B * 8) sid[e] = ids[e];
    __syncthreads();
    int n = 0;
    for (int t = 0; t < B * 8; ++t) n += (sid[t] == e);
    const unsigned m = __ballot_sync(0xffffffffu, n > 0);
    if (lane == 0) wtot[warp] = __popc(m);
    __syncthreads();
    int base = 0, total = 0;
    for (int w = 0; w < 4; ++w) { if (w < warp) base += wtot[w]; total += wtot[w]; }
    if (n > 0) {
        const int u = base + __popc(m & ((1u << lane) - 1u));
        grp_e[u] = e;
        grp_n[u] = n;
        int j = 0;
        for (int t = 0; t < B * 8; ++t) if (sid[t] == e) grp_t[u * 16 + j++] = t;
    }
    if (e == 0) *grp_nu = total;
}

// ---- fused router (B <= 2): logits GEMV (one block per expert row) + top-8 + expert grouping in ONE launch.
// The last block to finish (atomic ticket, modular so it never needs a reset) reads all logits (L2) and does what kb_router_top8
// (one 128-thread half per token) and kb_group do, with the same arithmetic. Launch with 256 threads, grid NEXP.
__device__ unsigned g_ticket[32];
__global__ void __launch_bounds__(256)
kb_router_fused(const float * __restrict__ W, const float * __restrict__ x, int xbs, float * __restrict__ logits_all, const float * __restrict__ bias,
                int * __restrict__ ids_all, float * __restrict__ wts_all, float scale, int K, int B,
                int * __restrict__ grp_e, int * __restrict__ grp_n, int * __restrict__ grp_t, int * __restrict__ grp_nu) {
    __shared__ float red[MB * 8];
    __shared__ float probs[2][128], sel[2][128];
    __shared__ int chosen[2][8], sid[16], wtot[8];
    __shared__ int is_last;
    const int row = blockIdx.x;
    {
        const float * wr = W + (size_t) row * K;
        float w[RMS_NV], acc[2] = {0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; w[j] = (i < K) ? wr[i] : 0.0f; }
#pragma unroll
        for (int bb = 0; bb < 2; ++bb) if (bb < B) {
#pragma unroll
            for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; if (i < K) acc[bb] += w[j] * x[(size_t) bb * xbs + i]; }
        }
        float v[MB];
#pragma unroll
        for (int bb = 0; bb < MB; ++bb) v[bb] = bb < 2 ? acc[bb] : 0.0f;
        block_sum_multi(v, B, red);
        if (threadIdx.x == 0) for (int bb = 0; bb < B; ++bb) logits_all[(size_t) bb * 128 + row] = v[bb];
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(&g_ticket[0], 1u);
        is_last = ((old & 127u) == 127u);
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    const int h = threadIdx.x >> 7, e = threadIdx.x & 127;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, hw = e >> 5;
    if (h < B) {
        const float pr = 1.0f / (1.0f + expf(-__ldcg(logits_all + h * 128 + e)));
        probs[h][e] = pr;
        sel[h][e] = pr + bias[e];
    }
    __syncthreads();
    // top-8 by rank counting: expert j precedes e if sel[j] > sel[e] or (equal and j < e) -- the same order the sequential arg-max rounds produce
    if (h < B) {
        const float se = sel[h][e];
        int rank = 0;
#pragma unroll 8
        for (int j = 0; j < 128; ++j) { const float sj = sel[h][j]; rank += (sj > se) || (sj == se && j < e); }
        if (rank < 8) { chosen[h][rank] = e; ids_all[h * 8 + rank] = e; sid[h * 8 + rank] = e; }
    }
    __syncthreads();
    if (h < B && e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k) sum += probs[h][chosen[h][k]];
        sum += 1e-20f;
        for (int k = 0; k < 8; ++k) wts_all[h * 8 + k] = probs[h][chosen[h][k]] / sum * scale;
    }
    // grouping (as kb_group; both 128-thread halves compute it redundantly, only the first writes)
    int n = 0;
    for (int t = 0; t < B * 8; ++t) n += (sid[t] == e);
    const unsigned m = __ballot_sync(0xffffffffu, n > 0);
    if (lane == 0) wtot[warp] = __popc(m);
    __syncthreads();
    int base = 0, total = 0;
    for (int w = 0; w < 4; ++w) { if (w < hw) base += wtot[h * 4 + w]; total += wtot[h * 4 + w]; }
    if (h == 0) {
        if (n > 0) {
            const int u = base + __popc(m & ((1u << lane) - 1u));
            grp_e[u] = e;
            grp_n[u] = n;
            int j = 0;
            for (int t = 0; t < B * 8; ++t) if (sid[t] == e) grp_t[u * 16 + j++] = t;
        }
        if (e == 0) *grp_nu = total;
    }
}

// SITU on all B*8 gate_up rows. grid (ceil(1152/256), B*8)
__global__ void kb_situ(const float * __restrict__ gu, const float * __restrict__ gu_scale, const int * __restrict__ ids,
                        float * __restrict__ h, float beta, float lb) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1152) return;
    const float sc = gu_scale[ids[s]];
    const float g = gu[(size_t) s * 2304 + i] * sc;
    const float u = gu[(size_t) s * 2304 + 1152 + i] * sc;
    const float act = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
    const float up = lb * tanhf(u / lb);
    h[(size_t) s * 1152 + i] = act * up;
}

// ---------------------------------------------------------------- output ----

__global__ void kb_softcap_cut(float * __restrict__ logits, int n, int stride, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float * l = logits + (size_t) blockIdx.y * stride; l[i] = cap * tanhf(l[i] / cap); }
}

// grid B, block 1024
__global__ void kb_argmax(const float * __restrict__ logits_all, int n, int stride, int * __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float * logits = logits_all + (size_t) blockIdx.x * stride;
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
        out[blockIdx.x] = bii;
    }
}

// Multi-block arg-max (same result as kb_argmax: largest value, lowest index on ties). grid (B, 32), block 1024; the last block of a row
// (atomic ticket, modular) merges the 32 partials.
__device__ float g_amv[MB * 32];
__device__ int g_ami[MB * 32];
__global__ void __launch_bounds__(1024)
kb_argmax_mb(const float * __restrict__ logits_all, int n, int stride, int * __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int last;
    const float * logits = logits_all + (size_t) blockIdx.x * stride;
    const int per = (n + gridDim.y - 1) / gridDim.y;
    const int lo = blockIdx.y * per, hi = min(n, lo + per);
    float v = -INFINITY;
    int idx = 0;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x)
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
        g_amv[blockIdx.x * 32 + blockIdx.y] = bvv;
        g_ami[blockIdx.x * 32 + blockIdx.y] = bii;
        __threadfence();
        const unsigned old = atomicAdd(&g_ticket[1 + blockIdx.x], 1u);
        last = ((old & 31u) == 31u);
    }
    __syncthreads();
    if (!last) return;
    if (threadIdx.x == 0) {
        __threadfence();
        float bvv = __ldcg(&g_amv[blockIdx.x * 32]);
        int bii = __ldcg(&g_ami[blockIdx.x * 32]);
        for (int y = 1; y < 32; ++y) {
            const float cv = __ldcg(&g_amv[blockIdx.x * 32 + y]);
            const int ci = __ldcg(&g_ami[blockIdx.x * 32 + y]);
            if (cv > bvv || (cv == bvv && ci < bii)) { bvv = cv; bii = ci; }
        }
        out[blockIdx.x] = bii;
    }
}
