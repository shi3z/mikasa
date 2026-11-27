// Small kernels of the prefill path (elementwise / conv / small GEMMs), all batched over T tokens of one chunk.
#pragma once
#include "pf_gemm.cuh"

// rms-norm of 3 rows per token, write FP32 and FP16. grid 3*T, block 256. in [T][3][n]
__global__ void pf_rms3_h(const float * __restrict__ in, const float * __restrict__ w0, const float * __restrict__ w1,
                          const float * __restrict__ w2, float * __restrict__ out, __half * __restrict__ outh, int n, float eps) {
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
        if (i < n) { const float o = v[j] * inv * w[i]; out[(size_t) blockIdx.x * n + i] = o; outh[(size_t) blockIdx.x * n + i] = __float2half(o); }
    }
}
// rms-norm rows [T][n] (same arithmetic as kb_rms_rows9) + FP16 copy. grid T.
__global__ void pf_rms_h(const float * __restrict__ in, const float * __restrict__ w, float * __restrict__ out, __half * __restrict__ outh,
                         int n, float eps) {
    __shared__ float red[32];
    in += (size_t) blockIdx.x * n; out += (size_t) blockIdx.x * n; outh += (size_t) blockIdx.x * n;
    float v[RMS_NV], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) { const int i = threadIdx.x + 256 * j; v[j] = (i < n) ? in[i] : 0.0f; ss += v[j] * v[j]; }
    ss = block_sum(ss, red);
    const float inv = rsqrtf(ss / n + eps);
#pragma unroll
    for (int j = 0; j < RMS_NV; ++j) {
        const int i = threadIdx.x + 256 * j;
        if (i < n) { const float o = (v[j] * inv) * (w ? w[i] : 1.0f); out[i] = o; outh[i] = __float2half(o); }
    }
}

__device__ __forceinline__ float pf_wf(float x) { return x; }
__device__ __forceinline__ float pf_wf(uint16_t x) { return __uint_as_float((uint32_t) x << 16); }
// Y[t*ybs + n] = sum_k W[n][k] * X[t*xbs + k], FP32 FMA. grid (ceil(N/32), ceil(T/32)), block 256.
template <typename WT>
__global__ void __launch_bounds__(256) pf_sgemm(const WT * __restrict__ W, const float * __restrict__ X, int xbs, float * __restrict__ Y, int ybs,
                                               int K, int N, int T) {
    __shared__ float Ws[32][33], Xs[32][33];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int n0 = blockIdx.x * 32, t0 = blockIdx.y * 32;
    float acc[2][2] = {{0, 0}, {0, 0}};
    for (int k0 = 0; k0 < K; k0 += 32) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + 256 * i, rr = idx >> 5, kk = idx & 31;
            Ws[rr][kk] = (n0 + rr < N && k0 + kk < K) ? pf_wf(W[(size_t) (n0 + rr) * K + k0 + kk]) : 0.0f;
            Xs[rr][kk] = (t0 + rr < T && k0 + kk < K) ? X[(size_t) (t0 + rr) * xbs + k0 + kk] : 0.0f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; ++kk) {
            const float a0 = Ws[2 * tx][kk], a1 = Ws[2 * tx + 1][kk], b0 = Xs[2 * ty][kk], b1 = Xs[2 * ty + 1][kk];
            acc[0][0] += a0 * b0; acc[0][1] += a1 * b0; acc[1][0] += a0 * b1; acc[1][1] += a1 * b1;
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int n = n0 + 2 * tx + i, t = t0 + 2 * ty + j;
            if (n < N && t < T) Y[(size_t) t * ybs + n] = acc[j][i];
        }
}

// SITU on sorted gate_up rows: gu [P][2304] fp32 -> h [P][1152] fp16. grid (ceil(1152/256), P)
__global__ void pf_situ(const float * __restrict__ gu, const float * __restrict__ gu_scale, const int * __restrict__ sexp,
                        __half * __restrict__ h, float beta, float lb) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1152) return;
    const float sc = gu_scale[sexp[s]];
    const float g = gu[(size_t) s * 2304 + i] * sc;
    const float u = gu[(size_t) s * 2304 + 1152 + i] * sc;
    const float act = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
    const float up = lb * tanhf(u / lb);
    h[(size_t) s * 1152 + i] = __float2half(act * up);
}

// weighted sum of the 8 expert outputs of each token -> H row 3+b. dn sorted [P][C]; inv[a] = sorted slot of assignment a.
__global__ void pf_moe_sum(const float * __restrict__ dn, const float * __restrict__ dn_scale, const int * __restrict__ ids,
                           const float * __restrict__ wts, const int * __restrict__ inv, float * __restrict__ H, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (c >= C) return;
    float acc = 0.0f;
#pragma unroll
    for (int s2 = 0; s2 < 8; ++s2) {
        const int t = b * 8 + s2;
        acc += wts[t] * dn_scale[ids[t]] * dn[(size_t) inv[t] * C + c];
    }
    H[(size_t) (3 + b) * C + c] = acc;
}

// ---- convolutions over time, parallel over (token, channel). H = [3 + T][C]: rows 0..2 = committed state, rows 3+b inputs.
__global__ void pf_conv_prep(float * __restrict__ H, const float * __restrict__ S, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) { H[c] = S[c]; H[C + c] = S[C + c]; H[2 * C + c] = S[2 * C + c]; }
}
__global__ void pf_conv_commit(const float * __restrict__ H, float * __restrict__ S, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) { S[c] = H[(size_t) T * C + c]; S[C + c] = H[(size_t) (T + 1) * C + c]; S[2 * C + c] = H[(size_t) (T + 2) * C + c]; }
}
// grid (ceil(C/256), T, 2): z = 0 K, 1 V. dyn taps dy[b*32 + head*4 + i]
__global__ void pf_conv_kv(const float * __restrict__ H0, const float * __restrict__ w0, const float * __restrict__ dyn0, float * __restrict__ out0,
                           const float * __restrict__ H1, const float * __restrict__ w1, const float * __restrict__ dyn1, float * __restrict__ out1,
                           int C, int dyn) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, bb = blockIdx.y;
    if (c >= C) return;
    const bool vv = blockIdx.z;
    const float * H = vv ? H1 : H0;
    const float * w = vv ? w1 : w0;
    const float * dy = vv ? dyn1 : dyn0;
    float * out = vv ? out1 : out0;
    const float w0c = w[c * 4 + 0], w1c = w[c * 4 + 1], w2c = w[c * 4 + 2], w3c = w[c * 4 + 3];
    const int head = c / (C / 8);
    const float oscale = dyn ? (1.0f / 3.0f) : 0.44721359549995794f;
    const float h0 = H[(size_t) bb * C + c], h1 = H[(size_t) (bb + 1) * C + c], h2 = H[(size_t) (bb + 2) * C + c], cur = H[(size_t) (bb + 3) * C + c];
    float y = w0c * h0 + w1c * h1 + w2c * h2 + w3c * cur;
    if (dyn) {
        const float * tap = dy + bb * 32 + head * 4;
        y += tap[0] * cur; y += tap[1] * h2; y += tap[2] * h1; y += tap[3] * h0;
    }
    out[(size_t) bb * C + c] = (cur + y) * oscale;
}
// out[b] = add ? res[b] + conv : conv. grid (ceil(C/256), T). res may alias out.
__global__ void pf_conv_res(const float * __restrict__ H, const float * __restrict__ w, const float * res, float * out, int C, int add) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, bb = blockIdx.y;
    if (c >= C) return;
    const float w0c = w[c * 4 + 0], w1c = w[c * 4 + 1], w2c = w[c * 4 + 2], w3c = w[c * 4 + 3];
    const float h0 = H[(size_t) bb * C + c], h1 = H[(size_t) (bb + 1) * C + c], h2 = H[(size_t) (bb + 2) * C + c], cur = H[(size_t) (bb + 3) * C + c];
    const float y = w0c * h0 + w1c * h1 + w2c * h2 + w3c * cur;
    const float v = (cur + y) * 0.44721359549995794f;
    out[(size_t) bb * C + c] = add ? res[(size_t) bb * C + c] + v : v;
}

// ---------------------------------------------------------------- split-K small GEMMs (deterministic) ----
// jobs: part[ks][t][n] = sum over the ks-th K slice of W[n][k] * X[t*xbs + k]. grid (ceil(N/32), ceil(T/32), KS), block 256.
template <typename WT>
__global__ void __launch_bounds__(256) pf_sgemm_sk(const WT * __restrict__ W, const float * __restrict__ X, int xbs, float * __restrict__ part,
                                                  int K, int N, int T) {
    // tile 64 n x 64 t, thread micro tile 4 x 4, k chunk 16, k-major smem (float4 reads)
    __shared__ __align__(16) float Ws[16][68], Xs[16][68];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
    const int kc = K / gridDim.z, kb = blockIdx.z * kc;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = kb; k0 < kb + kc; k0 += 16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + 256 * i, rr = idx >> 4, kk = idx & 15;
            Ws[kk][rr] = (n0 + rr < N) ? pf_wf(W[(size_t) (n0 + rr) * K + k0 + kk]) : 0.0f;
            Xs[kk][rr] = (t0 + rr < T) ? X[(size_t) (t0 + rr) * xbs + k0 + kk] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            const float4 a = *reinterpret_cast<const float4 *>(&Ws[kk][tx * 4]);
            const float4 b = *reinterpret_cast<const float4 *>(&Xs[kk][ty * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[j][i] += av[i] * bv[j];
        }
        __syncthreads();
    }
    float * P = part + (size_t) blockIdx.z * T * N;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int n = n0 + tx * 4 + i, t = t0 + ty * 4 + j;
            if (n < N && t < T) P[(size_t) t * N + n] = acc[j][i];
        }
}
// y[t*N + n] = sum_ks part[ks][t][n]. grid ceil(T*N/256)
__global__ void pf_sred(const float * __restrict__ part, float * __restrict__ y, int TN, int KS) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TN) return;
    float a = 0.0f;
    for (int k = 0; k < KS; ++k) a += part[(size_t) k * TN + i];
    y[i] = a;
}
// top-8 router; logits summed from KS partials [ks][T][128]. grid T, block 128 (same selection logic as kb_router_top8)
__global__ void pf_router_top8(const float * __restrict__ part, int KS, int T, const float * __restrict__ bias, int * __restrict__ ids_all,
                               float * __restrict__ wts_all, float scale) {
    __shared__ float probs[128];
    __shared__ float sel[128];
    __shared__ float wv[4];
    __shared__ int wi[4];
    __shared__ int chosen[8];
    int * ids = ids_all + blockIdx.x * 8;
    float * wts = wts_all + blockIdx.x * 8;
    const int e = threadIdx.x;
    float lg = 0.0f;
    for (int k = 0; k < KS; ++k) lg += part[((size_t) k * T + blockIdx.x) * 128 + e];
    const float pr = 1.0f / (1.0f + expf(-lg));
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
// loop control: out[b][i] = (1 - tanh(F_i . r[b])) * cur[b][i] + tanh(G_i . r[b]) * msg[b][i], r = sum of KS partials [ks][T][64].
// grid (E/32, ceil(T/32)), block 256
__global__ void __launch_bounds__(256) pf_ctrl_gf(const float * cur, int cur_bs, const float * __restrict__ msg, int msg_bs, const float * __restrict__ G,
                                                 const float * __restrict__ F, const float * __restrict__ rpart, int KS, float * out, int out_bs, int Ed, int T) {
    __shared__ float Gs[32][65], Fs[32][65], Rs[32][65];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int i0 = blockIdx.x * 32, t0 = blockIdx.y * 32;
    for (int idx = tid; idx < 32 * 64; idx += 256) {
        const int rr = idx >> 6, j = idx & 63;
        Gs[rr][j] = G[(size_t) (i0 + rr) * 64 + j];
        Fs[rr][j] = F[(size_t) (i0 + rr) * 64 + j];
        float a = 0.0f;
        if (t0 + rr < T) for (int k = 0; k < KS; ++k) a += rpart[((size_t) k * T + t0 + rr) * 64 + j];
        Rs[rr][j] = a;
    }
    __syncthreads();
    float g[2][2] = {{0, 0}, {0, 0}}, f[2][2] = {{0, 0}, {0, 0}};
#pragma unroll 8
    for (int j = 0; j < 64; ++j) {
        const float g0 = Gs[2 * tx][j], g1 = Gs[2 * tx + 1][j], f0 = Fs[2 * tx][j], f1 = Fs[2 * tx + 1][j], r0 = Rs[2 * ty][j], r1 = Rs[2 * ty + 1][j];
        g[0][0] += g0 * r0; g[0][1] += g1 * r0; g[1][0] += g0 * r1; g[1][1] += g1 * r1;
        f[0][0] += f0 * r0; f[0][1] += f1 * r0; f[1][0] += f0 * r1; f[1][1] += f1 * r1;
    }
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int ii = i0 + 2 * tx + i, bb = t0 + 2 * ty + b;
            if (bb < T) out[(size_t) bb * out_bs + ii] = (1.0f - tanhf(f[b][i])) * cur[(size_t) bb * cur_bs + ii] + tanhf(g[b][i]) * msg[(size_t) bb * msg_bs + ii];
        }
}

// K / V cache write with the clip / rms rule of kb_kv_write; one warp per (token, head vector). grid ceil(B*16/8), block 256.
template <typename KT>
__global__ void pf_kv_write(const float * __restrict__ Kconv, const float * __restrict__ Vconv, float thr, const int * __restrict__ pos_arr,
                            const int * __restrict__ slot, KVSlots kv, int ring, float eps, int B) {
    const int item = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (item >= B * 16) return;
    const int bb = item >> 4, j = item & 15;
    const int pos = pos_arr[bb], sl = slot[bb];
    const size_t row = ring ? (size_t) (pos % ring) : (size_t) pos;
    const bool isk = j < 8;
    const int h = j & 7;
    const float4 v = *reinterpret_cast<const float4 *>((isk ? Kconv : Vconv) + (size_t) bb * 1024 + h * 128 + lane * 4);
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float f = isk ? fminf(1.0f, thr / (sqrtf(ss) + 1e-6f)) : rsqrtf(ss / 128.0f + eps);
    KT * dst = (KT *) (isk ? kv.k[sl] : kv.v[sl]) + row * 1024 + h * 128 + lane * 4;
    float w0 = v.x * f, w1 = v.y * f, w2 = v.z * f, w3 = v.w * f;
    if (sizeof(KT) == 2 && (d_kv8sim & (isk ? 1 : 2))) { w0 = kv_r8(w0); w1 = kv_r8(w1); w2 = kv_r8(w2); w3 = kv_r8(w3); }
    kv_store(dst + 0, w0); kv_store(dst + 1, w1); kv_store(dst + 2, w2); kv_store(dst + 3, w3);
}
