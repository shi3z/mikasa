// Batched heavy kernels, lane-per-row layout (x broadcast across the warp, weights streamed once, B accumulators).
#pragma once
#include "batch_kernels.cuh"
#include <unordered_map>


// ------------------------------------------------------ interleaved weight copies ----
// The lane-per-row kernels want every warp load to be contiguous. The batch path therefore uses a second, interleaved copy
// of each weight matrix (the original SoA copy stays for forward_token):
//   Q8_0:  qsT[g][blk][h][lane] (16 B)  = bytes 16h..16h+15 of block blk of row 32g+lane;  scT[g][blk][lane] (half)
//   NVFP4: qsT[e][g][b][h][lane] (16 B) = bytes 16h..16h+15 of 64-k block b of row 32g+lane; scT[e][g][b][lane] (u32)
static std::unordered_map<const void *, const void *> g_tmap;   // original device pointer -> interleaved copy
static bool g_lean = false;   // lean mode: the weights are loaded directly in the interleaved layout (no original SoA copy, no forward_token)
template <typename T> static const T * tmap(const T * p) {
    if (g_lean) return p;     // pointers already are the interleaved ones
    auto it = g_tmap.find((const void *) p);
    if (it == g_tmap.end()) { std::fprintf(stderr, "no interleaved copy registered\n"); std::exit(1); }
    return (const T *) it->second;
}
// one thread per (matrix, row, k-block); NB16 = 16-byte chunks per block (2 for both formats), SCB = scale bytes per block
__global__ void k_repack_il(const uint4 * __restrict__ qs, const uint8_t * __restrict__ sc, uint4 * __restrict__ qT,
                            uint8_t * __restrict__ sT, size_t total, int N, int nb, int scb) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int b = (int) (i % nb);
    const size_t rr = i / nb;                  // global row over matrices (matrix-major)
    const size_t mat = rr / N;
    const int row = (int) (rr % N);
    const size_t g = (size_t) row / 32 + mat * (N / 32);
    const int lane = row & 31;
    for (int h = 0; h < 2; ++h) qT[((g * nb + b) * 2 + h) * 32 + lane] = qs[(rr * nb + b) * 2 + h];
    for (int k = 0; k < scb; ++k) sT[((g * nb + b) * 32 + lane) * scb + k] = sc[(rr * nb + b) * scb + k];
}
static void register_il_ret(const void * qs, const void * sc, size_t mats, int N, int nb, int scb, uint4 *& qT, uint8_t *& sT) {
    const size_t total = mats * N * nb;
    if (cudaMalloc(&qT, total * 32) != cudaSuccess || cudaMalloc(&sT, total * scb) != cudaSuccess) { std::fprintf(stderr, "out of memory (interleaved copy)\n"); std::exit(1); }
    k_repack_il<<<(total + 255) / 256, 256>>>((const uint4 *) qs, (const uint8_t *) sc, qT, sT, total, N, nb, scb);
}
static void register_il(const void * qs, const void * sc, size_t mats, int N, int nb, int scb) {
    uint4 * qT; uint8_t * sT;
    register_il_ret(qs, sc, mats, N, nb, scb, qT, sT);
    g_tmap[qs] = qT;
    g_tmap[sc] = sT;
}
// lean mode: convert a freshly uploaded SoA tensor to the interleaved layout, free the SoA copy and return the interleaved pointers
template <typename QT, typename ST>
static void lean_il(QT *& qs, ST *& sc, size_t mats, int N, int nb, int scb) {
    uint4 * qT; uint8_t * sT;
    register_il_ret(qs, sc, mats, N, nb, scb, qT, sT);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "repack failed\n"); std::exit(1); }
    cudaFree(qs); cudaFree(sc);
    qs = (QT *) qT; sc = (ST *) sT;
}
// Q8_0 matrix [N][K]; scales are halves (2 bytes per block)
static void register_q8_il(const int8_t * qs, const __half * sc, int N, int K, size_t mats = 1) { register_il(qs, sc, mats, N, K / 32, 2); }
// NVFP4 matrices [mats][N][K]; scale = 4 bytes per 64-k block
static void register_nv_il(const uint8_t * qs, const uint8_t * sc, int N, int K, size_t mats) { register_il(qs, sc, mats, N, K / 64, 4); }

// ------------------------------------------------------------ Q8_0 GEMV ----
// y[b][row] = dot(W[row], x[b]) for B tokens; x[b] = x + b*xbs, y[b] = y + b*ybs.
// A warp owns 32 consecutive rows (one per lane); the K blocks of 32 are split over KS warps (partials reduced in
// shared memory). All lanes read the same x float4 (broadcast), every weight block is converted once and applied to B tokens.
template <int BT, int KS, int RG>
__device__ __forceinline__ void q8_body(float * xs, float * red, const int8_t * __restrict__ qs, const __half * __restrict__ sc,
                                        const float * __restrict__ x, int xbs, float * __restrict__ y, int ybs, int K, int N, int B,
                                        int gblk) {
    constexpr int CB = 8;                       // K blocks of 32 per staged x chunk (256 floats per token)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kp = warp % KS, g = warp / KS;
    const int row = (gblk * RG + g) * 32 + lane;
    const int nb = K / 32;                      // multiple of CB
    const bool valid = row < N;
    const size_t grp = (size_t) (gblk * RG + g);     // 32-row group (interleaved layout)
    const uint4 * q = reinterpret_cast<const uint4 *>(qs) + grp * nb * 64 + lane;
    const __half * s = sc + grp * nb * 32 + lane;
    float acc[BT];
#pragma unroll
    for (int t = 0; t < BT; ++t) acc[t] = 0.0f;
    uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0;
    float d = 0.0f;
    if (valid) { a0 = __ldg(q + kp * 64); a1 = __ldg(q + kp * 64 + 32); d = __half2float(s[kp * 32]); }
    int nxt = kp;                               // next block of this warp (already loaded into a0, a1, d)
    for (int ch = 0; ch < nb / CB; ++ch) {
        __syncthreads();
        for (int i = threadIdx.x; i < BT * CB * 8; i += blockDim.x) {
            const int tb = i / (CB * 8), k4 = i - tb * (CB * 8);
            float4 v = make_float4(0, 0, 0, 0);
            if (tb < B) v = __ldg(reinterpret_cast<const float4 *>(x + (size_t) tb * xbs + ch * CB * 32) + k4);
            reinterpret_cast<float4 *>(xs)[i] = v;
        }
        __syncthreads();
        for (; nxt < (ch + 1) * CB; nxt += KS) {
            const uint4 c0 = a0, c1 = a1;
            const float cd = d;
            if (valid && nxt + KS < nb) { a0 = __ldg(q + (nxt + KS) * 64); a1 = __ldg(q + (nxt + KS) * 64 + 32); d = __half2float(s[(nxt + KS) * 32]); }
            const uint32_t w[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            float t[BT];
#pragma unroll
            for (int tb = 0; tb < BT; ++tb) t[tb] = 0.0f;
            const float4 * xl = reinterpret_cast<const float4 *>(xs) + (nxt - ch * CB) * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float w0 = (float) (int8_t) w[i], w1 = (float) (int8_t) (w[i] >> 8);
                const float w2 = (float) (int8_t) (w[i] >> 16), w3 = (float) (int8_t) (w[i] >> 24);
#pragma unroll
                for (int tb = 0; tb < BT; ++tb) {
                    const float4 v = xl[tb * (CB * 8) + i];
                    t[tb] += w0 * v.x + w1 * v.y + w2 * v.z + w3 * v.w;
                }
            }
#pragma unroll
            for (int tb = 0; tb < BT; ++tb) acc[tb] += cd * t[tb];
        }
    }
    if (KS == 1) {
        if (valid)
#pragma unroll
            for (int tb = 0; tb < BT; ++tb) if (tb < B) y[(size_t) tb * ybs + row] = acc[tb];
    } else {
#pragma unroll
        for (int tb = 0; tb < BT; ++tb) red[(warp * BT + tb) * 32 + lane] = acc[tb];
        __syncthreads();
        const int rows = RG * 32;
        for (int e = threadIdx.x; e < rows * B; e += blockDim.x) {
            const int rl = e % rows, tb = e / rows;
            const int gg = rl >> 5, ln = rl & 31;
            const int r = (gblk * RG + gg) * 32 + ln;
            if (r < N) {
                float sum = 0.0f;
                for (int p = 0; p < KS; ++p) sum += red[((gg * KS + p) * BT + tb) * 32 + ln];
                y[(size_t) tb * ybs + r] = sum;
            }
        }
    }
}

constexpr int Q8PD = 3;   // register prefetch depth of q8_body_full
// KS = 8, RG = 1, BT <= 2: x of the (<= 2) tokens is staged once (no chunk loop / extra barriers) and every warp streams its K blocks
// (block kp, kp + 8, ...) with a PD-deep register prefetch. Same per-row arithmetic and summation order as q8_body.
template <int BT, int PD>
__device__ __forceinline__ void q8_body_full(float * xs, float * red, const int8_t * __restrict__ qs, const __half * __restrict__ sc,
                                             const float * __restrict__ x, int xbs, float * __restrict__ y, int ybs, int K, int N, int B,
                                             int gblk) {
    constexpr int KS = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kp = warp;
    const int row = gblk * 32 + lane;
    const int nb = K / 32, ni = nb / KS;
    const bool valid = row < N;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs) + (size_t) gblk * nb * 64 + lane;
    const __half * s = sc + (size_t) gblk * nb * 32 + lane;
    uint4 r0[PD], r1[PD];
    float rd[PD];
#pragma unroll
    for (int p = 0; p < PD; ++p) {
        r0[p] = make_uint4(0, 0, 0, 0); r1[p] = r0[p]; rd[p] = 0.0f;
        if (valid && p < ni) { const int blk = kp + p * KS; r0[p] = __ldg(q + blk * 64); r1[p] = __ldg(q + blk * 64 + 32); rd[p] = __half2float(s[blk * 32]); }
    }
    for (int i = threadIdx.x; i < BT * (K / 4); i += blockDim.x) {
        const int tb = i / (K / 4), k4 = i - tb * (K / 4);
        float4 v = make_float4(0, 0, 0, 0);
        if (tb < B) v = __ldg(reinterpret_cast<const float4 *>(x + (size_t) tb * xbs) + k4);
        reinterpret_cast<float4 *>(xs)[i] = v;
    }
    __syncthreads();
    float acc[BT];
#pragma unroll
    for (int t = 0; t < BT; ++t) acc[t] = 0.0f;
    for (int it = 0; it < ni; ++it) {
        const uint4 c0 = r0[0], c1 = r1[0];
        const float cd = rd[0];
#pragma unroll
        for (int p = 0; p + 1 < PD; ++p) { r0[p] = r0[p + 1]; r1[p] = r1[p + 1]; rd[p] = rd[p + 1]; }
        if (valid && it + PD < ni) { const int blk = kp + (it + PD) * KS; r0[PD - 1] = __ldg(q + blk * 64); r1[PD - 1] = __ldg(q + blk * 64 + 32); rd[PD - 1] = __half2float(s[blk * 32]); }
        const uint32_t w[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float t[BT];
#pragma unroll
        for (int tb = 0; tb < BT; ++tb) t[tb] = 0.0f;
        const float4 * xl = reinterpret_cast<const float4 *>(xs) + (kp + it * KS) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float w0 = (float) (int8_t) w[i], w1 = (float) (int8_t) (w[i] >> 8);
            const float w2 = (float) (int8_t) (w[i] >> 16), w3 = (float) (int8_t) (w[i] >> 24);
#pragma unroll
            for (int tb = 0; tb < BT; ++tb) {
                const float4 v = xl[tb * (K / 4) + i];
                t[tb] += w0 * v.x + w1 * v.y + w2 * v.z + w3 * v.w;
            }
        }
#pragma unroll
        for (int tb = 0; tb < BT; ++tb) acc[tb] += cd * t[tb];
    }
#pragma unroll
    for (int tb = 0; tb < BT; ++tb) red[(warp * BT + tb) * 32 + lane] = acc[tb];
    __syncthreads();
    for (int e = threadIdx.x; e < 32 * B; e += blockDim.x) {
        const int ln = e & 31, tb = e >> 5;
        const int r = gblk * 32 + ln;
        if (r < N) {
            float sum = 0.0f;
            for (int p = 0; p < KS; ++p) sum += red[(p * BT + tb) * 32 + ln];
            y[(size_t) tb * ybs + r] = sum;
        }
    }
}

// single matrix; grid (row blocks, token groups of BT tokens)
template <int BT, int KS, int RG>
__global__ void __launch_bounds__(32 * KS * RG)
gemvb2_q8(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x, int xbs,
          float * __restrict__ y, int ybs, int K, int N, int B, int TG) {
    constexpr bool FULL = (KS == 8 && RG == 1 && BT <= 2);
    __shared__ __align__(16) float xs[FULL ? BT * 3072 : BT * 8 * 32];
    __shared__ float red[KS > 1 ? KS * RG * BT * 32 : 1];
    const int tok0 = (blockIdx.x % TG) * BT, gb = blockIdx.x / TG;   // the TG token groups of a row block are adjacent in the grid (weights shared through L2)
    if (FULL) q8_body_full<(FULL ? BT : 1), Q8PD>(xs, red, qs, sc, x + (size_t) tok0 * xbs, xbs, y + (size_t) tok0 * ybs, ybs, K, N, min(BT, B - tok0), gb);
    else q8_body<BT, KS, RG>(xs, red, qs, sc, x + (size_t) tok0 * xbs, xbs, y + (size_t) tok0 * ybs, ybs, K, N, min(BT, B - tok0), gb);
}

// several matrices with the same K in one launch (rows of consecutive jobs are laid out along grid.x); KS = 8, RG = 1
struct BfJob { const uint16_t * W; const float * x; int xbs; float * y; int ybs; int N; };
struct Q8Jobs {
    const int8_t * qs[5]; const __half * sc[5]; const float * x[5]; float * y[5];
    int xbs[5], ybs[5], N[5], blk0[6];
    int K;
    int tg;                   // token groups per row block (adjacent in the grid)
    int full;                 // 1: BT <= 2 uses q8_body_full (x staged once, deeper prefetch)
    BfJob bf[3]; int nbf;     // optional tiny BF16 GEMVs (one block per row, all B tokens) riding in the same launch, after the Q8 blocks
};
template <int BT>
__global__ void __launch_bounds__(256)
gemvb2_q8m(const Q8Jobs J, int B) {
    constexpr bool FULL = (BT <= 2);
    __shared__ __align__(16) float xs[FULL ? BT * 3072 : BT * 8 * 32];
    __shared__ float red[8 * BT * 32];
    if ((int) blockIdx.x >= J.blk0[5] * J.tg) {          // BF16 row job (extra blocks at the end of the grid)
        int r = blockIdx.x - J.blk0[5] * J.tg, jb = 0;
        while (jb + 1 < J.nbf && r >= J.bf[jb].N) { r -= J.bf[jb].N; ++jb; }
        const BfJob & f = J.bf[jb];
        __shared__ float redb[MB * 8];
        const uint16_t * wr = f.W + (size_t) r * J.K;
        float w[RMS_NV], acc[MB];
#pragma unroll
        for (int q = 0; q < RMS_NV; ++q) { const int i = threadIdx.x + 256 * q; w[q] = (i < J.K) ? bf16_to_f32(wr[i]) : 0.0f; }
#pragma unroll
        for (int bb = 0; bb < MB; ++bb) acc[bb] = 0.0f;
#pragma unroll
        for (int bb = 0; bb < MB; ++bb) if (bb < B) {
#pragma unroll
            for (int q = 0; q < RMS_NV; ++q) { const int i = threadIdx.x + 256 * q; if (i < J.K) acc[bb] += w[q] * f.x[(size_t) bb * f.xbs + i]; }
        }
        block_sum_multi(acc, B, redb);
        if (threadIdx.x == 0) for (int bb = 0; bb < B; ++bb) f.y[(size_t) bb * f.ybs + r] = acc[bb];
        return;
    }
    const int gb = blockIdx.x / J.tg, tok0 = (blockIdx.x % J.tg) * BT;
    int j = 0;
    while (gb >= J.blk0[j + 1]) ++j;
    if (FULL && J.full) q8_body_full<(FULL ? BT : 1), Q8PD>(xs, red, J.qs[j], J.sc[j], J.x[j] + (size_t) tok0 * J.xbs[j], J.xbs[j], J.y[j] + (size_t) tok0 * J.ybs[j], J.ybs[j],
                      J.K, J.N[j], min(BT, B - tok0), gb - J.blk0[j]);
    else q8_body<BT, 8, 1>(xs, red, J.qs[j], J.sc[j], J.x[j] + (size_t) tok0 * J.xbs[j], J.xbs[j], J.y[j] + (size_t) tok0 * J.ybs[j], J.ybs[j],
                      J.K, J.N[j], min(BT, B - tok0), gb - J.blk0[j]);
}

// tile choice: BT tokens per block, ceil(B / BT) token groups (weights are re-read from L2 by the other groups)
static int q8_bt_for(int B) {
    static int tab[17] = {0};
    if (!tab[0]) {
        const int def[17] = {1, 1, 2, 3, 4, 5, 6, 7, 8, 5, 5, 6, 6, 7, 7, 8, 8};
        for (int i = 0; i <= 16; ++i) tab[i] = def[i];
        if (const char * e = std::getenv("MIKASA_Q8BT")) { int v[16]; if (std::sscanf(e, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", v,v+1,v+2,v+3,v+4,v+5,v+6,v+7,v+8,v+9,v+10,v+11,v+12,v+13,v+14,v+15) == 16) for (int i = 1; i <= 16; ++i) tab[i] = v[i - 1]; }
    }
    return tab[B];
}

template <int BT>
static void launch_q8_bt(const int8_t * qs, const __half * sc, const float * x, int xbs, float * y, int ybs, int K, int N,
                         int B, cudaStream_t s) {
    qs = tmap(qs); sc = tmap(sc);
    const int tg = (B + BT - 1) / BT;
    if (N > 100000) gemvb2_q8<BT, 1, 8><<<dim3(((N + 255) / 256) * tg), 256, 0, s>>>(qs, sc, x, xbs, y, ybs, K, N, B, tg);
    else            gemvb2_q8<BT, 8, 1><<<dim3(((N + 31) / 32) * tg), 256, 0, s>>>(qs, sc, x, xbs, y, ybs, K, N, B, tg);
}

static void launch_gemvb_q8(const int8_t * qs, const __half * sc, const float * x, int xbs, float * y, int ybs, int K, int N,
                            int B, cudaStream_t s) {
    switch (q8_bt_for(B)) {
        case 1: launch_q8_bt<1>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
        case 2: launch_q8_bt<2>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
        case 3: launch_q8_bt<3>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
        case 4: launch_q8_bt<4>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
        case 5: launch_q8_bt<5>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
        case 6: launch_q8_bt<6>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
        case 7: launch_q8_bt<7>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
        default: launch_q8_bt<8>(qs, sc, x, xbs, y, ybs, K, N, B, s); break;
    }
}

template <int BT>
static void launch_q8m_bt(Q8Jobs & J, int total_blk, int B, cudaStream_t s) {
    const int tg = (B + BT - 1) / BT;
    J.tg = tg;
    gemvb2_q8m<BT><<<dim3(J.blk0[5] * tg + (total_blk - J.blk0[5])), 256, 0, s>>>(J, B);
}
struct Q8JobsBuilder {
    Q8Jobs J; int n = 0;
    Q8JobsBuilder(int K) { J.K = K; J.blk0[0] = 0; J.nbf = 0; static const int f = std::getenv("MIKASA_Q8MFULL") ? std::atoi(std::getenv("MIKASA_Q8MFULL")) : 1; J.full = f; }
    int bf_rows = 0;
    void add_bf16(const uint16_t * W, const float * x, int xbs, float * y, int ybs, int N) { J.bf[J.nbf++] = BfJob{W, x, xbs, y, ybs, N}; bf_rows += N; }
    void add(const int8_t * qs, const __half * sc, const float * x, int xbs, float * y, int ybs, int N) {
        J.qs[n] = tmap(qs); J.sc[n] = tmap(sc); J.x[n] = x; J.xbs[n] = xbs; J.y[n] = y; J.ybs[n] = ybs; J.N[n] = N;
        J.blk0[n + 1] = J.blk0[n] + (N + 31) / 32; ++n;
    }
    void launch(int B, cudaStream_t s) {
        for (int i = n; i < 5; ++i) { J.blk0[i + 1] = J.blk0[n]; }
        const int tb = J.blk0[n] + bf_rows;
        switch (q8_bt_for(B)) {
            case 1: launch_q8m_bt<1>(J, tb, B, s); break;
            case 2: launch_q8m_bt<2>(J, tb, B, s); break;
            case 3: launch_q8m_bt<3>(J, tb, B, s); break;
            case 4: launch_q8m_bt<4>(J, tb, B, s); break;
            case 5: launch_q8m_bt<5>(J, tb, B, s); break;
            case 6: launch_q8m_bt<6>(J, tb, B, s); break;
            case 7: launch_q8m_bt<7>(J, tb, B, s); break;
            default: launch_q8m_bt<8>(J, tb, B, s); break;
        }
    }
};

// ----------------------------------------------------------- NVFP4 experts ----
// grid (N/128, NUmax), 128 threads (4 warps x 32 rows, lane = row). Block u processes expert grp_e[u] for its grp_n[u]
// tokens, NT tokens at a time: the activation vectors of the NT tokens sit in shared memory (broadcast reads), every
// weight chunk is dequantised once and applied to all NT tokens. Token slot t reads x + (t >> x_shift) * x_stride and
// writes y[t][N]. Requires N % 128 == 0. The first weight block is issued before the activation staging.
template <int K, int NT>
__global__ void __launch_bounds__(128)
nvfp4_exp2(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x,
           int64_t x_stride, int x_shift, float * __restrict__ y, int N, const int * __restrict__ grp_e,
           const int * __restrict__ grp_n, const int * __restrict__ grp_t, const int * __restrict__ grp_nu) {
    extern __shared__ float xs[];
    __shared__ float kt[16];
    constexpr int NB = K / 64;
    const int ue = blockIdx.y;
    if (ue >= *grp_nu) return;
    const int64_t e = grp_e[ue];
    const int n = grp_n[ue];
    const int * toks = grp_t + ue * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * 4 + warp) * 32 + lane;
    const size_t grp = (size_t) e * (N / 32) + blockIdx.x * 4 + warp;      // interleaved layout: [e][N/32][NB][2][32] uint4
    const uint4 * q = reinterpret_cast<const uint4 *>(qs_all) + grp * NB * 64 + lane;
    const uint32_t * sp = reinterpret_cast<const uint32_t *>(sc_all) + grp * NB * 32 + lane;
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    for (int tc = 0; tc < n; tc += NT) {
        const int nt = min(NT, n - tc);
        uint4 c0 = __ldg(q), c1 = __ldg(q + 32);
        uint32_t cs = __ldg(sp);
        if (tc) __syncthreads();
        for (int i = threadIdx.x; i < (K / 4) * nt; i += 128) {
            const int tk = i / (K / 4), k4 = i - tk * (K / 4);
            const int t = toks[tc + tk];
            reinterpret_cast<float4 *>(xs)[tk * (K / 4) + k4] =
                reinterpret_cast<const float4 *>(x + (int64_t) (t >> x_shift) * x_stride)[k4];
        }
        __syncthreads();
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
        for (int b = 0; b < NB; ++b) {
            uint4 n0 = c0, n1 = c1;
            uint32_t ns = cs;
            if (b + 1 < NB) { n0 = __ldg(q + (b + 1) * 64); n1 = __ldg(q + (b + 1) * 64 + 32); ns = __ldg(sp + (b + 1) * 32); }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint4 a = half ? c1 : c0;
                const uint32_t w[4] = {a.x, a.y, a.z, a.w};
                float wl[2][8], wh[2][8];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = h * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        wl[h][j] = kt[byte & 15];
                        wh[h][j] = kt[byte >> 4];
                    }
                const float d0 = ue4m3_arith((cs >> (8 * (2 * half + 0))) & 0xFF);
                const float d1 = ue4m3_arith((cs >> (8 * (2 * half + 1))) & 0xFF);
#pragma unroll
                for (int tk = 0; tk < NT; ++tk) {
                    if (tk < nt) {
                        const float4 * xb = reinterpret_cast<const float4 *>(xs + tk * K + b * 64 + half * 32);
                        float4 xv[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) xv[i] = xb[i];
                        const float * xf = reinterpret_cast<const float *>(xv);
                        float tot = 0.0f;
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            float sub = 0.0f;
#pragma unroll
                            for (int j = 0; j < 8; ++j) sub += wl[h][j] * xf[h * 16 + j] + wh[h][j] * xf[h * 16 + j + 8];
                            tot += (h ? d1 : d0) * sub;
                        }
                        acc[tk] += tot;
                    }
                }
            }
            c0 = n0; c1 = n1; cs = ns;
        }
#pragma unroll
        for (int tk = 0; tk < NT; ++tk) if (tk < nt) y[(size_t) toks[tc + tk] * N + row] = acc[tk];
    }
}

// ---- MoE chain gate_up -> SITU -> down in ONE launch (B <= 2: every unique expert has n <= NT tokens, all blocks co-resident).
// Phase 1 = nvfp4_exp2<2304, NT> (gate_up rows of this block), then a grid barrier, phase 2 = nvfp4_exp2<1152, NT> with x = SITU(gu) computed
// in shared memory from the gate_up result (same arithmetic as kb_situ). The first down-weight block is loaded before the barrier.
__device__ unsigned g_bar_cnt = 0, g_bar_gen = 0;
template <int NT>
__global__ void __launch_bounds__(128)
nvfp4_moe_fused(const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc, const uint8_t * __restrict__ dn_qs,
                const uint8_t * __restrict__ dn_sc, const float * __restrict__ x, int64_t x_stride, int x_shift, float * __restrict__ gu_out,
                float * __restrict__ dn_out, const float * __restrict__ gu_scale, const int * __restrict__ ids, float beta, float lb, int N,
                const int * __restrict__ grp_e, const int * __restrict__ grp_n, const int * __restrict__ grp_t, const int * __restrict__ grp_nu) {
    extern __shared__ float xs[];
    __shared__ float kt[16];
    constexpr int K1 = 2304, K2 = 1152, NB1 = K1 / 64, NB2 = K2 / 64;
    const int ue = blockIdx.y;
    const int nu = *grp_nu;
    if (ue >= nu) return;
    const int64_t e = grp_e[ue];
    const int n = grp_n[ue];          // <= NT
    const int * toks = grp_t + ue * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * 4 + warp) * 32 + lane;
    const size_t grp1 = (size_t) e * (N / 32) + blockIdx.x * 4 + warp;
    const uint4 * q1 = reinterpret_cast<const uint4 *>(gu_qs) + grp1 * NB1 * 64 + lane;
    const uint32_t * sp1 = reinterpret_cast<const uint32_t *>(gu_sc) + grp1 * NB1 * 32 + lane;
    const uint4 * q2 = reinterpret_cast<const uint4 *>(dn_qs) + grp1 * NB2 * 64 + lane;
    const uint32_t * sp2 = reinterpret_cast<const uint32_t *>(dn_sc) + grp1 * NB2 * 32 + lane;
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    const int nt = n;
    // ---------------- phase 1: gate_up
    {
        uint4 c0 = __ldg(q1), c1 = __ldg(q1 + 32);
        uint32_t cs = __ldg(sp1);
        for (int i = threadIdx.x; i < (K1 / 4) * nt; i += 128) {
            const int tk = i / (K1 / 4), k4 = i - tk * (K1 / 4);
            const int t = toks[tk];
            reinterpret_cast<float4 *>(xs)[tk * (K1 / 4) + k4] =
                reinterpret_cast<const float4 *>(x + (int64_t) (t >> x_shift) * x_stride)[k4];
        }
        __syncthreads();
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
        for (int b = 0; b < NB1; ++b) {
            uint4 n0 = c0, n1 = c1;
            uint32_t ns = cs;
            if (b + 1 < NB1) { n0 = __ldg(q1 + (b + 1) * 64); n1 = __ldg(q1 + (b + 1) * 64 + 32); ns = __ldg(sp1 + (b + 1) * 32); }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint4 a = half ? c1 : c0;
                const uint32_t w[4] = {a.x, a.y, a.z, a.w};
                float wl[2][8], wh[2][8];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = h * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        wl[h][j] = kt[byte & 15];
                        wh[h][j] = kt[byte >> 4];
                    }
                const float d0 = ue4m3_arith((cs >> (8 * (2 * half + 0))) & 0xFF);
                const float d1 = ue4m3_arith((cs >> (8 * (2 * half + 1))) & 0xFF);
#pragma unroll
                for (int tk = 0; tk < NT; ++tk) {
                    if (tk < nt) {
                        const float4 * xb = reinterpret_cast<const float4 *>(xs + tk * K1 + b * 64 + half * 32);
                        float4 xv[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) xv[i] = xb[i];
                        const float * xf = reinterpret_cast<const float *>(xv);
                        float tot = 0.0f;
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            float sub = 0.0f;
#pragma unroll
                            for (int j = 0; j < 8; ++j) sub += wl[h][j] * xf[h * 16 + j] + wh[h][j] * xf[h * 16 + j + 8];
                            tot += (h ? d1 : d0) * sub;
                        }
                        acc[tk] += tot;
                    }
                }
            }
            c0 = n0; c1 = n1; cs = ns;
        }
#pragma unroll
        for (int tk = 0; tk < NT; ++tk) if (tk < nt) gu_out[(size_t) toks[tk] * N + row] = acc[tk];
    }
    // first down-weight block in flight while the grid synchronises
    uint4 c0 = __ldg(q2), c1 = __ldg(q2 + 32);
    uint32_t cs = __ldg(sp2);
    // ---------------- grid barrier over the nu * (N/128) active blocks (counter reset by the last arriver, generation flag releases the others)
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned total = (unsigned) nu * (N / 128);
        const unsigned gen = *((volatile unsigned *) &g_bar_gen);
        const unsigned old = atomicAdd(&g_bar_cnt, 1u);
        if (old + 1 == total) {
            *((volatile unsigned *) &g_bar_cnt) = 0u;
            __threadfence();
            atomicAdd(&g_bar_gen, 1u);
        } else {
            while (*((volatile unsigned *) &g_bar_gen) == gen) { }
        }
        __threadfence();
    }
    __syncthreads();
    // ---------------- phase 2: SITU into smem, then down
    for (int i = threadIdx.x; i < K2 * nt; i += 128) {
        const int tk = i / K2, c = i - tk * K2;
        const int t = toks[tk];
        const float sc = gu_scale[ids[t]];
        const float g = __ldcg(gu_out + (size_t) t * N + c) * sc;
        const float u = __ldcg(gu_out + (size_t) t * N + K2 + c) * sc;
        const float act = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
        const float up = lb * tanhf(u / lb);
        xs[tk * K2 + c] = act * up;
    }
    __syncthreads();
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int b = 0; b < NB2; ++b) {
        uint4 n0 = c0, n1 = c1;
        uint32_t ns = cs;
        if (b + 1 < NB2) { n0 = __ldg(q2 + (b + 1) * 64); n1 = __ldg(q2 + (b + 1) * 64 + 32); ns = __ldg(sp2 + (b + 1) * 32); }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint4 a = half ? c1 : c0;
            const uint32_t w[4] = {a.x, a.y, a.z, a.w};
            float wl[2][8], wh[2][8];
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int bi = h * 8 + j;
                    const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                    wl[h][j] = kt[byte & 15];
                    wh[h][j] = kt[byte >> 4];
                }
            const float d0 = ue4m3_arith((cs >> (8 * (2 * half + 0))) & 0xFF);
            const float d1 = ue4m3_arith((cs >> (8 * (2 * half + 1))) & 0xFF);
#pragma unroll
            for (int tk = 0; tk < NT; ++tk) {
                if (tk < nt) {
                    const float4 * xb = reinterpret_cast<const float4 *>(xs + tk * K2 + b * 64 + half * 32);
                    float4 xv[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) xv[i] = xb[i];
                    const float * xf = reinterpret_cast<const float *>(xv);
                    float tot = 0.0f;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float sub = 0.0f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) sub += wl[h][j] * xf[h * 16 + j] + wh[h][j] * xf[h * 16 + j + 8];
                        tot += (h ? d1 : d0) * sub;
                    }
                    acc[tk] += tot;
                }
            }
        }
        c0 = n0; c1 = n1; cs = ns;
    }
#pragma unroll
    for (int tk = 0; tk < NT; ++tk) if (tk < nt) dn_out[(size_t) toks[tk] * N + row] = acc[tk];
}

// One token at a time (B <= 2), K split over KS slices of warps: warp (slice, rg) handles rows rg*32..+31 of the 128-row block and the 64-weight
// blocks [slice*NB/KS, (slice+1)*NB/KS). The per-(block, half) values `tot` do not depend on the accumulator, so they are stored and then
// added in the original order (acc += tot for b = 0.., half 0, 1) by one thread per row: bit-identical to nvfp4_exp2<K, 1> with KS times the warps in flight.
// grid (N/128, NUmax), 128*KS threads, dynamic smem = (K + 2*NB*128) floats.
template <int K, int KS>
__global__ void __launch_bounds__(128 * KS)
nvfp4_exp_ks(const uint8_t * __restrict__ qs_all, const uint8_t * __restrict__ sc_all, const float * __restrict__ x,
             int64_t x_stride, int x_shift, float * __restrict__ y, int N, const int * __restrict__ grp_e,
             const int * __restrict__ grp_n, const int * __restrict__ grp_t, const int * __restrict__ grp_nu) {
    extern __shared__ float xs[];
    __shared__ float kt[16];
    constexpr int NB = K / 64;
    static_assert(NB % KS == 0, "NB must be divisible by KS");
    constexpr int LB = NB / KS;
    float * T = xs + K;                       // [2 * NB][128]
    const int ue = blockIdx.y;
    if (ue >= *grp_nu) return;
    const int64_t e = grp_e[ue];
    const int n = grp_n[ue];
    const int * toks = grp_t + ue * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp & 3, slice = warp >> 2;
    const int rl = rg * 32 + lane;
    const int row = blockIdx.x * 128 + rl;
    const size_t grp = (size_t) e * (N / 32) + blockIdx.x * 4 + rg;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs_all) + grp * NB * 64 + lane;
    const uint32_t * sp = reinterpret_cast<const uint32_t *>(sc_all) + grp * NB * 32 + lane;
    if (threadIdx.x < 16) kt[threadIdx.x] = kE2M1d[threadIdx.x];
    const int b0 = slice * LB;
    for (int tc = 0; tc < n; ++tc) {
        uint4 c0 = __ldg(q + b0 * 64), c1 = __ldg(q + b0 * 64 + 32);
        uint32_t cs = __ldg(sp + b0 * 32);
        if (tc) __syncthreads();
        {
            const int t = toks[tc];
            const float4 * src = reinterpret_cast<const float4 *>(x + (int64_t) (t >> x_shift) * x_stride);
            for (int i = threadIdx.x; i < K / 4; i += 128 * KS) reinterpret_cast<float4 *>(xs)[i] = src[i];
        }
        __syncthreads();
        for (int bl = 0; bl < LB; ++bl) {
            const int b = b0 + bl;
            uint4 n0 = c0, n1 = c1;
            uint32_t ns = cs;
            if (bl + 1 < LB) { n0 = __ldg(q + (b + 1) * 64); n1 = __ldg(q + (b + 1) * 64 + 32); ns = __ldg(sp + (b + 1) * 32); }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint4 a = half ? c1 : c0;
                const uint32_t w[4] = {a.x, a.y, a.z, a.w};
                float wl[2][8], wh[2][8];
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int bi = h * 8 + j;
                        const uint32_t byte = (w[bi >> 2] >> (8 * (bi & 3))) & 0xFF;
                        wl[h][j] = kt[byte & 15];
                        wh[h][j] = kt[byte >> 4];
                    }
                const float d0 = ue4m3_arith((cs >> (8 * (2 * half + 0))) & 0xFF);
                const float d1 = ue4m3_arith((cs >> (8 * (2 * half + 1))) & 0xFF);
                const float4 * xb = reinterpret_cast<const float4 *>(xs + b * 64 + half * 32);
                float4 xv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = xb[i];
                const float * xf = reinterpret_cast<const float *>(xv);
                float tot = 0.0f;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float sub = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) sub += wl[h][j] * xf[h * 16 + j] + wh[h][j] * xf[h * 16 + j + 8];
                    tot += (h ? d1 : d0) * sub;
                }
                T[(b * 2 + half) * 128 + rl] = tot;
            }
            c0 = n0; c1 = n1; cs = ns;
        }
        __syncthreads();
        if (threadIdx.x < 128) {
            float acc = 0.0f;
            for (int j = 0; j < 2 * NB; ++j) acc += T[j * 128 + threadIdx.x];
            y[(size_t) toks[tc] * N + blockIdx.x * 128 + threadIdx.x] = acc;
        }
    }
}

template <int K, int KS>
static void launch_nvfp4_ks(const uint8_t * qs_all, const uint8_t * sc_all, const float * x, int64_t x_stride, int x_shift, float * y,
                            int N, int B, const int * ge, const int * gn, const int * gt, const int * gnu, cudaStream_t s) {
    const size_t smem = (size_t) (K + 2 * (K / 64) * 128) * sizeof(float);
    dim3 grid(N / 128, std::min(128, 8 * B));
    nvfp4_exp_ks<K, KS><<<grid, 128 * KS, smem, s>>>(tmap(qs_all), tmap(sc_all), x, x_stride, x_shift, y, N, ge, gn, gt, gnu);
}

template <int K, int NT>
static void launch_nvfp4_nt(const uint8_t * qs_all, const uint8_t * sc_all, const float * x, int64_t x_stride, int x_shift, float * y,
                            int N, int B, const int * ge, const int * gn, const int * gt, const int * gnu, cudaStream_t s) {
    const size_t smem = (size_t) NT * K * sizeof(float);
    dim3 grid(N / 128, std::min(128, 8 * B));
    nvfp4_exp2<K, NT><<<grid, 128, smem, s>>>(tmap(qs_all), tmap(sc_all), x, x_stride, x_shift, y, N, ge, gn, gt, gnu);
}

constexpr int KS_B = 0, KS_GU = 4, KS_DN = 2;   // nvfp4_exp_ks used for B <= KS_B (K=2304 / 1152 slices)
template <int K>
static void launch_nvfp4_nch(const uint8_t * qs_all, const uint8_t * sc_all, const float * x, int64_t x_stride, int x_shift, float * y,
                             int N, int B, const int * ge, const int * gn, const int * gt, const int * gnu, cudaStream_t s) {
    if (B <= KS_B && !std::getenv("MIKASA_NOKS")) { launch_nvfp4_ks<K, (K >= 2304 ? KS_GU : KS_DN)>(qs_all, sc_all, x, x_stride, x_shift, y, N, B, ge, gn, gt, gnu, s); return; }
    if (B <= 1)      launch_nvfp4_nt<K, 1>(qs_all, sc_all, x, x_stride, x_shift, y, N, B, ge, gn, gt, gnu, s);
    else if (B <= 2) launch_nvfp4_nt<K, 2>(qs_all, sc_all, x, x_stride, x_shift, y, N, B, ge, gn, gt, gnu, s);
    else             launch_nvfp4_nt<K, 4>(qs_all, sc_all, x, x_stride, x_shift, y, N, B, ge, gn, gt, gnu, s);
}

// ------------------------------------------------------------- Q8_0 experts ----
// Same grouping, lane = row, NT tokens at a time, x broadcast from global/L1.
constexpr int QU = 4;
template <int NT>
__global__ void __launch_bounds__(128)
q8_experts_rows2b(const int8_t * __restrict__ qs_all, const __half * __restrict__ sc_all, const float * __restrict__ x,
                  int64_t x_stride, int x_shift, float * __restrict__ y, int K, int N, const int * __restrict__ grp_e,
                  const int * __restrict__ grp_n, const int * __restrict__ grp_t, const int * __restrict__ grp_nu) {
    const int ue = blockIdx.y;
    if (ue >= *grp_nu) return;
    const int64_t e = grp_e[ue];
    const int n = grp_n[ue];
    const int * toks = grp_t + ue * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x * 4 + warp) * 32 + lane;
    if (row >= N) return;
    const int nb = K / 32;
    const size_t grp = (size_t) e * (N / 32) + blockIdx.x * 4 + warp;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs_all) + grp * nb * 64 + lane;
    const __half * s = sc_all + grp * nb * 32 + lane;
    for (int tc = 0; tc < n; tc += NT) {
        const int nt = min(NT, n - tc);
        const float * xp[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) xp[k] = x + (int64_t) (toks[min(tc + k, n - 1)] >> x_shift) * x_stride;
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
        for (int blk0 = 0; blk0 < nb; blk0 += QU) {        // QU weight blocks are loaded before they are used (nb % QU == 0); arithmetic order unchanged
            uint4 a0[QU], a1[QU];
            float dd[QU];
#pragma unroll
            for (int u = 0; u < QU; ++u) { a0[u] = __ldg(q + (blk0 + u) * 64); a1[u] = __ldg(q + (blk0 + u) * 64 + 32); dd[u] = __half2float(s[(blk0 + u) * 32]); }
#pragma unroll
            for (int u = 0; u < QU; ++u) {
                const int blk = blk0 + u;
                const float d = dd[u];
                const uint32_t w[8] = {a0[u].x, a0[u].y, a0[u].z, a0[u].w, a1[u].x, a1[u].y, a1[u].z, a1[u].w};
                float t[NT];
#pragma unroll
                for (int k = 0; k < NT; ++k) t[k] = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float w0 = (float) (int8_t) w[i], w1 = (float) (int8_t) (w[i] >> 8);
                    const float w2 = (float) (int8_t) (w[i] >> 16), w3 = (float) (int8_t) (w[i] >> 24);
#pragma unroll
                    for (int k = 0; k < NT; ++k) {
                        if (k < nt) {
                            const float4 v = __ldg(reinterpret_cast<const float4 *>(xp[k]) + blk * 8 + i);
                            t[k] += w0 * v.x + w1 * v.y + w2 * v.z + w3 * v.w;
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < NT; ++k) acc[k] += d * t[k];
            }
        }
#pragma unroll
        for (int k = 0; k < NT; ++k) if (k < nt) y[(size_t) toks[tc + k] * N + row] = acc[k];
    }
}

// Q8_0 experts with the K blocks split over KW warps. The per-block partial t[tok] (a function of the weights and x only) is computed in
// parallel and stored; the ordered accumulation acc += d * t (one FMA per block, in block order) is then done once per (row, token),
// so the result is bit-identical to q8_experts_rows2b while the memory-level parallelism is KW times higher.
// grid (N / 32, NUmax), 256 threads (8 warps). K <= 2304 (nb <= 72).
template <int NT>
__global__ void __launch_bounds__(256)
q8_experts_ks(const int8_t * __restrict__ qs_all, const __half * __restrict__ sc_all, const float * __restrict__ x,
              int64_t x_stride, int x_shift, float * __restrict__ y, int K, int N, const int * __restrict__ grp_e,
              const int * __restrict__ grp_n, const int * __restrict__ grp_t, const int * __restrict__ grp_nu) {
    constexpr int KW = 8, NBMAX = 72;
    __shared__ float tt[NBMAX][NT][32];
    __shared__ float dd[NBMAX][32];
    const int ue = blockIdx.y;
    if (ue >= *grp_nu) return;
    const int64_t e = grp_e[ue];
    const int n = grp_n[ue];
    const int * toks = grp_t + ue * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = K / 32;
    const size_t grp = (size_t) e * (N / 32) + blockIdx.x;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs_all) + grp * nb * 64 + lane;
    const __half * s = sc_all + grp * nb * 32 + lane;
    for (int tc = 0; tc < n; tc += NT) {
        const int nt = min(NT, n - tc);
        const float * xp[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) xp[k] = x + (int64_t) (toks[min(tc + k, n - 1)] >> x_shift) * x_stride;
        if (tc) __syncthreads();
        uint4 a0 = __ldg(q + warp * 64), a1 = __ldg(q + warp * 64 + 32);
        float d = __half2float(s[warp * 32]);
        for (int blk = warp; blk < nb; blk += KW) {
            const uint4 c0 = a0, c1 = a1;
            const float cd = d;
            if (blk + KW < nb) { a0 = __ldg(q + (blk + KW) * 64); a1 = __ldg(q + (blk + KW) * 64 + 32); d = __half2float(s[(blk + KW) * 32]); }
            const uint32_t w[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            float t[NT];
#pragma unroll
            for (int k = 0; k < NT; ++k) t[k] = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float w0 = (float) (int8_t) w[i], w1 = (float) (int8_t) (w[i] >> 8);
                const float w2 = (float) (int8_t) (w[i] >> 16), w3 = (float) (int8_t) (w[i] >> 24);
#pragma unroll
                for (int k = 0; k < NT; ++k) {
                    if (k < nt) {
                        const float4 v = __ldg(reinterpret_cast<const float4 *>(xp[k]) + blk * 8 + i);
                        t[k] += w0 * v.x + w1 * v.y + w2 * v.z + w3 * v.w;
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < NT; ++k) tt[blk][k][lane] = t[k];
            dd[blk][lane] = cd;
        }
        __syncthreads();
        if (warp < NT && warp < nt) {
            float acc = 0.0f;
            for (int blk = 0; blk < nb; ++blk) acc += dd[blk][lane] * tt[blk][warp][lane];
            y[(size_t) toks[tc + warp] * N + blockIdx.x * 32 + lane] = acc;
        }
    }
}

// ---------------------------------------------------------------- attention ----
// grid (NKV, B), block 512 (16 warps). One block handles the 3 query heads of a KV head for token b (query position
// pos_arr[b], KV cache of sequence slot[b]; causal limit pos_arr[b]): each K / V row is read once for the 3 heads.
// Scores: warp per key, lane owns 4 head dims (coalesced 512 B row read, shuffle reduction). PV: 4 groups of 128 threads
// split the keys. Softmax with the zero sink logit as in k_attn_decode_gen.
__global__ void __launch_bounds__(512)
kb_attn2(const float * __restrict__ Q, const float * __restrict__ Kc0, const float * __restrict__ Vc0,
         const float * __restrict__ Kx, const float * __restrict__ Vx, size_t slot_stride,
         const float * __restrict__ R, const float * __restrict__ relW, const float * __restrict__ sink,
         const int * __restrict__ pos_arr, const int * __restrict__ slot, int window, float * __restrict__ O,
         const float * __restrict__ G, float * __restrict__ gated) {
    const int bb = blockIdx.y, kvh = blockIdx.x;
    const int pos = pos_arr[bb];
    const int sl = slot[bb];
    const float * Kc = sl ? Kx + (size_t) (sl - 1) * slot_stride : Kc0;
    const float * Vc = sl ? Vx + (size_t) (sl - 1) * slot_stride : Vc0;
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
    const float inv_sqrt_d = 0.08838834764831845f;
    const float4 q0 = reinterpret_cast<const float4 *>(q[0])[lane], q1 = reinterpret_cast<const float4 *>(q[1])[lane],
                 q2 = reinterpret_cast<const float4 *>(q[2])[lane];
    for (int j = j0 + warp; j <= pos; j += 16) {
        const float4 kv = __ldg(reinterpret_cast<const float4 *>(Kc + ((size_t) j * 8 + kvh) * 128) + lane);
        float d0 = q0.x * kv.x + q0.y * kv.y + q0.z * kv.z + q0.w * kv.w;
        float d1 = q1.x * kv.x + q1.y * kv.y + q1.z * kv.z + q1.w * kv.w;
        float d2 = q2.x * kv.x + q2.y * kv.y + q2.z * kv.z + q2.w * kv.w;
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
    int i = dg;
    for (; i + 12 < nk; i += 16) {
        float v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = Vc[((size_t) (j0 + i + 4 * u) * 8 + kvh) * 128 + d];
#pragma unroll
        for (int u = 0; u < 4; ++u) { o0 += sc[0][i + 4 * u] * v[u]; o1 += sc[1][i + 4 * u] * v[u]; o2 += sc[2][i + 4 * u] * v[u]; }
    }
    for (; i < nk; i += 4) {
        const float v = Vc[((size_t) (j0 + i) * 8 + kvh) * 128 + d];
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
