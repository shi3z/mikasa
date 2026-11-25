// Unit test of pf_gemm.cuh against naive references (synthetic weights in the real interleaved layout).
#include <cstdio>
#include <cstdlib>
#include "batch_kernels2.cuh"
#include "pf_gemm.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

static float e2m1d[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float ue4m3_h(int x) {
    if (x == 0 || x == 0x7F) return 0.f;
    int e = (x >> 3) & 0xF, m = x & 7;
    float raw = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.f + m / 8.f, e - 7);
    return raw * 0.5f;
}
struct Host { std::vector<int8_t> q8; std::vector<uint16_t> s8; std::vector<uint8_t> q4, s4; };

// reference value of weight (row n, k)
static float wval(const Host & h, int fmt, int n, int k, int K) {
    if (fmt == PF_Q8) {
        float d = __half2float(*reinterpret_cast<const __half *>(&h.s8[(size_t) n * (K / 32) + k / 32]));
        return d * h.q8[(size_t) n * K + k];
    }
    int blk = k / 64, kk = k % 64, sub = kk / 16, k2 = kk % 16, i = k2 % 8;
    uint8_t b = h.q4[(size_t) n * (K / 2) + 32 * blk + sub * 8 + i];
    int nib = k2 < 8 ? (b & 15) : (b >> 4);
    return e2m1d[nib] * ue4m3_h(h.s4[(size_t) n * (K / 16) + k / 16]);
}
static void gen(Host & h, int fmt, size_t rows, int K, std::mt19937 & rng) {
    if (fmt == PF_Q8) {
        h.q8.resize(rows * K); h.s8.resize(rows * (K / 32));
        for (auto & v : h.q8) v = (int8_t) ((int) (rng() % 255) - 127);
        for (auto & v : h.s8) { float f = 0.002f * (1 + (rng() % 100) / 50.f); __half hh = __float2half(f); v = *reinterpret_cast<uint16_t *>(&hh); }
    } else {
        h.q4.resize(rows * (K / 2)); h.s4.resize(rows * (K / 16));
        for (auto & v : h.q4) v = rng() & 255;
        for (auto & v : h.s4) { int e = 3 + rng() % 8, m = rng() % 8; v = (e << 3) | m; }
    }
}
template <class T> static T * up(const std::vector<T> & v) { T * d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }

int main(int argc, char ** argv) {
    std::mt19937 rng(1);
    const int K = 2304, N = 256, T = 150, NE = 16;
    int bad = 0;
    for (int fmt = 0; fmt < 2; ++fmt) {
        // ---- dense: two segments (N rows each) sharing x, with xoff
        Host h[2];
        gen(h[0], fmt, N, K, rng); gen(h[1], fmt, N, K, rng);
        std::vector<__half> x((size_t) T * 2 * K);
        std::vector<float> xf(x.size());
        for (size_t i = 0; i < x.size(); ++i) { x[i] = __float2half(((int) (rng() % 2001) - 1000) / 500.f); xf[i] = __half2float(x[i]); }
        __half * dx = up(x);
        uint4 * qT[2]; uint8_t * sT[2];
        for (int s = 0; s < 2; ++s) {
            if (fmt == PF_Q8) { int8_t * a = up(h[s].q8); uint16_t * b = up(h[s].s8); register_il_ret(a, b, 1, N, K / 32, 2, qT[s], sT[s]); }
            else { uint8_t * a = up(h[s].q4); uint8_t * b = up(h[s].s4); register_il_ret(a, b, 1, N, K / 64, 4, qT[s], sT[s]); }
        }
        CK(cudaDeviceSynchronize());
        float * dy[2]; for (int s = 0; s < 2; ++s) CK(cudaMalloc(&dy[s], (size_t) T * N * 4));
        PFDense pd; pd.nseg = 2; pd.x = dx; pd.xstride = 2 * K; pd.T = T; pd.K = K;
        for (int s = 0; s < 2; ++s) { pd.seg[s] = {(const uint32_t *) qT[s], sT[s], s * (N / 128), N, dy[s], N, s * K}; }
        const int TB = 64;
        size_t sm = (size_t) PF_NST * TB * PF_BK * 2;
        dim3 grid(2 * (N / 128), (T + TB - 1) / TB);
        if (fmt == PF_Q8) pf_gemm_dense<PF_Q8, TB><<<grid, 128, sm>>>(pd); else pf_gemm_dense<PF_NV4, TB><<<grid, 128, sm>>>(pd);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        for (int s = 0; s < 2; ++s) {
            std::vector<float> y((size_t) T * N);
            CK(cudaMemcpy(y.data(), dy[s], y.size() * 4, cudaMemcpyDeviceToHost));
            double md = 0, mx = 0;
            for (int t = 0; t < T; ++t) for (int n = 0; n < N; ++n) {
                double ref = 0;
                for (int k = 0; k < K; ++k) ref += (double) wval(h[s], fmt, n, k, K) * xf[(size_t) t * 2 * K + s * K + k];
                md = std::max(md, fabs(ref - y[(size_t) t * N + n])); mx = std::max(mx, fabs(ref));
            }
            printf("dense fmt=%d seg=%d: max abs diff %.3e (max |ref| %.3e, rel %.2e)\n", fmt, s, md, mx, md / mx);
            if (md / mx > 1e-5) ++bad;
        }
        // ---- moe: NE experts, ids random for T tokens x 8
        Host hm; gen(hm, fmt, (size_t) NE * N, K, rng);
        uint4 * mq; uint8_t * ms;
        if (fmt == PF_Q8) { int8_t * a = up(hm.q8); uint16_t * b = up(hm.s8); register_il_ret(a, b, NE, N, K / 32, 2, mq, ms); }
        else { uint8_t * a = up(hm.q4); uint8_t * b = up(hm.s4); register_il_ret(a, b, NE, N, K / 64, 4, mq, ms); }
        std::vector<int> ids(T * 8);
        for (int t = 0; t < T; ++t) for (int j = 0; j < 8; ++j) { int e; bool dup; do { e = (int) (rng() % NE); if (t % 7 == 0) e = j; dup = false; for (int u = 0; u < j; ++u) dup |= ids[t * 8 + u] == e; } while (dup); ids[t * 8 + j] = e; }
        int *dids, *perm, *inv, *rmap, *sexp, *te, *tp, *tc, *nt;
        CK(cudaMalloc(&dids, T * 8 * 4)); CK(cudaMemcpy(dids, ids.data(), T * 8 * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&perm, T * 32)); CK(cudaMalloc(&inv, T * 32)); CK(cudaMalloc(&rmap, T * 32)); CK(cudaMalloc(&sexp, T * 32));
        CK(cudaMalloc(&te, 4096 * 4)); CK(cudaMalloc(&tp, 4096 * 4)); CK(cudaMalloc(&tc, 4096 * 4)); CK(cudaMalloc(&nt, 4));
        const int TBM = 32;
        pf_group<TBM><<<1, 1024>>>(dids, T * 8, perm, inv, rmap, sexp, te, tp, tc, nt);
        float * dym; CK(cudaMalloc(&dym, (size_t) T * 8 * N * 4));
        size_t smm = (size_t) PF_NST * TBM * PF_BK * 2;
        dim3 gm(N / 128, T * 8 / TBM + NE);
        if (fmt == PF_Q8) pf_gemm_moe<PF_Q8, TBM><<<gm, 128, smm>>>((const uint32_t *) mq, ms, N, K, dx, 2 * K, rmap, dym, te, tp, tc, nt);
        else pf_gemm_moe<PF_NV4, TBM><<<gm, 128, smm>>>((const uint32_t *) mq, ms, N, K, dx, 2 * K, rmap, dym, te, tp, tc, nt);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> ym((size_t) T * 8 * N); std::vector<int> hperm(T * 8), hinv(T * 8);
        CK(cudaMemcpy(ym.data(), dym, ym.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hinv.data(), inv, T * 32, cudaMemcpyDeviceToHost));
        double md = 0, mx = 0;
        for (int a = 0; a < T * 8; a += 3) {
            int t = a >> 3, e = ids[a], p = hinv[a];
            for (int n = 0; n < N; n += 5) {
                double ref = 0;
                for (int k = 0; k < K; ++k) ref += (double) wval(hm, fmt, e * N + n, k, K) * xf[(size_t) t * 2 * K + k];
                md = std::max(md, fabs(ref - ym[(size_t) p * N + n])); mx = std::max(mx, fabs(ref));
            }
        }
        printf("moe fmt=%d: max abs diff %.3e (max |ref| %.3e, rel %.2e)\n", fmt, md, mx, md / mx);
        if (md / mx > 1e-5) ++bad;
    }
    printf(bad ? "FAILED\n" : "ALL OK\n");
    return bad;
}
