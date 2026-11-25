#include <cstdio>
#include <cstdlib>
#include "batch_kernels2.cuh"
#include "pf_gemm.cuh"
#include <vector>
#include <random>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
template <class F> static float timeit(F f) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) f();
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) { cudaEventRecord(a); for (int i = 0; i < 10; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms / 10); }
    return best;
}
int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 256;
    std::mt19937 rng(1);
    const int K = 2304, NE = 128, N = 2304;
    // random bytes weights directly in interleaved layout (content irrelevant for timing)
    const size_t rows = (size_t) NE * N;
    uint8_t *q4, *s4; int8_t* q8; __half* s8;
    CK(cudaMalloc(&q4, rows * K / 2)); CK(cudaMalloc(&s4, rows * K / 16)); CK(cudaMalloc(&q8, rows * K)); CK(cudaMalloc(&s8, rows * K / 32 * 2));
    CK(cudaMemset(q4, 0x37, rows * K / 2)); CK(cudaMemset(s4, 0x30, rows * K / 16)); CK(cudaMemset(q8, 3, rows * K)); CK(cudaMemset(s8, 0x1c, rows * K / 16));
    __half * x; CK(cudaMalloc(&x, (size_t) T * 3 * K * 2)); CK(cudaMemset(x, 0x3c, (size_t) T * 3 * K * 2));
    float * y; CK(cudaMalloc(&y, (size_t) T * 8 * 6528 * 4));
    int *ids, *perm, *inv, *rmap, *sexp, *te, *tp, *tc, *nt;
    std::vector<int> h(T * 8);
    for (int t = 0; t < T; ++t) for (int j = 0; j < 8; ++j) { int e; bool d; do { e = rng() % NE; d = false; for (int u = 0; u < j; ++u) d |= h[t * 8 + u] == e; } while (d); h[t * 8 + j] = e; }
    CK(cudaMalloc(&ids, T * 32)); CK(cudaMemcpy(ids, h.data(), T * 32, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&perm, T * 32)); CK(cudaMalloc(&inv, T * 32)); CK(cudaMalloc(&rmap, T * 32)); CK(cudaMalloc(&sexp, T * 32));
    CK(cudaMalloc(&te, 4096 * 4)); CK(cudaMalloc(&tp, 4096 * 4)); CK(cudaMalloc(&tc, 4096 * 4)); CK(cudaMalloc(&nt, 4));
    // dense: fused 6528 rows Q8
    PFDense pd; pd.nseg = 1; pd.x = x; pd.xstride = 3 * K; pd.T = T; pd.K = K;
    pd.seg[0] = {(const uint32_t *) q8, s8, 0, 6528, y, 6528, 0};
    auto dense = [&](auto tbc) {
        constexpr int TB = decltype(tbc)::value;
        dim3 g(6528 / 128, (T + TB - 1) / TB);
        pf_gemm_dense<PF_Q8, TB><<<g, 128, PF_NST * TB * PF_BK * 2>>>(pd);
    };
    float ms = timeit([&] { dense(std::integral_constant<int, 64>()); });
    printf("dense Q8 TB=64 T=%d N=6528: %.3f ms  %.1f TFLOP/s\n", T, ms, 2.0 * T * 6528 * K / ms / 1e9);
    ms = timeit([&] { dense(std::integral_constant<int, 128>()); });
    printf("dense Q8 TB=128 T=%d N=6528: %.3f ms  %.1f TFLOP/s\n", T, ms, 2.0 * T * 6528 * K / ms / 1e9);
    auto moe = [&](auto tbc, int fmt, int Kk, int Nn) {
        constexpr int TB = decltype(tbc)::value;
        dim3 g(Nn / 128, T * 8 / TB + 128);
        size_t sm = (size_t) PF_NST * TB * PF_BK * 2;
        if (fmt == 0) pf_gemm_moe<PF_Q8, TB><<<g, 128, sm>>>((const uint32_t *) q8, s8, Nn, Kk, x, 3 * K, rmap, y, te, tp, tc, nt);
        else pf_gemm_moe<PF_NV4, TB><<<g, 128, sm>>>((const uint32_t *) q4, s4, Nn, Kk, x, 3 * K, rmap, y, te, tp, tc, nt);
    };
    pf_group<16><<<1, 1024>>>(ids, T * 8, perm, inv, rmap, sexp, te, tp, tc, nt);
    ms = timeit([&] { moe(std::integral_constant<int, 16>(), 1, K, N); });
    printf("moe NV4 TB=16 gate_up T=%d: %.3f ms  %.1f TFLOP/s\n", T, ms, 2.0 * T * 8 * N * K / ms / 1e9);
    pf_group<32><<<1, 1024>>>(ids, T * 8, perm, inv, rmap, sexp, te, tp, tc, nt);
    ms = timeit([&] { moe(std::integral_constant<int, 32>(), 1, K, N); });
    printf("moe NV4 TB=32 gate_up T=%d: %.3f ms  %.1f TFLOP/s\n", T, ms, 2.0 * T * 8 * N * K / ms / 1e9);
    pf_group<64><<<1, 1024>>>(ids, T * 8, perm, inv, rmap, sexp, te, tp, tc, nt);
    ms = timeit([&] { moe(std::integral_constant<int, 64>(), 1, K, N); });
    printf("moe NV4 TB=64 gate_up T=%d: %.3f ms  %.1f TFLOP/s\n", T, ms, 2.0 * T * 8 * N * K / ms / 1e9);
    ms = timeit([&] { moe(std::integral_constant<int, 64>(), 1, 1152, N); });
    printf("moe NV4 TB=64 down T=%d: %.3f ms  %.1f TFLOP/s\n", T, ms, 2.0 * T * 8 * N * 1152 / ms / 1e9);
    ms = timeit([&] { moe(std::integral_constant<int, 64>(), 0, K, N); });
    printf("moe Q8 TB=64 gate_up T=%d: %.3f ms  %.1f TFLOP/s\n", T, ms, 2.0 * T * 8 * N * K / ms / 1e9);
    return 0;
}
