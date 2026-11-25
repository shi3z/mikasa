// Bandwidth sweep of the NVFP4 expert GEMV at the real Inkling shapes (random weights, L2 never reused: 128 experts rotate).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
constexpr int E = 2304, NEXP = 128;
#include "kc.cuh"
#include "variants.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

struct Case { const char * name; int K, N; };
static float timeit(cudaStream_t s, int reps, auto fn) {
    for (int i = 0; i < 8; ++i) fn(i);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a, s);
    for (int i = 0; i < reps; ++i) fn(i);
    cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.f / reps;
}
int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    for (Case c : { Case{"gate_up", 2304, 2304}, Case{"down", 1152, 2304} }) {
        size_t qb = (size_t) NEXP * c.N * (c.K / 2), sb = (size_t) NEXP * c.N * (c.K / 16);
        uint8_t *qs, *sc; float *x, *y; int * ids;
        CK(cudaMalloc(&qs, qb)); CK(cudaMalloc(&sc, sb)); CK(cudaMalloc(&x, 64 * c.K * 4)); CK(cudaMalloc(&y, 64 * c.N * 4)); CK(cudaMalloc(&ids, 64 * 4 * 16));
        std::vector<uint8_t> h(1 << 24); for (auto & v : h) v = (uint8_t) (rand() & 0x77);   // plausible nibbles, scale bytes < 0x78
        for (size_t o = 0; o < qb; o += h.size()) CK(cudaMemcpy(qs + o, h.data(), std::min(h.size(), qb - o), cudaMemcpyHostToDevice));
        for (size_t o = 0; o < sb; o += h.size()) CK(cudaMemcpy(sc + o, h.data(), std::min(h.size(), sb - o), cudaMemcpyHostToDevice));
        { std::vector<float> hx(64 * c.K); for (auto & v : hx) v = rand() / (float) RAND_MAX - 0.5f; CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice)); }
        std::vector<int> hid(16 * 64); for (int r = 0; r < 16; ++r) for (int i = 0; i < 64; ++i) hid[r * 64 + i] = (r * 8 + i) % NEXP;   // rotating expert sets
        CK(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice));
        const size_t smem = (size_t) (c.K / 64) * 68 * 4;
        printf("== %s (K=%d N=%d)\n", c.name, c.K, c.N);
        // reference output of the baseline for the correctness check
        std::vector<float> ref(8 * c.N), got(8 * c.N);
        { dim3 g(c.N / 16, 8); if (c.K == 2304) nvfp4_gemv_pack4<72><<<g, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, ids); else nvfp4_gemv_pack4<36><<<g, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, ids);
          cudaStreamSynchronize(s); cudaMemcpy(ref.data(), y, ref.size() * 4, cudaMemcpyDeviceToHost); }
        auto check = [&](const char * nm) { cudaStreamSynchronize(s); cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost); double md = 0, mr = 0; for (size_t i = 0; i < ref.size(); ++i) { md = std::max(md, (double) fabsf(got[i] - ref[i])); mr = std::max(mr, (double) fabsf(ref[i])); } printf("    [%s] max|diff|/max|ref| = %.2e\n", nm, md / mr); };
        auto runv = [&](const char * nm, auto launch) {
            cudaMemset(y, 0, 8 * c.N * 4); launch(0, 8); check(nm);
            for (int slots : {8}) { float us = timeit(s, 300, [&](int i) { launch(i, slots); }); double mb = (double) slots * c.N * c.K * 0.5625 / 1e6;
                printf("  %-26s slots=%2d: %7.2f us  %5.0f GB/s (%2.0f%%)\n", nm, slots, us, mb / us * 1e3, 100 * mb / us * 1e3 / 768); } };
        for (bool st : {false, true}) for (int wpb : {4, 2, 8}) {
            char nm[64]; snprintf(nm, 64, "v1 %s wpb=%d", st ? "stream" : "ldg   ", wpb);
            runv(nm, [&](int i, int slots) { const int * idp = ids + (i % 16) * 64; dim3 g((c.N / 4 + wpb - 1) / wpb / 1 / 1, slots); g.x = (c.N / 4 + wpb - 1) / wpb;
                if (c.K == 2304) { if (st) { if (wpb == 4) nv_v1<72, true, 4><<<g, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else if (wpb == 2) nv_v1<72, true, 2><<<g, 64, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else nv_v1<72, true, 8><<<g, 256, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); }
                                   else { if (wpb == 4) nv_v1<72, false, 4><<<g, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else if (wpb == 2) nv_v1<72, false, 2><<<g, 64, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else nv_v1<72, false, 8><<<g, 256, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); } }
                else { if (st) { if (wpb == 4) nv_v1<36, true, 4><<<g, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else if (wpb == 2) nv_v1<36, true, 2><<<g, 64, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else nv_v1<36, true, 8><<<g, 256, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); }
                       else { if (wpb == 4) nv_v1<36, false, 4><<<g, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else if (wpb == 2) nv_v1<36, false, 2><<<g, 64, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else nv_v1<36, false, 8><<<g, 256, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); } } });
        }
        for (int slots : {8, 16, 32, 64}) {
            float us = timeit(s, 200, [&](int i) { dim3 grid(c.N / 16, slots); const int * idp = ids + (i % 16) * 64;
                if (c.K == 2304) nvfp4_gemv_pack4<72><<<grid, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); else nvfp4_gemv_pack4<36><<<grid, 128, smem, s>>>(qs, sc, x, c.K, y, c.K, c.N, idp); });
            double mb = (double) slots * c.N * c.K * 0.5625 / 1e6;
            printf("  baseline slots=%2d: %7.2f us  %6.1f MB  %5.0f GB/s (%2.0f%% of 768)\n", slots, us, mb, mb / us * 1e3, 100 * mb / us * 1e3 / 768);
        }
    }
}
