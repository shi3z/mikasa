// Micro-benchmark of the exact router kernels of mikasa (gemv_f32_row9 + k_router_top8) in a CUDA graph, 48 applications per "token".
#include <cstdio>
#include <vector>
#include <cstdlib>
#include "inkling_common_wrap.h"
#include "fused.cuh"
int main() {
    const int LAYERS = 24, APP = 48;
    std::vector<float *> W(LAYERS); float *x, *lg, *bias, *wts; int *ids;
    for (auto & w : W) { cudaMalloc(&w, (size_t) NEXP * E * 4); std::vector<float> h((size_t) NEXP * E); for (auto & v : h) v = (rand() / (float) RAND_MAX - 0.5f) * 0.05f; cudaMemcpy(w, h.data(), h.size() * 4, cudaMemcpyHostToDevice); }
    cudaMalloc(&x, E * 4); cudaMalloc(&lg, NEXP * 4); cudaMalloc(&bias, NEXP * 4); cudaMalloc(&wts, 8 * 4); cudaMalloc(&ids, 8 * 4);
    { std::vector<float> h(E, 0.1f); cudaMemcpy(x, h.data(), E * 4, cudaMemcpyHostToDevice); cudaMemset(bias, 0, NEXP * 4); }
    cudaStream_t s; cudaStreamCreate(&s);
    auto runf = [&](int apps) { cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < apps; ++i) k_router_fused<<<NEXP, 256, 0, s>>>(W[i % LAYERS], x, bias, lg, ids, wts, 8.0f);
        cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0); for (int i = 0; i < 20; ++i) cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); const int R = 300; cudaEventRecord(a, s); for (int i = 0; i < R; ++i) cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms / R; };
    auto run = [&](int apps, bool gemv, bool top) { cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < apps; ++i) { if (gemv) gemv_f32_row9<<<NEXP, 256, 0, s>>>(W[i % LAYERS], x, lg, E, NEXP); if (top) k_router_top8<<<1, NEXP, 0, s>>>(lg, bias, ids, wts, 8.0f); }
        cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0);
        for (int i = 0; i < 20; ++i) cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); const int R = 300; cudaEventRecord(a, s);
        for (int i = 0; i < R; ++i) cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms / R; };
    // correctness: compare fused vs original on random inputs
    { int bad = 0; std::vector<float> hx(E), hb(NEXP); int *ids2; float *wts2; cudaMalloc(&ids2, 32); cudaMalloc(&wts2, 32);
      for (int trial = 0; trial < 300; ++trial) { for (auto & v : hx) v = (rand() / (float) RAND_MAX - 0.5f); for (auto & v : hb) v = (rand() / (float) RAND_MAX - 0.5f) * 0.1f;
        cudaMemcpy(x, hx.data(), E * 4, cudaMemcpyHostToDevice); cudaMemcpy(bias, hb.data(), NEXP * 4, cudaMemcpyHostToDevice);
        gemv_f32_row9<<<NEXP, 256, 0, s>>>(W[trial % LAYERS], x, lg, E, NEXP); k_router_top8<<<1, NEXP, 0, s>>>(lg, bias, ids, wts, 8.0f);
        k_router_fused<<<NEXP, 256, 0, s>>>(W[trial % LAYERS], x, bias, lg, ids2, wts2, 8.0f); cudaStreamSynchronize(s);
        int a[8], b[8]; float wa[8], wb[8]; cudaMemcpy(a, ids, 32, cudaMemcpyDeviceToHost); cudaMemcpy(b, ids2, 32, cudaMemcpyDeviceToHost); cudaMemcpy(wa, wts, 32, cudaMemcpyDeviceToHost); cudaMemcpy(wb, wts2, 32, cudaMemcpyDeviceToHost);
        for (int k = 0; k < 8; ++k) if (a[k] != b[k] || fabsf(wa[k] - wb[k]) > 1e-5f) { ++bad; break; } }
      printf("fused vs original: %d / 300 mismatching trials\n", bad); cudaMemcpy(x, std::vector<float>(E, 0.1f).data(), E * 4, cudaMemcpyHostToDevice); cudaMemset(bias, 0, NEXP * 4); }
    float full = run(APP, true, true), fused = runf(APP), g_only = run(APP, true, false), t_only = run(APP, false, true);
    printf("fused router: %.1f us per token (%.2f us per application)\n", fused * 1000, fused * 1000 / APP);
    printf("per token (48 router applications, CUDA graph replay):\n  gemv+top8: %.1f us   gemv only: %.1f us   top8 only: %.1f us\n  per application: %.2f us\n", full * 1000, g_only * 1000, t_only * 1000, full * 1000 / APP);
}
