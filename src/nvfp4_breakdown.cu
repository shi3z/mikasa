// Baseline timing of the production NVFP4 expert GEMV (fused over 8 experts), per shape. Graph-replayed.
#include "kernels_common.cuh"
#include <cstdio>
#include <vector>
#include <random>
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
template <typename F> static float graph_us(F f, cudaStream_t s, int R = 100) {
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < R; ++i) f(s);
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s); for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.f / (3.f * R);
}
int main() {
    const int NE = 128, E = 8;
    struct Cfg { const char * name; int K, N; } cfgs[] = {{"gate_up K2304 N2304", 2304, 2304}, {"down    K1152 N2304", 1152, 2304}};
    cudaStream_t s; cudaStreamCreate(&s);
    std::mt19937 rng(3); std::uniform_int_distribution<int> qd(0, 255);
    for (auto & c : cfgs) {
        const size_t qbytes = (size_t) NE * c.N * (c.K / 2), sbytes = (size_t) NE * c.N * (c.K / 16);
        std::vector<uint8_t> qs(qbytes), sc(sbytes);
        for (auto & v : qs) v = qd(rng);
        for (auto & v : sc) v = 0x38;  // a valid UE4M3 scale
        std::vector<float> x((size_t) c.K);
        for (auto & v : x) v = (float) (qd(rng) - 128) / 128.f;
        std::vector<int> ids = {3, 17, 42, 60, 81, 99, 101, 127};
        uint8_t *dq, *ds; float *dx, *dy; int *di;
        cudaMalloc(&dq, qbytes); cudaMalloc(&ds, sbytes); cudaMalloc(&dx, c.K * 4); cudaMalloc(&dy, E * c.N * 4); cudaMalloc(&di, E * 4);
        cudaMemcpy(dq, qs.data(), qbytes, cudaMemcpyHostToDevice); cudaMemcpy(ds, sc.data(), sbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), c.K * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, ids.data(), E * 4, cudaMemcpyHostToDevice);
        float us = graph_us([&](cudaStream_t st) {
            launch_nvfp4_fused(dq, ds, dx, c.K == 2304 ? 0 : c.K, dy, c.K, c.N, di, E, st); }, s);
        double bytes = (double) E * c.N * c.K * 36.0 / 64.0;
        std::printf("%-22s 8 experts: %8.2f us  %7.1f GB/s  (weights %.2f MB)\n", c.name, us, bytes / (us * 1e-6) / 1e9, bytes / 1e6);
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dy); cudaFree(di);
    }
    return 0;
}
