// Per-shape timing of the F32 GEMV calls in the forward pass (router vs loop control), graph-replayed.
#include "kernels_common.cuh"
#include <cstdio>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
template <typename F> static float gus(F f, cudaStream_t s, int R = 200) {
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
    cudaStream_t s; cudaStreamCreate(&s);
    float *W, *x, *y; cudaMalloc(&W, 4608 * 2304 * 4); cudaMalloc(&x, 4608 * 4); cudaMalloc(&y, 4096 * 4);
    cudaMemset(W, 0, 4608 * 2304 * 4); cudaMemset(x, 0, 4608 * 4);
    struct C { const char * name; int K, N, calls; };
    C cs[] = {{"router  K2304 N128 ", 2304, 128, 48}, {"ctrl_down K4608 N64", 4608, 64, 96},
              {"ctrl_gain K64 N2304 ", 64, 2304, 96}, {"ctrl_forget K64 N2304", 64, 2304, 96}};
    for (auto & c : cs) {
        float us = gus([&](cudaStream_t st) { gemv_f32<<<(c.N * 32 + 255) / 256, 256, 0, st>>>(W, x, y, c.K, c.N); }, s);
        std::printf("%s  %7.2f us/call  calls/token %3d  total %6.1f us/token\n", c.name, us, c.calls, us * c.calls);
    }
    {
        float us = gus([&](cudaStream_t st) { gemv_f32_row8<<<64, 256, 0, st>>>(W, x, y, 4608); }, s);
        std::printf("ctrl_down row8  K4608 N64   %7.2f us/call (current warp/row: 5.32)\n", us);
        float us2 = gus([&](cudaStream_t st) { gemv_f32_row8<<<128, 256, 0, st>>>(W, x, y, 2304); }, s);
        std::printf("router    row8  K2304 N128  %7.2f us/call (current warp/row: 3.56)\n", us2);
    }
    return 0;
}
