// (a) latency of a hand-rolled grid barrier with all blocks resident; (b) the 3-kernel chain gate_up -> k_situ -> down inside a CUDA graph vs the sum of the parts.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
constexpr int E = 2304, NEXP = 128;
#include "kc.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
__device__ unsigned int g_bar[2];
__global__ void k_bar(int rounds, float * out) {                 // `rounds` consecutive grid barriers (sense via monotonically growing target)
    __shared__ unsigned int tgt0;
    for (int r = 0; r < rounds; ++r) {
        __syncthreads();
        if (threadIdx.x == 0) { __threadfence(); const unsigned int t = (r + 1) * gridDim.x; atomicAdd(&g_bar[0], 1u); while (*((volatile unsigned int *) &g_bar[0]) < t) { } __threadfence(); }
        __syncthreads();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = 1.f;
}
int main() {
    cudaStream_t s; cudaStreamCreate(&s); float * d; cudaMalloc(&d, 4);
    for (int blocks : {84 * 4, 84 * 9, 84 * 12}) {
        for (int rounds : {1, 101}) { cudaMemset(g_bar, 0, 0); unsigned int z[2] = {0, 0}; cudaMemcpyToSymbol(g_bar, z, 8);
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a, s); k_bar<<<blocks, 128, 0, s>>>(rounds, d); cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
            static float t1 = 0; if (rounds == 1) t1 = ms; else printf("blocks=%d: %.2f us per barrier (100 barriers: %.1f us, 1 barrier incl. launch: %.1f us)\n", blocks, (ms - t1) * 1000 / 100, ms * 1000, t1 * 1000); }
    }
}
