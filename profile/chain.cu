#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
constexpr int E = 2304, NEXP = 128, FFN = 1152;
#include "kc.cuh"
#include "variants.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    size_t qg = (size_t) NEXP * 2304 * 1152, sg = (size_t) NEXP * 2304 * 144, qd = (size_t) NEXP * 2304 * 576, sd = (size_t) NEXP * 2304 * 72;
    uint8_t *gq, *gs, *dq, *ds; float *x, *gu, *h, *dn, *ones; int * ids;
    CK(cudaMalloc(&gq, qg)); CK(cudaMalloc(&gs, sg)); CK(cudaMalloc(&dq, qd)); CK(cudaMalloc(&ds, sd));
    std::vector<uint8_t> hb(1 << 24); for (auto & v : hb) v = (uint8_t) (rand() & 0x77);
    { std::pair<uint8_t *, size_t> parts[4] = {{gq, qg}, {gs, sg}, {dq, qd}, {ds, sd}}; for (auto & pr : parts) for (size_t o = 0; o < pr.second; o += hb.size()) CK(cudaMemcpy(pr.first + o, hb.data(), std::min(hb.size(), pr.second - o), cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&x, 8 * E * 4)); CK(cudaMalloc(&gu, 8 * 2304 * 4)); CK(cudaMalloc(&h, 8 * FFN * 4)); CK(cudaMalloc(&dn, 8 * E * 4)); CK(cudaMalloc(&ones, NEXP * 4)); CK(cudaMalloc(&ids, 16 * 64 * 4));
    { std::vector<float> hx(8 * E, 0.1f), o1(NEXP, 1.f); CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ones, o1.data(), NEXP * 4, cudaMemcpyHostToDevice)); }
    std::vector<int> hid(16 * 64); for (int r = 0; r < 16; ++r) for (int i = 0; i < 64; ++i) hid[r * 64 + i] = (r * 8 + i) % NEXP; CK(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice));
    const size_t sm1 = (size_t) (2304 / 64) * 68 * 4, sm2 = (size_t) (1152 / 64) * 68 * 4;
    auto kg = [&](int i) { nv_v4<72, 1><<<dim3(144, 8), 128, sm1, s>>>(gq, gs, x, E, gu, 2304, 2304, ids + (i % 16) * 64); };
    auto ks = [&](int) { k_situ<<<dim3((FFN + 255) / 256, 8), 256, 0, s>>>(gu, ones, ids, h, 4.0f, 25.0f); };
    auto kd = [&](int i) { nv_v4<36, 1><<<dim3(144, 8), 128, sm2, s>>>(dq, ds, h, FFN, dn, 1152, 2304, ids + (i % 16) * 64); };
    auto bench = [&](const char * nm, auto body) { cudaGraph_t g; cudaGraphExec_t ge; cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal); for (int i = 0; i < 44; ++i) body(i); cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0);
        for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, s); cudaStreamSynchronize(s); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a, s); for (int i = 0; i < 40; ++i) cudaGraphLaunch(ge, s); cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
        printf("%-34s %8.1f us per token (44 layers) = %6.2f us per layer\n", nm, ms * 1000 / 40, ms * 1000 / 40 / 44); };
    bench("gate_up only", [&](int i) { kg(i); }); bench("situ only", [&](int i) { ks(i); }); bench("down only", [&](int i) { kd(i); });
    bench("chain gate_up+situ+down", [&](int i) { kg(i); ks(i); kd(i); });
    printf("bytes per layer: %.1f MB -> at 704 GB/s %.1f us, at 768 GB/s %.1f us\n", (23.89 + 11.94), 35.83e6 / 704e9 * 1e6, 35.83e6 / 768e9 * 1e6);
}
