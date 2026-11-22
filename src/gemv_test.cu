// GEMV check on real weights: y = W x on the GPU, then compared with numpy in gemv_check.py.
// usage: gemv_test <model.gguf> <tensor-name> <K> <N> <expert-index or -1> <x.bin> <y.bin> [reps]
#include "gguf.hpp"
#include "gemv.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); std::exit(1); } } while (0)

int main(int argc, char ** argv) {
    if (argc < 8) { std::fprintf(stderr, "see header\n"); return 2; }
    const std::string path = argv[1], name = argv[2];
    const int K = std::atoi(argv[3]), N = std::atoi(argv[4]), expert = std::atoi(argv[5]);
    const std::string xpath = argv[6], ypath = argv[7];
    const int reps = argc > 8 ? std::atoi(argv[8]) : 50;

    gguf::File f = gguf::open(path);
    const gguf::Tensor * t = nullptr;
    for (const auto & tt : f.tensors) if (tt.name == name) t = &tt;
    if (!t) { std::fprintf(stderr, "tensor not found\n"); return 1; }
    const uint32_t type = t->type;
    const uint64_t blck = gguf::type_info(type).blck, bytes = gguf::type_info(type).bytes;
    // per expert: N rows of K/blck blocks
    const uint64_t per_expert = (uint64_t) N * (K / blck) * bytes;
    const uint8_t * host = f.base + f.data_start + t->offset + (expert > 0 ? (uint64_t) expert * per_expert : 0);

    uint8_t * dW = nullptr;
    float * dx = nullptr, * dy = nullptr;
    CHECK(cudaMalloc(&dW, per_expert));
    CHECK(cudaMemcpy(dW, host, per_expert, cudaMemcpyHostToDevice));

    std::vector<float> x(K);
    FILE * fx = std::fopen(xpath.c_str(), "rb");
    if (!fx || std::fread(x.data(), sizeof(float), K, fx) != (size_t) K) { std::fprintf(stderr, "x read\n"); return 1; }
    std::fclose(fx);
    CHECK(cudaMalloc(&dx, K * sizeof(float)));
    CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CHECK(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    auto run = [&]() {
        if (type == gguf::Q8_0) launch_gemv_q8_0(dW, dx, dy, K, N, s);
        else if (type == gguf::NVFP4) launch_gemv_nvfp4(dW, dx, dy, K, N, s);
        else std::fprintf(stderr, "type not handled\n");
    };
    run();
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaGetLastError());

    std::vector<float> y(N);
    CHECK(cudaMemcpy(y.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
    FILE * fy = std::fopen(ypath.c_str(), "wb");
    std::fwrite(y.data(), sizeof(float), N, fy);
    std::fclose(fy);

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0, s));
    for (int i = 0; i < reps; ++i) run();
    CHECK(cudaEventRecord(e1, s));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = ms * 1000.0 / reps;
    const double gbs = per_expert / (us * 1e-6) / 1e9;
    std::printf("%s K=%d N=%d expert=%d: %.2f us per call, weight %.2f MB, %.1f GB/s\n",
        name.c_str(), K, N, expert, us, per_expert / 1e6, gbs);

    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    gguf::close(f);
    return 0;
}
