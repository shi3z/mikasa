// Q8_0 GEMV shape benchmark: parallelization variants for the model's shapes (sm_86).
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -o build_q8_bench src/q8_bench.cu
// Run:   build_q8_bench [--ncu]
// Every variant computes y = W x with W in SoA (int8 [N][K], half scale [N][K/32]); results are checked
// against variant V0 (the production kernel). Times are device time per call from CUDA graphs.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

// ---------------------------------------------------------------- V0 (production) ----
// two rows per warp, x as float4, 8 blocks of x per iteration.
__global__ void v0_rows2(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                         float * __restrict__ y, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int row0 = warp * 2;
    if (row0 >= N) return;
    const int nb = K / 32;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc[2] = {0.0f, 0.0f};
    for (int b = lane; b < nb; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = __ldg(x4 + b * 8 + i);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
                const float d = __half2float(sc[(size_t) row * nb + b]);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float t = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = xv[i];
                    t += (float) (int8_t) w[i] * v.x + (float) (int8_t) (w[i] >> 8) * v.y
                       + (float) (int8_t) (w[i] >> 16) * v.z + (float) (int8_t) (w[i] >> 24) * v.w;
                }
                acc[r] += d * t;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = a;
    }
}

// ------------------------------------------------ V1: one row per warp, all loads of a lane issued first ----
// Each lane handles up to NBL blocks (b = lane + 32*i). Weights and scales are prefetched before the dot.
template <int NBL>
__global__ void v1_row_prefetch(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                                float * __restrict__ y, int K, int N) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= N) return;
    const int nb = K / 32;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
    const __half * s = sc + (size_t) row * nb;
    uint4 wa[NBL], wb[NBL];
    float d[NBL];
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int b = lane + 32 * i;
        if (b < nb) {
            wa[i] = __ldg(q + b * 2);
            wb[i] = __ldg(q + b * 2 + 1);
            d[i] = __half2float(s[b]);
        } else {
            wa[i] = make_uint4(0, 0, 0, 0);
            wb[i] = wa[i];
            d[i] = 0.0f;
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int b = lane + 32 * i;
        if (b < nb) {
            const float4 * xb = reinterpret_cast<const float4 *>(x + b * 32);
            const uint32_t w[8] = {wa[i].x, wa[i].y, wa[i].z, wa[i].w, wb[i].x, wb[i].y, wb[i].z, wb[i].w};
            float t = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 v = __ldg(xb + j);
                t += (float) (int8_t) w[j] * v.x + (float) (int8_t) (w[j] >> 8) * v.y
                   + (float) (int8_t) (w[j] >> 16) * v.z + (float) (int8_t) (w[j] >> 24) * v.w;
            }
            acc += d[i] * t;
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// --------------------------------- V2: WPR warps per row (block of 256 = 8 warps), smem reduction over warps ----
template <int WPR>
__global__ void v2_split_warps(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                               float * __restrict__ y, int K, int N) {
    __shared__ float red[8];
    constexpr int RPB = 8 / WPR;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rl = warp / WPR, part = warp % WPR;
    const int row = blockIdx.x * RPB + rl;
    const int nb = K / 32;
    float acc = 0.0f;
    if (row < N) {
        const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
        const __half * s = sc + (size_t) row * nb;
        for (int b = part * 32 + lane; b < nb; b += 32 * WPR) {
            const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
            const float d = __half2float(s[b]);
            const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float4 * xb = reinterpret_cast<const float4 *>(x + b * 32);
            float t = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 v = __ldg(xb + j);
                t += (float) (int8_t) w[j] * v.x + (float) (int8_t) (w[j] >> 8) * v.y
                   + (float) (int8_t) (w[j] >> 16) * v.z + (float) (int8_t) (w[j] >> 24) * v.w;
            }
            acc += d * t;
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (part == 0 && lane == 0 && row < N) {
        float t = 0.0f;
        for (int p = 0; p < WPR; ++p) t += red[rl * WPR + p];
        y[row] = t;
    }
}

// ---------------------------------------------- V3: split-K across blocks, deterministic partials + reduce ----
// grid = (ceil(N / 8), S): warp (row) handles blocks [s*nb/S, (s+1)*nb/S). partial[s][row].
template <int S>
__global__ void v3_splitk_part(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                               float * __restrict__ part, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int split = blockIdx.y;
    if (warp >= N) return;
    const int nb = K / 32;
    const int b0 = split * nb / S, b1 = (split + 1) * nb / S;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) warp * K);
    const __half * s = sc + (size_t) warp * nb;
    float acc = 0.0f;
    for (int b = b0 + lane; b < b1; b += 32) {
        const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
        const float d = __half2float(s[b]);
        const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float4 * xb = reinterpret_cast<const float4 *>(x + b * 32);
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 v = __ldg(xb + j);
            t += (float) (int8_t) w[j] * v.x + (float) (int8_t) (w[j] >> 8) * v.y
               + (float) (int8_t) (w[j] >> 16) * v.z + (float) (int8_t) (w[j] >> 24) * v.w;
        }
        acc += d * t;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) part[(size_t) split * N + warp] = acc;
}

template <int S>
__global__ void v3_reduce(const float * __restrict__ part, float * __restrict__ y, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float a = 0.0f;
    for (int s = 0; s < S; ++s) a += part[(size_t) s * N + i];
    y[i] = a;
}


// V8: one row per warp, plain loop unrolled by 2 (fine-grained blocks: N warps -> N/8 blocks).
__global__ void v8_row_unroll2(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                               float * __restrict__ y, int K, int N) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= N) return;
    const int nb = K / 32;
    const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
    const __half * s = sc + (size_t) row * nb;
    float acc = 0.0f;
#pragma unroll 2
    for (int b = lane; b < nb; b += 32) {
        const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
        const float d = __half2float(s[b]);
        const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float4 * xb = reinterpret_cast<const float4 *>(x + b * 32);
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 v = __ldg(xb + j);
            t += (float) (int8_t) w[j] * v.x + (float) (int8_t) (w[j] >> 8) * v.y
               + (float) (int8_t) (w[j] >> 16) * v.z + (float) (int8_t) (w[j] >> 24) * v.w;
        }
        acc += d * t;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// V9: V0 math with x staged once per block in shared memory (rows2, 256 threads).
__global__ void v9_rows2_smemx(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                               float * __restrict__ y, int K, int N) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int row0 = warp * 2;
    if (row0 >= N) return;
    const int nb = K / 32;
    const float4 * x4 = reinterpret_cast<const float4 *>(xs);
    float acc[2] = {0.0f, 0.0f};
    for (int b = lane; b < nb; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = x4[b * 8 + i];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
                const float d = __half2float(sc[(size_t) row * nb + b]);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float t = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = xv[i];
                    t += (float) (int8_t) w[i] * v.x + (float) (int8_t) (w[i] >> 8) * v.y
                       + (float) (int8_t) (w[i] >> 16) * v.z + (float) (int8_t) (w[i] >> 24) * v.w;
                }
                acc[r] += d * t;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = a;
    }
}


// Parametric V0: R rows per warp, one block = T threads. x as float4, weights as uint4 (SoA).
template <int R>
__global__ void q8_tpl(const int8_t * __restrict__ qs, const __half * __restrict__ sc, const float * __restrict__ x,
                       float * __restrict__ y, int K, int N) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int row0 = warp * R;
    if (row0 >= N) return;
    const int nb = K / 32;
    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float4 xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = __ldg(x4 + b * 8 + i);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < N) {
                const uint4 * q = reinterpret_cast<const uint4 *>(qs + (size_t) row * K);
                const float d = __half2float(sc[(size_t) row * nb + b]);
                const uint4 a0 = __ldg(q + b * 2), a1 = __ldg(q + b * 2 + 1);
                const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float t = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = xv[i];
                    t += (float) (int8_t) w[i] * v.x + (float) (int8_t) (w[i] >> 8) * v.y
                       + (float) (int8_t) (w[i] >> 16) * v.z + (float) (int8_t) (w[i] >> 24) * v.w;
                }
                acc[r] += d * t;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float a = acc[r];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = a;
    }
}

// ----------------------------------------------------------------- driver ----

struct Shape { const char * name; int K, N, calls; };

template <typename F>
static float graph_us(F launch, cudaStream_t s, int R = 200) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < R; ++i) launch(s);
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CHECK(cudaGraphLaunch(ge, s));
    CHECK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0, s));
    for (int i = 0; i < 3; ++i) CHECK(cudaGraphLaunch(ge, s));
    CHECK(cudaEventRecord(e1, s));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1000.0f / (3.0f * R);
}

template <typename K>
static int regs(K f) {
    cudaFuncAttributes a;
    CHECK(cudaFuncGetAttributes(&a, (const void *) f));
    return a.numRegs;
}

int main(int argc, char ** argv) {
    const bool ncu = argc > 1 && std::strcmp(argv[1], "--ncu") == 0;
    const Shape shapes[] = {
        {"wq/wgate  K2304 N3072", 2304, 3072, 96},
        {"wk/wv     K2304 N1024", 2304, 1024, 96},
        {"wrel      K2304 N384 ", 2304, 384, 48},
        {"wo        K3072 N2304", 3072, 2304, 48},
        {"head      K2304 N201024", 2304, 201024, 1},
    };
    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (const Shape & sh : shapes) {
        const int K = sh.K, N = sh.N, nb = K / 32;
        std::vector<int8_t> qs((size_t) N * K);
        std::vector<__half> sc((size_t) N * nb);
        std::vector<float> x(K);
        for (auto & v : qs) v = (int8_t) qd(rng);
        for (auto & v : sc) v = __float2half(0.01f * std::fabs(nd(rng)));
        for (auto & v : x) v = nd(rng);
        int8_t * dq; __half * ds; float * dx; float * dy; float * dp;
        CHECK(cudaMalloc(&dq, qs.size()));
        CHECK(cudaMalloc(&ds, sc.size() * sizeof(__half)));
        CHECK(cudaMalloc(&dx, K * sizeof(float)));
        CHECK(cudaMalloc(&dy, N * sizeof(float)));
        CHECK(cudaMalloc(&dp, 8ull * N * sizeof(float)));
        CHECK(cudaMemcpy(dq, qs.data(), qs.size(), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(ds, sc.data(), sc.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        const double bytes = (double) N * K * 34.0 / 32.0;

        // reference: V0
        std::vector<float> ref(N), out(N);
        v0_rows2<<<(N + 15) / 16, 256, 0, s>>>(dq, ds, dx, dy, K, N);
        CHECK(cudaStreamSynchronize(s));
        CHECK(cudaMemcpy(ref.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
        if (ncu) {
            v1_row_prefetch<3><<<(N * 32 + 255) / 256, 256, 0, s>>>(dq, ds, dx, dy, K, N);
            v2_split_warps<2><<<(N + 3) / 4, 256, 0, s>>>(dq, ds, dx, dy, K, N);
            v3_splitk_part<4><<<dim3((N * 32 + 255) / 256, 4), 256, 0, s>>>(dq, ds, dx, dp, K, N);
            CHECK(cudaStreamSynchronize(s));
            std::printf("ncu pass done for %s\n", sh.name);
            continue;
        }

        struct Row { const char * name; float us; int r; double err; };
        std::vector<Row> rows;
        auto err_of = [&](float * d) {
            CHECK(cudaStreamSynchronize(s));
            CHECK(cudaMemcpy(out.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost));
            double m = 0, mr = 0;
            for (int i = 0; i < N; ++i) { m = std::max(m, (double) std::fabs(out[i] - ref[i])); mr = std::max(mr, (double) std::fabs(ref[i])); }
            return m / (mr + 1e-30);
        };
        rows.push_back({"V0 rows2 (prod)", graph_us([&](cudaStream_t st) { v0_rows2<<<(N + 15) / 16, 256, 0, st>>>(dq, ds, dx, dy, K, N); }, s),
                        regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) v0_rows2), 0});
        rows.back().err = 0;
        rows.push_back({"V1 row/warp, NBL=3", graph_us([&](cudaStream_t st) { v1_row_prefetch<3><<<(N * 32 + 255) / 256, 256, 0, st>>>(dq, ds, dx, dy, K, N); }, s),
                        regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) v1_row_prefetch<3>), 0});
        rows.back().err = (v1_row_prefetch<3><<<(N * 32 + 255) / 256, 256, 0, s>>>(dq, ds, dx, dy, K, N), err_of(dy));
        rows.push_back({"V2 2 warps/row", graph_us([&](cudaStream_t st) { v2_split_warps<2><<<(N + 3) / 4, 256, 0, st>>>(dq, ds, dx, dy, K, N); }, s),
                        regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) v2_split_warps<2>), 0});
        rows.back().err = (v2_split_warps<2><<<(N + 3) / 4, 256, 0, s>>>(dq, ds, dx, dy, K, N), err_of(dy));
        rows.push_back({"V2 4 warps/row", graph_us([&](cudaStream_t st) { v2_split_warps<4><<<(N + 1) / 2, 256, 0, st>>>(dq, ds, dx, dy, K, N); }, s),
                        regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) v2_split_warps<4>), 0});
        rows.back().err = (v2_split_warps<4><<<(N + 1) / 2, 256, 0, s>>>(dq, ds, dx, dy, K, N), err_of(dy));
        rows.push_back({"V3 splitK=2 + reduce", graph_us([&](cudaStream_t st) {
                            v3_splitk_part<2><<<dim3((N * 32 + 255) / 256, 2), 256, 0, st>>>(dq, ds, dx, dp, K, N);
                            v3_reduce<2><<<(N + 255) / 256, 256, 0, st>>>(dp, dy, N); }, s), 0, 0});
        rows.back().err = (v3_splitk_part<2><<<dim3((N * 32 + 255) / 256, 2), 256, 0, s>>>(dq, ds, dx, dp, K, N),
                           v3_reduce<2><<<(N + 255) / 256, 256, 0, s>>>(dp, dy, N), err_of(dy));
        rows.push_back({"V3 splitK=4 + reduce", graph_us([&](cudaStream_t st) {
                            v3_splitk_part<4><<<dim3((N * 32 + 255) / 256, 4), 256, 0, st>>>(dq, ds, dx, dp, K, N);
                            v3_reduce<4><<<(N + 255) / 256, 256, 0, st>>>(dp, dy, N); }, s), 0, 0});
        rows.back().err = (v3_splitk_part<4><<<dim3((N * 32 + 255) / 256, 4), 256, 0, s>>>(dq, ds, dx, dp, K, N),
                           v3_reduce<4><<<(N + 255) / 256, 256, 0, s>>>(dp, dy, N), err_of(dy));

        {
            const size_t smem = (size_t) K * sizeof(float);
            rows.push_back({"V8 row/warp, unroll2", graph_us([&](cudaStream_t st) { v8_row_unroll2<<<(N * 32 + 255) / 256, 256, 0, st>>>(dq, ds, dx, dy, K, N); }, s),
                            regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) v8_row_unroll2), 0});
            rows.back().err = (v8_row_unroll2<<<(N * 32 + 255) / 256, 256, 0, s>>>(dq, ds, dx, dy, K, N), err_of(dy));
            rows.push_back({"V0 rows2, 128thr", graph_us([&](cudaStream_t st) { v0_rows2<<<(N + 7) / 8, 128, 0, st>>>(dq, ds, dx, dy, K, N); }, s),
                            regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) v0_rows2), 0});
            rows.back().err = (v0_rows2<<<(N + 7) / 8, 128, 0, s>>>(dq, ds, dx, dy, K, N), err_of(dy));
            rows.push_back({"V9 rows2 smem x", graph_us([&](cudaStream_t st) { v9_rows2_smemx<<<(N + 15) / 16, 256, smem, st>>>(dq, ds, dx, dy, K, N); }, s),
                            regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) v9_rows2_smemx), 0});
            rows.back().err = (v9_rows2_smemx<<<(N + 15) / 16, 256, smem, s>>>(dq, ds, dx, dy, K, N), err_of(dy));
        }
        // sweep: rows per warp R in {1,2,4} x block threads T in {64,128,256}
        {
            auto add_cfg = [&](const char * nm, int R, int T) {
                const int warps_needed = (N + R - 1) / R;
                const int blocks = (warps_needed * 32 + T - 1) / T;
                auto launch = [&](cudaStream_t st) {
                    if (R == 1) q8_tpl<1><<<blocks, T, 0, st>>>(dq, ds, dx, dy, K, N);
                    else if (R == 2) q8_tpl<2><<<blocks, T, 0, st>>>(dq, ds, dx, dy, K, N);
                    else q8_tpl<4><<<blocks, T, 0, st>>>(dq, ds, dx, dy, K, N);
                };
                rows.push_back({nm, graph_us(launch, s), 0, 0});
                launch(s);
                rows.back().err = err_of(dy);
                if (R == 1) rows.back().r = regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) q8_tpl<1>);
                else if (R == 2) rows.back().r = regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) q8_tpl<2>);
                else rows.back().r = regs((void (*)(const int8_t *, const __half *, const float *, float *, int, int)) q8_tpl<4>);
            };
            static char names[9][32];
            int k = 0;
            for (int R : {1, 2, 4})
                for (int T : {64, 128, 256}) {
                    std::snprintf(names[k], 32, "R%d T%d", R, T);
                    add_cfg(names[k], R, T);
                    ++k;
                }
        }
        std::printf("\n== %s  (weight %.2f MB, calls/token %d)\n", sh.name, bytes / 1e6, sh.calls);
        std::printf("%-22s %10s %10s %10s %12s\n", "variant", "us", "GB/s", "regs", "rel.err");
        for (auto & r : rows)
            std::printf("%-22s %10.2f %10.1f %10d %12.2e\n", r.name, r.us, bytes / (r.us * 1e-6) / 1e9, r.r, r.err);

        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dy); cudaFree(dp);
    }
    return 0;
}
